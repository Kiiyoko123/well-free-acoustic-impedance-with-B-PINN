# %% [markdown]
# # Field line M10SJI_6001: final runs, four windows (thesis only)
#
# **Thesis use only. Do not use any result, number or figure from this script in the IPA Convex abstract or any IPA material.**
#
# Applies the B-PINN (same network, forward model and variational inference as the Marmousi2 tests) to two windows of the 2D post-stack line, read directly from Google Drive.
#
# What is assumed (state each one in the thesis):
# - **Background:** compaction trend V(z) = V0 + k z measured below the picked seabed, converted to two-way time, with the Rock Physics Handbook density relation. V0 and k are assumed (no velocity data), so absolute impedance follows this assumption (Marmousi2 E3).
# - **Wavelet:** zero-phase Ricker. The peak frequency is set per window from the spectrum measured inside that window (W1 37 Hz, W2 28 Hz); the dominant frequency falls with depth, and the whole-line spectrum (46 Hz) overestimates it.
# - **Amplitude scale:** the seismic is rescaled so that its standard deviation equals an assumed value (the data may have been gain-balanced). The size of the impedance contrasts follows this assumption; the layering does not.
# - **No ground truth:** results are checked for plausibility and stability, not accuracy.
#
# Four windows, all with the same settings and a Ricker wavelet of **35 Hz**, chosen from the residual scan (the minimum was about 32 Hz in W1 and 33 Hz in W2, and the range 32-38 Hz was about equally good):
# - **W1** (km 40-52) and **W2** (km 108-120): the two windows used so far, now at 35 Hz, with wavelet sensitivity at 30 and 40 Hz.
# - **W3** (km 55-65): a more complex zone with a dipping seabed and stronger structure, added on purpose so the choice of windows does not favour easy data.
# - **W4**: a window drawn at random (fixed seed, printed) from the usable part of the line (km 5-150), not looked at beforehand.
#
# W3 and W4 are run once each. Every window starts below its own seabed (window start = deepest seabed pick + 0.15 s). Results are saved as .png, .json and .npz (posterior mean and intervals) for the lithology and porosity step. Eight runs, about 1.5 hours in total.
#
# **How to run:** point `FIELD_SGY` (environment variable) or `FIELD_FILE` in the config cell to the SEG-Y file on your own machine (the raw file is not in this repository), then run the cells in order (or `python field_line_four_windows.py`). About 1.5 hours on a CPU.

# %%
# Setup: shared code, paths, and a separate output folder (everything this script writes goes to outputs/field_line/)
import os
from pathlib import Path

from bpinn_core import *   # forward model, CNN, MAP/VI, metrics, run(), preview_model()

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # running cell by cell in the VS Code interactive window
    ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "field_line"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUT)              # relative output names below (figures, json, npz) land in OUT

# %% [markdown]
# ## Field-data functions (do not edit)

# %%
import json, time
import numpy as np, torch
import matplotlib.pyplot as plt
import segyio
from scipy.ndimage import gaussian_filter1d

DT = 0.004
# Table 1 of the notes (provisional analogue windows, MRayl)
TABLE1 = {"carbonate": (7.7, 14.4), "sandstone": (5.3, 11.1), "shale": (5.6, 11.7), "any analogue": (5.3, 14.4)}


def read_line_geometry(path):
    """Distance along the line (km) for every trace, plus the number of traces."""
    with segyio.open(path, ignore_geometry=True) as s:
        N = s.tracecount
        x = np.array([s.header[i][segyio.TraceField.CDP_X] for i in range(N)], float) / 100.0
        y = np.array([s.header[i][segyio.TraceField.CDP_Y] for i in range(N)], float) / 100.0
    dist = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] / 1000.0
    return dist


def pick_seabed(tr, thr=1e-3):
    """First sample whose |amplitude| exceeds thr * max|amplitude| (the water column is muted)."""
    a = np.abs(tr)
    return int(np.argmax(a > thr * a.max()))


def load_window(path, dist, x0, x1, t0, ntr=120, n=250):
    """Read ntr traces between x0 and x1 (km along the line) and samples t0 .. t0 + n*DT (s)."""
    sel = np.where((dist >= x0) & (dist <= x1))[0]
    assert len(sel) > 10, f"no traces between {x0} and {x1} km (line is {dist[-1]:.1f} km long)"
    idx = sel[np.linspace(0, len(sel) - 1, ntr).astype(int)]
    i0 = int(round(t0 / DT))
    d, tsb = [], []
    with segyio.open(path, ignore_geometry=True) as s:
        assert abs(segyio.tools.dt(s) / 1e6 - DT) < 1e-9, "sampling is not 4 ms"
        for i in idx:
            tr = np.array(s.trace.raw[int(i)], dtype=float)
            tsb.append(pick_seabed(tr) * DT)
            d.append(tr[i0:i0 + n])
    d, tsb = np.array(d), np.array(tsb)
    assert d.shape[1] == n, "window runs past the end of the traces"
    tau = (t0 + np.arange(n) * DT)[None, :] - tsb[:, None]         # TWT below the seabed (s)
    return d, tsb, tau, dist[idx]


def compaction_background(tau, V0, k, dens=(1.70, 0.263)):
    """ln Zp from V(z) = V0 + k z (V km/s, z km, k 1/s) in two-way time below the seabed.
    V(tau) = V0 exp(k tau / 2);  rho = d V^f (Rock Physics Handbook, mean of the sandstone and shale fits);  Zp in MRayl."""
    V = V0 * np.exp(k * np.clip(tau, 0, None) / 2.0)
    dd, ff = dens
    return np.log(dd) + (1.0 + ff) * np.log(V)


def spectrum_peak(d, fmax=100.0):
    w = np.hanning(d.shape[1])
    sp = np.abs(np.fft.rfft(d * w, axis=1)).mean(0)
    fr = np.fft.rfftfreq(d.shape[1], DT)
    sp = gaussian_filter1d(sp, 2)
    ok = (fr > 10) & (fr < fmax)
    return float(fr[ok][sp[ok].argmax()])


def run_field(d_raw, tau, name, out, V0=1.7, k=0.6, fm=45.0, amp=0.04, sigma_d_frac=0.10,
              ch=8, ksz=11, dil="1,1,1", sigma_p=0.1, map_steps=2000, vi_steps=3000, vi_mc=4, n_post=100,
              seed=0, show=True, v0_rng=(1.5, 1.9), k_rng=(0.4, 0.8)):
    n = d_raw.shape[1]
    torch.manual_seed(seed)
    d = d_raw / d_raw.std() * amp                       # amplitude scale is an ASSUMPTION (amp = std of the clean data)
    m_lf = compaction_background(tau, V0, k)
    G = build_G(n, DT, fm)
    d_t = torch.tensor(d)
    sd_d = float(d_t.std())
    sigma_d = sigma_d_frac * sd_d
    net = CNN1D(ch, ksz, [int(v) for v in dil.split(",")])
    x_in = (d_t / sd_d)[:, None, :]
    pb = Problem(net, G, d_t, torch.tensor(m_lf), x_in, sigma_d, sigma_p, unc=None, fm0=fm, dt=DT, n=n)
    gp = torch.Generator().manual_seed(seed + 99)
    pr = torch.stack([pb.delta_m(torch.randn(net.P, generator=gp) * sigma_p) for _ in range(30)])
    prior_sd = float(pr.std())
    print(f"[bg   ] V0={V0} km/s, k={k} 1/s -> Zp {np.exp(m_lf).min():.2f}-{np.exp(m_lf).max():.2f} MRayl (background)")
    print(f"[data ] window {d.shape[0]} traces x {n} samples, wavelet {fm:.0f} Hz, std(d)={sd_d:.4f}, sigma_d={sigma_d:.4f}")
    t0 = time.time()
    th0 = torch.randn(pb.P) * pb.sp_vec
    th_map = fit_map(pb, th0, map_steps, prior=True)
    print(f"[MAP  ] done ({time.time() - t0:.0f}s)")
    mu, s, hist = fit_vi(pb, vi_steps, vi_mc, seed=seed)
    gd = torch.Generator().manual_seed(seed + 1)
    lz = np.stack([pb.zp_log(mu + s * torch.randn(pb.P, generator=gd)).detach().numpy() for _ in range(n_post)])
    z = np.exp(lz)
    zm, lo, hi = z.mean(0), *np.quantile(z, [0.05, 0.95], axis=0)
    # Background uncertainty: the low-frequency part is outside the data band (null space of G), so the data do not
    # constrain it. Each posterior draw gets its own (V0, k) from the stated ranges: lnZp_j + [bg(V0_j, k_j) - bg(V0, k)].
    rg = np.random.default_rng(seed + 7)
    lzm_ = np.stack([lz[j] + compaction_background(tau, rg.uniform(*v0_rng), rg.uniform(*k_rng)) - m_lf for j in range(lz.shape[0])])
    zq = np.exp(lzm_)
    lo_m, hi_m = np.quantile(zq, [0.05, 0.95], axis=0)
    lzm = lz.mean(0)
    syn = lzm @ G.numpy().T
    fitc = float(np.corrcoef(syn.ravel(), d.ravel())[0, 1])
    rr = d - syn
    hp = lambda A: A - gaussian_filter1d(A, 10, axis=1)                    # noqa
    post_sd = float(lz.std(0).mean())
    res = dict(name=name, V0=V0, k=k, fm=fm, amp=amp, fit_corr=fitc, resid_std_over_sigma_d=float(rr.std() / sigma_d),
               mean_lnZp=float(lzm.mean()), post_sd_lnZp=post_sd, prior_pred_sd=prior_sd,
               post_over_prior=post_sd / prior_sd, mean_width90=float((hi - lo).mean()),
               inside={kk: float(np.mean((z >= a) & (z <= b))) for kk, (a, b) in TABLE1.items()},
               inside_marg={kk: float(np.mean((zq >= a) & (zq <= b))) for kk, (a, b) in TABLE1.items()},
               mean_width90_marg=float((hi_m - lo_m).mean()), v0_rng=list(v0_rng), k_rng=list(k_rng),
               zp_mean_range=[float(zm.min()), float(zm.max())])
    print(f"[VI   ] data fit corr={fitc:.3f}  residual std / sigma_d={res['resid_std_over_sigma_d']:.2f}  "
          f"posterior sd(ln Zp)={post_sd:.3f} (prior-predictive {prior_sd:.3f})  ({time.time() - t0:.0f}s)")
    print("[check] share of posterior samples inside the analogue windows of Table 1 (plausibility, not accuracy): "
          + ", ".join(f"{kk} {v:.0%}" for kk, v in res["inside"].items()))
    print(f"[bg   ] interval width incl. background uncertainty (V0 {v0_rng}, k {k_rng}): {res['mean_width90_marg']:.2f} MRayl "
          f"(conditional on the assumed background: {res['mean_width90']:.2f})")
    fig, ax = plt.subplots(2, 4, figsize=(22, 8), sharex=True, sharey=True)
    ext = [0, d.shape[0], tau[:, -1].mean(), tau[:, 0].mean()]
    vmax = np.percentile(np.abs(d), 98)
    panels = [(d, "observed (scaled)", "gray_r", -vmax, vmax), (syn, "synthetic from posterior mean", "gray_r", -vmax, vmax),
              (rr, "residual", "gray_r", -vmax, vmax), (np.exp(m_lf), "background Zp (MRayl)", "viridis", None, None),
              (zm, "posterior mean Zp (MRayl)", "viridis", None, None), (hi - lo, "90% width, background fixed (MRayl)", "magma", None, None),
              (hi_m - lo_m, "90% width incl. background uncertainty", "magma", None, None)]
    ax[1, 3].axis("off")
    for a_, (A, ttl, cm, v0, v1) in zip(ax.ravel()[:7], panels):
        if ttl.startswith("posterior mean") or ttl.startswith("background"):
            v0, v1 = np.percentile(zm, [1, 99])
        im = a_.imshow(A.T, aspect="auto", cmap=cm, vmin=v0, vmax=v1, extent=[0, A.shape[0], ext[2], ext[3]])
        a_.set_title(ttl); plt.colorbar(im, ax=a_)
    for a_ in ax[1, :3]: a_.set_xlabel("trace in window")
    for a_ in ax[:, 0]: a_.set_ylabel("TWT below seabed (s)")
    fig.suptitle(name); fig.tight_layout()
    import os
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/{name}.png", dpi=110)
    if show: plt.show()
    plt.close(fig)
    json.dump(res, open(f"{out}/{name}.json", "w"), indent=1)
    np.savez_compressed(f"{out}/{name}.npz", zp_mean=zm, zp_lo=lo, zp_hi=hi, zp_lo_marg=lo_m, zp_hi_marg=hi_m, lnzp_mean=lzm,
                        bg=np.exp(m_lf), d=d, syn=syn, tau=tau)
    return res, lzm


def auto_t0(path, dist, x0, x1, ntr=120, margin=0.15):
    """Window start = deepest seabed pick in the window + margin (s), on the 4 ms grid."""
    sel = np.where((dist >= x0) & (dist <= x1))[0]
    idx = sel[np.linspace(0, len(sel) - 1, ntr).astype(int)]
    with segyio.open(path, ignore_geometry=True) as s:
        tsb = np.array([pick_seabed(np.array(s.trace.raw[int(i)], float)) * DT for i in idx])
    return float(np.ceil((tsb.max() + margin) / DT) * DT)

# %% [markdown]
# ## Configuration

# %%
QUICK = False              # True = smoke test on a fake line
# The raw SEG-Y is NOT part of the repository. Keep it outside the repo or in data/ (which is git-ignored).
FIELD_FILE = os.environ.get("FIELD_SGY", str(DATA / "M10SJI_6001.sgy"))
assert os.path.exists(FIELD_FILE), f"field SEG-Y not found: {FIELD_FILE} (set FIELD_SGY or edit FIELD_FILE)"
WINDOWS = {'W1': dict(x0=40.0, x1=52.0, t0=2.55, fm=35.0), 'W2': dict(x0=108.0, x1=120.0, t0=4.45, fm=35.0), 'W3': dict(x0=55.0, x1=65.0, t0=None, fm=35.0)}   # x in km along the line, t0 = window start (s TWT); each window is 120 traces x 1.0 s
FM_STEP = 5.0      # wavelet sensitivity: base +/- this many Hz (the base wavelet frequency is set per window in WINDOWS)
V0, K = 1.7, 0.6   # compaction trend at the seabed: V0 (km/s) and k (1/s), assumed
AMP = 0.04         # assumed std of the clean data after rescaling (same scale as the Marmousi2 tests)
steps = dict(map_steps=300, vi_steps=400, vi_mc=2, n_post=20) if QUICK else dict(map_steps=2000, vi_steps=3000, vi_mc=4, n_post=100)
VARIANTS = {
    "base":      dict(),
    "fm_lo":     dict(fm_off=-FM_STEP),
    "fm_hi":     dict(fm_off=+FM_STEP),
}
RANDOM_SEED = 2026
dist = read_line_geometry(FIELD_FILE)
print(f"line length {dist[-1]:.1f} km, {len(dist)} traces")
# W4: random window, drawn once with a fixed seed, not overlapping the other windows
rg = np.random.default_rng(RANDOM_SEED)
lo_x, hi_x = (0.5, dist[-1] - 2.5) if QUICK else (5.0, 150.0)
WW = 2.0 if QUICK else 12.0
for _ in range(500):
    x0r = round(float(rg.uniform(lo_x, hi_x)), 1)
    if all(x0r + WW < w["x0"] - 1 or x0r > w["x1"] + 1 for w in WINDOWS.values()):
        break
WINDOWS["W4"] = dict(x0=x0r, x1=x0r + WW, t0=None, fm=35.0)
for nm, w in WINDOWS.items():
    if w["t0"] is None:
        w["t0"] = auto_t0(FIELD_FILE, dist, w["x0"], w["x1"])
    print(f"{nm}: km {w['x0']:.1f}-{w['x1']:.1f}, window start {w['t0']:.3f} s, wavelet {w['fm']:.0f} Hz")
RUNS = [("W1", "base"), ("W1", "fm_lo"), ("W1", "fm_hi"), ("W2", "base"), ("W2", "fm_lo"), ("W2", "fm_hi"), ("W3", "base"), ("W4", "base")]
if QUICK: RUNS = [("W1", "base"), ("W3", "base"), ("W4", "base")]

# %% [markdown]
# ## Preview: the windows on the section (check them before the long run)

# %%
sub = np.arange(0, len(dist), max(1, len(dist) // 700))
with segyio.open(FIELD_FILE, ignore_geometry=True) as s:
    sec = np.stack([np.array(s.trace.raw[int(i)], float) for i in sub]).T
tmax = min(sec.shape[0], 1500) * DT
v = np.percentile(np.abs(sec[:1500]), 98)
fig, ax = plt.subplots(figsize=(17, 6))
ax.imshow(sec[:1500], aspect="auto", cmap="gray_r", vmin=-v, vmax=v, extent=[0, dist[-1], tmax, 0])
for nm, w in WINDOWS.items():
    ax.add_patch(plt.Rectangle((w["x0"], w["t0"]), w["x1"] - w["x0"], 1.0, fill=False, ec="red", lw=2))
    ax.text(w["x0"], w["t0"] - 0.05, nm, color="red", fontsize=13)
ax.set_xlabel("distance along line (km)"); ax.set_ylabel("TWT (s)"); plt.show()
for nm, w in WINDOWS.items():
    d_, tsb_, tau_, _ = load_window(FIELD_FILE, dist, w["x0"], w["x1"], w["t0"])
    print(f"{nm}: seabed {tsb_.min():.2f}-{tsb_.max():.2f} s, window starts {tau_[:, 0].min():.2f}-{tau_[:, 0].max():.2f} s below the seabed, "
          f"spectral peak {spectrum_peak(d_):.0f} Hz")
    assert tau_[:, 0].min() > 0.05, f"{nm}: the window starts above or at the seabed, move t0 down"

# %% [markdown]
# ## Run

# %%
store, results = {}, {}
for wn, vn in RUNS:
    w = WINDOWS[wn]
    d_raw, tsb, tau, xk = load_window(FIELD_FILE, dist, w["x0"], w["x1"], w["t0"])
    kw = dict(V0=V0, k=K, fm=w["fm"], amp=AMP); kw.update(VARIANTS[vn]); kw.update(steps)
    kw["fm"] = kw["fm"] + kw.pop("fm_off", 0.0)
    print("=" * 20, wn, vn, "=" * 20)
    r, lzm = run_field(d_raw, tau, f"{wn}_{vn}", "field_out", **kw)
    results[(wn, vn)], store[(wn, vn)] = r, lzm
    json.dump({f"{a}_{b}": v for (a, b), v in results.items()}, open("field_summary.json", "w"), indent=1)
print()
hp = lambda A: A - gaussian_filter1d(A, 10, axis=1)       # noqa
print(f"{'run':14s} | {'fit corr':>8s} {'res/sd':>6s} | {'post sd':>7s} {'/prior':>6s} {'width90':>7s} {'incl.bg':>7s} | {'in range':>8s} {'incl.bg':>7s} | vs base: {'detail corr':>11s} {'mean offset':>11s}")
for (wn, vn), r in results.items():
    base = store.get((wn, "base"))
    dc = float(np.corrcoef(hp(store[(wn, vn)]).ravel(), hp(base).ravel())[0, 1]) if base is not None else float("nan")
    off = float(store[(wn, vn)].mean() - base.mean()) if base is not None else float("nan")
    print(f"{wn + '_' + vn:14s} | {r['fit_corr']:8.3f} {r['resid_std_over_sigma_d']:6.2f} | {r['post_sd_lnZp']:7.3f} {r['post_over_prior']:6.2f} "
          f"{r['mean_width90']:7.2f} {r['mean_width90_marg']:7.2f} | {r['inside']['any analogue']:8.0%} {r['inside_marg']['any analogue']:7.0%} | {dc:11.3f} {off:+11.3f}")

# %% [markdown]
# ## How to read the table
# - **fit corr, res/sd**: how well the posterior-mean impedance reproduces the observed trace. `res/sd` is the residual std divided by sigma_d; values far above 1 mean the model cannot explain part of the data (noise, multiples, gain), so the result there is less trustworthy.
# - **post sd / prior**: posterior std of ln Zp divided by the prior-predictive std. Close to 1 means the data barely narrowed the prior, so a narrow interval would reflect the prior rather than the data.
# - **in analogue range**: share of posterior samples inside the provisional analogue windows (Table 1 of the notes). This is plausibility only; those windows come from onshore laboratory relations and may sit above shallow in-situ values.
# - **detail corr (vs base)**: stability of the layering (high-frequency part) against the assumptions. Values near 1 for the background and wavelet variants mean the structure does not depend on them, as in the Marmousi2 tests.
# - **mean offset**: change in the average ln Zp. The background variants are expected to shift it (E3), so absolute impedance should be reported with that range.
