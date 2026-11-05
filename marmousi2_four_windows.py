# %% [markdown]
# # Marmousi2: four windows (same design as the field line)
#
# One script for the synthetic test. It mirrors the field-line study: **four windows**, the same network, the same MAP + VI inversion, the same settings, and the same kind of sensitivity tests.
#
# | Window | Role | Default location |
# |---|---|---|
# | M1 | simple geology, chosen | x 1-4.5 km, TWT 0.9-1.9 s (= window B) |
# | M2 | second chosen window | x 12-16 km, TWT 1.0-2.0 s (**check the preview and move it if it is flat or in the water**) |
# | M3 | complex, faulted | x 5-12 km, TWT 1.3-2.3 s (= window A) |
# | M4 | random (seed 2026), not chosen by eye | drawn by the code |
#
# For every window: baseline (correct background and wavelet). For the windows in `VARIANT_WINDOWS` (default M1 and M3): E3 (background +10%, -10%, slope x1.3) and E4 (wavelet 20 and 30 Hz, data 25 Hz).
#
# Seed 0, ensemble skipped. About 12 min per run: 4 baselines + 10 variants = about 3 hours on a CPU. For a shorter test set `VARIANT_WINDOWS = []` (about 50 min).
#
# **How to run**
# 1. Put the two unzipped Marmousi2 `.segy` files in the `data/` folder (see README, "Data"); check `VP_FILE` and `RHO_FILE` in the config cell.
# 2. Run the config cell, look at the four red boxes, change M2 if needed (and only M2).
# 3. Run the remaining cells (or `python marmousi2_four_windows.py`). Results are saved to `outputs/marmousi2/m4_summary.json` after every run, and the last cell zips everything.
# 4. Send the final table back to the supervisor / put it in the report.

# %% [markdown]
# ## Configuration

# %%
# Setup: shared code, paths, and a separate output folder (everything this script writes goes to outputs/marmousi2/)
import os
from pathlib import Path

from bpinn_core import *   # forward model, CNN, MAP/VI, metrics, run(), preview_model()

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # running cell by cell in the VS Code interactive window
    ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "marmousi2"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUT)              # relative output names below (figures, json, npz) land in OUT

# %%
import numpy as np
QUICK = False            # True = smoke test on the stand-in model (rough numbers, a few minutes)
VP_FILE  = str(DATA / "MODEL_P-WAVE_VELOCITY_1.25m.segy")   # Marmousi2 Vp (SEG-Y), see README
RHO_FILE = str(DATA / "MODEL_DENSITY_1.25m.segy")
SEED = 0                      # seed 0 reproduces the earlier windows A and B
MODEL = "standin" if QUICK else "real"
assert VP_FILE or QUICK, "set VP_FILE to your Marmousi2 Vp file"

# random window M4: seed 2026, width 6 km, start TWT drawn in 0.8-1.6 s (the model is 17 km wide)
_r = np.random.default_rng(2026)
M4_X0 = float(np.round(_r.uniform(0, 17000 - 6000), -1)); M4_T0 = float(np.round(_r.uniform(0.8, 1.6), 2))

WINDOWS = {                  # name: (x0 m, x1 m, TWT start s); every window is 1.0 s long
    "M1_simple":  (1000, 4500, 0.9),
    "M2_chosen":  (12000, 16000, 1.0),
    "M3_complex": (5000, 12000, 1.3),
    "M4_random":  (M4_X0, M4_X0 + 6000, M4_T0),
}
VARIANT_WINDOWS = ["M1_simple", "M3_complex"]    # windows that also get E3 and E4 ([] = baselines only)

def base_args(win):
    x0, x1, t0 = WINDOWS[win]
    b = ["--model", MODEL, "--seed", str(SEED), "--x0", str(x0), "--x1", str(x1), "--t_start", str(t0), "--skip_ens"]
    if VP_FILE and not QUICK:   # QUICK = stand-in model, no SEG-Y needed
        b += ["--vp_file", VP_FILE, "--dz", "1.25", "--dx", "1.25"]
        if RHO_FILE:
            b += ["--rho_file", RHO_FILE]
    if QUICK:
        b += ["--quick"]
    return b

for w, (x0, x1, t0) in WINDOWS.items():
    print(f"{w:11s} x {x0:.0f}-{x1:.0f} m, TWT {t0:.2f}-{t0 + 1.0:.2f} s")
    preview_model(get_args(base_args(w) + ["--out", "m4_preview_" + w]))   # red box = the window

# %% [markdown]
# ## Run (about 3 hours; baselines first, then variants)

# %%
import json, time, os
EXPS = {
    "E3a_shift_p10":    ["--bg_shift", "0.10"],
    "E3b_shift_m10":    ["--bg_shift", "-0.10"],
    "E3c_slope_x1.3":   ["--bg_slope_scale", "1.3"],
    "E4a_wavelet_20Hz": ["--fm_inv", "20"],
    "E4b_wavelet_30Hz": ["--fm_inv", "30"],
}
jobs = [(w, "baseline") for w in WINDOWS] + [(w, e) for w in VARIANT_WINDOWS for e in EXPS]
summary = {}

def show(summary):
    print(f"{'window / run':32s} | {'BG corr':>7s} {'BG MAE':>6s} | {'MAP corr':>8s} {'MAE':>6s} {'hp':>5s} | {'VI corr':>7s} {'MAE':>6s} {'hp':>5s} {'PICP90':>6s} {'width':>6s}")
    for k, r in summary.items():
        print(f"{k:32s} | {r['Background']['corr']:7.3f} {r['Background']['mae']:6.3f} | "
              f"{r['MAP']['corr']:8.3f} {r['MAP']['mae']:6.3f} {r['MAP']['corr_hp']:5.2f} | "
              f"{r['VI']['corr']:7.3f} {r['VI']['mae']:6.3f} {r['VI']['corr_hp']:5.2f} {r['VI']['picp90']:6.2f} {r['VI']['width90']:6.2f}")

for win, name in jobs:
    t0 = time.time()
    key = f"{win}/{name}"
    print("=" * 20, key, "=" * 20)
    extra = [] if name == "baseline" else EXPS[name]
    r = run(get_args(base_args(win) + extra + ["--out", f"m4_{win}_{name}"]))
    summary[key] = {k: r[k] for k in ["Background", "MAP", "VI"]}
    json.dump({k: {m: {x: y for x, y in v.items() if x != "coverage" and not x.startswith("elbo")} for m, v in d.items()} for k, d in summary.items()},
              open("m4_summary.json", "w"), indent=1)
    print(f"[done {key} in {time.time() - t0:.0f}s]")
print()
show(summary)

# %%
import numpy as np, matplotlib.pyplot as plt
WIN = {w: (WINDOWS[w][2], w.replace("_", " ")) for w in WINDOWS}

# 1) gambar lengkap 7 panel tiap window (true, background, seismik, MAP, VI mean, VI std, |error|)
for w in WIN:
    print(w, "->", OUT / f"m4_{w}_baseline" / "e2_sections.png")

# 2) satu gambar ringkasan: true / background / VI mean / VI std untuk 4 window
fig, ax = plt.subplots(4, 4, figsize=(18, 15))
for i, (w, (t0, lab)) in enumerate(WIN.items()):
    z = np.load(f"m4_{w}_baseline/e2_arrays.npz"); x = z["x"] / 1000; n = z["z_true"].shape[1]
    t = (t0 + np.arange(n) * 0.004) * 1000; ext = [x[0], x[-1], t[-1], t[0]]
    v0, v1 = np.percentile(z["z_true"], [1, 99])
    for j, (A, ttl, cm, a0, a1) in enumerate([(z["z_true"], "True Zp (MRayl)", "viridis", v0, v1),
                                              (np.exp(z["m_lf"]), "Background (trend)", "viridis", v0, v1),
                                              (z["z_vi_mean"], "VI posterior mean", "viridis", v0, v1),
                                              (z["z_vi_std"], "VI posterior std (MRayl)", "magma", None, None)]):
        h = ax[i, j].imshow(A.T, aspect="auto", extent=ext, cmap=cm, vmin=a0, vmax=a1)
        ax[i, j].set_title(f"{lab}: {ttl}", fontsize=10); ax[i, j].set_xlabel("x (km)"); ax[i, j].set_ylabel("TWT (ms)")
        plt.colorbar(h, ax=ax[i, j], fraction=0.045)
fig.tight_layout(); fig.savefig("m4_summary_figure.png", dpi=130); plt.show()

# %% [markdown]
# ## Baselines only (the four windows side by side)

# %%
show({k: v for k, v in summary.items() if k.endswith("/baseline")})

# %% [markdown]
# ## Download everything (figures + json)

# %%
import glob
import zipfile
dirs = sorted(d for d in glob.glob("m4_*") if not d.endswith(".zip"))
with zipfile.ZipFile("m4_results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for d in dirs:
        for p in ([Path(d)] if Path(d).is_file() else Path(d).rglob("*")):
            if p.is_file():
                zf.write(p)
print("saved", OUT / "m4_results.zip")


# %% [markdown]
# ## How to read the table
# - **corr_hp**: correlation of the high-frequency part (detail the seismic can resolve). Expected high (about 0.8) in every window and almost unchanged under E3.
# - **BG MAE vs MAP/VI MAE**: shows whether the level follows the background (E3) or the data.
# - **PICP90**: ideal 0.90. Compare windows: calibration is expected to differ between simple and complex geology.
# - **E4**: a wavelet frequency that is too low should hurt corr_hp; too high should hurt much less.
