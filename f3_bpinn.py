import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
from scipy.ndimage import gaussian_filter1d

from bpinn_core import CNN1D, Problem, build_G, fit_map, fit_vi

BASE = r"C:\Users\User\Downloads\F3_Demo_2023\F3_Demo_2023\Rawdata"
SGY = os.environ.get("F3_SGY", BASE + r"\Seismic_data.sgy")
NPZ = "f3_F03-4_prepared.npz"
DT = 0.004
V0, K = 1.7, 0.6
V0_RNG, K_RNG = (1.5, 1.9), (0.4, 0.8)
QUICK = os.environ.get("F3_QUICK") == "1"
STEPS = dict(map_steps=200, vi_steps=300, vi_mc=2, n_post=20) if QUICK else dict(map_steps=2000, vi_steps=3000, vi_mc=4, n_post=100)


def compaction_background(tau, V0, k, dens=(1.70, 0.263)):
    V = V0 * np.exp(k * np.clip(tau, 0, None) / 2.0)
    dd, ff = dens
    return np.log(dd) + (1.0 + ff) * np.log(V)


def hp(lz):
    return lz - gaussian_filter1d(lz, 10)


def lp(lz):
    return gaussian_filter1d(lz, 10)


def trace_metrics(z_post, z_bg, z_well, z_samp, z_samp_bg):
    lw = np.log(z_well)
    out = dict(
        mean_well=float(z_well.mean()), mean_post=float(z_post.mean()), mean_bg=float(z_bg.mean()),
        mae_post=float(np.mean(np.abs(z_post - z_well))), mae_bg=float(np.mean(np.abs(z_bg - z_well))),
        corr=float(np.corrcoef(z_post, z_well)[0, 1]),
        corr_hp=float(np.corrcoef(hp(np.log(z_post)), hp(lw))[0, 1]),
        corr_lp=float(np.corrcoef(lp(np.log(z_post)), lp(lw))[0, 1]),
    )
    for tag, S in (("", z_samp), ("_incl_bg", z_samp_bg)):
        lo, hi = np.quantile(S, [0.05, 0.95], axis=0)
        out["picp90" + tag] = float(np.mean((z_well >= lo) & (z_well <= hi)))
        out["width90" + tag] = float(np.mean(hi - lo))
    return out


def main():
    P = np.load(NPZ)
    t, sec, zp_well, fm, amp, xl_cols, il = P["t"], P["sec"], P["zp_well"], float(P["fm"]), float(P["amp"]), P["xl_cols"], int(P["il"])
    ntr, n = sec.shape
    iw = ntr // 2
    with segyio.open(SGY, ignore_geometry=True) as f:
        ilh, xlh = f.attributes(189)[:], f.attributes(193)[:]
        tsb = []
        for c in xl_cols:
            i = int(np.where((ilh == il) & (xlh == int(c)))[0][0])
            tr = np.abs(np.array(f.trace[i], float))
            tsb.append((7 + int(np.argmax(tr[7:50]))) * DT)
    tsb = np.array(tsb)
    tau = t[None, :] - tsb[:, None]
    print("seabed pick (s): median %.3f  min %.3f  max %.3f" % (np.median(tsb), tsb.min(), tsb.max()))
    print("wavelet %.0f Hz, amplitude scale %.4f, window %.2f-%.2f s, %d traces x %d samples" % (fm, amp, t[0], t[-1], ntr, n))
    torch.manual_seed(0)
    d = sec / sec.std() * amp
    m_lf = compaction_background(tau, V0, K)
    G = build_G(n, DT, fm)
    d_t = torch.tensor(d)
    sd_d = float(d_t.std())
    sigma_d = float(os.environ.get("F3_SD", "0.10")) * sd_d
    net = CNN1D(8, 11, [1, 1, 1])
    x_in = (d_t / sd_d)[:, None, :]
    pb = Problem(net, G, d_t, torch.tensor(m_lf), x_in, sigma_d, float(os.environ.get("F3_SP", "0.1")), unc=None, fm0=fm, dt=DT, n=n)
    t0 = time.time()
    th_map = fit_map(pb, torch.randn(pb.P) * pb.sp_vec, STEPS["map_steps"], prior=True)
    z_map = pb.zp(th_map).detach().numpy()
    print("[MAP] done (%.0f s)" % (time.time() - t0))
    mu, s, hist = fit_vi(pb, STEPS["vi_steps"], STEPS["vi_mc"], seed=0)
    gd = torch.Generator().manual_seed(1)
    lz = np.stack([pb.zp_log(mu + s * torch.randn(pb.P, generator=gd)).detach().numpy() for _ in range(STEPS["n_post"])])
    z = np.exp(lz)
    rg = np.random.default_rng(7)
    zq = np.exp(np.stack([lz[j] + compaction_background(tau, rg.uniform(*V0_RNG), rg.uniform(*K_RNG)) - m_lf for j in range(lz.shape[0])]))
    zm = z.mean(0)
    syn = lz.mean(0) @ G.numpy().T
    fit_corr = float(np.corrcoef(syn.ravel(), d.ravel())[0, 1])
    res_sd = float((d - syn).std() / sigma_d)
    print("[VI ] data fit corr=%.3f  residual std / sigma_d=%.2f  (%.0f s)" % (fit_corr, res_sd, time.time() - t0))
    zb = np.exp(m_lf)
    mv = trace_metrics(zm[iw], zb[iw], zp_well, z[:, iw], zq[:, iw])
    mm = trace_metrics(z_map[iw], zb[iw], zp_well, z[:, iw], zq[:, iw])
    print("\nAt the well (trace %d, crossline %d), %d samples:" % (iw, int(xl_cols[iw]), n))
    print("  mean Zp (MRayl):   well %.2f | background %.2f | VI posterior mean %.2f | MAP %.2f" % (mv["mean_well"], mv["mean_bg"], mv["mean_post"], mm["mean_post"]))
    print("  MAE (MRayl):       background %.2f | VI %.2f | MAP %.2f" % (mv["mae_bg"], mv["mae_post"], mm["mae_post"]))
    print("  corr, all / detail (hp) / trend (lp): VI %.2f / %.2f / %.2f ; MAP %.2f / %.2f / %.2f" % (mv["corr"], mv["corr_hp"], mv["corr_lp"], mm["corr"], mm["corr_hp"], mm["corr_lp"]))
    print("  PICP90 / width90 (MRayl): VI %.2f / %.2f ; incl. background uncertainty %.2f / %.2f" % (mv["picp90"], mv["width90"], mv["picp90_incl_bg"], mv["width90_incl_bg"]))
    res = dict(vi=mv, map=mm, fit_corr=fit_corr, resid_over_sigma_d=res_sd, fm=fm, amp=amp, V0=V0, k=K, seabed_median=float(np.median(tsb)))
    json.dump(res, open("f3_result.json", "w"), indent=1)
    np.savez_compressed("f3_result.npz", t=t, zp_mean=zm, zp_bg=zb, zp_well=zp_well, lo=np.quantile(z, 0.05, axis=0), hi=np.quantile(z, 0.95, axis=0),
                        lo_m=np.quantile(zq, 0.05, axis=0), hi_m=np.quantile(zq, 0.95, axis=0), z_map=z_map)
    lo, hi = np.quantile(z[:, iw], [0.05, 0.95], axis=0)
    lo_m, hi_m = np.quantile(zq[:, iw], [0.05, 0.95], axis=0)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12.5))
    ext = [0, ntr, t[-1], t[0]]
    vm = np.percentile(np.abs(d), 98)
    ax[0, 0].imshow(d.T, aspect="auto", cmap="gray_r", vmin=-vm, vmax=vm, extent=ext); ax[0, 0].set_title("F3 seismic (scaled)")
    lim = np.percentile(zm, [1, 99])
    im = ax[0, 1].imshow(zm.T, aspect="auto", cmap="viridis", vmin=lim[0], vmax=lim[1], extent=ext); ax[0, 1].set_title("posterior mean Zp (MRayl)")
    plt.colorbar(im, ax=ax[0, 1])
    im = ax[0, 2].imshow(zb.T, aspect="auto", cmap="viridis", vmin=lim[0], vmax=lim[1], extent=ext); ax[0, 2].set_title("background Zp (MRayl)")
    plt.colorbar(im, ax=ax[0, 2])
    for a_ in ax[0]:
        a_.axvline(iw, color="r", lw=0.8); a_.set_xlabel("trace in window"); a_.set_ylabel("TWT (s)")
    ax[1, 0].fill_betweenx(t, lo_m, hi_m, color="orange", alpha=0.25, label="90% incl. background uncertainty")
    ax[1, 0].fill_betweenx(t, lo, hi, color="tab:red", alpha=0.3, label="90% interval (background fixed)")
    ax[1, 0].plot(zm[iw], t, "r", lw=1.2, label="posterior mean")
    ax[1, 0].plot(zb[iw], t, "b--", lw=1.2, label="background")
    ax[1, 0].plot(zp_well, t, "k", lw=1.2, label="well F03-4 (log)")
    ax[1, 0].invert_yaxis(); ax[1, 0].legend(fontsize=8); ax[1, 0].set_xlabel("Zp (MRayl)"); ax[1, 0].set_ylabel("TWT (s)"); ax[1, 0].set_title("at the well")
    ax[1, 1].plot(hp(np.log(zp_well)), t, "k", lw=1, label="well")
    ax[1, 1].plot(hp(np.log(zm[iw])), t, "r", lw=1, label="VI mean")
    ax[1, 1].invert_yaxis(); ax[1, 1].legend(); ax[1, 1].set_title("detail (high-pass ln Zp), corr %.2f" % mv["corr_hp"])
    ax[1, 2].plot(lp(np.log(zp_well)), t, "k", lw=1, label="well")
    ax[1, 2].plot(lp(np.log(zm[iw])), t, "r", lw=1, label="VI mean")
    ax[1, 2].plot(lp(np.log(zb[iw])), t, "b--", lw=1, label="background")
    ax[1, 2].invert_yaxis(); ax[1, 2].legend(); ax[1, 2].set_title("trend (low-pass ln Zp)")
    fig.tight_layout(h_pad=4.0); fig.savefig("f3_result.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()



