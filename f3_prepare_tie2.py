import glob
import numpy as np
import segyio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

BASE = r"C:\Users\User\Downloads\F3_Demo_2023\F3_Demo_2023\Rawdata"
SGY = BASE + r"\Seismic_data.sgy"
WDIR = BASE + r"\Well_data"
WELL, IL, XL = "F03-4", 442, 1007
T0_GUESS = 0.0
WIN = (0.50, 1.50)
NTR = 120
DT = 0.004


def parse_las(path):
    names, rows, sec = [], [], None
    with open(path, errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("~"):
                sec = s[1].upper()
                continue
            if not s or s.startswith("#"):
                continue
            if sec == "C":
                names.append(s.split(".")[0].strip())
            elif sec == "A":
                rows.append([float(x) for x in s.split()])
    a = np.array(rows)
    a[np.isclose(a, -999.25)] = np.nan
    out = {}
    for i, n in enumerate(names):
        out.setdefault(n, a[:, i])
    return out


def ricker(fm, dt=DT, half=0.12):
    t = np.arange(-half, half + dt / 2, dt)
    a = (np.pi * fm * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def time_depth(z, dt_us_m, z_cs, t_cs, t0):
    ok = ~np.isnan(dt_us_m)
    z, s = z[ok], dt_us_m[ok] * 1e-6
    tw = 2 * np.concatenate([[0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(z))])
    tw = tw - np.interp(z_cs[0], z, tw)
    drift = t_cs - np.interp(z_cs, z, tw)
    t_rel = tw + np.interp(z, z_cs, drift)
    return z, t_rel + t0


def log_to_time(z_t, t_abs, zp, t_grid):
    fine = np.arange(t_grid[0] - DT / 2, t_grid[-1] + DT / 2, DT / 8)
    lz = np.interp(fine, t_abs, np.log(zp), left=np.nan, right=np.nan)
    n = (len(fine) // 8) * 8
    m = np.nanmean(lz[:n].reshape(-1, 8), axis=1)
    return m[: len(t_grid)]


BAND = (8.0, 60.0)


def bp(x):
    b, a = butter(4, BAND, btype="band", fs=1.0 / DT)
    return filtfilt(b, a, np.nan_to_num(x))


def wavelet(fm, ph):
    w = ricker(fm)
    h = np.imag(hilbert(w))
    p = np.deg2rad(ph)
    return np.cos(p) * w - np.sin(p) * h


def tie(seis, m, i0, i1, fms, phs, lags):
    r = np.zeros_like(m)
    r[1:] = 0.5 * np.diff(m)
    r = np.nan_to_num(r)
    sf = bp(seis)
    best = dict(c=0.0)
    grid = np.zeros((len(fms), len(lags)))
    for a, fm in enumerate(fms):
        for ph in phs:
            syn = bp(np.convolve(r, wavelet(fm, ph), mode="same"))
            for b, k in enumerate(lags):
                c = np.corrcoef(sf[i0:i1], syn[i0 - k:i1 - k])[0, 1]
                if abs(c) > abs(grid[a, b]):
                    grid[a, b] = c
                if abs(c) > abs(best["c"]):
                    best = dict(c=c, fm=fm, ph=ph, k=k, syn=syn)
    best["c0"] = np.corrcoef(sf[i0:i1], best["syn"][i0:i1])[0, 1]
    return best, grid, r, sf


def main():
    las = glob.glob(WDIR + r"\**\%s_logs.las" % WELL, recursive=True)[0]
    L = parse_las(las)
    z, ai, dtlog = L["DEPTH"], L["AI"], L["DT"]
    cs = np.loadtxt(glob.glob(WDIR + r"\**\Checkshot\%s_TD.txt" % WELL, recursive=True)[0])
    cs = cs[np.argsort(cs[:, 0])]
    _, u = np.unique(cs[:, 0], return_index=True)
    cs = cs[u]
    ok = ~np.isnan(ai) & ~np.isnan(dtlog)
    z, ai, dtlog = z[ok], ai[ok], dtlog[ok]
    zp = ai / 1e6
    zt, t_abs = time_depth(z, dtlog, cs[:, 0], cs[:, 1], T0_GUESS)
    with segyio.open(SGY, ignore_geometry=True) as f:
        il, xl = f.attributes(189)[:], f.attributes(193)[:]
        t_all = np.array(f.samples, float) / 1000.0
        near = [int(np.where((il == IL) & (xl == XL + d))[0][0]) for d in (-1, 0, 1)]
        seis = np.mean([f.trace[i] for i in near], axis=0).astype(float)
        cols = np.arange(XL - NTR // 2, XL + NTR // 2)
        tri = [int(np.where((il == IL) & (xl == c))[0][0]) for c in cols]
        sec = np.array([f.trace[i] for i in tri]).astype(float)
    m = log_to_time(zt, t_abs, zp, t_all)
    i0, i1 = int(round(WIN[0] / DT)) - 1, int(round(WIN[1] / DT)) - 1
    fms = np.arange(15, 61, 5)
    phs = np.arange(-90, 91, 30)
    lags = np.arange(-25, 26)
    best, grid, r, sf = tie(seis, m, i0, i1, fms, phs, lags)
    c, fm, ph, k, syn = best["c"], best["fm"], best["ph"], best["k"], best["syn"]
    sgn = 1.0 if c > 0 else -1.0
    m_al = np.full_like(m, np.nan)
    src = np.arange(len(m)) - k
    okk = (src >= 0) & (src < len(m))
    m_al[okk] = m[src[okk]]
    zp_al = np.exp(m_al)
    raw = np.convolve(np.nan_to_num(np.r_[0, 0.5 * np.diff(m)]), wavelet(fm, ph), mode="same")
    amp = float(np.std(raw[i0 - k:i1 - k]))
    print("well", WELL, "trace il/xl", IL, XL, " band %s Hz" % (BAND,))
    print("best corr %.3f  Ricker %d Hz  phase %d deg  shift %d samples = %.0f ms  polarity %+d" % (c, fm, ph, k, k * DT * 1000, sgn))
    print("corr at zero shift (same wavelet): %.3f" % best["c0"])
    print("synthetic std in window (amplitude scale for the data) %.4f ; seismic std %.1f" % (amp, np.std(seis[i0:i1])))
    print("well Zp in window: %.2f .. %.2f MRayl (mean %.2f)" % (np.nanmin(zp_al[i0:i1]), np.nanmax(zp_al[i0:i1]), np.nanmean(zp_al[i0:i1])))
    print("coverage of Zp in window: %.0f%%" % (100 * np.mean(~np.isnan(zp_al[i0:i1]))))
    np.savez_compressed("f3_%s_prepared.npz" % WELL, t=t_all[i0:i1], zp_well=zp_al[i0:i1], sec=sec[:, i0:i1] * sgn,
                        trace_well=seis[i0:i1] * sgn, fm=fm, phase=ph, shift_samples=k, corr=c, amp=amp, il=IL, xl_cols=cols)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(sf[i0:i1] / np.std(sf[i0:i1]) * sgn, t_all[i0:i1], "k", lw=1, label="seismic (band-passed)")
    ax[0].plot(syn[i0 - k:i1 - k] / np.std(syn[i0 - k:i1 - k]), t_all[i0:i1], "r", lw=1, label="synthetic %d Hz, %d deg" % (fm, ph))
    ax[0].invert_yaxis(); ax[0].legend(); ax[0].set_title("tie, corr %.2f, shift %.0f ms" % (c, k * DT * 1000)); ax[0].set_ylabel("TWT (s)")
    ax[1].plot(zp_al[i0:i1], t_all[i0:i1], "b"); ax[1].invert_yaxis(); ax[1].set_title("well Zp (MRayl) in time")
    ax[2].imshow(grid, aspect="auto", origin="lower", extent=[lags[0] * DT * 1000, lags[-1] * DT * 1000, fms[0], fms[-1]], cmap="RdBu_r", vmin=-1, vmax=1)
    ax[2].set_xlabel("shift (ms)"); ax[2].set_ylabel("Ricker peak (Hz)"); ax[2].set_title("correlation map (best over phase)")
    fig.tight_layout(); fig.savefig("f3_tie_%s.png" % WELL, dpi=130)
    plt.show()


if __name__ == "__main__":
    main()




