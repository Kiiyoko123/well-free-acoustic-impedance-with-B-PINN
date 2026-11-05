# %% [markdown]
# # Second window: simpler geology (Marmousi2, 2D)
#
# Robustness check of the first results (window A: x 5-12 km, complex faulted zone). Here **window B** is used: x 1-4.5 km, TWT 0.9-1.9 s, gently dipping layers. The same settings and the same tests are run, and both windows are reported.
#
# Runs (seed 0, MAP and VI, ensemble skipped, about 10-15 min each, 5 runs plus the baseline, about 1-1.5 hours): baseline, E3 and E4 as below.
#
# **E3**: the background (compaction trend) is wrong: shifted by +0.10 and -0.10 in ln Zp (about 10% impedance), and trend slope x1.3.
# **E4**: the inversion assumes the wrong wavelet: data use 25 Hz, the inversion assumes 20 Hz or 30 Hz.
#
# Compares MAP and VI (ensemble skipped, it halves the run time) against the correct-background, correct-wavelet case.
#
# **How to run**
# 1. Put the two unzipped Marmousi2 `.segy` files in the `data/` folder (see README, "Data").
# 2. Check the names in the config cell (`VP_FILE`, `RHO_FILE`).
# 3. Run the cells in order (or `python window_b_e2_e3_e4.py`). Total about 1-1.5 hours on a CPU. Results are saved to `outputs/window_b/e34_summary.json` after every run.
# 4. The final table is printed at the end.

# %%
# Setup: shared code, paths, and a separate output folder (everything this script writes goes to outputs/window_b/)
import os
from pathlib import Path

from bpinn_core import *   # forward model, CNN, MAP/VI, metrics, run(), preview_model()

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # running cell by cell in the VS Code interactive window
    ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "window_b"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUT)              # relative output names below (figures, json, npz) land in OUT

# %% [markdown]
# ## Configuration

# %%
QUICK = False            # True = smoke test (rough numbers, a few minutes)
VP_FILE  = str(DATA / "MODEL_P-WAVE_VELOCITY_1.25m.segy")   # Marmousi2 Vp (SEG-Y), see README
RHO_FILE = str(DATA / "MODEL_DENSITY_1.25m.segy")
X0, X1   = 1000, 4500      # section start/end (m)
T_START  = 0.9             # window start TWT (s); window is 1.0 s long
SEED     = 0               # seed 0 matches the E2 reference numbers; other seeds also run a fresh baseline
RUN_BASELINE = True        # True = also rerun the unperturbed case in this script (+12 min)
MODEL = "standin" if QUICK else "real"

assert VP_FILE or QUICK, "set VP_FILE to your Marmousi2 Vp file (otherwise this would run on the stand-in model)"
BASE = ["--model", MODEL, "--seed", str(SEED), "--x0", str(X0), "--x1", str(X1), "--t_start", str(T_START), "--skip_ens"]
if VP_FILE and not QUICK:   # QUICK = stand-in model, no SEG-Y needed
    BASE += ["--vp_file", VP_FILE, "--dz", "1.25", "--dx", "1.25"]
    if RHO_FILE:
        BASE += ["--rho_file", RHO_FILE]
if QUICK:
    BASE += ["--quick"]
preview_model(get_args(BASE + ["--out", "e34_preview"]))   # red box = window used by every run below

# %% [markdown]
# ## Run E3 and E4 (about 1 hour)

# %%
import json, time
EXPS = {
    "E3a_shift_p10":    ["--bg_shift", "0.10"],
    "E3b_shift_m10":    ["--bg_shift", "-0.10"],
    "E3c_slope_x1.3":   ["--bg_slope_scale", "1.3"],
    "E4a_wavelet_20Hz": ["--fm_inv", "20"],
    "E4b_wavelet_30Hz": ["--fm_inv", "30"],
}
RUN = list(EXPS)           # e.g. RUN = ["E3a_shift_p10"] to run only one
# E2 seed 0, correct background and wavelet (from the earlier E2 run, same settings)
REF = {"Background": dict(corr=0.595, mae=0.873, corr_hp=-0.021), "MAP": dict(corr=0.689, mae=0.827, corr_hp=0.793),
       "VI": dict(corr=0.672, mae=0.875, corr_hp=0.757, picp90=0.75, width90=2.68)}
need_baseline = RUN_BASELINE or QUICK or SEED != 0
todo = (["baseline"] if need_baseline else []) + RUN
summary = {}
if not need_baseline:
    summary["E2 reference"] = REF

def show(summary):
    print(f"{'experiment':18s} | {'BG corr':>7s} {'BG MAE':>6s} | {'MAP corr':>8s} {'MAE':>6s} {'hp':>5s} | {'VI corr':>7s} {'MAE':>6s} {'hp':>5s} {'PICP90':>6s} {'width':>6s}")
    for k, r in summary.items():
        print(f"{k:18s} | {r['Background']['corr']:7.3f} {r['Background']['mae']:6.3f} | "
              f"{r['MAP']['corr']:8.3f} {r['MAP']['mae']:6.3f} {r['MAP']['corr_hp']:5.2f} | "
              f"{r['VI']['corr']:7.3f} {r['VI']['mae']:6.3f} {r['VI']['corr_hp']:5.2f} {r['VI']['picp90']:6.2f} {r['VI']['width90']:6.2f}")

for name in todo:
    t0 = time.time()
    print("=" * 20, name, "=" * 20)
    extra = [] if name == "baseline" else EXPS[name]
    r = run(get_args(BASE + extra + ["--out", "e34_" + name]))
    summary[name] = {k: r[k] for k in ["Background", "MAP", "VI"]}
    json.dump({k: {m: {x: y for x, y in v.items() if x != "coverage" and not x.startswith("elbo")} for m, v in d.items()} for k, d in summary.items()},
              open("e34_summary.json", "w"), indent=1)
    print(f"[done {name} in {time.time() - t0:.0f}s]")
print()
show(summary)

# %% [markdown]
# ## How to read the table
# - **corr_hp**: correlation of the high-frequency part (the detail the seismic can resolve). If it stays high while MAE rises, the background error only shifts the result.
# - **PICP90**: ideal 0.90. If it drops when the background is wrong, the uncertainty is overconfident (dangerous). If the width grows instead, the interval is honest about the extra error.
# - **E4**: if the wavelet is wrong, the detail (corr_hp) is expected to suffer more than the background.
