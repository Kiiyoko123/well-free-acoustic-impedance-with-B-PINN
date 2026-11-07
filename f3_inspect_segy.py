import numpy as np
import segyio

p = r"C:\Users\User\Downloads\F3_Demo_2023\F3_Demo_2023\Rawdata\Seismic_data.sgy"
with segyio.open(p, ignore_geometry=True) as f:
    print(segyio.tools.wrap(f.text[0])[:1200])
    print("traces", f.tracecount, "samples", len(f.samples), "dt(ms)", segyio.tools.dt(f) / 1000, "t0", f.samples[0], "tmax", f.samples[-1])
    for name, b in [("inline(189)", 189), ("xline(193)", 193), ("cdp_x(181)", 181), ("cdp_y(185)", 185), ("scalar(71)", 71)]:
        v = f.attributes(b)[:]
        print(f"{name:12s} min {v.min()} max {v.max()}")
    idx = np.random.default_rng(0).choice(f.tracecount, 2000, replace=False)
    d = np.array([f.trace[int(i)] for i in idx])
    print("amp min", d.min(), "max", d.max(), "std", d.std())
