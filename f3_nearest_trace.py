import numpy as np
import segyio

p = r"C:\Users\User\Downloads\F3_Demo_2023\F3_Demo_2023\Rawdata\Seismic_data.sgy"
wells = {"F02-1": (606554, 6080126), "F03-2": (619101, 6089491),
         "F03-4": (623256, 6082586), "F06-1": (607903, 6077213)}
with segyio.open(p, ignore_geometry=True) as f:
    x = f.attributes(181)[:].astype(float) / 10.0
    y = f.attributes(185)[:].astype(float) / 10.0
    il = f.attributes(189)[:]
    xl = f.attributes(193)[:]
    for n, (wx, wy) in wells.items():
        d = np.hypot(x - wx, y - wy)
        i = int(d.argmin())
        tr = f.trace[i]
        print(n, "trace", i, "inline", int(il[i]), "xline", int(xl[i]),
              "dist(m)", round(float(d[i]), 1), "trace std", round(float(np.std(tr)), 1))
