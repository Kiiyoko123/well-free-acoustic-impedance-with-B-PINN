import glob
import numpy as np

root = r"C:\Users\User\Downloads\F3_Demo_2023\F3_Demo_2023\Rawdata\Well_data"

for f in sorted(glob.glob(root + r"\**\*.las", recursive=True)):
    names, rows, sec = [], [], None
    with open(f, errors="ignore") as fh:
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
    print("\n", f.split("\\")[-1], a.shape)
    for i, n in enumerate(names):
        v = a[:, i]
        ok = ~np.isnan(v)
        if ok.any():
            d = a[ok, 0]
            print(f"  col{i+1} {n:8s} valid {ok.sum():6d}  depth {d.min():7.1f}-{d.max():7.1f} m  range {np.nanmin(v):8.2f} .. {np.nanmax(v):8.2f}")
        else:
            print(f"  col{i+1} {n:8s} EMPTY")
