import numpy as np
from scipy.ndimage import gaussian_filter1d

R = np.load("f3_result.npz")
t, iw = R["t"], 60
zw, zm, zb = R["zp_well"], R["zp_mean"][iw], R["zp_bg"][iw]
lo, hi, lo_m, hi_m = R["lo"][iw], R["hi"][iw], R["lo_m"][iw], R["hi_m"][iw]
hp = lambda x: x - gaussian_filter1d(x, 10)
lp = lambda x: gaussian_filter1d(x, 10)
SPLIT = 1.195
rows = [("0.50-1.195 s (above the unconformity)", t < SPLIT), ("1.195-1.50 s (below the unconformity)", t >= SPLIT), ("whole window", t > 0)]
for name, k in rows:
    c = lambda a, b: np.corrcoef(a[k], b[k])[0, 1]
    hw, hm = hp(np.log(zw)), hp(np.log(zm))
    print(name, "- %d samples" % k.sum())
    print("   mean Zp: well %.2f | background %.2f | VI %.2f" % (zw[k].mean(), zb[k].mean(), zm[k].mean()))
    print("   MAE: background %.2f | VI %.2f" % (np.abs(zb - zw)[k].mean(), np.abs(zm - zw)[k].mean()))
    print("   corr detail %.2f | trend %.2f | detail amplitude ratio (VI/well) %.2f" % (c(hm, hw), c(lp(np.log(zm)), lp(np.log(zw))), hm[k].std() / hw[k].std()))
    print("   PICP90 %.2f (incl. background %.2f) | width90 %.2f MRayl" % (np.mean((zw >= lo)[k] & (zw <= hi)[k]), np.mean((zw >= lo_m)[k] & (zw <= hi_m)[k]), (hi - lo)[k].mean()))
