"""O4: sensitivity of realized FPR to the quantile-estimator choice for the frozen
threshold (99th pct of calibration-split honest scores). Realized FPR = 1-Phi(thr)
(fresh-data expectation), honest ~ N(0,1). 50k reps per (n, method).
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(99)
R = 50000
methods = ["linear", "lower", "higher", "inverted_cdf", "median_unbiased", "hazen"]

print(f"{'n_cal':>6} " + " ".join(f"{m:>16}" for m in methods) + f" {'k=ceil(.99(n+1))':>17}")
print(" " * 7 + "mean realized FPR % (SD %)")
for n in [200, 500, 1000, 1500, 2000]:
    x = rng.standard_normal((R, n))
    xs = np.sort(x, axis=1)
    row = []
    for m in methods:
        t = np.quantile(x, 0.99, axis=1, method=m)
        f = 100 * (1 - norm.cdf(t))
        row.append(f"{f.mean():6.3f} ({f.std():.3f})")
    k = int(np.ceil(0.99 * (n + 1)))  # exceedance-guarantee order statistic
    t = xs[:, min(k, n) - 1]
    f = 100 * (1 - norm.cdf(t))
    row.append(f"{f.mean():6.3f} ({f.std():.3f})")
    print(f"{n:>6} " + " ".join(f"{r:>16}" for r in row))
print("\ntarget = 1.000%; k-rule guarantees E[FPR] = (n+1-k)/(n+1) <= 1% exactly for continuous scores")
for n in [200, 500, 1000, 1500, 2000]:
    k = int(np.ceil(0.99*(n+1)))
    print(f"  n={n}: k={k}, E[FPR]={(n+1-k)/(n+1)*100:.3f}%")
