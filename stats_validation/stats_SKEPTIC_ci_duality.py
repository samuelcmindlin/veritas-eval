"""Skeptic re-check: per-leg error of 90% bootstrap CIs (percentile/basic/BCa)
for the mean of n=25 cluster values, under normal and skew-2 data.
Per-leg errors (translation-invariant, so evaluate at true mean mu):
  sig leg error  = P(L > mu)   (nominal 0.05)  -- false 'effect present'
  TOST leg error = P(U < mu)   (nominal 0.05)  -- false equivalence
Independent implementation; seed differs from original finding's script.
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(424242)
R = 3000      # MC reps
B = 1999      # bootstrap replicates
n = 25        # clusters
CHUNK = 100

def gen(dist, size):
    if dist == "normal":
        return rng.normal(0.0, 1.0, size)
    if dist == "exp_skew2":          # skewness = 2, mean 0, sd 1
        return rng.exponential(1.0, size) - 1.0
    raise ValueError(dist)

z05, z95 = norm.ppf(0.05), norm.ppf(0.95)

for dist in ["normal", "exp_skew2"]:
    res = {k: [0, 0] for k in ["percentile", "basic", "bca"]}  # [sig_err, tost_err]
    done = 0
    while done < R:
        c = min(CHUNK, R - done)
        X = gen(dist, (c, n))                      # data, true mean 0
        mhat = X.mean(axis=1)                      # (c,)
        idx = rng.integers(0, n, size=(c, B, n))
        boot = X[np.arange(c)[:, None, None], idx].mean(axis=2)  # (c,B)
        boot_sorted = np.sort(boot, axis=1)

        # percentile
        q05 = np.quantile(boot, 0.05, axis=1)
        q95 = np.quantile(boot, 0.95, axis=1)
        res["percentile"][0] += int((q05 > 0).sum())
        res["percentile"][1] += int((q95 < 0).sum())
        # basic
        Lb, Ub = 2 * mhat - q95, 2 * mhat - q05
        res["basic"][0] += int((Lb > 0).sum())
        res["basic"][1] += int((Ub < 0).sum())
        # BCa
        prop = (boot < mhat[:, None]).mean(axis=1)
        prop = np.clip(prop, 1.0 / (B + 1), B / (B + 1.0))
        z0 = norm.ppf(prop)
        S = X.sum(axis=1, keepdims=True)
        jack = (S - X) / (n - 1)                   # (c,n) leave-one-out means
        jm = jack.mean(axis=1, keepdims=True)
        d = jm - jack
        a = (d**3).sum(axis=1) / (6.0 * ((d**2).sum(axis=1)) ** 1.5 + 1e-300)
        a1 = norm.cdf(z0 + (z0 + z05) / (1 - a * (z0 + z05)))
        a2 = norm.cdf(z0 + (z0 + z95) / (1 - a * (z0 + z95)))
        a1 = np.clip(a1, 0.0, 1.0); a2 = np.clip(a2, 0.0, 1.0)
        rows = np.arange(c)
        i1 = np.clip((a1 * (B - 1)).round().astype(int), 0, B - 1)
        i2 = np.clip((a2 * (B - 1)).round().astype(int), 0, B - 1)
        Lc = boot_sorted[rows, i1]
        Uc = boot_sorted[rows, i2]
        res["bca"][0] += int((Lc > 0).sum())
        res["bca"][1] += int((Uc < 0).sum())
        done += c
    se = (0.05 * 0.95 / R) ** 0.5
    print(f"\n{dist}  (R={R}, B={B}, n={n}; MC se ~ {se:.4f} at p=0.05)")
    for k, (s, t) in res.items():
        print(f"  {k:11s}  sig-leg err = {s/R:.4f}   TOST-leg err = {t/R:.4f}")
