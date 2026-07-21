"""O2: bootstrap CI coverage for paired Delta(recall@1%FPR):
threshold re-estimated inside each replicate vs fixed at full-sample estimate.
90% percentile CIs; true Delta known analytically. iid instance bootstrap
(cluster layer orthogonal to the threshold question).
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(7)
Z99 = norm.ppf(0.99)
N_NEG, N_POS, RHO = 1500, 300, 0.8
B = 500
NDAT = 600

def corr_pair(shape, rho, rng, mu_a=0.0, mu_b=0.0):
    z = rng.standard_normal(shape)
    a = np.sqrt(rho)*z + np.sqrt(1-rho)*rng.standard_normal(shape) + mu_a
    b = np.sqrt(rho)*z + np.sqrt(1-rho)*rng.standard_normal(shape) + mu_b
    return a, b

for label, (muA, muB) in {"null (Delta=0), base recall .65":
                          (Z99+norm.ppf(.65), Z99+norm.ppf(.65)),
                          "effect (Delta=0.10)":
                          (Z99+norm.ppf(.65), Z99+norm.ppf(.55))}.items():
    trueD = norm.cdf(muA - Z99) - norm.cdf(muB - Z99)
    cov = {"re": 0, "fix": 0}
    wid = {"re": [], "fix": []}
    for d in range(NDAT):
        hA, hB = corr_pair((N_NEG,), RHO, rng)
        pA, pB = corr_pair((N_POS,), RHO, rng, muA, muB)
        tA_full = np.quantile(hA, 0.99); tB_full = np.quantile(hB, 0.99)
        idxH = rng.integers(0, N_NEG, (B, N_NEG))
        idxP = rng.integers(0, N_POS, (B, N_POS))
        # re-estimated thresholds per replicate (paired resampling: same idx both conds)
        tA_b = np.quantile(hA[idxH], 0.99, axis=1)
        tB_b = np.quantile(hB[idxH], 0.99, axis=1)
        dA = (pA[idxP] > tA_b[:, None]).mean(axis=1)
        dB = (pB[idxP] > tB_b[:, None]).mean(axis=1)
        D_re = dA - dB
        # fixed thresholds
        fA = (pA[idxP] > tA_full).mean(axis=1)
        fB = (pB[idxP] > tB_full).mean(axis=1)
        D_fix = fA - fB
        for k, Db in (("re", D_re), ("fix", D_fix)):
            lo, hi = np.percentile(Db, [5, 95])
            cov[k] += (lo <= trueD <= hi)
            wid[k].append(hi - lo)
    print(f"{label}: true Delta={trueD:.3f}")
    for k in ("re", "fix"):
        c = cov[k]/NDAT
        se = np.sqrt(c*(1-c)/NDAT)
        print(f"  {k:>3}: coverage of 90% CI = {c:.3f} (+-{se:.3f}), mean width = {np.mean(wid[k]):.4f}")
