"""Skeptic re-check: skewness of the sampling distribution of paired
Delta-recall@1%FPR (per-condition threshold), design-faithful-ish:
25 clusters x 40 paired positives, 1600 negatives per condition,
cluster random effect on positive scores, per-condition threshold =
99th pct of that condition's negatives.
"""
import numpy as np
from scipy.stats import skew

rng = np.random.default_rng(777001)
REPS = 10000
K, M = 25, 40          # clusters x positives per cluster
NNEG = 1600
mu_pos_A, mu_pos_B = 2.6, 2.3   # detector separation; B slightly degraded
tau = 0.4                        # cluster RE sd
d = np.zeros(REPS)
for r in range(REPS):
    negA = rng.normal(0, 1, NNEG)
    negB = rng.normal(0, 1, NNEG)
    thrA = np.quantile(negA, 0.99)
    thrB = np.quantile(negB, 0.99)
    re = rng.normal(0, tau, (K, 1))
    eps = rng.normal(0, 1, (K, M))          # shared instance noise (paired)
    posA = mu_pos_A + re + eps + rng.normal(0, 0.3, (K, M))
    posB = mu_pos_B + re + eps + rng.normal(0, 0.3, (K, M))
    d[r] = (posA > thrA).mean() - (posB > thrB).mean()
print(f"Delta-recall: mean={d.mean():.4f} sd={d.std():.4f} skew={skew(d):.3f}")
