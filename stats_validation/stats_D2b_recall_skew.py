"""Is the sampling distribution of paired Delta-recall@1%FPR (threshold
re-estimated per condition) actually skewed at the prereg's design size?
25 clusters x 40 positives/condition, 1600 negatives/condition,
cluster random effects on positives."""
import numpy as np
from scipy.stats import skew

rng = np.random.default_rng(42)
R, NC, NPOS, NNEG = 20_000, 25, 40, 1600

def delta_hat(mu_pos_A, mu_pos_B, base_recall_note=""):
    ds = np.empty(R)
    for s in range(0, R, 2000):
        c = min(2000, R - s)
        negA = rng.standard_normal((c, NNEG)); negB = rng.standard_normal((c, NNEG))
        thrA = np.quantile(negA, 0.99, axis=1, keepdims=True)
        thrB = np.quantile(negB, 0.99, axis=1, keepdims=True)
        u = 0.4 * rng.standard_normal((c, NC, 1))          # cluster effect (shared, paired)
        posA = mu_pos_A + u + rng.standard_normal((c, NC, NPOS))
        posB = mu_pos_B + u + rng.standard_normal((c, NC, NPOS))
        rA = (posA > thrA[:, :, None]).mean(axis=(1, 2))
        rB = (posB > thrB[:, :, None]).mean(axis=(1, 2))
        ds[s:s+c] = rA - rB
    return ds

for muA, muB, tag in [(3.2, 2.8, "high recall (~0.85/0.75)"),
                      (2.6, 2.2, "mid recall (~0.6/0.45)")]:
    d = delta_hat(muA, muB)
    print(f"{tag}: mean={d.mean():.3f} sd={d.std():.3f} skew={skew(d):+.2f} "
          f"p5={np.quantile(d,0.05):.3f} p95={np.quantile(d,0.95):.3f}")
