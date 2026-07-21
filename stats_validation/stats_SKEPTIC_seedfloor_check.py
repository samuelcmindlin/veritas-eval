"""Skeptic re-implementation: size of one-sided alpha=0.05 test (lower bound of
two-sided 90% percentile bootstrap CI > 0) under the null, for a crossed
cluster x seed design.

Model: X_{g,s} = c_g + u_s + e_{g,s}, Delta-hat = mean(X).
Var(Delta-hat) = sc^2/G + su^2/S + se^2/(GS). Seed-variance SHARE rho refers to
(su^2/S) / Var(Delta-hat); small residual share eta for e.

Methods:
  A "joint"   : outer resample clusters, inner ONE global seed resample per replicate (correct)
  B "fixed"   : cluster resample only, 5 seeds held fixed (seed variance omitted)
  C "percluster": seed indices redrawn independently PER CLUSTER (misreading of 'nested')
"""
import numpy as np

rng = np.random.default_rng(20260707)

def run(G, S, rho, eta, mc=4000, B=499, method="joint"):
    # variance components scaled so Var(Delta-hat)=1
    su2 = rho * S           # su^2 such that su^2/S = rho
    sc2 = (1 - rho - eta) * G
    se2 = eta * G * S
    su, sc, se = np.sqrt(su2), np.sqrt(sc2), np.sqrt(se2)
    rej_lo = 0
    rej_hi = 0
    for m in range(mc):
        c = rng.normal(0, sc, G)
        u = rng.normal(0, su, S)
        e = rng.normal(0, se, (G, S))
        X = c[:, None] + u[None, :] + e
        ig = rng.integers(0, G, size=(B, G))
        if method == "fixed":
            rowm = X.mean(axis=1)
            db = rowm[ig].mean(axis=1)
        elif method == "joint":
            isd = rng.integers(0, S, size=(B, S))
            db = X[ig[:, :, None], isd[:, None, :]].mean(axis=(1, 2))
        elif method == "percluster":
            isd = rng.integers(0, S, size=(B, G, S))
            db = X[ig[:, :, None], isd].mean(axis=(1, 2))
        lo, hi = np.percentile(db, [5, 95])
        rej_lo += lo > 0
        rej_hi += hi < 0
    return rej_lo / mc, rej_hi / mc

configs = [
    # (G, S, rho, eta, method)
    (25, 5, 0.30, 0.05, "joint"),
    (25, 5, 0.60, 0.05, "joint"),
    (25, 10, 0.30, 0.05, "joint"),
    (25, 10, 0.60, 0.05, "joint"),
    (25, 5, 0.30, 0.05, "fixed"),
    (25, 5, 0.60, 0.05, "fixed"),
    (25, 5, 0.60, 0.05, "percluster"),
    (25, 5, 0.30, 0.05, "percluster"),
    (25, 5, 0.00, 0.05, "joint"),   # baseline: cluster-only variance, joint method
]
print(f"{'G':>3} {'S':>3} {'rho':>5} {'method':>11} {'size_lo':>8} {'size_hi':>8}  (nominal 0.05/tail)")
for G, S, rho, eta, meth in configs:
    lo, hi = run(G, S, rho, eta, method=meth)
    print(f"{G:>3} {S:>3} {rho:>5.2f} {meth:>11} {lo:>8.4f} {hi:>8.4f}")
