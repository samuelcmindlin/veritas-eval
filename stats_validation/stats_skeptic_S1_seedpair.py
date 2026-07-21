"""Skeptic re-check: S1 difference-of-Deltas, inner seed resampling paired vs independent.

Model: per (cluster i, seed s), detector a in {A,B} contributes
  x_{a,i,s} = mu_a + c_{a,i} + u_{a,s} + e_{a,i,s}
Cluster effects c correlated across detectors with rho_c (shared instances),
seed effects u correlated with rho_u (shared training seeds).
Statistic D = mean(x_A - x_B). Null: mu_A = mu_B = 0.

Nested bootstrap: outer resamples clusters PAIRED (prereg: shared instances,
whole clusters intact). Inner resamples seeds either PAIRED (same seed indices
for A and B) or INDEPENDENT. Percentile two-sided 90% CI; record per-tail
rejection (nominal 0.05/tail) and mean CI width.
"""
import numpy as np

rng = np.random.default_rng(20260707)

n_clu, K = 25, 5
B = 499
MC = 2500
rho_c, rho_u = 0.8, 0.8
sigma_e = 0.5

def run(sigma_c, sigma_u, mode):
    lo_rej = hi_rej = 0
    widths = np.empty(MC)
    for m in range(MC):
        # correlated cluster effects
        zc = rng.standard_normal((n_clu, 2))
        cA = sigma_c * zc[:, 0]
        cB = sigma_c * (rho_c * zc[:, 0] + np.sqrt(1 - rho_c**2) * zc[:, 1])
        # correlated seed effects
        zu = rng.standard_normal((K, 2))
        uA = sigma_u * zu[:, 0]
        uB = sigma_u * (rho_u * zu[:, 0] + np.sqrt(1 - rho_u**2) * zu[:, 1])
        eA = sigma_e * rng.standard_normal((n_clu, K))
        eB = sigma_e * rng.standard_normal((n_clu, K))
        xA = cA[:, None] + uA[None, :] + eA
        xB = cB[:, None] + uB[None, :] + eB
        # bootstrap (vectorized over B)
        ci = rng.integers(0, n_clu, size=(B, n_clu))          # paired cluster idx
        siA = rng.integers(0, K, size=(B, K))
        siB = siA if mode == "paired" else rng.integers(0, K, size=(B, K))
        # D* = mean over resampled clusters & seeds
        dA = xA[ci[:, :, None], siA[:, None, :]].mean(axis=(1, 2))
        dB = xB[ci[:, :, None], siB[:, None, :]].mean(axis=(1, 2))
        dstar = dA - dB
        L, U = np.percentile(dstar, [5, 95])
        widths[m] = U - L
        if L > 0:
            lo_rej += 1
        if U < 0:
            hi_rej += 1
    return lo_rej / MC, hi_rej / MC, widths.mean()

for sigma_c, sigma_u, tag in [(1.0, 0.3, "seed-minor"), (1.0, 1.0, "seed-equal")]:
    rp = run(sigma_c, sigma_u, "paired")
    ri = run(sigma_c, sigma_u, "indep")
    print(f"[{tag}] sigma_c={sigma_c} sigma_u={sigma_u} rho_u={rho_u}")
    print(f"  paired: tails=({rp[0]:.3f},{rp[1]:.3f}) width={rp[2]:.4f}")
    print(f"  indep : tails=({ri[0]:.3f},{ri[1]:.3f}) width={ri[2]:.4f}"
          f"  width ratio indep/paired={ri[2]/rp[2]:.3f}")
