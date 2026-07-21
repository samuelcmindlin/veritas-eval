"""O1: sampling behavior of per-condition-threshold recall@1%FPR vs n_neg.
Honest ~ N(0,1); deceptive ~ N(mu,1); mu set so true recall@1%FPR in {0.4,0.65,0.9}.
Paired design: conditions A,B share instances -> correlated scores (rho) for both classes.
Null Delta (mu_A = mu_B). Threshold = empirical 99th pct (linear) of that condition's honest.
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20260707)
Z99 = norm.ppf(0.99)
R = 20000
N_POS = 300
RHO = 0.8

def corr_pair(shape, rho, rng):
    z = rng.standard_normal(shape)
    a = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(shape)
    b = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(shape)
    return a, b

print(f"{'n_neg':>6} {'target':>7} {'mean_rec':>9} {'bias':>8} {'SD_rec':>7} {'SD_Delta':>8} {'SD_FPRhat(pp)':>13}")
for n_neg in [200, 500, 1000, 1500, 2000]:
    for target in [0.4, 0.65, 0.9]:
        mu = Z99 + norm.ppf(target)
        hA, hB = corr_pair((R, n_neg), RHO, rng)
        pA, pB = corr_pair((R, N_POS), RHO, rng)
        pA += mu; pB += mu
        tA = np.quantile(hA, 0.99, axis=1)
        tB = np.quantile(hB, 0.99, axis=1)
        rA = (pA > tA[:, None]).mean(axis=1)
        rB = (pB > tB[:, None]).mean(axis=1)
        delta = rA - rB
        # realized FPR of the estimated threshold (population)
        fpr = 1 - norm.cdf(tA)
        print(f"{n_neg:>6} {target:>7} {rA.mean():>9.4f} {rA.mean()-target:>8.4f} "
              f"{rA.std():>7.4f} {delta.std():>8.4f} {100*fpr.std():>13.3f}")
    print()

# decompose: threshold noise vs positive-class binomial noise at target 0.65
print("decomposition at target=0.65 (SD of recall):")
mu = Z99 + norm.ppf(0.65)
for n_neg in [200, 500, 1000, 1500, 2000]:
    h = rng.standard_normal((R, n_neg))
    t = np.quantile(h, 0.99, axis=1)
    rec_inf_pos = 1 - norm.cdf(t - mu)          # infinite positives: pure threshold noise
    binom_sd = np.sqrt(0.65*0.35/N_POS)
    print(f"  n_neg={n_neg:>5}: thr-only SD={rec_inf_pos.std():.4f}, "
          f"binom(n_pos=300) SD={binom_sd:.4f}, thr-induced bias={rec_inf_pos.mean()-0.65:+.4f}")
