import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20260707)
REPS = 20000

# Gaussian score model: neg ~ N(0,1), pos ~ N(d,1).
# True threshold at 1% FPR = z_0.99; true recall = Phi(d - z99).
z99 = norm.ppf(0.99)

def thr_only_sd(n_neg, d, reps=REPS):
    # threshold noise only: estimate empirical 1%FPR threshold from n_neg negatives,
    # evaluate TRUE recall at that threshold (infinite positives)
    neg = rng.standard_normal((reps, n_neg))
    thr = np.quantile(neg, 0.99, axis=1)
    recall = norm.cdf(d - thr)
    return recall.std(), recall.mean()

def full_sd(n_neg, n_pos, d, reps=REPS):
    neg = rng.standard_normal((reps, n_neg))
    thr = np.quantile(neg, 0.99, axis=1)
    pos = rng.standard_normal((reps, n_pos)) + d
    recall = (pos > thr[:, None]).mean(axis=1)
    return recall.std(), recall.mean()

for target_recall in [0.5, 0.65, 0.8]:
    d = z99 + norm.ppf(target_recall)
    print(f"--- true recall={target_recall} (d={d:.3f}) ---")
    for n_neg in [1500, 2000]:
        sd_t, m = thr_only_sd(n_neg, d)
        # analytic: SD(thr) = sqrt(p(1-p)/n)/phi(z99); SD(R) = phi(Phi^-1(R)) * SD(thr)
        sd_thr_an = np.sqrt(0.01*0.99/n_neg)/norm.pdf(z99)
        sd_an = norm.pdf(norm.ppf(target_recall)) * sd_thr_an
        print(f"  n_neg={n_neg}: thr-only SD sim={sd_t:.4f} analytic={sd_an:.4f} (mean recall {m:.3f})")
    for n_pos in [300, 500, 1000]:
        binom = np.sqrt(target_recall*(1-target_recall)/n_pos)
        print(f"  n_pos={n_pos}: binomial SD={binom:.4f}")
    # combined at n_neg=1500, n_pos=300
    sd_f, m = full_sd(1500, 300, d)
    print(f"  full (1500 neg, 300 pos): SD={sd_f:.4f}")
