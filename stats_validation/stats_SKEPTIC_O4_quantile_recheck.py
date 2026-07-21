"""Skeptic re-check of frozen-threshold quantile estimator finding.

Scores iid continuous -> WLOG Uniform(0,1); realized FPR of threshold t is 1-t.
Estimators compared:
  - numpy 'linear' (default)
  - numpy 'inverted_cdf'
  - numpy 'higher'
  - proposed order-statistic rule: k = ceil(0.99*(n+1))
Analytic check: E[1-F(X_(k))] = (n+1-k)/(n+1).
"""
import numpy as np
import math

rng = np.random.default_rng(20260707)
REPS = 40000
q = 0.99

print(f"{'n_cal':>6} {'linear':>8} {'inv_cdf':>8} {'higher':>8} {'k-rule':>8} {'k':>5} {'analytic k-rule':>16}")
for n in (200, 500, 1500):
    X = rng.random((REPS, n))
    t_lin = np.quantile(X, q, axis=1, method="linear")
    t_inv = np.quantile(X, q, axis=1, method="inverted_cdf")
    t_hi  = np.quantile(X, q, axis=1, method="higher")
    k = math.ceil(q * (n + 1))
    Xs = np.sort(X, axis=1)
    t_k = Xs[:, k - 1]
    fpr = lambda t: np.mean(1.0 - t) * 100
    analytic = (n + 1 - k) / (n + 1) * 100
    print(f"{n:>6} {fpr(t_lin):8.3f} {fpr(t_inv):8.3f} {fpr(t_hi):8.3f} {fpr(t_k):8.3f} {k:>5} {analytic:16.4f}")

# also: analytic for inverted_cdf's order statistic k=ceil(q*n)
print("\nanalytic E[FPR]% for inverted_cdf k=ceil(0.99n):",
      [( n, round((n+1-math.ceil(q*n))/(n+1)*100, 3)) for n in (200,500,1500)])
