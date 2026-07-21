"""Skeptic re-check of BH FDR control for the S1-S6 family:
6 tests, S1-S2 two-sided, S3-S6 one-sided, under several dependence structures.
Independent implementation, seed 123 (finding used seed 7)."""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(123)
m = 6
alpha = 0.05
NREP = 300_000

two_sided = np.array([True, True, False, False, False, False])

def pvals(z):
    p = np.empty_like(z)
    p[:, two_sided] = 2 * norm.sf(np.abs(z[:, two_sided]))
    p[:, ~two_sided] = norm.sf(z[:, ~two_sided])
    return p

def bh_reject(p, a=alpha):
    n, mm = p.shape
    o = np.sort(p, axis=1)
    thresh = a * np.arange(1, mm + 1) / mm
    ok = o <= thresh
    # largest k satisfying
    kmax = np.where(ok.any(axis=1), mm - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
    cutoff = np.where(kmax >= 0, o[np.arange(n), np.clip(kmax, 0, mm - 1)], -1.0)
    return p <= cutoff[:, None]

def chol_samples(Sigma, mu, n):
    L = np.linalg.cholesky(Sigma)
    return mu + rng.standard_normal((n, m)) @ L.T

def fdr(rej, is_null):
    V = rej[:, is_null].sum(axis=1)
    R = rej.sum(axis=1)
    return (V / np.maximum(R, 1)).mean()

# dependence structures
structs = {}
structs["independent"] = np.eye(m)
for r in (0.5, 0.8):
    S = np.full((m, m), r); np.fill_diagonal(S, 1.0)
    structs[f"exch r={r}"] = S
# mixed-sign via signed one-factor model: corr_ij = l_i*l_j
lam = np.array([0.85, 0.8, -0.75, 0.8, -0.75, 0.7])
S = np.outer(lam, lam); np.fill_diagonal(S, 1.0)
structs["mixed-sign factor (+/-0.5..0.7)"] = S

print(f"{'structure':38s} {'globalFDR':>10s} {'partialFDR':>11s}")
for name, Sig in structs.items():
    # global null
    z = chol_samples(Sig, np.zeros(m), NREP)
    rej = bh_reject(pvals(z))
    g = fdr(rej, np.ones(m, bool))
    # partial null: S3, S4 have true effects (mean 3), rest null
    mu = np.zeros(m); mu[2] = 3.0; mu[3] = 3.0
    z2 = chol_samples(Sig, mu, NREP)
    rej2 = bh_reject(pvals(z2))
    is_null = np.ones(m, bool); is_null[2] = is_null[3] = False
    p_ = fdr(rej2, is_null)
    print(f"{name:38s} {g:10.4f} {p_:11.4f}")

# MC standard error scale: FDR values ~0.04, per-rep var <= 0.25 -> se <~ 0.0009
print("approx MC se ~", (0.04 * 0.96 / NREP) ** 0.5)
