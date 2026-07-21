"""Skeptic re-check of finding C2: does per-class mean shift distinguish
pure-shift vs separation-loss vs variance-inflation, and does frozen-map
Delta-ECE fail to distinguish them? Independent implementation.
Control: honest ~ N(0,1), deceptive ~ N(d,1), 50/50 mix, d s.t. AUROC=0.898.
Platt map fit by logistic regression on control sample, then FROZEN.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

rng = np.random.default_rng(20260707)
N = 2_000_000  # per class per scenario

d = norm.ppf(0.898) * np.sqrt(2)   # separation for AUROC .898
print(f"d = {d:.4f}")

def sample(mu0, mu1, sd, n=N):
    s0 = rng.normal(mu0, sd, n)
    s1 = rng.normal(mu1, sd, n)
    s = np.concatenate([s0, s1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return s, y, s0, s1

# ---- fit Platt on control (logistic regression, Newton) ----
def fit_platt(s, y):
    a, b = 1.0, 0.0
    for _ in range(50):
        z = a * s + b
        p = 1 / (1 + np.exp(-z))
        g = np.array([np.sum((p - y) * s), np.sum(p - y)])
        w = p * (1 - p)
        H = np.array([[np.sum(w * s * s), np.sum(w * s)],
                      [np.sum(w * s), np.sum(w)]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return a, b

s_c, y_c, s0c, s1c = sample(0, d, 1)
a, b = fit_platt(s_c, y_c)
print(f"Platt a={a:.4f} b={b:.4f} (theory a={d:.4f}, b={-d*d/2:.4f})")

def platt(s):
    return 1 / (1 + np.exp(-(a * s + b)))

def ece_equal_mass(p, y, nbins=15):
    idx = np.argsort(p)
    p, y = p[idx], y[idx]
    bins = np.array_split(np.arange(len(p)), nbins)
    e = 0.0
    for bidx in bins:
        e += len(bidx) / len(p) * abs(p[bidx].mean() - y[bidx].mean())
    return e

def resolution(p, y, nbins=15):
    idx = np.argsort(p)
    p, y = p[idx], y[idx]
    ybar = y.mean()
    bins = np.array_split(np.arange(len(p)), nbins)
    r = 0.0
    for bidx in bins:
        r += len(bidx) / len(p) * (y[bidx].mean() - ybar) ** 2
    return r

def auroc_gauss(mu0, mu1, sd):
    return norm.cdf((mu1 - mu0) / (sd * np.sqrt(2)))

def report(name, s0, s1, mu0_ref=0.0, mu1_ref=d):
    s = np.concatenate([s0, s1]); y = np.concatenate([np.zeros(len(s0)), np.ones(len(s1))])
    p = platt(s)
    # empirical AUROC via rank method on subsample for speed
    sub = rng.choice(len(s), 400_000, replace=False)
    ss, yy = s[sub], y[sub]
    order = np.argsort(ss); ranks = np.empty(len(ss)); ranks[order] = np.arange(1, len(ss) + 1)
    n1 = yy.sum(); n0 = len(yy) - n1
    auc = (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    print(f"{name:28s} ECE={ece_equal_mass(p, y):.4f}  "
          f"res={resolution(p, y):.4f}  AUROC={auc:.3f}  "
          f"dmean0={s0.mean()-mu0_ref:+.4f}  dmean1={s1.mean()-mu1_ref:+.4f}")
    return ece_equal_mass(p, y)

ece0 = report("control (frozen-map home)", s0c, s1c)

# --- Scenario C: variance inflation, means fixed; k s.t. AUROC -> .801 ---
k = d / (np.sqrt(2) * norm.ppf(0.801))
print(f"\nk (variance inflation) = {k:.4f}, implied AUROC = {auroc_gauss(0, d, k):.4f}")
_, _, s0v, s1v = sample(0, d, k)
ece_v = report("variance inflation", s0v, s1v)
dtarget = ece_v - ece0
print(f"Delta-ECE target from variance scenario: {dtarget:+.4f}")

# --- Scenario A: pure both-class shift c, tuned to match that Delta-ECE ---
def ece_shift(c):
    return ece_equal_mass(platt(s_c + c), y_c)
c = brentq(lambda c: ece_shift(c) - ece_v, 0.01, 3.0, xtol=1e-3)
print(f"\npure-shift c matched: c = {c:.4f}")
report("pure shift (both classes)", s0c + c, s1c + c)

# --- Scenario B: separation loss, deceptive mean d -> d2, matched Delta-ECE ---
def ece_sep(d2):
    _, _, a0, a1 = sample(0, d2, 1, n=500_000)
    s = np.concatenate([a0, a1]); y = np.concatenate([np.zeros(len(a0)), np.ones(len(a1))])
    return ece_equal_mass(platt(s), y)
d2 = brentq(lambda x: ece_sep(x) - ece_v, 0.1, d, xtol=1e-3)
print(f"\nseparation-loss d2 matched: d2 = {d2:.4f}  (AUROC={auroc_gauss(0,d2,1):.3f})")
_, _, s0s, s1s = sample(0, d2, 1)
report("separation loss", s0s, s1s)
