"""Skeptic re-verification of O3 (discrete-score 1%FPR operating point).

Independent implementation. Latent: honest N(0,1), deceptive N(mu,1),
mu s.t. continuous recall@1%FPR = 0.65. Instrument = fixed-bin discretizer.
Condition B = pure location shift eps of BOTH classes (continuous estimand
exactly invariant => any Delta is pure artifact). Per-condition thresholds.

Rules:
  cons  : deterministic, most liberal threshold with FPR <= alpha
  rand  : randomized/interpolated -- exact alpha FPR in expectation
          (linear ROC interpolation between straddling achievable points)

Checks: (a) population artifact vs their numbers; (b) other discretizations;
(c) finite-sample paired MC, n_neg=1500/cond, n_pos=300/cond.
"""
import numpy as np
from scipy.stats import norm

ALPHA = 0.01
MU = norm.ppf(0.99) + norm.ppf(0.65)   # 2.7116; continuous recall@1% = 0.65

def pop_rates(edges, shift=0.0):
    """(FPR, TPR) at each candidate threshold 'score > edge_c'."""
    fpr = norm.sf(np.asarray(edges) - shift)
    tpr = norm.sf(np.asarray(edges) - shift - MU)
    return fpr, tpr

def rule_cons(fpr, tpr, a=ALPHA):
    ok = np.where(fpr <= a)[0]
    if len(ok) == 0:
        return 0.0, 0.0
    i = ok[np.argmax(fpr[ok])]          # largest achievable FPR still <= a
    return fpr[i], tpr[i]

def rule_rand(fpr, tpr, a=ALPHA):
    # append (0,0) [reject none] and (1,1); interpolate ROC at FPR=a
    f = np.concatenate([[0.0], fpr, [1.0]])
    t = np.concatenate([[0.0], tpr, [1.0]])
    o = np.argsort(f)
    return float(np.interp(a, f[o], t[o]))

grids = {
    "theirs K=10 linspace(-2.5,3.5)": np.linspace(-2.5, 3.5, 9),
    "K=10 honest-decile edges":       norm.ppf(np.arange(1, 10) / 10),
    "K=8  linspace(-2,4)":            np.linspace(-2, 4, 7),
    "K=20 linspace(-3,4.5)":          np.linspace(-3, 4.5, 19),
}
print(f"mu={MU:.4f}, continuous recall@1%FPR = {norm.sf(norm.ppf(0.99)-MU):.4f}\n")
for name, edges in grids.items():
    fA, tA = pop_rates(edges)
    achiev = np.sort(fA[(fA > 1e-4) & (fA < 0.08)])[::-1]
    cfA, crA = rule_cons(fA, tA)
    rrA = rule_rand(fA, tA)
    print(f"[{name}] achievable FPRs near 1%: {[f'{100*x:.2f}%' for x in achiev]}")
    print(f"  cond A: cons realFPR={100*cfA:.2f}% recall={crA:.3f} | rand recall={rrA:.3f}")
    for eps in (0.05, 0.10, 0.15):
        fB, tB = pop_rates(edges, shift=eps)
        cfB, crB = rule_cons(fB, tB)
        rrB = rule_rand(fB, tB)
        print(f"  eps={eps:.2f}: cons Delta={crA-crB:+.4f} "
              f"(realFPR B={100*cfB:.2f}%) | rand Delta={rrA-rrB:+.4f}")
    print()

# ---- finite-sample paired MC: their K=10 grid, empirical thresholds ----
rng = np.random.default_rng(20260707)
edges = np.linspace(-2.5, 3.5, 9)
R, n_neg, n_pos = 4000, 1500, 300

def emp_recalls(hon_lat, dec_lat):
    """Empirical cons + rand recall from discretized samples (one condition)."""
    h = np.digitize(hon_lat, edges)     # scores 0..9
    d = np.digitize(dec_lat, edges)
    K = len(edges) + 1
    ch = np.bincount(h, minlength=K); cd = np.bincount(d, minlength=K)
    fpr_tail = ch[::-1].cumsum()[::-1] / len(hon_lat)   # P(score >= c)
    tpr_tail = cd[::-1].cumsum()[::-1] / len(dec_lat)
    # thresholds 'score >= c', c=0..K (c=K -> reject none)
    f = np.concatenate([fpr_tail, [0.0]])
    t = np.concatenate([tpr_tail, [0.0]])
    ok = np.where(f <= ALPHA)[0]
    cons = t[ok[np.argmax(f[ok])]] if len(ok) else 0.0
    o = np.argsort(f)
    rand = float(np.interp(ALPHA, f[o], t[o]))
    return cons, rand

for eps in (0.0, 0.10):
    dc = np.empty(R); dr = np.empty(R)
    ca = np.empty(R)
    for i in range(R):
        cA, rA = emp_recalls(rng.standard_normal(n_neg),
                             rng.standard_normal(n_pos) + MU)
        cB, rB = emp_recalls(rng.standard_normal(n_neg) + eps,
                             rng.standard_normal(n_pos) + MU + eps)
        dc[i], dr[i], ca[i] = cA - cB, rA - rB, cA
    print(f"finite-sample eps={eps:.2f} (R={R}, n_neg={n_neg}): "
          f"cons recall A mean={ca.mean():.3f} SD={ca.std():.3f}; "
          f"cons Delta mean={dc.mean():+.4f} SD={dc.std():.3f}; "
          f"rand Delta mean={dr.mean():+.4f} SD={dr.std():.3f}")
