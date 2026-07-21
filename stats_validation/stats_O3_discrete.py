"""O3: coarse discrete scores and the '1% FPR operating point'.
Latent honest N(0,1), deceptive N(mu,1) (mu -> recall .65 at true 1% FPR),
score = binned latent, K bins. Population-level analysis + a small-shift stress test:
condition B honest distribution shifted by eps; compare conservative rule
(largest threshold with FPR<=1%) vs interpolated/randomized rule (exact 1% FPR).
"""
import numpy as np
from scipy.stats import norm

Z99 = norm.ppf(0.99)
mu = Z99 + norm.ppf(0.65)

def analyze(K, edges):
    # bins: (-inf,e1],(e1,e2],...,(eK-1,inf); threshold candidates = bin boundaries
    # FPR(c) = P(honest > e_c), recall(c) = P(dec > e_c)
    fprs = 1 - norm.cdf(edges)
    recs = 1 - norm.cdf(edges - mu)
    return fprs, recs

def conservative(fprs, recs, alpha=0.01):
    ok = fprs <= alpha
    if not ok.any():
        return np.nan, np.nan
    i = np.argmax(ok)  # first edge with FPR<=alpha (fprs decreasing in edge)
    return fprs[i], recs[i]

def interpolated(fprs, recs, alpha=0.01):
    # randomize between adjacent edges straddling alpha -> exact alpha in expectation
    order = np.argsort(fprs)
    f, r = fprs[order], recs[order]
    if alpha <= f[0]:
        return r[0] * alpha / f[0]  # mix with 'reject none'
    j = np.searchsorted(f, alpha)
    w = (alpha - f[j-1]) / (f[j] - f[j-1])
    return (1-w)*r[j-1] + w*r[j]

for K in [5, 10, 20]:
    edges = np.linspace(-2.5, 3.5, K-1)  # K bins
    fprs, recs = analyze(K, edges)
    near = sorted(fprs[(fprs > 0.0005) & (fprs < 0.06)], reverse=True)
    f0, r0 = conservative(fprs, recs)
    ri0 = interpolated(fprs, recs)
    print(f"K={K}: achievable FPRs near 1%: {[f'{100*f:.2f}%' for f in near]}")
    print(f"   conservative: realized FPR={100*f0:.2f}%, recall={r0:.3f} "
          f"(true recall@1%={0.65:.2f}); interpolated recall={ri0:.3f}")
    # stress: condition B honest shifted by eps (tiny distribution shift, same edges)
    print("   paired Delta under NULL discrimination change, honest shift eps:")
    for eps in [0.0, 0.05, 0.10, 0.15]:
        fprsB = 1 - norm.cdf(edges - eps)   # honest B ~ N(eps,1)
        recsB = 1 - norm.cdf(edges - eps - mu)  # dec B ~ N(mu+eps,1): pure location shift
        fB, rB = conservative(fprsB, recsB)
        riB = interpolated(fprsB, recsB)
        # per-condition-threshold recall is shift-invariant in the continuous case ->
        # true Delta = 0 for every eps
        print(f"     eps={eps:.2f}: cons Delta={r0-rB:+.3f} (realFPR A={100*f0:.2f}% B={100*fB:.2f}%), "
              f"interp Delta={ri0-riB:+.3f}")
    print()

# finite-sample wobble of the conservative rule at n_neg=1500, K=10
rng = np.random.default_rng(3)
K = 10; edges = np.linspace(-2.5, 3.5, K-1)
R = 5000; n_neg, n_pos = 1500, 300
h = np.digitize(rng.standard_normal((R, n_neg)), edges)
p = np.digitize(rng.standard_normal((R, n_pos)) + mu, edges)
recs_emp = np.empty(R)
fprs_emp = np.empty(R)
for i in range(R):
    # empirical conservative threshold: smallest bin c with emp FPR(score>=c)<=1%
    cnt = np.bincount(h[i], minlength=K)
    tailfpr = cnt[::-1].cumsum()[::-1] / n_neg  # P(score>=c)
    cands = np.where(tailfpr <= 0.01)[0]
    c = cands[0] if len(cands) else K  # K = reject none
    recs_emp[i] = (p[i] >= c).mean()
    fprs_emp[i] = tailfpr[c] if c < K else 0.0
print(f"finite-sample (K=10, n_neg=1500): recall mean={recs_emp.mean():.3f}, SD={recs_emp.std():.3f}; "
      f"realized-FPR mean={100*fprs_emp.mean():.2f}%, SD={100*fprs_emp.std():.2f}pp")
print(f"  distinct empirical thresholds used: {len(np.unique(recs_emp.round(4)))} recall values")
