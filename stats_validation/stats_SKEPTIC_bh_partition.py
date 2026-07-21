"""Skeptic re-verification, independent implementation.
(1) Exact check: union of cells 1-3 == {L>0} over a dense grid of (L,U), U>=L.
(2) BH vs fixed-90%-CI presence: under a global null with N=6 one-sided tests,
    compare P(any 'present' by L>0 rule) vs BH FWER/FDR; and show BH rejection
    is NOT determined by the 0.05 one-sided cut (rank-dependent thresholds).
"""
import numpy as np

SESOI = 0.10

# ---------- (1) exact partition check ----------
g = np.linspace(-0.5, 0.5, 2000)
L, U = np.meshgrid(g, g, indexing="ij")
mask = U >= L
c1 = (L >= SESOI)
c2 = (L > 0) & (L < SESOI) & (U >= SESOI)
c3 = (L > 0) & (U < SESOI)
c4 = (L <= 0) & (U < SESOI)
c5 = (L <= 0) & (U >= SESOI)
union123 = c1 | c2 | c3
present = L > 0
mism = np.sum((union123 != present) & mask)
# exclusivity/totality
tot = c1.astype(int)+c2.astype(int)+c3.astype(int)+c4.astype(int)+c5.astype(int)
excl_viol = np.sum((tot != 1) & mask)
print(f"[1] pairs checked={mask.sum()}, union(1-3)!= (L>0): {mism}, partition-count!=1: {excl_viol}")

# ---------- (2) BH vs uncorrected-90%-CI presence ----------
rng = np.random.default_rng(20260707)
N = 6
reps = 200_000
# one-sided p-values under global null: Uniform(0,1)
p = rng.uniform(size=(reps, N))

# rule A: 'present' if L>0 on 90% two-sided CI  <=>  one-sided p < 0.05
anyA = (p < 0.05).any(axis=1)

# rule B: BH at q=0.05
ps = np.sort(p, axis=1)
thr = 0.05 * np.arange(1, N+1) / N
rej_rank = np.where((ps <= thr).any(axis=1),
                    np.argmax((ps <= thr)[:, ::-1], axis=1), -1)
# BH: reject all p <= p_(k) where k = max index with p_(k)<=thr_k
hit = ps <= thr
k = np.where(hit.any(axis=1), N - 1 - np.argmax(hit[:, ::-1], axis=1), -1)
anyB = k >= 0
print(f"[2] global null, N=6: P(any present, uncorrected L>0 rule) = {anyA.mean():.4f} "
      f"(theory 1-0.95^6 = {1-0.95**6:.4f})")
print(f"    P(any BH rejection) = {anyB.mean():.4f} (should be ~0.05)")

# rank-dependence: fraction of cases where a test has p<0.05 (so 'present' by CI)
# but is NOT BH-rejected
ci_present = p < 0.05
cut = np.where(k >= 0, ps[np.arange(reps), np.maximum(k, 0)], -1.0)
bh_rej = p <= cut[:, None]
disagree = ci_present & ~bh_rej
print(f"    among CI-'present' calls, fraction NOT BH-rejected = "
      f"{disagree.sum()/max(ci_present.sum(),1):.3f}")
print(f"    BH per-rank thresholds (one-sided): {np.round(thr,5)} -> "
      f"two-sided CI levels {np.round(100*(1-2*thr),2)}%")
print(f"    min achievable bootstrap p with B outer reps = 1/(B+1); "
      f"need B >= {int(np.ceil(N/0.05))-1} for min p <= 0.05/{N}={0.05/N:.5f}")
