"""D3: FDR of BH at q=0.05 over the S1-S6 family (2 two-sided, 4 one-sided)
under correlated Gaussian test statistics, incl. negative dependence
(contrast statistics sharing Delta_eval terms with opposite signs).
Under the global null FDR = FWER = P(any rejection)."""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(7)
M, Q, R = 6, 0.05, 400_000
two_sided = np.array([True, True, False, False, False, False])  # S1,S2 two-sided

def nearest_psd(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C2)); return C2 / np.outer(d, d)

def corr_cases():
    yield "independent", np.eye(M)
    for r in (0.5, 0.8):
        C = np.full((M, M), r); np.fill_diagonal(C, 1.0)
        yield f"exchangeable r={r}", C
    # mixed-sign: S1/S2 (family contrasts) negatively corr. with S3-S5 (share -Delta_ctrl etc.)
    C = np.eye(M)
    for i in range(M):
        for j in range(M):
            if i != j:
                C[i, j] = -0.5 if (i < 2) != (j < 2) else 0.6
    yield "mixed-sign +-0.5/0.6", nearest_psd(C)

def bh_reject(p, q=Q):
    ps = np.sort(p, axis=1)
    thr = q * np.arange(1, M+1) / M
    ok = ps <= thr
    kmax = np.where(ok.any(axis=1), M - np.argmax(ok[:, ::-1], axis=1), 0)  # largest k
    cut = np.where(kmax > 0, ps[np.arange(len(p)), np.maximum(kmax-1, 0)], -1.0)
    return p <= cut[:, None]

for name, C in corr_cases():
    Lc = np.linalg.cholesky(C)
    Z = rng.standard_normal((R, M)) @ Lc.T
    # global null
    p = np.where(two_sided, 2*norm.sf(np.abs(Z)), norm.sf(Z))
    rej = bh_reject(p)
    fdr_global = rej.any(axis=1).mean()
    # partial null: S3,S4,S5 true effects (mean 3); nulls = S1,S2,S6
    Z2 = Z.copy(); Z2[:, 2:5] += 3.0
    p2 = np.where(two_sided, 2*norm.sf(np.abs(Z2)), norm.sf(Z2))
    rej2 = bh_reject(p2)
    V = rej2[:, [0, 1, 5]].sum(axis=1); Rtot = np.maximum(rej2.sum(axis=1), 1)
    fdr_partial = (V / Rtot).mean()
    print(f"{name:22s} global-null FDR={fdr_global:.4f}  partial-null(3 true) FDR={fdr_partial:.4f}")
print(f"\nnominal q=0.05; MC SE ~ {np.sqrt(0.05*0.95/R):.5f}")
print("BY threshold factor for m=6: 1/sum(1/i)=", 1/sum(1/i for i in range(1, 7)))
