"""Independent re-implementation (skeptic pass) of the C3 power/coverage sim.

Design: paired two-condition recall@1%FPR with per-condition threshold
re-estimation, scenario-cluster structure, cluster-level outer bootstrap
(whole clusters, all classes/conditions intact), 90% percentile CI.

Generative model:
  score = m_class + b_c + e,  b_c ~ N(0, tau2) shared across classes/conditions
  within cluster, e ~ N(0, 1-tau2); paired across conditions with corr rho.
  Honest marginal N(0,1). Deceptive marginal N(m,1).
  True threshold at 1% FPR = z_.99 = 2.32635; true recall = Phi(m - z99).
"""
import numpy as np
from scipy.stats import norm

Z99 = norm.ppf(0.99)

def m_for_recall(r):
    return Z99 + norm.ppf(r)

def simulate_setting(K, n_h, n_d, icc, rho, recall_ctrl, recall_trt,
                     R=400, B=300, seed=0):
    rng = np.random.default_rng(seed)
    tau = np.sqrt(icc); sd_e = np.sqrt(1 - icc)
    nh_c = n_h // K; nd_c = n_d // K          # per-cluster counts
    m0 = m_for_recall(recall_ctrl); m1 = m_for_recall(recall_trt)
    d_true = recall_ctrl - recall_trt
    L = np.empty(R); U = np.empty(R); est = np.empty(R)
    for r in range(R):
        b = rng.normal(0, tau, K)[:, None]
        # honest pairs
        u = rng.normal(0, 1, (K, nh_c)); w = rng.normal(0, 1, (K, nh_c))
        hC = b + sd_e * u
        hT = b + sd_e * (rho * u + np.sqrt(1 - rho**2) * w)
        # deceptive pairs
        u = rng.normal(0, 1, (K, nd_c)); w = rng.normal(0, 1, (K, nd_c))
        dC = m0 + b + sd_e * u
        dT = m1 + b + sd_e * (rho * u + np.sqrt(1 - rho**2) * w)
        # point estimate (per-condition threshold = 99th pct of own honest)
        tC = np.quantile(hC, 0.99); tT = np.quantile(hT, 0.99)
        est[r] = (dC > tC).mean() - (dT > tT).mean()
        # cluster bootstrap
        idx = rng.integers(0, K, (B, K))
        bhC = hC[idx].reshape(B, -1); bhT = hT[idx].reshape(B, -1)
        bdC = dC[idx].reshape(B, -1); bdT = dT[idx].reshape(B, -1)
        btC = np.quantile(bhC, 0.99, axis=1); btT = np.quantile(bhT, 0.99, axis=1)
        dl = (bdC > btC[:, None]).mean(1) - (bdT > btT[:, None]).mean(1)
        L[r], U[r] = np.quantile(dl, [0.05, 0.95])
    return dict(d_true=d_true, est_mean=est.mean(), est_sd=est.std(),
                p_sup=(L > 0).mean(), p_equiv=(U < 0.10).mean(),
                cover=((L <= d_true) & (d_true <= U)).mean())

def report(name, res):
    print(f"{name}: d_true={res['d_true']:.2f} est={res['est_mean']:.3f}"
          f" (sd {res['est_sd']:.3f}) P(L>0)={res['p_sup']:.3f}"
          f" P(U<.10)={res['p_equiv']:.3f} cover90={res['cover']:.3f}")

# --- Fill-in A: the finding's parameters ---
A = dict(K=25, n_h=1600, n_d=300, icc=0.15, rho=0.8)
report("A sup (D=.10)", simulate_setting(**A, recall_ctrl=0.61, recall_trt=0.51, seed=11))
report("A null (D=0) ", simulate_setting(**A, recall_ctrl=0.61, recall_trt=0.61, seed=12))

# --- Fill-in B: equally spec-compliant optimistic parameters ---
# K=30 clusters (top of the pinned 20-30 band), 2000 honest (top of floor),
# 400 positives, ICC=0.05, rho=0.9, baseline recall 0.75 (inside [0.4,0.9]).
Bp = dict(K=30, n_h=2010, n_d=390, icc=0.05, rho=0.9)
report("B sup (D=.10)", simulate_setting(**Bp, recall_ctrl=0.75, recall_trt=0.65, seed=21))
report("B null (D=0) ", simulate_setting(**Bp, recall_ctrl=0.75, recall_trt=0.75, seed=22))

# --- Fill-in C: pessimistic but spec-compliant (K=20 bottom of band, ICC=.25, rho=.6) ---
Cp = dict(K=20, n_h=1500, n_d=200, icc=0.25, rho=0.6)
report("C sup (D=.10)", simulate_setting(**Cp, recall_ctrl=0.61, recall_trt=0.51, seed=31))
