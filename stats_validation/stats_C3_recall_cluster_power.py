"""C3 support: is SESOI=0.10 on paired Delta-recall@1%FPR plausibly powered at the
stated floors (25 clusters, 1600 honest + 300 deceptive per condition), and does the
25-cluster outer bootstrap CI have acceptable coverage/level?
Per-condition threshold = 99th pct of that condition's honest scores, re-estimated in
every bootstrap replicate. Cluster RE ICC=0.15, within-pair corr rho=0.8.
Baseline recall ~0.54 (inside the [0.4,0.9] band). Seeded.
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20260708)
K, NH, ND = 25, 1600, 300           # clusters, honest/condition, deceptive/condition
tau2, rho = 0.15, 0.8
sig = np.sqrt(1 - tau2)             # marginal var 1
MU_D = 2.6
R1, RB = 400, 300

# choose treated deceptive shift so TRUE paired delta-recall = 0.10
t_pop = norm.ppf(0.99)              # marginal honest ~ N(0,1)
rec_c = 1 - norm.cdf(t_pop - MU_D)  # ~0.54... (sd 1 marginal)
delta_shift = MU_D - (t_pop - norm.ppf(1 - (rec_c - 0.10)))
print(f"baseline recall={rec_c:.3f}, shift for true drop 0.10: {delta_shift:.3f}")

nh_k, nd_k = NH // K, ND // K

def draw_study(true_shift, rng):
    """returns per-cluster arrays: hc,ht (honest ctrl/treat), dc,dt"""
    u = rng.normal(0, np.sqrt(tau2), K)
    e1h = rng.normal(0, sig, (K, nh_k)); e2h = rng.normal(0, sig, (K, nh_k))
    e1d = rng.normal(0, sig, (K, nd_k)); e2d = rng.normal(0, sig, (K, nd_k))
    hc = u[:, None] + e1h
    ht = u[:, None] + rho * e1h + np.sqrt(1 - rho**2) * e2h
    dc = MU_D + u[:, None] + e1d
    dt = MU_D - true_shift + u[:, None] + rho * e1d + np.sqrt(1 - rho**2) * e2d
    return hc, ht, dc, dt

def delta_recall(hc, ht, dc, dt):
    tc = np.quantile(hc.ravel(), 0.99); tt = np.quantile(ht.ravel(), 0.99)
    return (dc.ravel() > tc).mean() - (dt.ravel() > tt).mean()

def one_rep(true_shift, rng):
    hc, ht, dc, dt = draw_study(true_shift, rng)
    ks = rng.integers(0, K, (RB, K))                    # cluster resample
    Ls, Us = np.empty(RB), np.empty(RB)
    boots = np.empty(RB)
    for b in range(RB):
        k = ks[b]
        boots[b] = delta_recall(hc[k], ht[k], dc[k], dt[k])
    L, U = np.quantile(boots, [0.05, 0.95])             # two-sided 90% CI
    return L, U

for true_shift, label in [(delta_shift, "TRUE DELTA = 0.10 (SESOI)"), (0.0, "TRUE DELTA = 0")]:
    res = np.array([one_rep(true_shift, rng) for _ in range(R1)])
    L, U = res[:, 0], res[:, 1]
    true_d = 0.10 if true_shift > 0 else 0.0
    print(f"\n{label}: 25 clusters, {NH}h/{ND}d per condition, {R1} reps x {RB} boot")
    print(f"  P(L>0)  [superiority power / type-I] = {(L > 0).mean():.3f}")
    print(f"  P(U<0.10) [equivalence power at SESOI] = {(U < 0.10).mean():.3f}")
    print(f"  CI coverage of true delta = {((L <= true_d) & (true_d <= U)).mean():.3f} (nominal 0.90)")
    print(f"  mean CI = [{L.mean():.3f}, {U.mean():.3f}]  mean width = {(U - L).mean():.3f}")
