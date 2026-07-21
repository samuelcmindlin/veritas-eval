"""Independent re-implementation (skeptic pass) of the percentile-vs-basic
bootstrap size question for paired Delta-recall@1%FPR with per-replicate
threshold re-estimation, cluster (outer) + seed (inner) nested bootstrap.

Deliberately different from the original sim:
  - explicit cluster-index resampling (not multinomial weights)
  - np.quantile (linear interpolation) for the 1%FPR threshold
  - uniform-ish honest allocation across clusters
  - cluster sizes ~ randint(2,17) (not lognormal)
  - includes a real seed layer (K=5, seed x condition effects on positives,
    statistic = mean over seeds, inner bootstrap resamples seeds)

Under H0 (conditions A,B exchangeable), measure one-sided per-tail rejection
rates at nominal 0.05 from the 90% interval endpoints:
  percentile: q05 > 0  /  q95 < 0
  basic:      2T - q95 > 0  /  2T - q05 < 0
"""
import sys, time
import numpy as np

def run(G, icc, mc, B, K, sig_seed, seed, label=""):
    rng = np.random.default_rng(seed)
    tau = np.sqrt(icc)
    sig_pair = np.sqrt(0.5 * (1 - icc))   # shared pair effect across conditions
    sig_eps = np.sqrt(0.5 * (1 - icc))    # condition-specific
    mu_d = 2.9
    NH = 1750
    rej = np.zeros(4)  # perc_lo, perc_hi, basic_lo, basic_hi
    Ts, bsd = [], []
    for m in range(mc):
        sizes = rng.integers(2, 17, G)
        npos = sizes.sum()
        nh_g = np.full(G, NH // G); nh_g[: NH % G] += 1
        nh = nh_g.sum()
        u = rng.normal(0, tau, G)
        clh = np.repeat(np.arange(G), nh_g)
        clp = np.repeat(np.arange(G), sizes)
        # honest: cluster effect + rest to unit variance, per condition
        sh = np.sqrt(1 - icc)
        hA = u[clh] + rng.normal(0, sh, nh)
        hB = u[clh] + rng.normal(0, sh, nh)
        # positives: paired across conditions
        ep = rng.normal(0, sig_pair, npos)
        pA = mu_d + u[clp] + ep + rng.normal(0, sig_eps, npos)
        pB = mu_d + u[clp] + ep + rng.normal(0, sig_eps, npos)
        # seed x condition effects on positives (detector seeds)
        sA = rng.normal(0, sig_seed, K)
        sB = rng.normal(0, sig_seed, K)
        # per-cluster index slices
        h_idx = [np.where(clh == g)[0] for g in range(G)]
        p_idx = [np.where(clp == g)[0] for g in range(G)]

        def delta(gsel, ksel):
            hi = np.concatenate([h_idx[g] for g in gsel])
            pi = np.concatenate([p_idx[g] for g in gsel])
            thrA = np.quantile(hA[hi], 0.99)
            thrB = np.quantile(hB[hi], 0.99)
            # recall per seed, mean over selected seeds
            rA = (pA[pi][:, None] + sA[None, ksel] > thrA).mean()
            rB = (pB[pi][:, None] + sB[None, ksel] > thrB).mean()
            return rA - rB

        allg = np.arange(G); allk = np.arange(K)
        T = delta(allg, allk)
        gs = rng.integers(0, G, (B, G))
        ks = rng.integers(0, K, (B, K))
        d = np.array([delta(gs[b], ks[b]) for b in range(B)])
        q05, q95 = np.quantile(d, [0.05, 0.95])
        rej += [q05 > 0, q95 < 0, 2 * T - q95 > 0, 2 * T - q05 < 0]
        Ts.append(T); bsd.append(d.std())
    Ts = np.array(Ts); r = rej / mc
    se = np.sqrt(0.05 * 0.95 / mc)
    print(f"{label} G={G:2d} icc={icc:.2f} K={K} sig_seed={sig_seed:.2f} mc={mc} B={B} | "
          f"perc {r[0]:.3f}/{r[1]:.3f} basic {r[2]:.3f}/{r[3]:.3f} | "
          f"empSD={Ts.std():.4f} bootSD/empSD={np.mean(bsd)/Ts.std():.3f} "
          f"(MC se~{se:.3f})", flush=True)

if __name__ == "__main__":
    t0 = time.time()
    mc, B = 700, 249
    # nested (cluster+seed) versions
    run(20, 0.5, mc, B, K=5, sig_seed=0.15, seed=71, label="nest")
    run(20, 0.2, mc, B, K=5, sig_seed=0.15, seed=72, label="nest")
    run(30, 0.35, mc, B, K=5, sig_seed=0.15, seed=73, label="nest")
    # cluster-only (seed variance off) to compare with original sim directly
    run(20, 0.5, mc, B, K=1, sig_seed=0.0, seed=74, label="clus")
    run(30, 0.2, mc, B, K=1, sig_seed=0.0, seed=75, label="clus")
    print(f"total {time.time()-t0:.1f}s")
