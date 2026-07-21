"""Skeptic add-on: per-tail size of the BCa interval (scipy's default method)
for the same statistic, cluster jackknife for acceleration."""
import time, math
import numpy as np

class norm:
    @staticmethod
    def cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    @staticmethod
    def ppf(p):
        # bisection, fine for this use
        lo, hi = -10.0, 10.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if norm.cdf(mid) < p: lo = mid
            else: hi = mid
        return (lo + hi) / 2

def run(G, icc, mc, B, K, sig_seed, seed, label=""):
    rng = np.random.default_rng(seed)
    tau = np.sqrt(icc)
    sig_pair = np.sqrt(0.5 * (1 - icc)); sig_eps = np.sqrt(0.5 * (1 - icc))
    mu_d = 2.9; NH = 1750
    z05, z95 = norm.ppf(0.05), norm.ppf(0.95)
    rej = np.zeros(6)  # perc lo/hi, basic lo/hi, bca lo/hi
    for m in range(mc):
        sizes = rng.integers(2, 17, G)
        nh_g = np.full(G, NH // G); nh_g[: NH % G] += 1
        u = rng.normal(0, tau, G)
        clh = np.repeat(np.arange(G), nh_g); nh = clh.size
        clp = np.repeat(np.arange(G), sizes); npos = clp.size
        sh = np.sqrt(1 - icc)
        hA = u[clh] + rng.normal(0, sh, nh)
        hB = u[clh] + rng.normal(0, sh, nh)
        ep = rng.normal(0, sig_pair, npos)
        pA = mu_d + u[clp] + ep + rng.normal(0, sig_eps, npos)
        pB = mu_d + u[clp] + ep + rng.normal(0, sig_eps, npos)
        sA = rng.normal(0, sig_seed, K); sB = rng.normal(0, sig_seed, K)
        h_idx = [np.where(clh == g)[0] for g in range(G)]
        p_idx = [np.where(clp == g)[0] for g in range(G)]

        def delta(gsel, ksel):
            hi = np.concatenate([h_idx[g] for g in gsel])
            pi = np.concatenate([p_idx[g] for g in gsel])
            thrA = np.quantile(hA[hi], 0.99); thrB = np.quantile(hB[hi], 0.99)
            rA = (pA[pi][:, None] + sA[None, ksel] > thrA).mean()
            rB = (pB[pi][:, None] + sB[None, ksel] > thrB).mean()
            return rA - rB

        allg = np.arange(G); allk = np.arange(K)
        T = delta(allg, allk)
        gs = rng.integers(0, G, (B, G)); ks = rng.integers(0, K, (B, K))
        d = np.array([delta(gs[b], ks[b]) for b in range(B)])
        q05, q95 = np.quantile(d, [0.05, 0.95])
        # BCa: z0 from boot dist, acceleration from leave-one-cluster-out jackknife
        frac = np.clip((d < T).mean(), 1 / (B + 1), B / (B + 1))
        z0 = norm.ppf(frac)
        Tj = np.array([delta(np.delete(allg, g), allk) for g in range(G)])
        dj = Tj.mean() - Tj
        denom = (dj ** 2).sum() ** 1.5
        a = (dj ** 3).sum() / (6 * denom) if denom > 0 else 0.0
        a1 = norm.cdf(z0 + (z0 + z05) / (1 - a * (z0 + z05)))
        a2 = norm.cdf(z0 + (z0 + z95) / (1 - a * (z0 + z95)))
        lo_bca, hi_bca = np.quantile(d, np.clip([a1, a2], 0, 1))
        rej += [q05 > 0, q95 < 0, 2 * T - q95 > 0, 2 * T - q05 < 0,
                lo_bca > 0, hi_bca < 0]
    r = rej / mc
    print(f"{label} G={G:2d} icc={icc:.2f} K={K} | perc {r[0]:.3f}/{r[1]:.3f} "
          f"basic {r[2]:.3f}/{r[3]:.3f} BCa {r[4]:.3f}/{r[5]:.3f}", flush=True)

if __name__ == "__main__":
    t0 = time.time()
    run(20, 0.5, 600, 249, K=1, sig_seed=0.0, seed=81, label="clus")
    run(20, 0.2, 600, 249, K=5, sig_seed=0.15, seed=82, label="nest")
    print(f"total {time.time()-t0:.1f}s")
