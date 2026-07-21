"""Sims B & C (R1/R2/R3), Gaussian components (smooth-mean statistic).

B: nested bootstrap (clusters outer via multinomial counts, seeds inner via
   multinomial counts) on T = grand mean over instances x S seeds, crossed
   random effects (cluster, instance, seed, cluster-x-seed interaction, resid).
   Varies seed-variance share of Var(T) and S. Also the 'seeds fixed' (no inner
   resample) implementation to quantify the ambiguity cost.
C-a: DiD (three conditions per cluster, shared cluster effects), cluster
   percentile bootstrap keeping triplets intact.
C-b: cross-family difference-of-Deltas, same instances, seed effects correlated
   across families (rho_s=0.8): paired vs independent inner seed resampling.

All under H0 (true effect 0). One-sided nominal size 0.05 per tail from the
90% percentile interval endpoints.
"""
import numpy as np, time

rng_global = np.random.default_rng(20260707)

def sizes(rng, G):
    return np.clip(np.round(rng.lognormal(1.8, 0.6, G)).astype(int), 2, 30)

def simB(G=25, S=5, seed_share=0.3, mc=4000, B=499, inner="boot", seed=1):
    rng = np.random.default_rng(seed)
    sc2, se2, si2, sr2 = 1.0, 1.0, 0.5, 1.0
    rej_lo = rej_hi = 0
    for m in range(mc):
        n = sizes(rng, G); N = n.sum()
        # base variance of T excluding seed term
        V0 = sc2 * (n**2).sum() / N**2 + se2 / N + si2 * (n**2).sum() / (N**2 * S) + sr2 / (N * S)
        ss2 = S * (seed_share / (1 - seed_share)) * V0 if seed_share > 0 else 0.0
        c = rng.normal(0, np.sqrt(sc2), G)
        s = rng.normal(0, np.sqrt(ss2), S) if ss2 > 0 else np.zeros(S)
        inter = rng.normal(0, np.sqrt(si2), (G, S))
        E = rng.normal(0, np.sqrt(se2 * n))          # sum of instance effects per cluster
        Z = rng.normal(0, np.sqrt(sr2 * n)[:, None], (G, S))
        M = n[:, None] * (c[:, None] + s[None, :] + inter) + E[:, None] + Z  # G x S cluster-sums
        W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(float)
        if inner == "boot":
            V = rng.multinomial(S, np.full(S, 1 / S), size=B).astype(float)
        else:
            V = np.ones((B, S))
        num = np.einsum('bg,gk,bk->b', W, M, V)
        den = (W @ n) * V.sum(1)
        d = num / den
        q05, q95 = np.quantile(d, [0.05, 0.95])
        rej_lo += q05 > 0; rej_hi += q95 < 0
    return rej_lo / mc, rej_hi / mc

def simCa(G=25, mc=5000, B=499, seed=2):
    rng = np.random.default_rng(seed)
    rej_lo = rej_hi = 0
    for m in range(mc):
        n = sizes(rng, G)
        c = rng.normal(0, 1, G)                       # shared cluster effect
        a = rng.normal(0, 0.7, G); b = rng.normal(0, 0.7, G)  # condition-pair specific
        # cluster-mean deltas (instance noise folded in, var/n_g)
        d1 = c + a + rng.normal(0, 1 / np.sqrt(n))
        d2 = c + b + rng.normal(0, 1 / np.sqrt(n))
        W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(float)
        den = W @ n
        stat = (W @ (n * d1) - W @ (n * d2)) / den    # triplets intact: same W
        T = (n * (d1 - d2)).sum() / n.sum()
        q05, q95 = np.quantile(stat, [0.05, 0.95])
        rej_lo += q05 > 0; rej_hi += q95 < 0
    return rej_lo / mc, rej_hi / mc

def simCb(G=25, S=5, rho_s=0.8, mc=4000, B=499, paired=True, seed=3):
    rng = np.random.default_rng(seed)
    ss = 0.15   # seed effect sd per family (statistic-level)
    rej_lo = rej_hi = 0; widths = []
    for m in range(mc):
        n = sizes(rng, G)
        c = rng.normal(0, 1, G)
        c1 = c + rng.normal(0, 0.5, G); c2 = c + rng.normal(0, 0.5, G)  # corr cluster effects
        z = rng.normal(0, ss, S)
        s1 = rho_s * z + np.sqrt(1 - rho_s**2) * rng.normal(0, ss, S)
        s2 = rho_s * z + np.sqrt(1 - rho_s**2) * rng.normal(0, ss, S)
        e1 = rng.normal(0, 1 / np.sqrt(n)[:, None], (G, S))
        e2 = rng.normal(0, 1 / np.sqrt(n)[:, None], (G, S))
        D1 = c1[:, None] + s1[None, :] + e1          # G x S per-cluster Delta, family 1
        D2 = c2[:, None] + s2[None, :] + e2
        W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(float)
        V1 = rng.multinomial(S, np.full(S, 1 / S), size=B).astype(float)
        V2 = V1 if paired else rng.multinomial(S, np.full(S, 1 / S), size=B).astype(float)
        den1 = (W @ n) * V1.sum(1); den2 = (W @ n) * V2.sum(1)
        t1 = np.einsum('bg,gk,bk->b', W, n[:, None] * D1, V1) / den1
        t2 = np.einsum('bg,gk,bk->b', W, n[:, None] * D2, V2) / den2
        d = t1 - t2
        q05, q95 = np.quantile(d, [0.05, 0.95])
        rej_lo += q05 > 0; rej_hi += q95 < 0; widths.append(q95 - q05)
    return rej_lo / mc, rej_hi / mc, float(np.mean(widths))

if __name__ == "__main__":
    t0 = time.time()
    print("== B: nested seed x cluster (G=25) ==")
    for S, share, inner in [(5, 0.0, "boot"), (5, 0.3, "boot"), (5, 0.6, "boot"),
                            (10, 0.3, "boot"), (10, 0.6, "boot"),
                            (5, 0.3, "fixed"), (5, 0.6, "fixed")]:
        lo, hi = simB(S=S, seed_share=share, inner=inner, seed=int(10 + S + share * 10))
        print(f"S={S:2d} seed_share={share:.1f} inner={inner:5s} -> size {lo:.3f}/{hi:.3f}", flush=True)
    print("== B extra: smooth mean, cluster-only, G sweep (seed_share=0) ==")
    for G in (20, 25, 30):
        lo, hi = simB(G=G, S=5, seed_share=0.0, inner="boot", seed=100 + G)
        print(f"G={G} -> size {lo:.3f}/{hi:.3f}", flush=True)
    print("== C-a: DiD triplets intact ==")
    for G in (20, 25, 30):
        lo, hi = simCa(G=G, seed=200 + G)
        print(f"G={G} -> size {lo:.3f}/{hi:.3f}", flush=True)
    print("== C-b: cross-family, shared seeds rho_s=0.8 ==")
    for paired in (True, False):
        lo, hi, w = simCb(paired=paired, seed=300 + paired)
        print(f"paired_seed_resample={paired} -> size {lo:.3f}/{hi:.3f} meanCIwidth={w:.4f}", flush=True)
    print(f"total {time.time()-t0:.1f}s")
