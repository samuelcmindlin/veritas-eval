"""Sim A2 (R2 robustness): same pipeline as Sim A but honest-class cluster effect
is attenuated (honest ICC ~0.05) while deceptive class keeps ICC 0.2/0.5.
Checks whether the percentile interval's behavior (conservative in Sim A) flips
anticonservative when threshold estimation is less cluster-dominated.
"""
import os, sys, time
import numpy as np
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling sim modules regardless of cwd
from stats_R1R4_simA_recall_boot import stat_from_weights

def run(G, icc_dec, icc_hon, mc, B, seed):
    rng = np.random.default_rng(seed)
    tcrit = st.t.ppf(0.95, G - 1)
    mu_d, n_h_total = 2.9, 1750
    rej = np.zeros(6); Ts, bsds = [], []
    for m in range(mc):
        npos_g = np.clip(np.round(rng.lognormal(1.8, 0.6, G)).astype(int), 2, 30)
        nh_g = np.maximum(np.round(n_h_total * npos_g / npos_g.sum()).astype(int), 2)
        ch = rng.normal(0, np.sqrt(icc_hon), G)
        cd = rng.normal(0, np.sqrt(icc_dec), G)
        seh = np.sqrt(0.4 * (1 - icc_hon)); snh = np.sqrt(0.6 * (1 - icc_hon))
        sed = np.sqrt(0.4 * (1 - icc_dec)); snd = np.sqrt(0.6 * (1 - icc_dec))
        cl_h = np.repeat(np.arange(G), nh_g); nh = cl_h.size
        eh = rng.normal(0, seh, nh)
        hA = ch[cl_h] + eh + rng.normal(0, snh, nh)
        hB = ch[cl_h] + eh + rng.normal(0, snh, nh)
        cl_p = np.repeat(np.arange(G), npos_g); npv = cl_p.size
        ep = rng.normal(0, sed, npv)
        pA = mu_d + cd[cl_p] + ep + rng.normal(0, snd, npv)
        pB = mu_d + cd[cl_p] + ep + rng.normal(0, snd, npv)
        oA = np.argsort(hA); oB = np.argsort(hB)
        args = (hA[oA], cl_h[oA], hB[oB], cl_h[oB], pA, pB, cl_p, npos_g)
        T = stat_from_weights(np.ones((1, G)), *args)[0]
        Tj = stat_from_weights(1.0 - np.eye(G), *args)
        se_jk = np.sqrt((G - 1) / G * ((Tj - Tj.mean()) ** 2).sum())
        d = stat_from_weights(rng.multinomial(G, np.full(G, 1 / G), size=B).astype(float), *args)
        q05, q95 = np.quantile(d, [0.05, 0.95])
        rej += [q05 > 0, q95 < 0, 2 * T - q95 > 0, 2 * T - q05 < 0,
                (se_jk > 0) and (T / se_jk > tcrit), (se_jk > 0) and (T / se_jk < -tcrit)]
        Ts.append(T); bsds.append(d.std())
    Ts = np.array(Ts); r = rej / mc
    print(f"G={G:2d} iccD={icc_dec} iccH={icc_hon} | perc {r[0]:.3f}/{r[1]:.3f} "
          f"basic {r[2]:.3f}/{r[3]:.3f} jack-t {r[4]:.3f}/{r[5]:.3f} | "
          f"empSD={Ts.std():.4f} bootSD/empSD={np.mean(bsds)/Ts.std():.3f}", flush=True)

if __name__ == "__main__":
    t0 = time.time()
    for G in (20, 25, 30):
        run(G, 0.35, 0.05, mc=1500, B=399, seed=500 + G)
    print(f"total {time.time()-t0:.1f}s")
