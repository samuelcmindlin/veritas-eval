"""Sim A (R2): size of the one-sided alpha=0.05 test from a cluster percentile
bootstrap on paired Delta-recall@1%FPR, with per-condition threshold re-estimated
inside every replicate, under H0 (identical conditions, paired instances).

Full pipeline is simulated: honest (negative) scores per condition supply the 1%FPR
threshold; deceptive (positive) scores supply recall; both classes clustered by the
same G scenario clusters (shared cluster effect); instances paired across conditions.

Also computes: basic (reflected) bootstrap interval, and a cluster jackknife-t
(leave-one-cluster-out SE + t_{G-1}) as the candidate fix.

Rejection rules under H0 (nominal one-sided size 0.05 per tail, from the two-sided
90% interval endpoints as pre-registered):
  perc: 5th pct of boot dist > 0   (and 95th < 0 for other tail)
  basic: 2*T - q95 > 0             (and 2*T - q05 < 0)
  jack-t: T/SE_jk > t_{.95,G-1}    (and < -t)
"""
import sys, time
import numpy as np
from scipy import stats as st

def stat_from_weights(Wc, hA_s, clhA_s, hB_s, clhB_s, pA, pB, cl_p, npos_g):
    # Wc: R x G nonneg cluster weights. Returns delta = recallA - recallB per row.
    out = []
    for h_s, clh_s, p in ((hA_s, clhA_s, pA), (hB_s, clhB_s, pB)):
        Wh = Wc[:, clh_s].astype(np.float64)          # R x nh, honest sorted asc
        cw = np.cumsum(Wh, axis=1)
        tot = cw[:, -1:]
        idx = np.argmax(cw >= 0.99 * tot, axis=1)      # smallest score w/ cumfrac>=.99
        thr = h_s[idx]                                  # R
        Wp = Wc[:, cl_p].astype(np.float64)             # R x npos
        num = ((p[None, :] > thr[:, None]) * Wp).sum(1)
        den = Wp.sum(1)
        out.append(num / den)
    return out[0] - out[1]

def run_config(G, icc, mc, B, seed):
    rng = np.random.default_rng(seed)
    tcrit = st.t.ppf(0.95, G - 1)
    tau = np.sqrt(icc); se_ = np.sqrt(0.4 * (1 - icc)); sn = np.sqrt(0.6 * (1 - icc))
    mu_d = 2.9
    n_h_total = 1750
    rej = np.zeros(6)  # perc_lo, perc_hi, basic_lo, basic_hi, jack_lo, jack_hi
    Ts, bsds = [], []
    for m in range(mc):
        npos_g = np.clip(np.round(rng.lognormal(1.8, 0.6, G)).astype(int), 2, 30)
        nh_g = np.maximum(np.round(n_h_total * npos_g / npos_g.sum()).astype(int), 2)
        c = rng.normal(0, tau, G)
        cl_h = np.repeat(np.arange(G), nh_g); nh = cl_h.size
        eh = rng.normal(0, se_, nh)
        hA = c[cl_h] + eh + rng.normal(0, sn, nh)
        hB = c[cl_h] + eh + rng.normal(0, sn, nh)
        cl_p = np.repeat(np.arange(G), npos_g); npv = cl_p.size
        ep = rng.normal(0, se_, npv)
        pA = mu_d + c[cl_p] + ep + rng.normal(0, sn, npv)
        pB = mu_d + c[cl_p] + ep + rng.normal(0, sn, npv)
        oA = np.argsort(hA); hA_s, clhA_s = hA[oA], cl_h[oA]
        oB = np.argsort(hB); hB_s, clhB_s = hB[oB], cl_h[oB]
        args = (hA_s, clhA_s, hB_s, clhB_s, pA, pB, cl_p, npos_g)
        T = stat_from_weights(np.ones((1, G)), *args)[0]
        # jackknife
        Wj = 1.0 - np.eye(G)
        Tj = stat_from_weights(Wj, *args)
        se_jk = np.sqrt((G - 1) / G * ((Tj - Tj.mean()) ** 2).sum())
        # cluster bootstrap
        Wb = rng.multinomial(G, np.full(G, 1 / G), size=B)
        d = stat_from_weights(Wb, *args)
        q05, q95 = np.quantile(d, [0.05, 0.95])
        rej += [q05 > 0, q95 < 0,
                2 * T - q95 > 0, 2 * T - q05 < 0,
                (se_jk > 0) and (T / se_jk > tcrit), (se_jk > 0) and (T / se_jk < -tcrit)]
        Ts.append(T); bsds.append(d.std())
    Ts = np.array(Ts)
    r = rej / mc
    print(f"G={G:2d} ICC={icc:.1f} mc={mc} B={B} | "
          f"perc {r[0]:.3f}/{r[1]:.3f} basic {r[2]:.3f}/{r[3]:.3f} "
          f"jack-t {r[4]:.3f}/{r[5]:.3f} | empSD(T)={Ts.std():.4f} "
          f"meanBootSD={np.mean(bsds):.4f} ratio={np.mean(bsds)/Ts.std():.3f}", flush=True)

if __name__ == "__main__":
    mc = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 299
    t0 = time.time()
    for icc in (0.2, 0.5):
        for G in (20, 25, 30):
            run_config(G, icc, mc, B, seed=1000 + G + int(icc * 10))
    print(f"total {time.time()-t0:.1f}s")
