"""Independent re-implementation: equivalence power of the Robust branch at the
cluster floor. Full pipeline: paired conditions on shared instances, cluster
random effects + cluster x condition interaction, 5 detector seeds, per-condition
1%-FPR threshold re-estimated from ~1750 negatives, nested cluster (outer) +
seed (inner) bootstrap with threshold re-estimated in every replicate.
90% two-sided percentile CI [L,U]; equivalence claim = U < SESOI (0.10).
"""
import numpy as np, time

SESOI = 0.10
ALPHA_Q = (5.0, 95.0)

# ---- variance components (latent score scale; negatives ~ total SD ~1) ----
SIG_U   = 0.50   # cluster main effect (positives), shared across conditions
SIG_W   = 0.15   # cluster x condition interaction (does NOT cancel in Delta)
SIG_PI  = 0.60   # positive instance effect, shared across cond & seeds
SIG_PQ  = 0.50   # positive instance x condition
SIG_PR  = 0.40   # positive instance x seed idiosyncratic
SIG_DS  = 0.05   # seed effect on separation (shared across cond)
SIG_DSC = 0.04   # seed x condition effect on separation
NA_A    = 0.75   # negative instance effect shared across cond & seeds
NA_B    = 0.45   # negative instance x condition
NA_C    = 0.50   # negative idiosyncratic (x seed)
MU_POS  = 2.60   # positive mean -> baseline recall in the [0.4,0.9] band

def true_recall(shift, n=4_000_000, seed=1):
    rng = np.random.default_rng(seed)
    negsd = np.sqrt(NA_A**2 + NA_B**2 + NA_C**2)
    thr = 2.3263 * negsd  # true 1% quantile of negatives
    tot = np.sqrt(SIG_U**2 + SIG_W**2 + SIG_PI**2 + SIG_PQ**2 + SIG_PR**2
                  + SIG_DS**2 + SIG_DSC**2)
    x = rng.standard_normal(n) * tot + MU_POS - shift
    return (x > thr).mean()

def calibrate_shift(target_delta):
    if target_delta == 0.0:
        return 0.0
    r0 = true_recall(0.0)
    lo, hi = 0.0, 1.2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if r0 - true_recall(mid) < target_delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def run(G, true_delta, mc=300, B=250, m=10, S=5, Nneg=1750, seed0=12345):
    rng = np.random.default_rng(seed0)
    shift = calibrate_shift(true_delta)
    k = int(np.ceil(0.99 * Nneg)) - 1  # 0-indexed order stat for 99th pct
    hits_L, hits_Ueq, widths, dhats = [], [], [], []
    for it in range(mc):
        # ---------- generate one dataset ----------
        u  = rng.normal(0, SIG_U, G)                      # cluster main
        w  = rng.normal(0, SIG_W, (2, G))                 # cluster x cond
        pi = rng.normal(0, SIG_PI, (G, m))                # instance shared
        pq = rng.normal(0, SIG_PQ, (2, G, m))             # instance x cond
        pr = rng.normal(0, SIG_PR, (S, 2, G, m))          # x seed
        ds = rng.normal(0, SIG_DS, S)                     # seed on separation
        dsc= rng.normal(0, SIG_DSC, (S, 2))
        mu_c = np.array([MU_POS, MU_POS - shift])
        pos = (mu_c[None, :, None, None] + u[None, None, :, None]
               + w[None, :, :, None] + pi[None, None, :, :]
               + pq[None, :, :, :] + pr
               + ds[:, None, None, None] + dsc[:, :, None, None])  # (S,2,G,m)
        na = rng.normal(0, NA_A, Nneg)
        nb = rng.normal(0, NA_B, (2, Nneg))
        nc = rng.normal(0, NA_C, (S, 2, Nneg))
        neg = na[None, None, :] + nb[None, :, :] + nc      # (S,2,Nneg)
        # ---------- point estimate ----------
        thr = np.partition(neg, k, axis=-1)[..., k]        # (S,2)
        rec = (pos > thr[:, :, None, None]).mean(axis=(2, 3))  # (S,2)
        dhat = (rec[:, 0] - rec[:, 1]).mean()
        dhats.append(dhat)
        # ---------- nested bootstrap ----------
        cidx = rng.integers(0, G, (B, G))                  # outer: clusters
        nidx = rng.integers(0, Nneg, (B, Nneg))            # outer: neg instances
        sidx = rng.integers(0, S, (B, S))                  # inner: seeds
        # thresholds per replicate: neg (S,2,Nneg) -> (S,2,B,Nneg)
        negb = neg[:, :, nidx]                             # (S,2,B,Nneg)
        thrb = np.partition(negb, k, axis=-1)[..., k]      # (S,2,B)
        # per-cluster recall at replicate thresholds
        above = pos[:, :, None, :, :] > thrb[:, :, :, None, None]  # (S,2,B,G,m)
        crec = above.mean(axis=-1)                         # (S,2,B,G)
        # gather resampled clusters, mean over G
        b_ar = np.arange(B)
        crec_b = crec[:, :, b_ar[:, None], cidx]           # (S,2,B,G)
        rec_b = crec_b.mean(axis=-1)                       # (S,2,B)
        # inner seed resample & average, Delta per replicate
        d_seed = rec_b[:, 0, :] - rec_b[:, 1, :]           # (S,B)
        d_star = d_seed[sidx.T, b_ar].mean(axis=0)         # (B,)
        L, U = np.percentile(d_star, ALPHA_Q)
        hits_L.append(L > 0)
        hits_Ueq.append(U < SESOI)
        widths.append(U - L)
    dhats = np.array(dhats)
    return dict(G=G, true_delta=true_delta,
                sd_dhat=dhats.std(ddof=1), mean_dhat=dhats.mean(),
                p_L_gt0=np.mean(hits_L), p_U_lt_SESOI=np.mean(hits_Ueq),
                mean_width=np.mean(widths), mc=mc)

if __name__ == "__main__":
    t0 = time.time()
    print("baseline true recall:", round(true_recall(0.0), 4))
    for G, td in [(20, 0.0), (30, 0.0), (20, 0.10), (50, 0.0)]:
        r = run(G, td, seed0=1000 + G + int(td * 100))
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in r.items()}, f"[{time.time()-t0:.0f}s]")
