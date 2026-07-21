"""D2: one-sided error at each endpoint of a two-sided 90% cluster bootstrap CI,
for percentile / basic / BCa, under skewed cluster-level effect distributions.
Duality claim: P(L > delta_true) = 0.05 (significance leg at delta=0) and
P(U < delta_true) = 0.05 (TOST leg at delta=SESOI). n=25 clusters.
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20260707)
NC = 25          # clusters
B = 1999         # bootstrap reps
R = 3000         # Monte Carlo reps
ALPHA = 0.05     # per tail (two-sided 90%)

def make_data(delta, skew_k, R, NC, rng):
    if skew_k is None:  # normal
        noise = rng.standard_normal((R, NC))
    else:               # standardized gamma: skewness = 2/sqrt(k)
        noise = (rng.gamma(skew_k, 1.0, (R, NC)) - skew_k) / np.sqrt(skew_k)
    return delta + 0.15 * noise   # cluster-level deltas, sd ~0.15 (recall-scale)

def endpoints(data, rng):
    """Return (L,U) for percentile, basic, BCa. data: (R, NC)."""
    R, NC = data.shape
    that = data.mean(axis=1)                       # (R,)
    Lp = np.empty(R); Up = np.empty(R)
    Lb = np.empty(R); Ub = np.empty(R)
    La = np.empty(R); Ua = np.empty(R)
    chunk = 250
    for s in range(0, R, chunk):
        d = data[s:s+chunk]                        # (c, NC)
        c = d.shape[0]
        idx = rng.integers(0, NC, (c, B, NC))
        boot = np.take_along_axis(d[:, None, :], idx, axis=2).mean(axis=2)  # (c,B)
        th = that[s:s+chunk][:, None]
        qlo = np.quantile(boot, ALPHA, axis=1)
        qhi = np.quantile(boot, 1-ALPHA, axis=1)
        Lp[s:s+c], Up[s:s+c] = qlo, qhi                       # percentile
        Lb[s:s+c], Ub[s:s+c] = 2*th[:,0]-qhi, 2*th[:,0]-qlo   # basic
        # BCa: z0 + jackknife acceleration over clusters
        z0 = norm.ppf(np.clip((boot < th).mean(axis=1), 1e-4, 1-1e-4))
        tot = d.sum(axis=1, keepdims=True)
        jack = (tot - d) / (NC - 1)                # (c, NC) leave-one-cluster-out
        jm = jack.mean(axis=1, keepdims=True)
        num = ((jm - jack)**3).sum(axis=1)
        den = 6.0 * (((jm - jack)**2).sum(axis=1))**1.5
        a = np.where(den > 0, num/np.maximum(den,1e-12), 0.0)
        for tail, (Lout, Uout) in [(ALPHA,(La,None)), (1-ALPHA,(None,Ua))]:
            z = norm.ppf(tail)
            adj = norm.cdf(z0 + (z0 + z)/(1 - a*(z0 + z)))
            q = np.take_along_axis(np.sort(boot,axis=1),
                                   np.clip((adj*(B-1)).astype(int),0,B-1)[:,None],axis=1)[:,0]
            if Lout is not None: Lout[s:s+c] = q
            else: Uout[s:s+c] = q
    return (Lp,Up),(Lb,Ub),(La,Ua)

print(f"{'skew':>8} {'leg':>6} | {'percentile':>10} {'basic':>10} {'BCa':>10}   (nominal 0.05)")
for skew_k, name in [(None,'normal'), (4,'sk=1.0'), (1,'sk=2.0')]:
    # significance leg: true delta = 0, error = P(L > 0)
    d0 = make_data(0.0, skew_k, R, NC, rng)
    (Lp,_),(Lb,_),(La,_) = endpoints(d0, rng)
    print(f"{name:>8} {'L>0':>6} | {np.mean(Lp>0):10.4f} {np.mean(Lb>0):10.4f} {np.mean(La>0):10.4f}")
    # TOST leg: true delta = SESOI = 0.10, error = P(U < 0.10)
    d1 = make_data(0.10, skew_k, R, NC, rng)
    (_,Up),(_,Ub),(_,Ua) = endpoints(d1, rng)
    print(f"{name:>8} {'U<S':>6} | {np.mean(Up<0.10):10.4f} {np.mean(Ub<0.10):10.4f} {np.mean(Ua<0.10):10.4f}")
mc_se = np.sqrt(0.05*0.95/R)
print(f"\nMC SE at p=0.05: {mc_se:.4f} (2SE band: {0.05-2*mc_se:.3f}-{0.05+2*mc_se:.3f})")
