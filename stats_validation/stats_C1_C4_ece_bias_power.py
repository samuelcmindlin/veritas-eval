"""C1/C4: debiased equal-mass ECE under a FROZEN Platt map at small n.
Design mirror: Platt fit once on a control-condition calibration split (frozen),
applied to eval sets of n per condition (half honest / half deceptive).
Treated condition = honest-class score shift delta_h (the realistic miscalibration mode).
Outputs: (i) bias/SD of naive vs debiased ECE; (ii) power of one-sided paired-bootstrap
DeltaECE test at alpha=0.05 and BH-worst 0.05/6; (iii) Brier reliability-term bias (C4).
Seeded. Runtime target < 2 min.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

rng = np.random.default_rng(20260707)
MU_DEC = 1.8          # AUROC = Phi(1.8/sqrt2) ~= 0.898
B = 10                # equal-mass bins (low end of 10-15)

# ---- frozen Platt map: fit on calibration split (control condition), n_cal=2000
n_cal = 2000
s_cal = np.concatenate([rng.normal(0, 1, n_cal // 2), rng.normal(MU_DEC, 1, n_cal // 2)])
y_cal = np.concatenate([np.zeros(n_cal // 2), np.ones(n_cal // 2)])

def nll(ab):
    a, b = ab
    z = a * s_cal + b
    return np.mean(np.logaddexp(0, z) - y_cal * z)

res = minimize(nll, [1.0, -MU_DEC / 2], method="Nelder-Mead")
A, Bb = res.x
platt = lambda s: 1.0 / (1.0 + np.exp(-(A * s + Bb)))
print(f"frozen Platt: a={A:.3f} b={Bb:.3f}")

def ece_batch(P, Y):
    """P,Y: (R,n). Equal-mass B bins per row. Returns naive, debiased L1-ECE, REL(sq), RES(sq)."""
    R, n = P.shape
    idx = np.argsort(P, axis=1)
    Ps = np.take_along_axis(P, idx, 1).reshape(R, B, n // B)
    Ys = np.take_along_axis(Y, idx, 1).reshape(R, B, n // B)
    pb, yb = Ps.mean(2), Ys.mean(2)
    d = pb - yb
    naive = np.abs(d).mean(1)                       # equal-mass -> equal weights
    v = (Ys - Ps).var(2, ddof=1) / (n // B)         # Var(ybar-pbar) per bin
    deb = np.sqrt(np.clip(d * d - v, 0, None)).mean(1)
    rel = (d * d).mean(1)                           # Brier reliability (plugin)
    resl = ((yb - Ys.mean((1, 2), keepdims=True)[:, :, 0]) ** 2).mean(1)
    return naive, deb, rel, resl

def draw(Rn, n, dh, rng):
    """paired scores; honest shifted by dh in treated. within-pair corr rho=0.8 on noise."""
    rho = 0.8
    nh = n // 2
    y = np.concatenate([np.zeros(nh), np.ones(n - nh)])
    mu = np.where(y == 1, MU_DEC, 0.0)
    e1 = rng.normal(0, 1, (Rn, n)); e2 = rng.normal(0, 1, (Rn, n))
    sc = mu + e1
    st = mu + rho * e1 + np.sqrt(1 - rho ** 2) * e2 + dh * (y == 0)
    return sc, st, np.broadcast_to(y, (Rn, n)).copy()

# ---- population truths under the frozen map (N=2e6)
Npop = 2_000_000
def true_stats(dh):
    s, _, y = draw(1, Npop, dh, np.random.default_rng(7))
    _, st, _ = draw(1, Npop, dh, np.random.default_rng(7))
    p = platt(st)
    naive, deb, rel, resl = ece_batch(p, y)
    return naive[0], rel[0]

truths = {dh: true_stats(dh) for dh in [0.0, 0.25, 0.5]}
for dh, (e, r) in truths.items():
    print(f"true frozen-map ECE(dh={dh}): {e:.4f}  REL={r:.5f}")

# ---- (i) bias/SD of estimators
print("\n== bias/SD of ECE estimators (2000 reps) ==")
print(f"{'n':>5} {'dh':>5} {'trueECE':>8} {'naive bias':>11} {'naive SD':>9} {'deb bias':>9} {'deb SD':>7} {'REL bias':>9}")
for n in [200, 500, 1000]:
    for dh in [0.0, 0.25, 0.5]:
        _, st, y = draw(2000, n, dh, rng)
        p = platt(st)
        naive, deb, rel, _ = ece_batch(p, y)
        tE, tR = truths[dh]
        print(f"{n:>5} {dh:>5} {tE:8.4f} {naive.mean()-tE:11.4f} {naive.std():9.4f} "
              f"{deb.mean()-tE:9.4f} {deb.std():7.4f} {rel.mean()-tR:9.5f}")

# ---- (ii) power of one-sided DeltaECE test (paired cluster-free bootstrap; best case)
print("\n== power: one-sided DeltaECE>0, paired percentile bootstrap ==")
R1, RB = 400, 300
alphas = {"a=.05": 0.05, "BH-worst a=.0083": 0.05 / 6}
print(f"{'n':>5} {'dh':>5} {'trueDECE':>9} " + " ".join(f"{k:>17}" for k in alphas))
for n in [200, 500, 1000]:
    for dh in [0.0, 0.25, 0.5]:
        tD = truths[dh][0] - truths[0.0][0]
        rej = {k: 0 for k in alphas}
        for r in range(R1):
            sc, st, y = draw(1, n, dh, rng)
            sc, st, y = sc[0], st[0], y[0]
            bidx = rng.integers(0, n, (RB, n))
            pc, pt, yb = platt(sc[bidx]), platt(st[bidx]), y[bidx]
            dec = ece_batch(pt, yb)[0] - ece_batch(pc, yb)[0]
            for k, a in alphas.items():
                if np.quantile(dec, a) > 0:
                    rej[k] += 1
        print(f"{n:>5} {dh:>5} {tD:9.4f} " + " ".join(f"{rej[k]/R1:17.3f}" for k in alphas))
