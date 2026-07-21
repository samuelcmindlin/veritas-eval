"""SKEPTIC re-check of Brier reliability plugin bias finding.
Independent implementation. Setup mirrors prereg: scores ~ mix N(0,1) honest /
N(1.8,1) deceptive (AUROC~.90), Platt map frozen on a 2000-pt control calibration
split, B=10 equal-mass bins. Check:
 (1) plugin REL bias at n=200,500,1000,2000 vs binned-population truth (dh=0)
 (2) analytic prediction bias ~ mean_b Var_b(y-p)/(n/B)
 (3) two-component CAL+REF vs Brier: residual = within-bin var(p) - 2cov(p,y)
Seeded rng(42). Also n=3500 (realistic full eval set: 1750 neg + positives).
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(42)
MU, B = 1.8, 10

# frozen Platt on control calibration split
nc = 2000
sc = np.concatenate([rng.normal(0, 1, nc // 2), rng.normal(MU, 1, nc // 2)])
yc = np.concatenate([np.zeros(nc // 2), np.ones(nc // 2)])
def nll(ab):
    z = ab[0] * sc + ab[1]
    return np.mean(np.logaddexp(0, z) - yc * z)
a, b = minimize(nll, [1.4, -1.6], method="Nelder-Mead").x
print(f"Platt a={a:.3f} b={b:.3f}")
platt = lambda s: 1 / (1 + np.exp(-(a * s + b)))

def decomp(P, Y):
    """P,Y (R,n) -> plugin REL, REF(=mean yb(1-yb)), Brier, per-row."""
    R, n = P.shape
    o = np.argsort(P, 1)
    Ps = np.take_along_axis(P, o, 1).reshape(R, B, n // B)
    Ys = np.take_along_axis(Y, o, 1).reshape(R, B, n // B)
    pb, yb = Ps.mean(2), Ys.mean(2)
    rel = ((pb - yb) ** 2).mean(1)
    ref = (yb * (1 - yb)).mean(1)
    bs = ((P - Y) ** 2).mean(1)
    return rel, ref, bs

def draw(R, n, g):
    y = np.broadcast_to(np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]), (R, n)).copy()
    s = np.where(y == 1, MU, 0.0) + g.normal(0, 1, (R, n))
    return platt(s), y

# population truth (binned REL under frozen map, dh=0)
gp = np.random.default_rng(7)
Pp, Yp = draw(1, 2_000_000, gp)
trel, tref, tbs = decomp(Pp, Yp)
print(f"true binned REL={trel[0]:.6f}  Brier={tbs[0]:.4f}  CAL+REF-BS={trel[0]+tref[0]-tbs[0]:+.5f}")

# analytic bias predictor: mean p(1-p) / (n/B)
mp = Pp[0]
avg_var = np.mean(mp * (1 - mp))
print(f"avg p(1-p)={avg_var:.4f}")

print(f"\n{'n':>5} {'REL bias':>10} {'analytic':>10} {'CAL+REF-BS mean':>16}")
for n in [200, 500, 1000, 2000, 3500]:
    P, Y = draw(3000, n - n % B, rng)
    rel, ref, bs = decomp(P, Y)
    print(f"{n:>5} {rel.mean()-trel[0]:+10.5f} {avg_var/((n - n % B)/B):+10.5f} "
          f"{(rel+ref-bs).mean():+16.5f}")
