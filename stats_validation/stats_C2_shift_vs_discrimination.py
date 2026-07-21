"""C2: what DeltaECE under a frozen Platt map responds to.
Population (N=2e6) signatures for:
  (a) pure location shift, both classes +0.5 (discrimination INTACT)
  (b) separation loss: deceptive mean 1.8 -> 1.1 (genuine discrimination loss)
  (c) variance inflation: both class SDs 1 -> 1.5, means fixed (discrimination loss,
      per-class MEAN score shift = 0)  <- the case mean-shift reporting cannot flag
Reports: AUROC, frozen-map ECE, per-class mean raw-score shift, per-class mean
predicted-prob shift, Brier + reliability/resolution.
"""
import numpy as np

rng0 = np.random.default_rng(11)
MU, B = 1.8, 10
N = 2_000_000

# frozen Platt fit on control calibration split (same as C1 script, refit here)
from scipy.optimize import minimize
nc = 2000
s_cal = np.concatenate([rng0.normal(0, 1, nc // 2), rng0.normal(MU, 1, nc // 2)])
y_cal = np.concatenate([np.zeros(nc // 2), np.ones(nc // 2)])
def nll(ab):
    a, b = ab
    z = a * s_cal + b
    return np.mean(np.logaddexp(0, z) - y_cal * z)
A, Bb = minimize(nll, [1.0, -MU / 2], method="Nelder-Mead").x
platt = lambda s: 1.0 / (1.0 + np.exp(-(A * s + Bb)))

def stats(s, y):
    # AUROC via rank
    r = np.argsort(np.argsort(s))
    auroc = (r[y == 1].mean() - (y == 1).sum() / 2 + 0.5) / (y == 0).sum()
    p = platt(s)
    idx = np.argsort(p)
    Ps, Ys = p[idx].reshape(B, -1), y[idx].reshape(B, -1)
    d = Ps.mean(1) - Ys.mean(1)
    ece = np.abs(d).mean()
    rel = (d ** 2).mean()
    resl = ((Ys.mean(1) - y.mean()) ** 2).mean()
    brier = np.mean((p - y) ** 2)
    return auroc, ece, brier, rel, resl, s[y == 0].mean(), s[y == 1].mean(), p[y == 0].mean(), p[y == 1].mean()

def make(mu_h, mu_d, sd, seed=3):
    r = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(N // 2), np.ones(N // 2)])
    s = np.where(y == 1, mu_d, mu_h) + sd * r.normal(0, 1, N)
    return s, y

rows = {
    "control            ": make(0.0, MU, 1.0),
    "(a) both +0.5      ": make(0.5, MU + 0.5, 1.0),
    "(b) sep loss d->1.1": make(0.0, 1.1, 1.0),
    "(c) var infl sd=1.5": make(0.0, MU, 1.5),
}
base = stats(*rows["control            "])
print(f"{'scenario':<21} {'AUROC':>6} {'ECE':>6} {'Brier':>6} {'REL':>7} {'RES':>7} {'dMean_h':>8} {'dMean_d':>8} {'dP_h':>7} {'dP_d':>7}")
for k, (s, y) in rows.items():
    a, e, br, rl, rs, mh, md, ph, pd = stats(s, y)
    print(f"{k:<21} {a:6.3f} {e:6.3f} {br:6.3f} {rl:7.4f} {rs:7.4f} "
          f"{mh - base[5]:8.3f} {md - base[6]:8.3f} {ph - base[7]:7.3f} {pd - base[8]:7.3f}")
