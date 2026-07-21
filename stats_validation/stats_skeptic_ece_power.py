"""Independent re-implementation: power of one-sided paired-bootstrap Delta-ECE test
under a frozen Platt map, 10 equal-mass bins, debiased (Kumar-style l2) and plug-in l1 ECE.

Skeptic pass: written from scratch, seed 12345 (different from original finding's 20260707).

Setup:
 - detector scores: honest ~ N(0,1), deceptive ~ N(2,1)  (AUC ~= 0.92, plausible strong detector)
 - Platt map fit ONCE on a control-condition calibration split (1000+1000), frozen.
 - Treated condition: honest-class scores shift by +delta (in SD units) toward deceptive;
   deceptive unchanged. Paired instances (same latent draw, shift added).
 - Eval set: n per condition, 50/50 class mix, paired across conditions.
 - ECE per condition with 10 equal-mass bins on that condition's own predicted probs.
 - Paired nonparametric bootstrap over instances (B reps), one-sided test:
   reject if alpha-quantile of bootstrap Delta-ECE > 0.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

rng = np.random.default_rng(12345)

MU_D = 2.0
NBINS = 10

def fit_platt(s, y):
    def nll(theta):
        a, b = theta
        z = a * s + b
        # stable logloss
        return np.mean(np.logaddexp(0, z) - y * z)
    res = minimize(nll, x0=[1.0, -1.0], method="Nelder-Mead")
    return res.x

# ---- frozen Platt map from control calibration split ----
ncal = 1000
s_cal = np.concatenate([rng.normal(0, 1, ncal), rng.normal(MU_D, 1, ncal)])
y_cal = np.concatenate([np.zeros(ncal), np.ones(ncal)])
a_hat, b_hat = fit_platt(s_cal, y_cal)
print(f"frozen Platt: a={a_hat:.4f} b={b_hat:.4f}  (Bayes-optimal a=2, b=-2 for these gaussians)")

def platt(s):
    return expit(a_hat * s + b_hat)

def ece_batch(P, Y):
    """P,Y: (B, n). Equal-mass NBINS bins per row on P. Returns (plugin_l1, debiased_l2sqrt) each (B,)."""
    B, n = P.shape
    nb = n // NBINS
    order = np.argsort(P, axis=1)
    Ps = np.take_along_axis(P, order, axis=1).reshape(B, NBINS, nb)
    Ys = np.take_along_axis(Y, order, axis=1).reshape(B, NBINS, nb)
    pbar = Ps.mean(axis=2)
    ybar = Ys.mean(axis=2)
    diff = pbar - ybar
    plugin_l1 = np.abs(diff).mean(axis=1)              # equal-mass -> equal weights
    D = Ps - Ys                                        # per-item p - y
    var_mean = D.var(axis=2, ddof=1) / nb              # var of the bin mean difference
    deb_sq = (diff**2 - var_mean).mean(axis=1)
    debiased = np.sqrt(np.clip(deb_sq, 0, None))
    return plugin_l1, debiased

# ---- true (population, binned) Delta-ECE under the frozen map ----
def true_delta(shift, N=2_000_000):
    r = np.random.default_rng(777)
    sh = r.normal(0, 1, N // 2)
    sd = r.normal(MU_D, 1, N // 2)
    y = np.concatenate([np.zeros(N // 2), np.ones(N // 2)])
    for label, s in (("control", np.concatenate([sh, sd])),
                     ("treated", np.concatenate([sh + shift, sd]))):
        l1, l2 = ece_batch(platt(s)[None, :], y[None, :])
        print(f"  shift={shift} {label}: ECE_l1={l1[0]:.4f}  ECE_deb={l2[0]:.4f}")
        if label == "control":
            c1, c2 = l1[0], l2[0]
    return  # printed inline

for sh in (0.5, 0.2):
    true_delta(sh)

# ---- power simulation ----
def power(n, shift, reps=400, B=400):
    r = np.random.default_rng(42 + n + int(shift * 1000))
    rej = {("l1", 0.05): 0, ("l1", 0.05/6): 0, ("deb", 0.05): 0, ("deb", 0.05/6): 0}
    for _ in range(reps):
        nh = n // 2
        sh_c = r.normal(0, 1, nh)
        sd_c = r.normal(MU_D, 1, n - nh)
        s_c = np.concatenate([sh_c, sd_c])
        s_t = np.concatenate([sh_c + shift, sd_c])   # paired: same draws, honest shifted
        y = np.concatenate([np.zeros(nh), np.ones(n - nh)])
        p_c, p_t = platt(s_c), platt(s_t)
        idx = r.integers(0, n, size=(B, n))          # paired resampling
        l1_c, deb_c = ece_batch(p_c[idx], y[idx])
        l1_t, deb_t = ece_batch(p_t[idx], y[idx])
        d_l1 = l1_t - l1_c
        d_deb = deb_t - deb_c
        for name, d in (("l1", d_l1), ("deb", d_deb)):
            for al in (0.05, 0.05/6):
                if np.quantile(d, al) > 0:
                    rej[(name, al)] += 1
    return {k: v / reps for k, v in rej.items()}

print("\nn/cond  shift   pow_l1@.05  pow_l1@.0083  pow_deb@.05  pow_deb@.0083")
for shift in (0.5, 0.2):
    for n in (200, 500, 1000):
        p = power(n, shift)
        print(f"{n:5d}  {shift:.1f}    {p[('l1',0.05)]:.3f}       {p[('l1',0.05/6)]:.3f}         "
              f"{p[('deb',0.05)]:.3f}        {p[('deb',0.05/6)]:.3f}")
