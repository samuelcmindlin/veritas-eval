"""Independent verification of the ECE-estimator-dependence finding.

Setup: frozen Platt-style map produces predicted probs p; true conditional
prob q(p) differs slightly (miscalibration). Equal-mass binning (B bins,
boundaries from sample quantiles). Compare:
  - plugin ECE: sum w_b |acc_b - conf_b|
  - variance-debiased (KLM-style L1): sum w_b sqrt(max(d_b^2 - acc_b(1-acc_b)/(n_b-1), 0))
at n=200, across true ECE = ~0.01 (null) and larger (~0.08-0.10).
"""
import numpy as np

rng = np.random.default_rng(20260707)

def q_of_p(p, shift):
    # miscalibration: logit shift
    lo = np.log(p/(1-p)) + shift
    return 1/(1+np.exp(-lo))

def true_binned_ece(shift, B, a=1.3, b=4.0, N=2_000_000):
    p = rng.beta(a, b, N)
    q = q_of_p(p, shift)
    idx = np.argsort(p)
    p, q = p[idx], q[idx]
    bins = np.array_split(np.arange(N), B)
    ece = 0.0
    for bi in bins:
        ece += len(bi)/N * abs(np.mean(q[bi] - p[bi]))
    return ece

def sim(shift, n, B, reps=4000, a=1.3, b=4.0):
    plug = np.empty(reps); deb = np.empty(reps)
    for r in range(reps):
        p = rng.beta(a, b, n)
        q = q_of_p(p, shift)
        y = (rng.random(n) < q).astype(float)
        idx = np.argsort(p)
        p_s, y_s = p[idx], y[idx]
        bins = np.array_split(np.arange(n), B)
        e_p = 0.0; e_d = 0.0
        for bi in bins:
            nb = len(bi)
            acc = y_s[bi].mean(); conf = p_s[bi].mean()
            d = acc - conf
            e_p += nb/n * abs(d)
            var = acc*(1-acc)/(nb-1) if nb > 1 else 0.0
            e_d += nb/n * np.sqrt(max(d*d - var, 0.0))
        plug[r] = e_p; deb[r] = e_d
    return plug, deb

for shift, label in [(0.0, "perfect"), (0.12, "small ECE (null-ish)"), (1.0, "large ECE")]:
    t = true_binned_ece(shift, 10)
    for B in (10, 15):
        tB = true_binned_ece(shift, B)
        pl, de = sim(shift, 200, B)
        print(f"{label:22s} shift={shift:4.2f} B={B:2d} trueECE={tB:.4f} | "
              f"plugin mean={pl.mean():.4f} bias={pl.mean()-tB:+.4f} SD={pl.std():.4f} | "
              f"debiased mean={de.mean():.4f} bias={de.mean()-tB:+.4f} SD={de.std():.4f}")

# Delta attenuation: control true ECE small, treated true ECE large
tc = true_binned_ece(0.12, 10); tt = true_binned_ece(1.0, 10)
plc, dec = sim(0.12, 200, 10); plt_, det = sim(1.0, 200, 10)
print(f"\ntrue Delta = {tt-tc:.4f}")
print(f"plugin   Delta mean = {(plt_-plc).mean():.4f}  (bias {(plt_-plc).mean()-(tt-tc):+.4f})")
print(f"debiased Delta mean = {(det-dec).mean():.4f}  (bias {(det-dec).mean()-(tt-tc):+.4f})")

# same at n=1800 (honest-class-floor scale, in case ECE is computed on full condition set)
pl, de = sim(0.12, 1800, 10, reps=1500)
t = true_binned_ece(0.12, 10)
print(f"\nn=1800 null: trueECE={t:.4f} plugin bias={pl.mean()-t:+.4f} SD={pl.std():.4f} "
      f"debiased bias={de.mean()-t:+.4f} SD={de.std():.4f}")
