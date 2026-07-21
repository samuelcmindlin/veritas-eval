"""Skeptic re-verification of the attenuation/DiD finding.

Checks:
 (B1) On a shared triple-matched instance set, DiD = D(eval-deploy) - D(ctrl-deploy)
      collapses exactly to R_ctrl - R_eval (deploy leg cancels), and
      attenuation = uncontrolled - DiD = R_deploy - R_ctrl (= the partial-specificity
      control-condition statistic).
 (B2) With per-pair matched subsets (eval-deploy matched set != ctrl-deploy matched set,
      selection depending on condition-specific behavior), the cancellation breaks and
      the estimand changes (bias vs the triple-matched estimand).
 (C1) Joint (same-replicate) cluster bootstrap of attenuation vs combining two
      independent bootstraps: correlation between uncontrolled and controlled stats,
      SD comparison -> independent resampling overstates variance (conservative).
 (C2) Degenerate-replicate frequency: replicates where the controlled subset has too
      few negatives for a stable 1%-FPR threshold.
"""
import numpy as np

rng = np.random.default_rng(20260707)

# ---------- data generator: clustered, paired-across-conditions scores ----------
def gen_data(n_clusters=60, pos_per=5, neg_per=30,
             d_deploy=2.2, d_eval=1.7, d_ctrl=2.05, rho_cluster=0.3, rng=rng):
    """Returns per-condition scores for shared instances.
    Instance effect shared across conditions (pairing); cluster effect too."""
    cl = np.repeat(np.arange(n_clusters), pos_per + neg_per)
    n = len(cl)
    is_pos = np.tile(np.r_[np.ones(pos_per), np.zeros(neg_per)].astype(bool), n_clusters)
    u_cl = rng.normal(0, np.sqrt(rho_cluster), n_clusters)[cl]
    u_inst = rng.normal(0, 0.6, n)                      # shared across conditions -> pairing
    eps = {c: rng.normal(0, 0.7, n) for c in ('deploy', 'eval', 'ctrl')}
    d = {'deploy': d_deploy, 'eval': d_eval, 'ctrl': d_ctrl}
    scores = {c: u_cl + u_inst + eps[c] + d[c] * is_pos for c in d}
    return cl, is_pos, scores

def recall_at_fpr(pos_scores, neg_scores, fpr=0.01):
    if len(neg_scores) < int(np.ceil(1 / fpr)):
        return np.nan  # degenerate: cannot realize the FPR
    thr = np.quantile(neg_scores, 1 - fpr)
    return np.mean(pos_scores > thr)

def cond_recall(scores_c, is_pos, mask=None):
    m = np.ones_like(is_pos, bool) if mask is None else mask
    return recall_at_fpr(scores_c[m & is_pos], scores_c[m & ~is_pos])

# ================= B1: exact cancellation on a shared triple-matched set ==========
cl, is_pos, S = gen_data()
R = {c: cond_recall(S[c], is_pos) for c in S}
D_eval = R['deploy'] - R['eval']          # uncontrolled effect (degradation positive)
D_ctrl = R['deploy'] - R['ctrl']          # control-condition effect (partial-specificity stat)
DiD = D_eval - D_ctrl
print("== B1: shared triple-matched set ==")
print(f"  DiD = {DiD:.6f};  R_ctrl - R_eval = {R['ctrl'] - R['eval']:.6f}; "
      f"diff = {DiD - (R['ctrl'] - R['eval']):.2e}")
att = D_eval - DiD
print(f"  attenuation = uncontrolled - DiD = {att:.6f};  D_ctrl = {D_ctrl:.6f}; "
      f"diff = {att - D_ctrl:.2e}")

# ============ B2: per-pair matching breaks cancellation / shifts estimand =========
# behavior-matching modeled as: instance retained in the (X,deploy) pair iff its
# X-condition noise didn't flip behavior; selection depends on the X-condition score.
reps = 2000
did_pair, did_triple = [], []
for r in range(reps):
    cl, is_pos, S = gen_data(rng=np.random.default_rng(1000 + r))
    # selection correlated with the condition-specific realized score (post-treatment)
    keep_ed = S['eval'] < np.quantile(S['eval'], 0.85)   # eval-deploy matched set
    keep_cd = S['ctrl'] < np.quantile(S['ctrl'], 0.85)   # ctrl-deploy matched set
    keep_tri = keep_ed & keep_cd                          # triple-matched set
    D_e_p = cond_recall(S['deploy'], is_pos, keep_ed) - cond_recall(S['eval'], is_pos, keep_ed)
    D_c_p = cond_recall(S['deploy'], is_pos, keep_cd) - cond_recall(S['ctrl'], is_pos, keep_cd)
    did_pair.append(D_e_p - D_c_p)
    D_e_t = cond_recall(S['deploy'], is_pos, keep_tri) - cond_recall(S['eval'], is_pos, keep_tri)
    D_c_t = cond_recall(S['deploy'], is_pos, keep_tri) - cond_recall(S['ctrl'], is_pos, keep_tri)
    did_triple.append(D_e_t - D_c_t)
    if r == 0:
        # deploy-leg non-cancellation under per-pair matching:
        rd_ed = cond_recall(S['deploy'], is_pos, keep_ed)
        rd_cd = cond_recall(S['deploy'], is_pos, keep_cd)
        print("\n== B2: per-pair matched subsets ==")
        print(f"  deploy recall on eval-matched set = {rd_ed:.4f}, on ctrl-matched set = {rd_cd:.4f}"
              f"  (cancellation requires equality)")
dp, dt = np.array(did_pair), np.array(did_triple)
print(f"  E[DiD per-pair]  = {dp.mean():.4f} (MC se {dp.std()/np.sqrt(reps):.4f})")
print(f"  E[DiD triple]    = {dt.mean():.4f} (MC se {dt.std()/np.sqrt(reps):.4f})")
print(f"  estimand gap     = {dp.mean() - dt.mean():.4f}")

# ====== C1: joint vs independent cluster bootstrap of attenuation (subset case) ====
# controlled = analysis on an inherited verified subset (e.g. 60% of instances,
# membership fixed per instance); uncontrolled = full set. Attenuation = unc - con.
def one_dataset_bootstrap(seed, B=800, n_clusters=60):
    rg = np.random.default_rng(seed)
    cl, is_pos, S = gen_data(n_clusters=n_clusters, rng=rg)
    subset = rg.random(len(cl)) < 0.6                     # inherited membership
    unc_b, con_b = np.empty(B), np.empty(B)
    for b in range(B):
        pick = rg.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([np.flatnonzero(cl == c) for c in pick])
        ip, sb = is_pos[idx], subset[idx]
        Se, Sd = S['eval'][idx], S['deploy'][idx]
        unc_b[b] = (recall_at_fpr(Sd[ip], Sd[~ip]) - recall_at_fpr(Se[ip], Se[~ip]))
        con_b[b] = (recall_at_fpr(Sd[ip & sb], Sd[~ip & sb])
                    - recall_at_fpr(Se[ip & sb], Se[~ip & sb]))
    return unc_b, con_b

unc_b, con_b = one_dataset_bootstrap(7)
ok = ~np.isnan(con_b)
att_joint = unc_b[ok] - con_b[ok]
corr = np.corrcoef(unc_b[ok], con_b[ok])[0, 1]
sd_joint = att_joint.std(ddof=1)
sd_indep = np.sqrt(unc_b[ok].var(ddof=1) + con_b[ok].var(ddof=1))
print("\n== C1: joint vs independent bootstrap of attenuation (60% subset) ==")
print(f"  corr(uncontrolled, controlled) across joint replicates = {corr:.3f}")
print(f"  SD(attenuation) joint = {sd_joint:.4f};  independent-combination SD = {sd_indep:.4f}"
      f"  (ratio {sd_indep/sd_joint:.2f}x)")

# ====== C2: degenerate replicates when the subset negative count is thin ==========
print("\n== C2: degenerate-replicate frequency ==")
for frac, ncl in [(0.6, 60), (0.1, 60), (0.05, 60)]:
    rg = np.random.default_rng(42)
    cl, is_pos, S = gen_data(n_clusters=ncl, rng=rg)
    subset = rg.random(len(cl)) < frac
    n_deg = 0; B = 500
    for b in range(B):
        pick = rg.integers(0, ncl, ncl)
        idx = np.concatenate([np.flatnonzero(cl == c) for c in pick])
        negs = np.sum(subset[idx] & ~is_pos[idx])
        if negs < 100:  # cannot realize 1% FPR
            n_deg += 1
    print(f"  subset frac {frac:.2f}: subset negatives ~{np.sum(subset & ~is_pos)}, "
          f"degenerate replicates {n_deg}/{B}")
