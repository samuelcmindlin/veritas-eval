# Statistical validation simulations — 2026-07-07 deep-stats pass

Monte Carlo evidence behind the statistical pins in **PREREGISTRATION.md** §§7/10/12/13/21 — every `[STATS-PASS 2026-07-07]` tag in the prereg traces to a script here. Files prefixed `stats_SKEPTIC_*`/`stats_skeptic_*` are **independent re-implementations** written during adversarial verification (different authors, different seeds/code); a claim was only accepted where both implementations agreed. `sim*_out.txt` are captured outputs. Python 3.11, numpy 2.2.4, scipy 1.17.1; scripts are seeded.

## What was CLEARED (design decisions validated as-is)

| Question | Script(s) | Headline result |
|---|---|---|
| Do 20–30 clusters suffice for the one-sided α=0.05 significance test? | `stats_R1R4_simA_recall_boot.py`, `_simA2_lowhonesticc.py`, `stats_SKEPTIC_boot_method.py` | Yes, with the **percentile** interval: per-tail size 0.013–0.039, never above nominal |
| Is the 1,500–2,000 negative-class floor the right order? | `stats_O1_negfloor.py` | Yes: SD(paired Δ) 0.10 @ n=200 → 0.047 @ 1,500 → 0.043 @ 2,000 |
| Does per-replicate threshold re-estimation give correct coverage? | `stats_O2_reest_coverage.py` | Yes: 90% CI coverage 0.947–0.952; fixed-threshold approach materially wrong |
| Is the §12 five-cell partition total and exclusive? | `stats_D1_partition.py`, `stats_SKEPTIC_bh_partition.py` | Yes: exhaustive enumeration ~8.4M (L,U) pairs incl. exact boundaries — 0 gaps, 0 overlaps |
| Is BH valid for the S1–S6 family under realistic dependence? | `stats_D3_bh_dependence.py`, `stats_SKEPTIC_D3_bh_recheck.py` | Yes: FDR 0.036–0.051 across independence/exchangeable/mixed-sign; BY not warranted (reported as sensitivity only) |

## What was FIXED (defects found by simulation, now pinned in the prereg)

| Defect | Script(s) | Headline result → pin |
|---|---|---|
| Bootstrap CI construction unpinned; BCa (scipy default) anticonservative | `stats_R1R4_simA_recall_boot.py`, `stats_SKEPTIC_bca.py`, `stats_SKEPTIC_boot_method.py` | basic 0.063–0.115, BCa 0.077–0.105 vs percentile 0.013–0.055 per tail → **percentile pinned, B ≥ 2,000** (§12/F27) |
| Seed resampling ambiguity + ≥5-seed floor too low | `stats_R1R4_simB_seed_did_xfam.py`, `stats_SKEPTIC_seedfloor_check.py` | fixed seeds: size 0.10–0.17; per-cluster redraws: 0.086–0.147; S=5 at 30–60% seed share: 0.060–0.077; S=10 restores baseline → **one global seed set per outer replicate; ≥10 seeds primary cell** (§10/§12/F28) |
| Equivalence (Robust) branch underpowered at base floors | `stats_skeptic_equiv_power.py`, `stats_C3_recall_cluster_power.py` | P(U<SESOI given Δ=0) ≈ 0.55–0.75 at base floors; threshold noise caps it ~0.75–0.8 regardless of clusters → **numeric ≥0.8 power gate on the Robust row; joint floors ~G≥50 + n_neg≈3–4k** (§10/§13) |
| Discrete-score 1%FPR ill-defined; deterministic tie rule breaks shift-invariance | `stats_O3_discrete.py`, `stats_SKEPTIC_O3_discrete_verify.py` | spurious Δ −0.06…+0.125 (grid-dependent sign, > SESOI) → **interpolated thresholding + achievable-FPR reporting + granularity gate [0.5%,2%]** (§7/F29) |
| Interpolated quantile for the frozen threshold overshoots FPR | `stats_O4_quantile_est.py`, `stats_SKEPTIC_O4_quantile_recheck.py` | realized FPR 1.49% @ n_cal=200 vs nominal 1% → **order statistic k=⌈0.99(n+1)⌉ + calibration-split floor ≥500** (§7/F30) |
| H4 ΔECE untestable at plausible n; estimator unpinned; mean-shift diagnostic blind to variance-driven loss | `stats_C1_C4_ece_bias_power.py`, `stats_C2_shift_vs_discrimination.py`, `stats_skeptic_ece_*.py`, `stats_SKEPTIC_C2_recheck.py` | power ≈0.005–0.04 @ n=200 BH-worst; naive bias +0.05; AUROC 0.898→0.801 with mean shift −0.001 → **ΔECE SESOI 0.05 + ≥80% power gate + exact estimator pins + Brier-resolution as the designated diagnostic** (§7/F31) |
| Attenuation/DiD estimand shifts under per-pair matching; independent bootstraps inflate SD | `stats_skeptic_attenuation_check.py` | per-pair vs triple-matched E[DiD] gap ≈ 0.035 ≈ SESOI/3; corr(legs) ≈ 0.58 → **single triple-matched set + within-replicate joint computation + degenerate-replicate rule** (§12) |
| Shared-seed pairing unstated for S1 | `stats_skeptic_S1_seedpair.py` | independent draws inflate CI up to ~2× (conservative) → **paired seed draws when detectors share seeds** (§12 S1) |
| Positive class co-binding at the negative floor | `stats_skeptic_posfloor.py` | n_pos=300 contributes ~35–40% of variance → **presumptive ≥300 positive floor** (§10/F32) |
| Skew sensitivity of the TOST leg (residual, disclosed not fixed) | `stats_D2_ci_duality.py`, `stats_D2b_recall_skew.py`, `stats_SKEPTIC_ci_duality.py`, `stats_SKEPTIC_recall_skew.py` | ~0.06/leg small-sample error at G=25 even under normality; design-faithful skew ≈ 0.02 → **F17 CI-level check reports realized per-leg error with the primary result** (§10/§19-7) |
| Brier reliability plugin bias | `stats_SKEPTIC_brier_rel_bias.py` | +0.006 @ n=200 vs true ~0.0002 → **debiased Murphy decomposition** (§7/F31) |

These scripts are candidates to seed `tests/` and `src/analysis/power.py` known-answer suites (DEV plan §3) — the coverage/size checks here are exactly the F17 CI-level check the prereg mandates at the realized design.
