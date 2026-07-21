# Analyst caveat on the F17 first pass — read before citing
# (updated 2026-07-08 after generator v2; v1 text below retained)
# (review addendum 2026-07-15 appended at end; corrects diagnosis item 2 and
#  restates the v3 acceptance band at matched seed count)

## v2 result: acceptance band FAILED — numbers still not decision-grade

Generator v2 (honest-cluster effects + a seed×condition component) moved the
design point from 1.00/1.00 to superiority 0.96 / equivalence 0.99, with
type-I and TOST-leg size still conservative (0.00–0.03) and the undersized
cell still correctly underpowered (0.38/0.43). But base-floor equivalence
power of 0.99 remains far above the stats_validation-predicted 0.55–0.75
band, so the acceptance test set in this file's v1 FAILED and the headline
power numbers remain untrustworthy for design decisions.

## Diagnosis v2 (from reading stats_skeptic_equiv_power.py directly)

The validated reference sim's generative model differs structurally, not by
tuning:

1. **Non-canceling instance and cluster interactions.** The reference model
   gives positives an instance×condition component (SIG_PQ = 0.50), an
   instance×seed component (SIG_PR = 0.40), and a cluster×condition
   interaction (SIG_W = 0.15) — none of which cancel in Δ̂. Our generator's
   item and cluster effects are shared across conditions, so they cancel;
   our only non-canceling term is the small seed×condition one (0.15).
2. **Negatives are not clustered — they carry instance×condition and
   instance×seed structure** (NA_B = 0.45, NA_C = 0.50 on top of NA_A = 0.75).
   Our v2 guess (cluster effects on honest scores) was the wrong form; the
   reference's threshold noise comes from per-condition/per-seed negative
   variation, which our generator lacks entirely.
3. **Estimator semantics: per-seed thresholds, averaged — not pooled.** The
   reference computes the 1%-FPR threshold per (seed, condition), takes the
   per-seed recall difference, and averages over seeds. Our
   joint_bootstrap_delta pools all seeds' scores into one recall at one
   threshold per condition. Since each detector seed is a separate frozen
   artifact with its own operating point (ARCHITECTURE / Pre-Reg §7), the
   per-seed-then-average semantics is the pinned interpretation (§19-7: the
   stats_validation suite is the verified fix set).

**Item 3 is bigger than the power sim:** the metrics module's multi-seed
handling must implement per-(seed, condition) thresholds with per-seed deltas
averaged inside each bootstrap replicate before any Stage-C confirmatory use.
Tracked as a required enhancement; the 0b toy usage (identical scores per
seed at fit time aside, 3 seeds, pipeline validation only) is unaffected in
its conclusions.

## Second-pass plan (v3, before any design decision leans on these numbers)

- Port the reference generative model's components verbatim (SIG_* / NA_* /
  MU_POS) as the pinned synthetic priors.
- Implement the per-seed-threshold statistic in the bootstrap layer and use it
  here and in the engine.
- Acceptance: reproduce the reference's ≈0.55–0.75 base-floor equivalence
  power and its ≈0.96 at (G=50, n_neg≈4,000) before trusting any other cell;
  then rerun the full sweep at B = 2000 for the freeze-deposited computation
  with Stage-A-updated priors.

---

## v1 caveat (2026-07-08, pre-v2) — retained for the record

The first-pass sweep VALIDATES THE MACHINERY (power.py runs the real §12
estimator; the undersized cell correctly reads as underpowered at 0.47/0.39;
type-I 0.00–0.02 and TOST-leg size ≤0.01 are conservative, consistent with the
percentile construction's known behavior; coverage 0.97–0.99). Its absolute
power numbers should NOT be trusted for design decisions. Red flag:
equivalence power read 1.00 at the BASE floors (G=25, n_neg=1750),
contradicting the simulation-backed Pre-Reg §10 statement (≈0.55–0.75, capped
≈0.75–0.8 by threshold noise). v1 diagnosis (partially right, superseded by
the v2 diagnosis above): no honest-class cluster structure; seed effects
canceling in the delta.

---

## Review addendum (2026-07-15) — two corrections to the v2 record

Both from direct re-verification during the 2026-07-15 repo review; the v2
FAIL verdict is unchanged by either.

**1. Diagnosis item 2 overstates.** "our generator lacks [per-condition/
per-seed negative variation] entirely" is wrong as written: generator v2
draws fresh iid noise (sd_noise = 1.0) per (condition, seed, item) row
(power.py `make_dataset`), which IS negative variation of the reference's
idiosyncratic NA_C form (0.50), at larger magnitude. What the generator
actually lacks is the reference's **instance×condition component shared
across seeds (NA_B = 0.45)** — condition-consistent negative structure that
survives seed averaging — and its v2 honest-CLUSTER effects
(sd_cluster_honest) have no counterpart in the reference, whose negatives are
unclustered. The v2 conclusion (structural mismatch, headline numbers not
decision-grade) stands.

**2. The acceptance band is seed-count-dependent; the v1 acceptance test
compared mismatched designs.** Pre-Reg §10's ≈0.55–0.75 equivalence-power
band is stated at "~5 seeds", and stats_skeptic_equiv_power.py's shipped
cells run S=5 — but the failing design point runs S=10. Re-running the
reference model itself (mc=300, B=250, n_neg=1750, Δ=0):

| G  | S=5 (shipped) | S=10 (design point's S) |
|----|---------------|--------------------------|
| 20 | 0.69          | 0.71                     |
| 25 | 0.67          | 0.82                     |
| 30 | 0.72          | 0.84                     |

(S=5 reproduces the §10 band; MC standard error ≈0.02–0.03.) The v3
acceptance criterion should therefore be S-matched: reproduce **≈0.67–0.72 at
(G=20–30, S=5)** and **≈0.82 at the design point (G=25, n_neg=1750, S=10)** —
not "0.55–0.75" at the design point. The v2 sweep's 0.99 fails the S-matched
target too, so this correction changes the target, not the verdict. The §10
prose band itself is a prereg statement; updating it (e.g. to note its S
dependence) is a human/prereg decision, not made here.

**Bookkeeping:** item 3's per-(seed, condition)-threshold enhancement,
described above as "Tracked as a required enhancement", had no tracker entry
until 2026-07-15; it is now BACKLOG **B14**, with the pinned-semantics
decision flagged [RESEARCHER/human] there.
