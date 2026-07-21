# F34 workspace-band check — dated note (2026-07-08)

**Purpose** (Pre-Reg §21-F34; DEV plan Stage 0b `[WSP]` item): decide whether
lens-based diagnostics — the §4 `[WSP-1]` component-level overlap diagnostic,
the `[WSP-2]` control-direction loading reports, and the B11/B12 backlog
instruments — are *available at the project's model scale*, or whether the
geometry gate proceeds on raw-cos/norm-profile with the limitation stated.

## What was checked

1. **Neuronpedia hosted J-lens** (neuronpedia.org, "Jacobian Lens" section,
   checked 2026-07-08): interactive J-lens is live, with a model selector
   listing **Qwen 3.6 27B** and **Gemma 3 12B** (plus an un-enumerated "Other
   Models" entry). Neuronpedia hosts gemma-2-2b/9b for *other* tools (SAEs),
   but they do **not** appear in the J-lens selector. **The smallest confirmed
   J-lens-hosted model today is 12B — the project's Gemma-2 2B/9B substrate is
   not hosted.**
2. **Open-source release**: Gurnee et al. 2026 ship a J-lens
   training/inference implementation (linked from the Neuronpedia page and the
   paper's replication section), so constructing J_ℓ locally for a 2B model is
   possible: a one-off backward-pass sweep (paper default ~1,000 prompts of
   128 tokens; their ablations show the lens beats logit/tuned-lens baselines
   with as few as ~10 prompts, so a reduced-prompt variant is viable on a
   single small GPU / M1-class machine).
3. **Scale caveat carried from the paper** (recorded, not resolvable here):
   directed-modulation success *increases with model size*; on their smallest
   tested model (Haiku 4.5) workspace ablation degrades coherence before
   producing qualitative change; "whether smaller models have an equally rich
   workspace... or none at all" is explicitly open. A constructed 2B lens may
   therefore reveal a weak or absent workspace band — which is itself the
   informative outcome of the band check (kurtosis / next-token-accuracy /
   CKA-block metrics per the paper's layer-band analysis).

## Recommendation (feeds the F34 freeze decision)

- **Do not** make any gating machinery depend on lens diagnostics (already the
  v0.4 posture: `[WSP-1/2]` rows are descriptive).
- **If** Stage C wants the component-level overlap diagnostic and loading
  reports: budget a **one-off Gemma-2-2B J_ℓ construction** using the
  open-source release (reduced-prompt variant first; validate against the
  paper's known-intermediate pass@k probes before trusting readouts), then run
  the band metrics to establish whether a workspace band exists at 2B. Bounded,
  Colab-feasible; not on the Stage-0 critical path.
- **Else** resolve F34 to the **raw-cos / norm-profile fallback**, with the
  limitation stated in any steering-specificity claim (per §4 `[WSP-1]`).
- **Re-check Neuronpedia before the freeze** — the J-lens section is new and
  models are being added; a hosted gemma-2 lens would collapse the cost of
  option (a) to zero.

Provenance: checks performed via Neuronpedia public pages on 2026-07-08;
paper claims per the cached full text of Gurnee et al. 2026 (see RESEARCH_REVIEW
near-overlap entry). This note is the Stage-0b deliverable named in DEV §4 0b.
