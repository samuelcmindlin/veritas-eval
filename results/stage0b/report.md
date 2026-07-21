# Stage 0b — pipeline validation (activation → probe → metrics)

Generated 2026-07-08T15:57:13.771235+00:00 | our SHA `5cfbe64f70a1` | config `be166d7fbc78` | **PASS**

**Substrate:** `google/gemma-2-2b-it` @ `299a8560bedf` (26 layers; probe read layer 13; diagnostics [6, 20]). Pipeline validation only — not a baseline reproduction (DEV §4 0b).

- Generation smoke: 6 generations, decoding params + seed logged per row (see report.json).
- Extraction: 120 items streamed to `activations/stage0b` (0 cache hits); peak RSS 0.785 GB (bounded-RAM check).
- Toy probe (18 train / 12 test clusters), per-seed held-out AUROC: s0=0.972, s1=0.972, s2=0.972
- Null ΔAUROC between template pseudo-conditions: +0.0417 [+0.0000, +0.0972] — covers 0: True (known-answer).
- recall@1%FPR guardrail at toy scale: fired = True (expected True — §10 metric-ladder trigger).
- Steering known-answer tests: same-site shift rel_err=9.89e-04 (pass=True); above-read bit-identity pass=True (max|diff|=0); below-read carry pass=True.

Probe artifact hashes (frozen per seed): s0=`2ac2a16ed9`, s1=`45f63ba7bb`, s2=`b6e70f5eba`
Dataset hash: `356dabc4cb8fe197` (120 toy items — NOT a deception organism).

Reproduce: `PYTHONPATH=src .venv/bin/python -m analysis.stage0b`
