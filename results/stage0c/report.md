# Stage 0c — Pacchiardi anchor qualification (scoring-level)

Generated 2026-07-08T18:10:24.280115+00:00 | our SHA `e605d6f2e522` | anchor pinned `c5689fa26153` (BSD-3-Clause) | **PASS**

Reproduced the in-distribution elicitation-question detector from the repo's shipped feature files (916 paired questions, 11 datasets) with independent implementations (scipy optimizer on the sklearn C=1 objective; our AUROC). Published targets = the canonical notebook's saved outputs at the pinned commit. Worst |ΔAUC| = 2.36e-05 (tolerance 0.002; qualification floor 0.9).

| features | probe group | n | AUC (ours) | AUC (published) | |Δ| | match |
|---|---|---|---|---|---|---|
| logprobs | all_probes | 65 | 0.99978 | 0.99978 | 4.8e-06 | ✓ |
| logprobs | no_lie | 14 | 0.98479 | 0.98479 | 3.4e-06 | ✓ |
| logprobs | lie | 24 | 0.99996 | 0.99996 | 3.3e-07 | ✓ |
| logprobs | knowable | 10 | 0.99160 | 0.99160 | 3.3e-06 | ✓ |
| logprobs | subsets_union | 48 | 0.99976 | 0.99976 | 2.0e-06 | ✓ |
| binary | all_probes | 65 | 0.99988 | 0.99988 | 9.9e-07 | ✓ |
| binary | no_lie | 14 | 0.94757 | 0.94757 | 2.5e-07 | ✓ |
| binary | lie | 24 | 0.99973 | 0.99973 | 1.1e-06 | ✓ |
| binary | knowable | 10 | 0.97884 | 0.97886 | 2.4e-05 | ✓ |
| binary | subsets_union | 48 | 0.99984 | 0.99984 | 1.3e-06 | ✓ |

**F11 operating-point color** (48-probe logprob detector, our §7 per-condition policy): recall@1%FPR = 0.9964 on 275 lie / 275 truth test items; achievable-FPR count 276; granularity gate passes: True. 48-probe logprob detector under our §7 per-condition policy; binary-feature variants are coarser — F11's tie rule exists for that case

Reproduce: `PYTHONPATH=src .venv/bin/python -m analysis.stage0c`
