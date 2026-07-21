# F17 power simulation — first pass (synthetic priors)

Generated 2026-07-08T19:23:26.390384+00:00 | SHA `53e6f929ab95` | priors: baseline recall 0.65, SESOI 0.1

| config | G | n_pos | n_neg | S | sup. power | equiv. power | type-I | TOST size | deff |
|---|---|---|---|---|---|---|---|---|---|
| design_point | 25 | 300 | 1750 | 10 | 0.96 | 0.99 | 0.000 | 0.000 | 1.84 |
| low_clusters | 20 | 300 | 1750 | 10 | 0.97 | 0.95 | 0.020 | 0.030 | 1.43 |
| equivalence_floors | 50 | 500 | 3500 | 10 | 0.99 | 0.96 | 0.030 | 0.030 | 1.68 |
| undersized | 25 | 150 | 500 | 5 | 0.38 | 0.43 | 0.010 | 0.010 | 1.15 |

Targets (F17 pin 1): superiority ≥ 0.8 at the design point; equivalence ≥ 0.8 required for the §13 Robust row (expected to FAIL at base floors per §10 — that expectation is itself under test).

Reproduce: `PYTHONPATH=src .venv/bin/python -m analysis.power`
