# Evidence scripts for `ideas/IMPLEMENTATION_PLAN.md`, appendix A

Run from the repository root with the project environment, for example
`nice -n 19 .venv/bin/python ideas/evidence/level_selection.py 2`.
Results quoted in the plan were produced on 2026-09-17 (CPU, bartz 0.12.1, jax 0.11.1).

| Script | Appendix | What it shows |
|---|---|---|
| `level_selection.py T` | A.1 | Current `LongBet` against plain DiD when treated units differ in level (`T` = 2 or 6). Minutes per run. |
| `bartz_prior_check.py` | A.2 | The oracle's enumerated tree prior equals `bartz`'s GROW/PRUNE law under a flat likelihood. |
| `prior_defaults.py` | A.4 | Closed-form root-only dose model: RMSE and coverage for the proposed basis and smoothness defaults. |
| `float32_check.py` | A.5 | float32 against float64 for the GROW likelihood ratio of a realistic node. |
| `mv_bart_check.py` | A.6 | `bartz` accepts a dense leaf-prior precision and a frozen dense error precision, so the multivariate equivalence oracle of section 8.4 is constructible. |
| `wishart_check.py` | section 4.3 | `bartz`'s Bartlett sampler takes the inverse-Wishart scale as its third argument and runs under x64. |

A.3 is `ideas/dose_engine_prototype.py` (add `--mutants` for the broken kernels).
