# Tree-move investigation: CHANGE and REGROW (2026-09-13)

## Diagnosis on a randomized-encouragement panel

On a single-wave encouragement pilot (N = 300 accounts, T = 8 weeks, a two-threshold
interaction in the treatment effect, noise well below the effect), four dispersed
chains of the public sampler each froze at a different set of split thresholds and
tree topologies: decoding the treatment forest showed one chain placing the team-scale
step at -0.20, another adding a spurious intermediate leaf at -0.40, a third stepping
at -0.10, and a fourth carrying the interaction in a single depth-5 tree. Within-chain
draws barely moved (lag-1 autocorrelation 0.7-0.98), so the population effects had
bulk ESS below 15 and rank R-hat 1.3-2.0 after 10,000 iterations. Fixing or removing
the coding scales, decoupling the innovations, longer chains, more or shallower trees,
coarser cutpoints, noisier data, and intercept priors did not change this. The cause is
that `bartz` proposes only GROW and PRUNE moves: a rule cannot slide to a neighbouring
cutpoint without passing through a much worse tree, and a topology cannot change into
another that represents nearly the same function.

## Moves

- `change_move`: at a random parent of two leaves, re-draw the rule in place, half of
  the time from the ancestor-restricted rule prior (proposal cancels prior) and half of
  the time as a symmetric local cutpoint perturbation; accept with the exact
  leaf-integrated likelihood ratio and the children's admissibility prior factors.
  Acceptance on the pilot panel: 10-14% (prognostic forest), about 1% (treatment
  forest), because every split there is load-bearing.
- `regrow_trees_per_sweep`: propose a whole tree anew from the root with an XBART-style
  recursion (stop or split at every node with probability proportional to prior times
  integrated leaf likelihood, count thresholds applied to candidates). The recursion's
  density is the product of the categorical probabilities it sampled and is evaluated
  for the current tree under the same residual, giving an exact Metropolis-Hastings
  independence ratio. A depth-first traversal with a fixed visit budget keeps the cost
  static; trees needing more visits are never proposed and never left through this
  kernel. Acceptance on the pilot panel: about 90% (treatment forest) and 67%
  (prognostic forest). Cost with four chains: roughly 12 ms per regrown tree and sweep.

## Validation

- `tests/test_geweke.py::test_geweke_joint_distribution[change_and_regrow]` passes.
- `tests/test_change_move.py` and `tests/test_regrow_move.py`: memberships match rules
  after every sweep, caches match memberships, trees are valid, residual plus forest
  fit is conserved to float tolerance, moves are actually accepted, and posterior means
  agree with the grow/prune sampler.
- `tests/test_regrow_distribution.py`: on a diffuse posterior where grow/prune mixes,
  fitted-value and tree-size laws agree between samplers (identical data).
- `benchmarks/tree_moves_validation.py`, strong signal (noise 0.5, effects of 2-3):
  a 30,000-sweep grow/prune reference and a 12,000-sweep regrow run agree on six fitted
  values (|z| <= 1.2) and on tree size (6.31 vs 5.99 internal nodes); tree-size ESS
  20 (grow/prune) vs 669 (regrow).

## What the moves do and do not change on the pilot panel

The moves leave the posterior unchanged and raise within-chain ESS on small problems by
more than an order of magnitude. On the pilot panel, regrowing one tree per forest and
sweep spread the chains over coarser and finer treatment-forest topologies instead of
freezing each on a sharp one, and the pooled posterior mean's out-of-sample error
against the expected truth rose (0.78 with the frozen chains, 1.0-1.3 with regrow).
That is the posterior under the default shallow-tree treatment prior speaking, not a
sampler error: the frozen chains sat on the sharpest modes. Whether a denser treatment
prior restores accuracy with converged chains is evaluated in the chapter that motivated
this work (`~/vignettes/panel-iv.qmd`). Neither move is a default; archives record them.

## Block proposals do not escape shared states

A block version (two or more trees proposed jointly from the residual with the
whole block removed, structures sequential with deterministic mean-fit residuals,
leaves integrated jointly through a small dense Gaussian marginal, exact joint
independence ratio) was implemented and tested. It never accepted on the test
fixture. The debug trace shows why: the reverse density of the current block is
tiny (log q around -270 against -16 for the proposal) because a current tree that
stops where the full residual says to keep splitting is nearly impossible under a
data-driven recursion, so the ratio rejects even better-fitting proposals. Shared
decompositions, the states the block move was meant to leave, are exactly the
states an independence proposal cannot leave. The option was removed rather than
shipped. Escaping forest-level decomposition modes would need a tempered or
auxiliary-variable scheme, which is outside this pass.

## Pilot-panel results with the moves on

Single-tree regrow with one tree per forest and sweep, four chains, 3,000 burn-in
and 1,000 retained: plain reduced form bulk ESS 7-10 and rank R-hat 1.3-1.5 on the
week-8 population effect, out-of-sample RMSE 1.0 against 0.78 for frozen
grow/prune chains; joint encouragement model ESS 6-8, R-hat 1.45-1.7, RMSE 1.15
against 0.48 (1.01 with two regrown trees per forest and sweep plus the CHANGE move). Denser treatment priors (`alpha_split_trt = 0.95, beta_split_trt = 2`)
or 50 treatment trees made accuracy worse (RMSE 2.0 and 1.2). The frozen chains'
sharper estimates are not posterior summaries; the correct sampler's are, and they
have not converged either.

## Addendum (2026-09-14): CHANGE at internal nodes, and the number of treatment trees

On the chapter's second example (a nudge whose effect flips sign across a step in two
covariates, 300 units, 8 weeks) the adoption-clock model's outcome effect failed the gates
at 4 chains x (2,000 + 6,000) sweeps: bulk ESS 57-140 and R-hat 1.025-1.05 for the
population ITT at five of six horizons. Decoding the chains showed each one holding one
or two `x1` cutpoints near the true threshold for the entire run, at different values
per chain; the units whose predicted effect differed between chains were control-arm
accounts that never adopted and sit on the threshold, whose effects only the forest prior
places. Those cutpoints sat at internal nodes (an `x1` rule with `x2` rules below), which
neither GROW/PRUNE nor the leaf-parent CHANGE can move.

`src/longbet/_change_internal_move.py` adds a CHANGE at any nonterminal node: the tree
shape is kept, the subtree is re-routed, and the move accepts with the exact ratio of the
full tree prior times the leaf-integrated likelihood, with the fresh-rule proposal density
divided out. Validation: `tests/test_change_internal_move.py` (no-op proposals are exact,
re-routing agrees with `bartz.grove.traverse_forest`, inadmissible descendant rules get
zero prior, residual and caches stay consistent) and a with/without comparison of the
successive-conditional forest marginals; `tests/test_geweke.py` passes with the move in
the sweep.

Results at 4 x (1,000 + 2,000) sweeps on the same panel (18 gated quantities):

| treatment trees | internal CHANGE | passed | min bulk ESS | max R-hat | units with chain disagreement > 1 |
|---|---|---|---|---|---|
| 20 | off | 5 | 52 | 1.069 | 17 |
| 20 | on | 6 | 26 | 1.104 | 17 |
| 60 | off | 16 | 487 | 1.011 | 14 |
| 60 | on | 17 | 462 | 1.011 | 5 |

The number of treatment trees is what matters: with 20 each tree carries a large share of
the surface and the forest sits in one decomposition; with 60 the chains agree. The
internal CHANGE removes the remaining frozen cutpoints. A steep-but-smooth boundary
instead of a step did not help at 20 trees (8 of 18). Regrowing every tree each sweep was
also tried and was far slower without fixing the freeze. Defaults are now 60 treatment
trees and a maximum depth of 8 (trees are stored as heaps of `2 ** depth` nodes, so the
cap bounds archive size; on the onboarding panel at 60 trees, depth 6 and depth 10 mixed
alike over two seeds, and depth 10 was slightly more accurate, 0.29 against 0.32 held-out
RMSE, so the cap stays above the depths posterior trees use).
