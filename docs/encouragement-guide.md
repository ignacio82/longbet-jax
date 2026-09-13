# Randomized encouragement in LongBet

LongBet separates the effect of encouragement on the outcome (`itt_y`), its
effect on adoption (`itt_d`), and their Wald ratio. Begin with the reference
analysis. The Bayesian wrapper adds covariate adjustment and time smoothing,
with diagnostics on the actual effects you report.

The Bayesian implementation is available, but its posterior interval calibration
is **not established**. An explicit `first_stage` choice is required. Check the
[calibration report](../benchmarks/encouragement/README.md) before interpreting
posterior ratios. Passing an ESS or R-hat threshold does not establish coverage.

The follow-up [sampler repair study](../benchmarks/encouragement/repair-report.md)
contains earlier research candidates. The repaired direct-smoothing model is now
available as `engine="direct_smooth"`, with explicit `direct_config` controls.
Its sampling tests and the [completed matched comparison](../benchmarks/encouragement/comparison-report.md)
separate computational correctness from empirical accuracy and coverage. Neither
evaluated covariance configuration establishes an advantage over both adjusted
IV baselines, and reduced-form coverage remains inadequate in tested settings.

## Data and interpretation

Use complete `(N, T)` panels for outcomes `y`, encouragement `z`, and actual
adoption `d`. Both indicators are binary and absorbing. Calendar time `t` has
strictly positive whole-unit gaps; an unobserved calendar period does not become
an observed outcome. Baseline covariates `x` have shape `(N, P)`.
Fractional calendar origins are preserved. Shift a very large calendar origin
before fitting if float32 model storage would collapse periods or change exposure
indices; the wrapper checks for that loss of precision.

The model supports a single randomized encouragement wave and a permanent
holdout, with complete individual assignment. Adoption can precede encouragement
or occur later. Experiments without baseline periods and perfect compliance are
valid. Actual uptake is a response in the first-stage equation; it is not used
as an unconfounded treatment in the outcome equation.

Randomization identifies encouragement ITTs. Interpreting their ratio as CACE
also requires exclusion, monotonicity, relevance, and appropriate assumptions
about treatment history. With adoption acceleration, someone can have adopted
in both assignments by a later horizon while still contributing a duration
effect to the numerator. Neither an observed lag nor a negative sample first
stage verifies an individual's compliance type. See the motivating
[RED chapter](https://book.martinez.fyi/iv.html) and the
[structural identification assessment](../benchmarks/encouragement/structural_research.md).

## Single-wave reference analysis

```python
from longbet import encouragement_effects, encouragement_summary

adoption = encouragement_summary(z, d, t)
reference = encouragement_effects(y, d, z, t)
print(reference[["horizon", "itt_y", "itt_d", "wald", "wald_set_type",
                 "wald_lower_1", "wald_upper_1", "wald_lower_2", "wald_upper_2"]])
```

```r
adoption <- encouragement_summary(z, d, t)
reference <- encouragement_effects(y, d, z, t)
```

These are assigned-arm differences in means. Uncertainty uses the joint
within-arm covariance of outcomes and adoption, with independent unit counts.
The Wald confidence set inverts a normal test of the effect on `Y - w*D`.
It can be bounded, unbounded, disjoint, all real numbers, or empty. Read both
interval components; do not drop negative denominators or replace an all-real
set with a finite interval. The interval is pointwise and asymptotic, not an
exact permutation interval. Normal intervals can undercover for discrete or
sparse first stages; the calibration report retains those failures.

## Joint Bayesian reduced forms

```python
from longbet import LongBetEncourage

# Explicit working-model choice; compare LPM and probit in sensitivity analysis.
fit = LongBetEncourage(first_stage="lpm").fit(y, d, z, x, t=t)
pred = fit.predict(groups=baseline_group, summary_only=True)

print(pred.table)                      # ITTs, posterior quantiles and diagnostics
print(pred.first_stage())              # sign/precision, separate from convergence
print(pred.wald())                     # withheld if either ITT or ratio checks fail
print(pred.reference)                  # separate frequentist confidence sets
print(pred.itt_covariance())           # empirical joint posterior ITT covariance
print(pred.comparison())               # descriptive same-target differences

fit.save("encouragement.npz")
replayed = LongBetEncourage.load("encouragement.npz").predict(groups=baseline_group)
```

```r
fit <- longbet_encourage(y, d, z, x, t = t, first_stage = "lpm")
pred <- predict(fit, groups = baseline_group, summary_only = TRUE)

pred$effects
encouragement_first_stage(pred)
encouragement_wald(pred)
pred$reference
pred$itt_covariance
encouragement_comparison(pred)

saveRDS(fit, "encouragement.rds")
replayed <- predict(readRDS("encouragement.rds"), groups = baseline_group)
```

`first_stage="lpm"` uses binary adoption as a continuous working response. It is
simple but may imply values outside probability bounds. `"probit"` computes
probability contrasts from a binary latent-response model. For binary outcomes,
also set `outcome="binary"`; both numerator and denominator are then on their
natural probability scales. This does not add an absorbing adoption-hazard
likelihood: these are working reduced forms of adoption stocks at each period.

Default wrapper innovation priors are inverse-gamma with shape 2 and scale 1
on the standardized continuous-response scale. They have mean 1 and infinite
variance. A supplied `LongBetConfig` must explicitly contain proper positive
innovation hyperparameters. Custom priors with `standardize=False` apply in
raw response units. Unit-intercept variance priors remain those in the config.
R accepts a named `config` list with Python sampler-option names.

```python
from longbet import LongBetConfig

config = LongBetConfig(sigma_prior_a=2, sigma_prior_b=1,
                      num_chains=4, num_burnin=4000, num_sweeps=1000)
fit = LongBetEncourage(config, first_stage="probit", outcome="binary").fit(y, d, z, x, t)
```

```r
fit <- longbet_encourage(y, d, z, x, t = t, first_stage = "probit",
  outcome = "binary", config = list(sigma_prior_a = 2, sigma_prior_b = 1,
    num_chains = 4L, num_burnin = 4000L, num_sweeps = 1000L))
```

Longer chains are a sensitivity check, not a guarantee. Examine both ITTs and
each reported subgroup ratio. Quantile MCSE and tail ESS concern the actual
interval endpoints; the wrapper omits a ratio mean and its mean MCSE because
those moments may not exist. Raw draws remain available even when summaries
are withheld. `pred.wald(require_convergence=False)` permits explicit inspection
of unconverged empirical quantiles. An exactly zero denominator draw makes the
interval unavailable instead of silently conditioning on the remaining draws.

`pred.first_stage(practical_threshold=0.03)` reports the posterior probability
of an adoption ITT above three percentage points. This is an analyst-specified
practical threshold, separate from instrument precision and convergence. R
accepts `practical_threshold` in `predict()`.

## Targets, dependence and memory

Every posterior draw compares encouragement-from-the-common-start with never
encouraged for **all original study units**, then averages both equations with
the same weights. The existing treated-only `get_att()` is not substituted for
this target. Prespecified groups use their own common units; groups missing an
observed arm have unavailable reference inference and model extrapolation flags.
Selecting groups after observing effects requires additional selection-aware
methods. There is no multiplicity correction across groups or horizons.

`pred.draws["itt_y"]`, `pred.draws["itt_d"]` and `pred.draws["wald"]` have axes
`(group, horizon, chain, retained_draw)`. The overall group is `"all"` and
subgroups follow first appearance. Joint draws stay paired. SUR models
within-period innovation dependence; the fitted unit-intercept priors remain
independent across equations. Matching draw indices is not a substitute for
unit-level covariance calibration.
For binary Y with a probit adoption equation, both innovation-loading rows are
fixed to zero. With the default separate treatment forests, these equations
have independent posteriors even if `sur=True`; metadata records
`innovation_coupling="independent_binary"`. Opt-in shared treatment partitions
can link forest uncertainty but do not add correlated unit intercepts or binary
innovation errors. The calibration report distinguishes this case explicitly.

With `summary_only=True`, prediction accumulates natural-scale binary contrasts
in blocks and retains only small group/horizon draw arrays plus per-cell
summaries. Set `summary_only=False` to retain `(N,T,chains*draws)` cell draws.
In Python these are `pred.standardized.cell_draws`; in R they are `pred$cell_draws`.

The default `standardization="conditional"` holds each original unit's sampled
intercept fixed under both assignments. `"population"` integrates a fresh
normal unit intercept for binary outcomes using `Phi(eta/sqrt(1+variance))`.
That is a different target, so reference comparison is marked unavailable.
Continuous additive contrasts coincide under the two choices.

Archives contain the original outcomes, adoption, assignment, covariates and
calendar times as well as posterior traces and provenance. Python archives are
versioned, pickle-free and checked on load. R stores raw archive bytes and plain
result objects so `saveRDS` replay does not depend on a surviving Python handle.

## Comparing model and reference estimates

`pred.comparison()` returns descriptive differences. It does not add independent
standard errors for two estimates computed from the same data. To estimate the
covariance of their difference, explicitly run paired refits:

```python
comparison = fit.bootstrap_comparison(replicates=200, groups=baseline_group,
                                      random_seed=2026)
print(comparison.table)
print(comparison.failures)
```

```r
comparison <- encouragement_bootstrap_comparison(fit, replicates = 200L,
  groups = baseline_group, random_seed = 2026L)
comparison$table
comparison$failures
```

This resamples whole longitudinal unit records within assigned arm and baseline
group. Each resample refits the joint model and recomputes the reference on the
same population. The comparison uses their paired sampling covariance and
compares only ITTs. It requires many complete model fits. Failures are preserved;
flags are withheld if any fit fails diagnostics, reference uncertainty is
unavailable, or fewer than 100 replications are run. This is an empirical
bootstrap approximation with pointwise normal critical values, not exact
randomization inference, a weak-IV ratio interval, or proof of misspecification.

## Blocks, clusters and known probabilities

The richer designs currently provide reference inference. Their explicit HT
targets and covariance differ from the single-wave wrapper's conditional
subgroup arm means.

```python
from longbet import EncouragementDesign, design_encouragement_effects

design = EncouragementDesign(blocks=block, clusters=cluster, target="unit")
result = design_encouragement_effects(y, d, z, t, design=design, groups=baseline_group)
print(result.table)
print(result.covariance)  # index/columns: (contrast_id, itt_y or itt_d)

bernoulli = EncouragementDesign(assignment="bernoulli", probabilities=known_p)
weighted = design_encouragement_effects(y, d, z, t, design=bernoulli)
```

```r
design <- encouragement_design(blocks = block, clusters = cluster, target = "unit")
result <- design_encouragement_effects(y, d, z, t, design = design, groups = baseline_group)
result$table
result$covariance
result$covariance_index

bernoulli <- encouragement_design(assignment = "bernoulli", probabilities = known_p)
weighted <- design_encouragement_effects(y, d, z, t, design = bernoulli)
```

`target="unit"` weights participants equally; `"cluster"` weights clusters
equally, then subgroup members within a cluster equally. A cluster must share
one assignment and belong wholly to one block. Optional `block_weights` fixes
the target mass in each block. Every positive-weight block must contain the
subgroup of interest. Covariance uses whole randomization units and retains
cross-horizon and cross-group terms.

Under complete assignment, inference uses blockwise Neyman covariance of scaled
cluster totals. Subgroups and unequal-size participant-weighted clusters use
fixed-denominator HT estimates, which may change under outcome location shifts.
Known-probability independent assignment uses a conservative uncentered joint
covariance bound; it can be loose and depends on outcome location. HT estimates
of binary contrasts are not clipped to parameter bounds. Small-arm uncertainty
is explicitly unavailable.

## Staggered cohorts

Declare independent categorical cohort assignment, known probabilities for
each randomization unit, and fixed cohort weights. Keys use **zero-based
observed period indices in both Python and R**, with `"never"` for a holdout.
For example, if `T >= 4`, cohorts starting in columns 1 and 2 can be analyzed as:

```python
design = EncouragementDesign(assignment="staggered",
    probabilities={1: .3, 2: .3, "never": .4},
    cohort_weights={1: .5, 2: .5}, control="not_yet")
result = design_encouragement_effects(y, d, z, t, design=design)
```

```r
design <- encouragement_design(assignment = "staggered",
  probabilities = list(`1` = .3, `2` = .3, never = .4),
  cohort_weights = list(`1` = .5, `2` = .5), control = "not_yet")
result <- design_encouragement_effects(y, d, z, t, design = design)
```

Each cohort/calendar contrast uses eligible later cohorts and holdouts, or only
holdouts with `control="never"`. This requires no anticipation. Covariance
retains reused controls and the fixed-weight horizon aggregates. Missing cohort
horizons or zero control probability yield explicit unavailable contrasts;
weights are never renormalized to hide a lost target. Fixed cohort quotas,
adaptive re-randomization, missing panels and nonabsorbing adoption are outside
this contract. These cases need their actual design covariance and identification
argument, rather than a generic staggered-IV label.

## Randomization-based Anderson–Rubin inference

`randomization_ar` tests candidate values under the sharp exclusion null
`Y_i(1,t) - Y_i(0,t) = beta * (D_i(1,t) - D_i(0,t))`, using complete individual
randomization with fixed arm counts. Random assignment alone does not make this
a test of an arbitrary heterogeneous average effect. With duration effects,
apply it only at horizons where the specified current-adoption null is valid.
For a baseline and first follow-up in columns 0 and 1:

```python
from longbet import randomization_ar
import numpy as np

ar = randomization_ar(y[:, :2], d[:, :2], z[:, :2],
    beta=np.arange(-5, 5.01, .1), method="monte_carlo",
    permutations=1999, seed=42)
print(ar.table)
```

Enumeration is exact under this sharp null; Monte Carlo uses a plus-one p-value.
The returned table covers only evaluated candidates. Preserve separate accepted
grid runs. Acceptance at a grid boundary does not establish an infinite tail,
and Monte Carlo p-value bounds are not treatment-effect intervals.

## Binary principal-stratum bounds

`encouragement_bounds` takes binary outcome, adoption, and assignment vectors.
It assumes monotone adoption and bounds the direct effect of encouragement on
outcome probabilities by `delta`: zero imposes exclusion; one leaves direct
effects unrestricted. Prespecify any threshold for a continuous outcome.

```python
from longbet import encouragement_bounds

bounds = encouragement_bounds(y_binary, d_at_horizon, assigned,
    delta=.1, draws=2000, chains=4, seed=42)
print(bounds.table)
print(bounds.metadata)
```

Read median identification endpoints separately from their posterior uncertainty.
`[lower_ci_lower, upper_ci_upper]` is an envelope of endpoint quantiles, not an
automatically calibrated confidence interval. Report available and compatible
draw fractions; they are posterior summaries, not specification-test p-values.
These bounds target the three named complier estimands, not an unrestricted ATE.

## Direct smooth Gaussian reduced forms

The direct backend fits assignment effects using sums of Gaussian-process stump
leaves. It integrates baseline and unit-intercept blocks and jointly samples
effect leaves. The repaired implementation averages unweighted counterfactual
contrasts over all fitted units. Optional correlated unit intercepts use the
other equation's current conditional prior mean and variance.

```python
from longbet import LongBetEncourage, LongBetConfig, DirectSmoothConfig

fit = LongBetEncourage(
    LongBetConfig(num_chains=4, num_burnin=500, num_sweeps=1000, random_seed=42),
    first_stage="lpm", engine="direct_smooth",
    direct_config=DirectSmoothConfig(baseline_trees=4, effect_trees=4,
                                    correlated_intercepts=False),
).fit(y, d, z, x, t=t)
pred = fit.predict()
print(pred.effects())
print(pred.wald())
fit.save("direct-encouragement.npz")
replayed = LongBetEncourage.load("direct-encouragement.npz").predict()
```

```r
fit <- longbet_encourage(y, d, z, x, t = t, first_stage = "lpm",
  engine = "direct_smooth",
  direct_config = list(baseline_trees = 4L, effect_trees = 4L,
                       correlated_intercepts = FALSE),
  config = list(num_chains = 4L, num_burnin = 500L,
                num_sweeps = 1000L, random_seed = 42L))
pred <- predict(fit)
encouragement_wald(pred)
```

Prediction returns the saved conditional-mean assignment contrasts averaged over
observed study units, with posterior uncertainty in those mean functions. It does
not impute realized finite-population missing potential outcomes. An earlier
implementation added independent Gaussian noise with variance `sigma^2/N` and
called it sample-effect imputation; that omitted the observed residual contribution
and an explicit cross-assignment residual model. That noise has been removed.
Refit or re-predict saved fits before reusing their old interval summaries. The
Python direct backend uses `target="conditional_mean"`; `"population"` remains a
legacy alias for the same observed-covariate average, and `"sample"` is rejected.
Prediction metadata records this target. This correction does not establish
frequentist coverage of the resulting posterior intervals.

The default time kernel is Matérn 3/2; `direct_config` also permits RBF and
Matérn 1/2 kernels. Smoothing does not constrain monotonicity, the sign of the
first stage, or its probability bounds. Each tree is a stump, so sums of these
trees provide additive covariate functions, not arbitrary interactions without
supplied interaction features. Innovations are independent across equations;
optional correlated unit intercepts provide the cross-equation dependence.

This backend supports continuous outcomes, an LPM first stage, and summaries for
the fitted population. It rejects binary/probit, subgroup and new-covariate
prediction requests. Forest and prior options belong in `direct_config`;
`config` supplies sampling controls. Serialized fits preserve covariance traces.
Legacy direct-backend archives must be refit: their saved contrasts and sampler
semantics precede these repairs, and the new loader rejects them explicitly.

The working Gaussian adoption likelihood does not represent all dependence
created by absorbing binary adoption. Correlated intercepts model a shared unit
component but do not establish the whole joint error distribution. ESS and R-hat
assess computation, not likelihood adequacy or coverage. The matched benchmark
uses independent intercepts as its prespecified primary model; different priors
or forest sizes require separate evaluation.

## Adoption hazard as an exploratory first stage

```python
from longbet import hazard_adoption_effects, HazardConfig

hazard = hazard_adoption_effects(d, z, x, t=t,
    config=HazardConfig(trees=4), chains=4, burnin=500, draws=1000, seed=42)
print(hazard.table)
print(hazard.metadata)
```

```r
hazard <- hazard_adoption_effects(d, z, x, t = t,
  trees = 4L, chains = 4L, burnin = 500L, draws = 1000L, seed = 42L)
hazard$table
```

This extension requires equally spaced observations, even though the general
reference API permits unequal whole-unit gaps. Adoption already present in the
first column is treated as a first-observation event; the model does not recover
earlier adoption histories or implement left truncation.

The hazard likelihood respects absorbing adoption. Its risk set contains units
not yet adopted, a group whose composition can depend on assignment. A hazard
contrast needs assumptions beyond randomization for causal interpretation.
Compare implied population adoption-stock and exposure contrasts with reference
ITTs. The relevance probability is model- and prior-dependent; it does not
eliminate weak-instrument bias or supply robust treatment-effect intervals.
This exploratory module is excluded from the superiority benchmark.

## A common duration response using conventional panel 2SLS

When encouragement accelerates adoption, a current-adoption denominator may
vanish while cumulative exposure differs. Define `S_it = sum_{s<=t} D_is`
and explicitly posit `Y_it = alpha_i + lambda_t + g(S_it) + error_it`.
`duration_deconvolution_effects` uses assignment-by-post-wave instruments after
absorbing unit and time effects:

```python
from longbet import duration_deconvolution_effects

result = duration_deconvolution_effects(y, d, z, t=t, model_type="linear")
print(result.summary_table)
print(result.instrument_rank, result.regressor_rank)
print(result.first_stage_diagnostics)
```

```r
result <- duration_deconvolution_effects(y, d, z, t = t, model_type = "linear")
result$summary_table
result$first_stage_diagnostics
```

This is unregularized 2SLS under a common structural response, not a new source
of identification. It rejects deficient rank and nonzero ridge penalties and
requires finite balanced panels, binary absorbing indicators, common assignment
onset and equally spaced periods. Duration counts observed periods. `linear`
estimates one slope; `stepwise` estimates cumulative duration increments;
`quadratic` uses duration and its square. The legacy `spline` alias is quadratic
and has no knots.

Unit-cluster CR1 covariance counts absorbed fixed effects in the residual degrees
of freedom; pointwise intervals use t(N−1) critical values. Units must be
independent assignment/sampling clusters. Scalar `first_stage_f` is a cluster
Wald statistic divided by instrument rank, not the homoskedastic F statistic.
Multiple regressors have separate diagnostics but no single joint-strength
statistic here. Singular cluster covariance is explicitly unavailable. These
intervals still require strong instruments and enough independent units;
randomization alone does not identify unrestricted heterogeneous duration effects.

## Experimental modular hazard–outcome model

`CoupledHazardIV` is retained as a modular working model. Its hazard module is
fit without outcome feedback; its outcome module samples the exact conditional
Gaussian regression on adoption and `D - Dhat`, with time effects. This is a
cut distribution, not a joint hazard–outcome posterior.

```python
from longbet import CoupledHazardIV, CoupledHazardConfig

experimental = CoupledHazardIV(CoupledHazardConfig(
    num_sweeps=1000, num_burnin=500, seed=42)).fit(y, d, z, t=t)
print(experimental.summary_table)  # working regression coefficients
print(experimental.metadata)
assert experimental.causal_ci is None
```

`D - Dhat` is not generally a valid control function for endogenous binary
adoption. At zero relevance, it and adoption are collinear after time effects,
so the likelihood cannot separate their coefficients. Proper priors can give
finite draws without causal identification. The implementation warns and
withholds causal intervals for every fit. Legacy `beta_ci` describes only the
working coefficient. Covariate adjustment is unsupported and rejected.

## Repeated-sampling evaluation

The historical filename `sbc_calibration.py` now describes its actual task as
fixed-parameter coverage, not Bayesian simulation-based calibration. It uses
known binary-outcome targets, checks truth containment, preserves disjoint grid
runs, reports truncation, and never pads intervals. Exploratory coupled-model
summaries are marked as lacking supported causal coverage. Older numerical
claims from the unrepaired script are invalid evidence.

The [matched comparison](../benchmarks/encouragement/comparison-protocol.md) uses
separate pilot and evaluation seeds, the same finite-population estimands for
all methods, and unadjusted, linear-ANCOVA and spline-ANCOVA IV comparators.
It records failures and diagnostics, pointwise coverage with Monte Carlo
uncertainty, complete Fieller sets, paired accuracy comparisons and runtime.
A sampler repair, narrower intervals, or a correct exposure definition does not
by itself demonstrate improvement over established IV alternatives.
