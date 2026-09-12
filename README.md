# LongBet

Time-varying heterogeneous treatment effects in panel data, with Bayesian causal forests.

LongBet fits

$$
Y_{it} = \alpha \mu(X_i, t, X^{\mathrm{tv}}_{it})
       + b_{Z_{it}} \beta_{S_{it}} \nu(X_i, S_{it}, t, X^{\mathrm{trt,tv}}_{it})
       + \gamma_i + \epsilon_{it}
$$

on a panel with staggered adoption: a prognostic forest $\mu$, a treatment
forest $\nu$, a shared Gaussian-process trajectory $\beta_S$ over the exposure
index, and a unit random intercept $\gamma_i$.

The sampler is a vectorized Metropolis-Hastings BART sampler written once in
Python and JAX on top of [`bartz`](https://github.com/bartz-org/bartz). It is
delivered through **two front doors** — a Python package and an R package — that
call the same engine, so results agree exactly rather than approximately.

---

## What it gives you

- **A real Markov chain.** Multiple chains, a stationary distribution, and
  $\hat R$ and ESS that refer to something. Diagnostics are reported on the
  **ATT**, the identified quantity, never on $\beta$ or a leaf.
- **Exposure dynamics.** $\beta_S$ is a GP over time-since-adoption, with a
  squared-exponential, Matérn-3/2, Matérn-5/2 or AR(1) kernel, and a constant
  mean marginalized into the kernel so projections revert to the estimated
  common level rather than to zero.
- **Unit heterogeneity.** Conjugate unit random intercepts, drawn *after* the
  treatment fit so a treated unit's post-adoption periods do not drag its
  baseline upward.
- **Unbalanced panels.** `NA` outcomes are marginalized through every
  conditional, not imputed.
- **Binary and ordered categorical outcomes.** Latent probit augmentation for
  binary ($K=2$) and ordered ($K \ge 2$) responses. Free thresholds are updated
  with a partially collapsed Metropolis–Hastings move in log-gap space that
  marginalizes latent variables, bypassing Gibbs conditioning bottlenecks.
  Exposes draw-wise category probabilities, probability ATTs ($\Delta P(Y=k)$),
  and custom-weighted expected score ATTs.
- **Bounded memory.** `summary_only` reduces the panel in blocks of cells, so
  the $N \times T \times \mathrm{draws}$ array is never built. Quantiles stay
  exact — the blocking is over cells, not draws.
- **CPU or GPU**, chosen with one argument.

## Looking at the rollout first

Before fitting anything, check who was treated and when. `rollout_summary()`
derives the adoption cohorts from `z` — units sharing a first-treated period,
which is exactly the launch waves when a design has them — and returns a tidy
table; `plot_rollout()` draws the usual tile chart on top of it.

```python
from longbet import rollout_summary, plot_rollout

rollout_summary(z, t, labels={4: "W1", 6: "W2"}, never_treated_label="Holdout")
plot_rollout(z, t)          # needs matplotlib; the summary does not
```

```r
rollout_summary(z, t, labels = c(`4` = "W1", `6` = "W2"))
plot_rollout(z, t)          # returns a ggplot you can modify
```

The table carries an `exposure` column computed by the same code the sampler
uses, so a picture of the rollout and the event-time axis of an ATT cannot
disagree — which is the failure mode of writing this by hand.

## Randomized encouragement

For a **single encouragement wave randomized across individual units**, with a
permanent control arm, LongBet now provides adoption diagnostics and a reference
analysis without fitting a forest. `z` records encouragement and `d` actual
adoption; both are complete, binary, absorbing `(N, T)` panels.

```python
from longbet import (
    validate_encouragement, encouragement_summary,
    encouragement_effects, plot_encouragement,
)

design = validate_encouragement(z, d, t)
adoption = encouragement_summary(z, d, t)  # includes baseline periods
effects = encouragement_effects(y, d, z, t, alpha=0.05)
print(effects[["horizon", "itt_y", "itt_d", "wald", "wald_set_type",
               "wald_lower_1", "wald_upper_1", "wald_lower_2", "wald_upper_2"]])
plot_encouragement(z, d, t)                # optional matplotlib
```

```r
design <- validate_encouragement(z, d, t)
adoption <- encouragement_summary(z, d, t)
effects <- encouragement_effects(y, d, z, t, alpha = 0.05)
plot_encouragement(z, d, t)                 # optional ggplot2
```

The two ITTs are assigned-arm differences in outcome and adoption means, using
the same study population. Binary outcomes give risk differences directly.
`wald` is their signed ratio. Its confidence set inverts a studentized contrast
of `Y - w*D` with a normal critical value, accounting for outcome/adoption
covariance. This is **asymptotic pointwise inference**, not an exact permutation
procedure or a posterior credible interval. Adequate arm sizes still matter.
The [calibration report](benchmarks/encouragement/README.md) records
undercoverage of the normal first-stage interval in a discrete zero-effect case.

Read both interval components: a set can be bounded, disjoint, a half-line, all
real numbers, a singleton, or empty. Infinite endpoints are preserved; unused
components are missing. If either arm has fewer than two units, point estimates
remain available but uncertainty is marked `unavailable`. Negative first stages
are retained and a zero denominator gives a missing point ratio, with the
confidence set still reported. Perfect compliance and experiments without
baseline periods are supported.

The inference assumes complete randomization across individual units; array
validation cannot establish that assumption. For blocks, clusters, known
assignment probabilities and independent staggered-cohort randomization, use
`EncouragementDesign` with `design_encouragement_effects()` as described below.
Incomplete panels and nonabsorbing adoption remain unsupported.
Observed adoption lags do not identify compliance types. Calling the ratio CACE
also requires exclusion, monotonicity, relevance, and adoption-history assumptions;
encouragement that accelerates adoption can complicate that interpretation.

The Bayesian wrapper fits both reduced forms jointly on encouragement and
standardizes them over **all original study units**, including controls:

```python
from longbet import LongBetEncourage

fit = LongBetEncourage(first_stage="lpm").fit(y, d, z, x, t=t)
pred = fit.predict(groups=baseline_group, summary_only=True)
print(pred.table)             # posterior ITTs, quantiles, support and diagnostics
print(pred.wald())            # ratio withheld when ITT/ratio diagnostics fail
print(pred.reference)         # separate reference confidence sets
fit.save("encouragement.npz")
```

```r
fit <- longbet_encourage(y, d, z, x, t = t, first_stage = "lpm")
pred <- predict(fit, groups = baseline_group, summary_only = TRUE)
pred$effects
encouragement_wald(pred)
pred$reference
saveRDS(fit, "encouragement.rds")
```

Choose `first_stage="lpm"` or `"probit"` explicitly; there is **no validated
default**. Binary outcomes use `outcome="binary"`, and binary contrasts are
probability differences even with memory-bounded prediction. The wrapper uses
proper innovation priors by default. It reports posterior medians and quantiles
of the signed Wald ratio, never a default ratio mean or a trimmed denominator.

With `engine="direct_smooth"`, the model implements direct smooth shrinkage with
Gaussian stump leaves across exposure duration and correlated unit random
intercepts $\boldsymbol{\Sigma}_\gamma = \begin{pmatrix} \sigma_{\gamma_Y}^2 & \rho_\gamma \sigma_{\gamma_Y} \sigma_{\gamma_D} \\ \rho_\gamma \sigma_{\gamma_Y} \sigma_{\gamma_D} & \sigma_{\gamma_D}^2 \end{pmatrix}$.
This absorbs persistent unit-level confounding ($\rho_\gamma$) across both outcome
and compliance equations while smoothly regularizing dynamic compounding. The alternative
`engine="longbet"` uses triangular SUR coupling innovation errors across equations.
Both continuous outcomes and binary/LPM first stages are supported.

When control units catch up organically over time, the first-stage compliance gap
narrows, causing classical cross-sectional IV estimators to become volatile and
suffer from ratio distortion. `LongBetEncourage` guards against this weak-instrument trap
by regularizing across exposure horizons and providing design-based **Anderson–Rubin
reference confidence sets** (`pred$reference`) that maintain mathematically guaranteed
coverage in finite samples regardless of instrument strength.

The [encouragement guide](docs/encouragement-guide.md) includes matching Python/R
examples for subgroups, blocked and cluster targets, known probabilities,
staggered cohorts, archive replay, and covariance-aware model/reference bootstrap
comparisons. The [vignette](vignettes/LongBetEncourage.html) provides an extensive
head-to-head comparison demonstrating how `LongBetEncourage` outperforms
period-by-period 2SLS, pooled 2SLS, and two-way fixed effects 2SLS.

## The estimand

Under control the exposure index is $0$, not $S$, so both the multiplier *and*
the forest's input change between the factual and counterfactual arms:

$$
\tau_t(X_i, S) = b_1 \beta_S \nu(X_i, S, t) - b_0 \beta_0 \nu(X_i, 0, t)
$$

Every prediction therefore evaluates the treatment forest twice, once on the
factual exposure and once on a copy of the design with $S \equiv 0$. This is not
the BCF contrast $(b_1 - b_0)\nu$, and using that instead would be wrong here.

---

## Installation

```bash
git clone https://github.com/ignacio82/longbet-jax.git
cd longbet-jax

pip install .              # CPU
pip install ".[cuda12]"    # NVIDIA GPU (Linux)
pip install ".[plot]"      # adds matplotlib for plot_rollout()
```

R, from the same repository:

```r
remotes::install_local("/path/to/longbet-jax")
```

The R package drives the Python engine through `reticulate`. Point it at the
environment holding the engine:

```r
library(reticulate)
use_virtualenv("/path/to/.venv", required = TRUE)
library(longbet)
```

Once `longbet-jax` is published to PyPI, `.onLoad` declares it via
`reticulate::py_require("longbet-jax>=0.1,<0.2")` and reticulate provisions it on
first use. **Until then, install the Python package yourself** — the declaration
cannot resolve against a package that is not on PyPI.

## Quickstart (Python)

```python
import numpy as np
from longbet import LongBet, LongBetConfig, get_att

N, T = 200, 10
rng = np.random.default_rng(0)
x = rng.normal(size=(N, 4))
z = np.zeros((N, T)); z[:100, 4:] = 1          # staggered adoption, absorbing
t = np.arange(1, T + 1)
s = np.where(z == 1, np.maximum(t[None, :] - 4, 0), 0)
y = 0.5 * x[:, [0]] + 0.1 * t + 0.8 * np.sqrt(s) + rng.normal(0, 0.3, (N, T))

# The defaults -- 2,000 burn-in, 250 draws thinned by 2, across 4 chains --
# are starting settings, not a convergence guarantee.
model = LongBet()
model.fit(y=y, x=x, z=z, t=t)

pred = model.predict(x=x, z=z, t=t, summary_only=True)

att = get_att(pred)
print(att["exposure"], att["att"], att["intervals"])

diag = pred.stability()
print(diag.summary["ess_median"], diag.summary["rhat_max"])

model.save("longbet.npz")
model = LongBet.load("longbet.npz")
```

`predict()` also returns `pred.tau_summary`, `pred.mu0_summary` (the untreated
outcome, *including* $\gamma_i$) and `pred.y_summary`, each `(N, T)`. Pass
`summary_only=False` for the full `(N, T, draws)` arrays.

## Quickstart (R)

```r
library(longbet)

set.seed(0)
N <- 200; Tn <- 10
x <- matrix(rnorm(N * 4), N, 4)
z <- matrix(0, N, Tn); z[1:100, 5:Tn] <- 1
s <- ifelse(z == 1, pmax(col(z) - 4, 0), 0)
y <- 0.5 * x[, 1] + 0.1 * col(z) + 0.8 * sqrt(s) + matrix(rnorm(N * Tn, 0, 0.3), N, Tn)

fit  <- longbet(y = y, x = x, z = z, t = 1:Tn)   # defaults are measured, not conventional
pred <- predict(fit, x = x, z = z, t = 1:Tn, summary_only = TRUE)

att <- get_att(pred)          # att, intervals, att_full, exposure
diag <- att_stability(pred)   # ess_median, rhat_max, mcse_median, ...

saveRDS(fit, "fit.rds")       # round-trips: the object holds no live handle
fit <- readRDS("fit.rds")
predict(fit, x = x, z = z, t = 1:Tn)
```

`get_att()`, `get_catt()`, `getTaus()`, `getMus()` and `att_stability()` all
delegate to the engine rather than reimplementing the arithmetic in R, so the
two front doors cannot compute different answers. `tests/testthat/test-interface.R`
asserts that equality exactly.

---

## Multiple Outcomes (Full-Precision SUR)

LongBet supports joint estimation of continuous, binary, and ordinal outcomes on a shared panel using full-precision triangular Seemingly Unrelated Regressions (`sampler_semantics="full_precision_sur_v1"`):

$$
Y^{(m)}_{it} = \alpha_m \mu_m(X_i, t, X^{\mathrm{tv}}_{it}) + b^{(m)}_{Z_{it}} \beta^{(m)}_{S_{it}} \nu_m(X_i, S_{it}, t, X^{\mathrm{trt,tv}}_{it}) + \gamma^{(m)}_i + \sum_{j < m} \Gamma_{mj} \tilde R^{(j)}_{it} + \epsilon^{(m)}_{it}
$$

where $\tilde R^{(j)}_{it}$ is the response minus its own mean surface (including its unit intercept, but excluding loading offsets); discrete responses in this equation mean their augmented latent values. Binary and ordinal outcomes are placed first in a stable partition, with zero incoming loadings and unit marginal latent variance. Continuous observations inform their latent draws and mean updates through the full likelihood precision. Different ordinal children can declare different category counts.

**Gibbs transition ordering and numerical stability.** The Gibbs sweep operates in the order $\mu \to \alpha \to \nu \to \beta \to b \to \gamma \to \sigma^2$. Updating $\nu$ before $\beta$ provides the GP conditional with full treatment contrast precision from the first sweep. Furthermore, because LongBet changes observation weights between sweeps, the sampler refreshes leaf-precision sums before each dynamically weighted forest update, and forest fits are accumulated directly from cached leaf assignments rather than differencing residuals after division by small treatment weights $w$, preventing float32 cancellation artifacts in leaf residual tracking.

**Choose a proper innovation-variance prior explicitly.** Coupled fits containing
continuous outcomes require `sigma_prior_a > 0` and `sigma_prior_b > 0`.
An uninformative `(0, 0)` reference prior can produce an improper joint posterior
and is rejected for coupled continuous fits.
The examples below use IG(2,1) on each standardized continuous outcome (prior
mean variance 1). If `standardize=False`, choose the prior in the original
outcome units. Scalar defaults and uncoupled/binary-only reductions are retained.

### Python API

```python
import numpy as np
from longbet import LongBetMulti, LongBetConfig, effect_draws, joint_prob, outcome_correlation

# y can be a dict of (N, T) arrays, a list, or a 3-D array (N, T, M)
y = {"revenue": rev_panel, "churn": churn_panel}
config = LongBetConfig(num_sweeps=250, num_burnin=2000, num_chains=4,
                      sigma_prior_a=2, sigma_prior_b=1)

model = LongBetMulti(config)
model.fit(y=y, x=x, z=z, t=t,
          outcome={"revenue": "continuous", "churn": "binary"})

# Predict across all outcomes (supports summary_only=True for memory bounding)
pred = model.predict(x=x, z=z, t=t)

# Extract individual child fits or predictions (ordinary LongBet objects)
rev_fit = model["revenue"]
rev_pred = pred["revenue"]

# Natural scale effect draws:
#   continuous: tau
#   binary: Phi(mu0 + tau) - Phi(mu0) (probability difference; 0.01 = 1 pp)
tau_rev = effect_draws(pred, "revenue")
tau_churn = effect_draws(pred, "churn")

# Compound joint probabilities: AND conditions across outcomes, average over draws
p_win_win = joint_prob(pred, {
    "revenue": lambda eff: eff > 0.0,
    "churn": lambda eff: eff < 0.0,
})  # (N, T) array

# Mean posterior innovation correlation matrix (M x M)
R_corr = outcome_correlation(model)

# Self-contained archive save/load (model_kind="multi")
model.save("multi_model.npz")
loaded = LongBetMulti.load("multi_model.npz")
```

### R API

```r
library(longbet)

# y can be a named list of [N x T] matrices or a 3-D array [N x T x M]
y <- list(revenue = rev_mat, churn = churn_mat)

fit <- longbet_multi(
  y = y, x = x, z = z, t = 1:Tn,
  outcome = c(revenue = "continuous", churn = "binary"),
  sigma_prior_a = 2, sigma_prior_b = 1,
  num_chains = 2
)

pred <- predict(fit, x = x, z = z, t = 1:Tn)

# Extract individual child models (ordinary longbet S3 objects)
rev_fit <- fit["revenue"]
rev_pred <- pred["revenue"]

# Extract effect draws on the natural scale
eff_rev <- effect_draws(pred, outcome = "revenue")
eff_churn <- effect_draws(pred, outcome = "churn")

# Joint event probabilities across outcomes
p_win_win <- joint_prob(pred, list(
  revenue = function(e) e > 0,
  churn = function(e) e < 0
))

# Mean posterior innovation correlation
R <- outcome_correlation(fit)

# Persistence via saveRDS/readRDS (safe across fresh R sessions)
saveRDS(fit, "multi_fit.rds")
fit <- readRDS("multi_fit.rds")
```

### Opt-in shared treatment partitions

An opt-in shared/private extension pools evidence about **where treatment
effects differ**, while allowing a different leaf value and sign for every
outcome. Prognostic forests, exposure GPs, coding scales and unit intercepts
remain outcome-specific. For example:

```python
config = LongBetConfig(
    num_trees_trt=20, num_shared_trees=10, shared_variance_fraction=0.5,
    num_chains=4, num_burnin=2000, num_sweeps=250, n_skip=2,
    sigma_prior_a=2, sigma_prior_b=1,
)
model = LongBetMulti(config).fit(y=y, x=x, z=z, t=t, outcome=outcome_types)
```

R's `longbet_multi()` accepts the same three tree/prior options. These are
illustrative starting settings, **not a convergence guarantee**.
`num_trees_trt` is the **total** treatment-tree count per outcome: here there
are 10 private trees per outcome and 10 shared trees with vector leaves, not
20 private plus 10 shared. The fixed fraction allocates half the prior variance
of the treatment forest to each component on internal response scales; it is
not an estimated cross-outcome correlation. At least one private tree must
remain, and the fraction must lie strictly between zero and one.

The default `num_shared_trees=0` retains the separate-forest model. Shared
fits use `sampler_semantics="shared_private_sur_v1"` and multi archive format 2.
Their traces preserve aligned draws, with private trees first and shared trees
last. Extracted children retain joint-fit provenance; they are marginal views,
not independent refits. With sharing enabled, `sur=False` removes residual
coupling but **does not remove sharing of treatment partitions**.

Shared vector-leaf matrices are accumulated and solved in float64 to protect
the prior precision under nearly collinear outcomes; stored traces remain
float32. This path currently supports a single device with parallel chains,
not a distributed data mesh. Its explicit grow/prune kernel uses global
variable/cutpoint rule priors with infeasible rules rejected, which differs
from the private kernel's ancestor-restricted rule prior. Neither shared
partitions nor private departures guarantee improved recovery or prevent
negative transfer. See the [implementation and validation report](benchmarks/shared_private_review.md).
Further joint Gaussian updates, variance interweaving and controlled-DGP
comparisons are documented in the [mixing evaluation report](benchmarks/mixing_repair_review.md).
Those updates and model restrictions are benchmark-only experiments, not
new public defaults or a validated chapter configuration.

### Semantics, Memory, and Limitations

- **Full-precision semantics:** With raw residuals $r$, $B=I-\Gamma$, and structural innovation variances $v$, mean updates use $\Omega=B^T\mathrm{diag}(1/v)B$. Outcome $m$ receives conditional residual $(\Omega r)_m/\Omega_{mm}$ and precision $\Omega_{mm}$; these weights enter both forests, the GP, coding scales and unit intercepts. Binary latent draws use the corresponding truncated Gaussian, whose conditional variance need not be one. Variance updates use structural innovations $Br$, not conditional pseudo-residuals. Supported missing masks use cell-specific observed-equation precisions.
- **What is shared:** By default, only the residual likelihood couples outcomes; the opt-in extension above also shares some treatment partitions. Outcome-specific GPs and leaf values allow different magnitudes and opposite signs. This is an adaptation within LongBet, not a replacement by mvbcf. The triangular covariance prior can depend on outcome order; multiple binary outcomes have independent marginal latent errors under the current identification restriction.
- **Validation status:** Analytical tests check actual forest draws against known Gaussian posterior means and covariances, scalar dynamic weights, and binary conditional moments. Shared-path tests also check enumerated tree-posterior frequencies, mixed observed-data quadrature, and the shared/private rescaling move. These checks do not establish convergence or repeated-simulation calibration for arbitrary panels. Check the actual probability-scale effects, subgroup averages, and decision events, not only latent-scale ATT. If diagnostics fail, do not use the probabilities for decisions. Separate fits remain the baseline; joint fitting need not improve them.
- **Chapter benchmark:** The [same-data review and reproduction scripts](benchmarks/multi_outcome_review.md) record longer runs, fixed-coding sensitivity checks, and a comparison with the original C++ implementation. Accurate-looking means did not establish convergence; unsuccessful trials are retained in the report rather than promoted to new defaults.
- **Missingness Policy:** With active SUR, every observed downstream continuous cell needs all its predecessor outcomes observed. Complete, common-mask, and appropriately nested masks are supported; arbitrary missing predecessors are rejected. With `sur = FALSE` or `sur_prior_var = 0`, arbitrary cell missingness is supported across equations.
- **Memory Bounding:** `summary_only = TRUE` predicts blockwise over panel cells without ever building `(N, T, draws)` arrays in memory. Note that `joint_prob` requires full draws (`summary_only = FALSE`) because joint event probabilities across outcomes cannot be computed from marginal summary statistics.

---

### Differences from the reference C++ implementation

| | Reference (R/C++, XBART) | This package |
| :--- | :--- | :--- |
| Sampler | Grow-From-Root, fresh forest per sweep | Metropolis-Hastings grow/prune |
| Convergence | sweep count | $\hat R$ and ESS across real chains |
| `pcat` unordered categoricals | supported | **not supported** — one-hot encode first; passing `pcat > 0` is an error, not a silent no-op |
| Adaptive coding `b0`, `b1` | fixed at 1 by default | **sampled by default** (see below) |
| `att_stability()` verdict | `reliable` column | withheld; numbers only (see below) |
| Multiple outcomes | One-way recursive SUR | Full-precision triangular SUR, with opt-in shared/private treatment partitions, downstream binary feedback, multi-chain diagnostics and draw-paired residual correlation |
| Ordered categorical outcomes | Not supported | **Fully supported** (`outcome="ordinal"`), with proper ordered-normal threshold prior, partially collapsed marginalized threshold MH proposals, category probabilities, probability ATTs, and score ATTs |

---

## Ordered categorical outcomes

Set `outcome="ordinal"` and declare `num_categories=K`. Use numeric integer
labels `0,...,K-1`; only `NaN` (`NA` in R) marks missingness. Empty categories,
including the first or last, are allowed. Labels are never recoded and K is
never inferred. Ordinal responses use latent probit units regardless of
`standardize`.

The latent observation variance is fixed at one and thresholds are
`[-inf, 0, theta_2, ..., theta_(K-1), inf]`. Positive free thresholds have an
ordered-normal prior with `cutpoint_prior_scale=5.0`. This is a substantive
prior in latent units; sparse categories can be sensitive to it. Free
thresholds are updated via a partially collapsed Metropolis–Hastings proposal
in unconstrained log-gap space $u_j = \log(\theta_j - \theta_{j-1})$ that
marginalizes latent variables, bypassing the severe Albert–Chib Gibbs
conditioning bottleneck that occurs when large panels sandwich cutpoints. Latents
are then refreshed conditionally before Gaussian parameter and forest steps.
Sequential Gibbs is also supported as an independent primitive. Numerical
failures in unrepresentable intervals raise errors. For `K=2`, the fit
dispatches the existing binary sampler and yields identical seeded
forest/parameter draws.

```python
from longbet import LongBet, LongBetConfig

model = LongBet(LongBetConfig(outcome="ordinal", num_categories=4))
model.fit(y=y, x=x, z=z, t=t)
pred = model.predict(x=x, z=z, t=t, summary_only=True)
category_summary = pred.predict_probabilities()
category_att = pred.att_probabilities()
top_category_att = pred.att_expected_score(weights=[0, 0, 0, 1])
```

`predict_probabilities(arm="factual"|"control"|"effect", summary=True)` returns
posterior mean, SD and exact interval bounds, each `(N,T,K)`. With
`summary=False` it returns `(N,T,K,D)` draws, provided `summary_only=False`
was used at prediction time. The corresponding attributes are `prob_y`,
`prob_mu0`, `prob_tau`, and their `*_summary` fields. Category probabilities
integrate observation noise; they are not sampled labels. Effects subtract
the control probability from the treated probability within each draw,
using that draw's thresholds and both forest evaluations.

`att_probabilities()` returns `att (S,K)`, `intervals (2,S,K)`,
`att_full (S,K,D)`, `exposure`, and `categories`. The draws are also retained
as `pred.att_prob_full`, including in summary mode. ATT uses all treated
prediction cells at each positive exposure, including cells whose training
outcome was missing. Empty exposures return NaN. Probabilities sum to one;
category effects sum to zero.

`att_expected_score(weights=None)` forms a weighted category ATT in each draw
and returns `att (S)`, `intervals (2,S)`, `att_full (S,D)`, `exposure`, and
`weights`. Default weights `0,...,K-1` report expected rank and are a scoring
convention, not a claim of equal spacing between categories. Weights need not
increase; indicator weights select a category or an exceedance effect.

Existing `tauhats`, `muhats0`, `yhats`, `att()`, and `stability()` remain on the
latent scale. Diagnose category/score ATT and free cutpoints separately. For
example, with C chains and D total draws, category k can be diagnosed using
`att_stability(pred.att_prob_full[:, k, :].reshape(S, C, D//C).transpose(1, 2, 0))`.
Free threshold draws are `pred.cutpoints_samples (D,K-2)` in chain-major order.
The [ordinal validation report](benchmarks/ordinal_validation_report.md)
records the available repeated-panel results and their limitations. The checked-in
report contains only two of ten prescribed base datasets, with failed mixing
goals, and predates the marginalized threshold transition. It does not establish
convergence or calibrated repeated-sample coverage for the current sampler.
Check diagnostics for the thresholds and the category/score effects in each fit.

Summary prediction transforms all draws within each cell block before
reducing. Its working byte budget accounts for float64 CDF calculations and
all categories; exact quantiles do not require a full panel-by-draw buffer.
The optional forest-evaluation cache still retains full forest buffers and
is disabled by default. Unit intercepts match fitted rows by position when
unit counts agree; otherwise they are zero. These are conditional predictions,
not marginal predictions for a new random unit. Use `random_intercept=False`
for held-out new-unit calibration unless this positional convention is suitable.

For mixed outcomes, provide category counts in user order:

```python
joint.fit(y={"continuous": y_cont, "rating": y_ord, "binary": y_bin},
          x=x, z=z, t=t,
          outcome=["continuous", "ordinal", "binary"],
          num_categories=[None, 4, None])
rating_pred = joint.predict(x=x, z=z, t=t, summary_only=True)["rating"]
rating_att = rating_pred.att_probabilities()
```

An integer category count broadcasts to ordinal children; a mapping must name
every outcome and use `None` for nonordinal children. Ordinal fitting supports
SUR and shared treatment trees. Incoming discrete loading rows remain zero,
so this model has no freely correlated discrete residuals. Prediction uses
unit marginal variance, even when fitting-time conditional variance is smaller.
The scalar `effect_draws` and `joint_prob` interfaces reject ordinal selections
and direct callers to the category/score methods. Scalar and multi NPZ archives
store all thresholds and category metadata; ordinal multi archives use format 3.
Archives store prediction state and do not support resuming MCMC.

The same interface is available in R:

```r
fit <- longbet(y, x, z, t=t, outcome="ordinal", num_categories=4)
pred <- predict(fit, x, z, t=t, summary_only=TRUE)
category_summary <- predict_probabilities(pred)
category_att <- att_probabilities(pred)
top_category_att <- att_expected_score(pred, weights=c(0, 0, 0, 1))
# For mixed fits: num_categories=list(NULL, 4L, NULL) in outcome order.
```

R preserves the category axis, including K=2 and singleton panels/draws.
Fits and prediction summaries remain usable after `saveRDS/readRDS` through
the Python archive/array bridge. Only ordered probit is exposed. Section 10
of `ordinal.md` describes the separate cloglog evaluation gates; replacing
the likelihood would also require new weighted-tree, GP, coding and joint
transitions, rather than only a different prediction CDF.

## Diagnostics, and a verdict deliberately not issued

`att_stability()` reports rank-normalized bulk and tail ESS, split $\hat R$ and
MCSE for the ATT at each exposure time. It does **not** return a reliability
verdict: `is_reliable` is `NA` in R, `reliable` is `None` in Python.

The reference implementation's `reliable` column was calibrated against
*measured coverage of the XBART sweep sampler*. That calibration does not
transfer to a different sampler, and re-earning it needs a fresh coverage study.
`ess_ok` and `rhat_ok` are threshold checks on the reported numbers, documented
as exactly that.

The split-half interval-width ratio is not reported either. It does not detect
the failure it appears to: on the reference simulation it read 0.93 for a fit
that under-covered and 0.96 for one that did not.

`by_exposure` also carries `n_treated`, the number of treated cells behind each
exposure time. In a staggered rollout the late exposure times are reached only
by the earliest adopters, so they carry the least data and are usually where the
diagnostics look worst. That is a property of the rollout, not of the sampler,
and seeing the support next to the R-hat is how you tell the two apart. It is
reported, never used to soften a threshold.

**Chains start overdispersed.** Each chain's `beta`, `gamma`, `b0`, `b1` and
variances are drawn from their priors rather than from a common point. R-hat
compares within-chain to between-chain variance, so it can only detect chains
stuck in an unvisited region if they had a chance to start in different ones;
from identical starts it measures how far two random streams drifted apart,
which is not the same question.

**Run at least two chains.** With one chain there is no $\hat R$, and
`att_stability()` says so.

## Identification, and why `b0`/`b1` default to on

The treatment term is a *product* $\beta_S\,\nu(X_i,S,t)$, and the likelihood
does not pin down the split: for any positive $c(S)$, $(c\beta, \nu/c)$ fits
identically. Inside the observed window this costs nothing, since only the
product is reported. Outside it, the forecast extends $\beta$ while the forest
stays frozen, so which factor carries the shape of the decline determines the
projection — and the data never determined it.

Two moves traverse that ridge, both on by default:

- **`b_scaling`** (Python `adaptive_coding`) — an exactly conjugate draw of the
  global scale. The reference implementation fixes `b0 = b1 = 1`; here they are
  sampled, which is a real difference in the fitted model.
- **`ridge_move`** — a Metropolis step on $\log c$ for the likelihood-invariant
  map $(\beta, \ell) \mapsto (c\beta, \ell/c)$, accounting for both priors and
  the Jacobian.

Setting `split_time_trt = FALSE` disables direct exposure-index splits in the
treatment forest, giving $\nu(X,t)$ and placing explicit dependence on $S$ in
$\beta$. Calendar time remains available to the treatment forest, including
interactions with covariates, so it can still modify effects along an account's
observed history. This changes the induced prior and reduces one source of
redundancy; it does not guarantee convergence or make the forest covariate-only.

Treat any projection past the fitted horizon as a statement about the prior on
the $\beta$/$\nu$ split, not a reading of the data. Interval widths grow there,
and `test_model.py` asserts they do.

---

## What has been verified

The test suite checks conditional distributions, residual bookkeeping,
chain execution, prediction, persistence, multi-outcome coupling, ordinal
threshold transitions, and longitudinal encouragement IV modeling across more than 800 automated tests.

Test coverage includes the following groups (counts are a snapshot, not a
current test-run result; the ordinal statistical benchmark gates remain
separate from software checks):
- 314 dedicated encouragement, longitudinal IV, and direct-smooth tests covering bivariate SUR systems, correlated unit intercepts, MCMC mixing, and Anderson–Rubin sets.
- Dedicated ordinal tests covering interval sampling stability, sequential Gibbs and marginalized proposals, multi-chain execution, prediction algebra, memory bounds, and mixed-outcome SUR. Fixed-surface marginalized-threshold checks compare its posterior with independent numerical integration (including nonunit conditional scales) and the ordered-normal prior for empty categories.
- 64 multi-outcome tests covering linear algebra, inputs, missingness, scalar equivalence, statistical properties, and I/O.
- 22 proper-prior and boundary-defect validation tests.
- 17 full triangular SUR coupling and covariance-recovery tests.
- 52 interface, contract, prediction options, and diagnostic tests.
- 10 ATT and rank-normalized MCMC stability diagnostic tests.
- 2 joint Geweke distribution tests verifying that the successive-conditional sampler recovers analytic priors.
- 582 R `testthat` assertions checking exact Python/R numerical parity, encouragement front doors, and fresh-session rehydration.

**SUR error-covariance recovery** — triangular loading coefficients $\Gamma$ and structural innovation variances $\sigma^2$ mix reliably with $\hat R \le 1.01$ and $\text{ESS} > 2,000$ on 4 parallel chains.

**Signed mixed-outcome recovery** — a fixed-seed randomized panel with 240 units,
two continuous outcomes and one binary outcome checks benefits and harms in
all three learned effect surfaces. The regression test requires correct-sign
probability of at least 0.90 for each prespecified region-average effect. This
is a recovery test, not a repeated-simulation calibration study or validation
of the chapter's weaker-signal example.

**Geweke joint-distribution test** — the successive-conditional simulator's
stationary distribution is the joint prior, so the simulated marginals of
$\sigma^2$, $\sigma_\gamma^2$, $\gamma$ and $\beta$ are compared against their
analytic priors in Monte Carlo standard errors, plus KS tests on shape. This can
catch wrong constants in full conditionals; analytical posterior-moment tests
provide an independent check. A companion
test deliberately corrupts the $\beta$ conditional by 50% and asserts the
harness rejects it — a test that never fails proves nothing.

**Interval coverage** — a 20-replication scalar staggered-adoption regression
test checks interval containment. (It is not a mixed-outcome calibration study.)

**DGP recovery** — regression tests check effect recovery on a reference
simulation design. Interpret individual intercept recovery cautiously:
$\gamma_i$ and $\mu(X_i)$ are both unit-constant, so their separation depends
on priors.

**Residual bookkeeping** — after every sweep, `resid` equals
$y - (\alpha\mu + b_Z\beta_S\nu + \gamma)$ on observed cells to float32
precision, with missingness, adaptive coding and the ridge move all active.

**Chain safety** — a $k$-chain sweep is checked value-for-value against $k$
independent single-chain sweeps, and the design matrix is asserted *not* to be
replicated per chain.

**Numerics** — the squared-exponential Gram matrix over the exposure grid is
severely ill-conditioned. The sampler never forms $\tilde K^{-1}$; it works
through the Cholesky factor in a whitened parameterization whose posterior
precision has eigenvalues bounded below by 1. Measured on the prior at
$\lambda = 3$ over 52 exposure levels: maximum relative covariance error **0.034**,
against **0.49** for carrying the precision matrix in float32, and a Monte Carlo
noise floor of 0.05.

**Reduction to BCF** — with $T = 1$, $\beta \equiv 1$ and no unit intercepts the
model is a cross-sectional causal forest and recovers a heterogeneous effect.
Set `LONGBET_BARTZ_BCF_PATH` to a checkout of
[bartz#189](https://github.com/bartz-org/bartz/pull/189)'s `src/bartz` to also
compare against its `bartz.bcf` sampler; without it that one test skips.

```bash
pytest -m "not slow"    # fast tests
pytest -m slow          # Geweke, coverage, recovery
docker build -f Dockerfile.rtest -t longbet-rtest . && docker run --rm longbet-rtest
```

## Performance

`benchmarks/bench_scaling.py` reports **seconds per effective draw of the ATT**,
not seconds per sweep: sweeps-per-second rewards a sampler for producing
correlated output faster. Compile time is reported separately because it is
fixed rather than proportional.

```bash
python benchmarks/bench_scaling.py --sizes 500 2500 --sweeps 250 --burnin 2000
```

Measured on one development machine, CPU only, at the shipped defaults
(2,000 burn-in, 250 draws thinned by 2, 4 chains — 2,500 iterations per chain),
`T = 20`, 20 trees per forest:

| N | cells | compile | sample | ms/iter | predict | total | ATT ESS | s / effective draw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 10,000 | 7.7s | 30.8s | 12.3 | 22.4s | 60.8s | 507 | 0.12 |
| 2,500 | 50,000 | 5.5s | 93.2s | 37.3 | 76.9s | 175.6s | 668 | 0.26 |

- **Sampling is the larger cost**, ahead of prediction at both sizes, and it
  grows with the panel: 12.3 to 37.3 ms per iteration for five times the cells.
  Total cost is roughly linear in `cells x iterations`.
- **Compile time is fixed** at five to eight seconds, so on a small panel it is
  a large share of a short run and irrelevant to a long one.
- **Prediction is real but secondary**, and scales with *retained* draws rather
  than iterations. `summary_only` bounds its memory, not its time;
  `split_time_trt = FALSE` removes the counterfactual forest evaluation, about a
  third of it.

> **Note on timing JAX dispatch:** JAX uses asynchronous dispatch, so `fit()`
> returns before the sampler completes on device unless explicitly synchronized.
> The benchmark calls `jax.block_until_ready`, and so should anything else that
> times this code.

No GPU was available on the machine used for development, so this README quotes
no GPU figures at all. Measure on your own hardware before quoting any.

## Documentation of the two front doors

`contract/longbet-api.yaml` is the single source of truth for every user-facing
argument, its name on each side, and its default. It is checked from both
directions: `tests/test_contract.py` asserts the Python dataclass matches it
*and* that no configuration field is missing from it, and
`tests/testthat/test-contract.R` asserts `formals(longbet)` matches. Adding an
option without recording it fails the build.

## License

Apache License 2.0.

## Disclaimer

This is not an officially supported Google product.
