# LongBet

Bayesian causal forests for panel data, in JAX, with a Python front door and an R
front door that call the same engine. Three models are supported.

## 1. `LongBet`: time-varying heterogeneous treatment effects in a panel

### The model

For unit $i$ in a unit-spaced period $t$, exposure $S_{it}$ is $1$ in the
first treated period, increases by one thereafter, and is $0$ before the
unit's absorbing treatment starts. For a continuous outcome,

```math
Y_{it} = \mu(X_i, X^{\mathrm{tv}}_{it}) + \eta_{it}
       + \beta_{S_{it}}\, \nu(X_i, t, X^{\mathrm{trt,tv}}_{it})\, \mathbf{1}\{S_{it} \ge 1\}
       + \gamma_i + \epsilon_{it},
\qquad \epsilon_{it} \sim \mathcal{N}(0, \sigma^2).
```

- **Prognostic and treatment forests ($\mu, \nu$).** $\mu$ is a prognostic sum of
  trees over baseline covariates $X_i$ and optional time-varying covariates
  $`X^{\mathrm{tv}}_{it}`$; $\nu$ is a treatment sum of trees over baseline
  covariates $X_i$ (or a separate moderation matrix `x_trt`), calendar time $t$
  (`split_calendar_trt = True`), and optional time-varying covariates
  $`X^{\mathrm{trt,tv}}_{it}`$. Each tree has the BART
  prior: a node at depth $d$ splits with probability $\alpha/(1+d)^{\beta}$
  ($0.95/(1+d)^2$ prognostic, $0.25/(1+d)^3$ treatment) and leaf values are
  $\mathcal{N}(0, \tau^2)$ with $\tau = 3/(2\sqrt{m_\mu})$ and
  $1/\sqrt{m_\nu}$ for $m$ trees on the standardized response.
- **Conjugate calendar-time block ($\eta_{it}$, `use_trend_block = True`).**
  On treated cells, calendar time, adoption period $E_i$, and exposure satisfy
  $t = E_i - 1 + S_{it}$. Separating untreated trends from exposure effects
  can therefore be difficult for the sampler. By default, the prognostic
  forest excludes direct calendar-time splits and the treatment forest allows
  them. Common period effects and smooth differential trends are represented by
  $`\eta_{it} = \Phi^{\mathrm{time}}_t c_0 + \sum_{k=1}^{d} \phi_k(t)\, b(X_i)^\top c_k`$,
  where $\Phi^{\mathrm{time}} \in \mathbb{R}^{T \times (T-1)}$
  (`trend_period_effects = True`) is a complete centered DCT-II basis with
  orthogonal columns of unit mean square. Its coefficients have independent
  Gaussian priors with standard deviation `trend_period_scale = 2.0`;
  the induced marginal standard deviation of a period effect is
  $2\sqrt{T-1}$, not $2$.
  The centered polynomial basis $\phi(t)$ has the same normalization and
  default degree `trend_degree = 2`, limited by the number of periods.
  Common period effects are unrestricted on the observed grid; differential
  trends lie in this finite polynomial-by-feature span.
  The feature map has $q=p_0+F$ columns, where $p_0=\min(p,24)$ and
  `trend_num_features = 64` sets $F$. It combines the first $p_0$ standardized
  covariate columns with random Fourier features drawn once from a fixed seed;
  nondegenerate columns are centered and scaled to unit mean square.
  Interaction coefficients have independent Gaussian priors with standard
  deviation $\sigma_c/\sqrt{dq}$, where `trend_prior_scale = 2.0` sets
  $\sigma_c$. Empty interaction blocks are omitted.
  With **`trend_horseshoe=True`**, those interaction coefficients instead have
  $c_j\mid\lambda_j,\tau\sim\mathcal N(0,\tau^2\lambda_j^2)$,
  $\lambda_j\sim C^+(0,1)$ and
  $\tau\sim C^+(0,\sigma_c/\sqrt{dq})$. This is a horseshoe prior without a
  regularized-horseshoe slab; there is no `trend_horseshoe_tau0` option.
  The implementation samples its inverse-Gamma auxiliary variables with
  numerical bounds. Common-period coefficients retain their Gaussian priors.
  Conditional on these scales, trend coefficients, unit intercepts, and the
  exposure trajectory are drawn jointly using Gaussian elimination. This
  addresses their conditional dependence but does not guarantee convergence.
  `split_calendar_mu=None` resolves to `False` when both the trend block and
  common-period basis are enabled, and to `True` otherwise.
- **Exposure trajectory ($\beta_S$).** $\beta_S$ is a Gaussian process over the
  exposure clock,
  $`\beta \sim \mathcal{N}\!\big(0,\, K + \sigma_m^2 \mathbf{1}\mathbf{1}^\top\big)`$,
  with a squared-exponential (default), Matérn-3/2, Matérn-5/2 or AR(1) kernel
  $K$ of marginal variance $\sigma_k^2$ and lengthscale $\lambda$, and a constant
  mean marginalized into the kernel so projections beyond the fitted horizon
  revert to the estimated common level. By default the treatment forest does
  not split on $S$ (`split_exposure_trt = False`): the trajectory carries the
  exposure profile at fixed covariates and calendar time. The factors retain a
  scale/sign ambiguity; their priors regularize the decomposition. This
  restriction alone does not establish causal identification. Defaults are
  `sig_knl=1.0`, `lambda_knl=1.0`, and `sigma_m=1.0`. The implemented covariance
  also includes diagonal jitter $\max(10^{-6}\sigma_k^2,10^{-8})I$.
- **Random intercepts and variances.** $\gamma_i \sim \mathcal{N}(0, \sigma_\gamma^2)$
  is a unit random intercept (`random_intercept = True`),
  $\sigma^2 \sim \mathrm{IG}(2, 1)$ on the standardized scale and
  $\sigma_\gamma^2 \sim \mathrm{IG}(1, 0.1)$.
- **Binary and ordinal outcomes.** Binary outcomes:
  $`Y_{it} = \mathbf{1}\{Y^*_{it} > 0\}`$ with the same mean surface (including
  $\eta_{it}$) for the latent $`Y^*_{it}`$ and unit innovation variance (probit).
  Ordinal outcomes require `outcome="ordinal"`, `num_categories=K`, and numeric
  labels $0,\ldots,K-1$, with $K\ge2$:
  $`Y_{it}=k\iff\kappa_k<Y^*_{it}\le\kappa_{k+1}`$, where
  $\kappa_0=-\infty$, $\kappa_1=0$, and $\kappa_K=\infty$.
  The free ordered positive cutpoints have an ordered-normal prior with
  `cutpoint_prior_scale=5.0`; their log-gaps are updated by a marginalized
  Metropolis move. The latent innovation variance is fixed at one.

For continuous outcomes under the default factorization, the treated-versus-untreated mean contrast is
$`\tau_t(X_i,S)=\beta_S\nu(X_i,t,X^{\mathrm{trt,tv}}_{it})`$; the baseline and
unit intercept cancel. `att()` averages this over treated cells at each
exposure. For binary and ordinal outcomes, `att()` and `catt()` summarize
**latent-scale** effects. Binary probability differences are
$\Phi(m^0_{it}+\tau_{it})-\Phi(m^0_{it})$, where $m^0_{it}$ includes the
untreated surface and unit intercept. Obtain their full draws with
`effect_draws(pred)` after `predict(..., summary_only=False)`. For ordinal
outcomes, use `pred.att_probabilities()` or `pred.att_expected_score(weights=...)`.
Probability contrasts generally do not share the latent effect's factorization.

For causal interpretation, the continuous additive model requires valid
untreated counterfactuals, no anticipation/interference, and treatment/control
overlap for the target profiles. Conditional parallel trends additionally
requires adoption-ignorable mean innovations; an additive time-invariant unit
level alone does not establish it. Finite-history shrinkage can leave bias
when adoption selects on that level. Time-varying adjustment requires appropriate
causal assumptions; being unaffected by treatment alone does not establish
that a covariate is valid to adjust for.
Scalar missing outcomes (`NaN`) are omitted from the likelihood under an
ignorable missingness assumption; informative dropout is not modeled.

Each sweep combines forest GROW/PRUNE, local/internal CHANGE and REGROW
proposals, Gaussian parameter updates, a scale-ridge Metropolis move, and
two-stage inter-ensemble Gaussian transfers (`use_inter_ensemble_move=True`).
With the trend block enabled, the joint $(c,\gamma,\beta)$ update follows the
transfer moves; variance updates complete the sweep. Defaults are 4 chains,
2,000 burn-in and 1,000 retained sweeps per chain (`n_skip=1`), 20 prognostic
and 60 treatment trees, and `max_depth_pr=max_depth_trt=8`.

The examples use `t=1,...,T`. With gaps in `t`, exposure is rounded elapsed
time since the last observation before adoption; for adoption in the first
column, the reference is one median time step before that column. Thus the
first treated exposure need not be one on an irregular grid. The calendar-time
basis is constructed over the ordered panel columns. Use unit-spaced indices
when the intended exposure clock counts observed periods.

### Python

```python
import numpy as np
from longbet import LongBet, LongBetConfig, get_att

rng = np.random.default_rng(0)
N, T = 200, 10
x = rng.normal(size=(N, 4))
z = np.zeros((N, T)); z[:100, 4:] = 1                 # absorbing, staggered adoption is fine
t = np.arange(1, T + 1)
s = np.where(z == 1, np.maximum(t[None, :] - 4, 0), 0)
y = 0.5 * x[:, [0]] + 0.1 * t + 0.8 * np.sqrt(s) + rng.normal(0, 0.3, (N, T))

model = LongBet(LongBetConfig(num_chains=4, num_burnin=2000, num_sweeps=1000, n_skip=1))
model.fit(y=y, x=x, z=z, t=t)                          # choose outcome="binary" / "ordinal" in LongBetConfig

pred = model.predict(x=x, z=z, t=t, summary_only=True)  # any counterfactual schedule z may be passed
att = get_att(pred)                                     # att, intervals, exposure
diag = pred.stability()                                 # bulk/tail ESS, rank R-hat, MCSE, and treated-cell counts
print(att["exposure"], att["att"], diag.summary["ess_min"], diag.summary["rhat_max"])
model.save("longbet.npz"); model = LongBet.load("longbet.npz")
```

### R

```r
library(longbet)
set.seed(0)
N <- 200; Tn <- 10
x <- matrix(rnorm(N * 4), N, 4)
z <- matrix(0, N, Tn); z[1:100, 5:Tn] <- 1
s <- ifelse(z == 1, pmax(col(z) - 4, 0), 0)
y <- 0.5 * x[, 1] + 0.1 * col(z) + 0.8 * sqrt(s) + matrix(rnorm(N * Tn, 0, 0.3), N, Tn)

fit  <- longbet(y = y, x = x, z = z, t = 1:Tn, num_chains = 4, num_burnin = 2000, num_sweeps = 1000)
pred <- predict(fit, x = x, z = z, t = 1:Tn, summary_only = TRUE)
att  <- get_att(pred)                 # att, intervals, att_full, exposure
diag <- att_stability(pred)           # ess_min, rhat_max, mcse per exposure
saveRDS(fit, "fit.rds"); fit <- readRDS("fit.rds")
```

## 2. `LongBetMulti`: several outcomes on one panel

### The model

Outcomes share the panel, covariate design, treatment schedule, and
calendar-time basis. Each has its own forests, trend coefficients, exposure
trajectory, and unit intercepts: there is no shared treatment forest or shared
exposure curve across outcomes. Write $`f^{(m)}_{it}`$ for outcome $m$'s full
LongBet mean surface, including calendar-time moderation. With `sur=True`
(the default), a triangular seemingly unrelated regression couples the residuals:

```math
Y^{*(m)}_{it} = f^{(m)}_{it}
             + \sum_{j < m} \Gamma_{mj}\big(Y^{*(j)}_{it}-f^{(j)}_{it}\big) + \epsilon^{(m)}_{it},
\qquad \epsilon^{(m)}_{it} \sim \mathcal{N}(0, \sigma_m^2),
```

where $`Y^*`$ denotes the internally standardized continuous response or the
latent probit response. Binary and ordinal outcomes are placed first, preserving
their relative input order, with zero incoming loadings and unit latent variance.
Consequently, discrete outcomes have no freely estimated residual correlation
with one another. Continuous outcomes retain their relative input order and
may load on predecessors. Free loadings have independent
$\mathcal N(0,\texttt{sur_prior_var})$ priors, with default variance one.
The full SUR precision enters the conditional updates, so downstream continuous
outcomes can inform earlier mean surfaces and latent responses. `sur=False`
removes residual coupling. Outcomes' unit-intercept priors remain separate.

With SUR enabled, an observed continuous outcome requires all its predecessors
in the internal ordering to be observed in that cell. Arbitrary missingness
patterns are not supported; incompatible masks raise an error.

Posterior draws are aligned across outcomes, permitting joint events such as
$`\Pr(\tau^{(1)}>0,\tau^{(2)}<0)`$. `effect_draws()` and `joint_prob()` require
full prediction draws (`summary_only=False`). Continuous effects stay in the
supplied outcome units; binary effects are probability differences
$\Phi(m_0+\tau)-\Phi(m_0)$. Those helpers do not accept ordinal outcomes;
select category probabilities or a score through the individual prediction's
`att_probabilities()` or `att_expected_score()` methods. Aligned draws do not
establish convergence or calibration of a joint event; inspect the relevant
effects and event draws as well as `pred.stability()` for each outcome.

### Python

```python
from longbet import LongBetMulti, LongBetConfig, effect_draws, joint_prob, outcome_correlation

y = {"revenue": revenue_panel, "churn": churn_panel}     # (N, T) arrays, or a list, or an (N, T, M) array
model = LongBetMulti(LongBetConfig(num_chains=4, num_burnin=2000, num_sweeps=1000))
model.fit(y=y, x=x, z=z, t=t, outcome={"revenue": "continuous", "churn": "binary"})

pred = model.predict(x=x, z=z, t=t, summary_only=False)  # pred["revenue"] is a LongBetPrediction
tau_revenue = effect_draws(pred, "revenue")              # (N, T, draws)
tau_churn = effect_draws(pred, "churn")                  # probability differences
p_win_win = joint_prob(pred, {"revenue": lambda e: e > 0, "churn": lambda e: e < 0})   # (N, T)
R = outcome_correlation(model)                           # posterior mean residual correlation, (M, M)
model["revenue"]                                         # the outcome's own LongBet fit
model.save("multi.npz"); model = LongBetMulti.load("multi.npz")
```

### R

```r
fit <- longbet_multi(y = list(revenue = revenue_mat, churn = churn_mat), x = x, z = z, t = 1:Tn,
                     outcome = c(revenue = "continuous", churn = "binary"),
                     num_chains = 4, num_burnin = 2000, num_sweeps = 1000)
pred <- predict(fit, x = x, z = z, t = 1:Tn, summary_only = FALSE)
eff_revenue <- effect_draws(pred, outcome = "revenue")
eff_churn   <- effect_draws(pred, outcome = "churn")
p_win_win   <- joint_prob(pred, list(revenue = function(e) e > 0, churn = function(e) e < 0))
R <- outcome_correlation(fit)
fit["revenue"]                                            # the outcome's own longbet fit
saveRDS(fit, "multi.rds"); fit <- readRDS("multi.rds")
```

## 3. `LongBetEncourage`: a randomized offer with absorbing adoption

### The model

A single individually randomized encouragement wave $Z_{it}$ (offered from a
common launch period onward, with a permanent control arm) and an absorbing
adoption indicator $D_{it}$ are observed. This interface requires complete
outcome, adoption, and offer panels and supports continuous or binary outcomes.
Two LongBet equations, with calendar-time blocks enabled by default, are fitted
separately and composed. Their unit-intercept priors are independent; this is
not a joint model with estimated outcome/frailty correlation.

**Outcome on the adoption clock.** With $S^D_{it}$ the periods since unit $i$
adopted (0 before adoption),

```math
Y_{it} = \mu_Y(X_i) + \eta^Y_{it} + \beta^Y_{S^D_{it}}\, \nu_Y(X_i,t)\, \mathbf{1}\{S^D_{it} \ge 1\}
       + \gamma^Y_i + \epsilon^Y_{it}.
```

This is the model of Section 1 with adoption as the treatment; both arms
contribute adoption events. The treatment term models the adoption response,
its exposure profile and its covariate/calendar-time heterogeneity. A binary
outcome uses the probit form. A causal interpretation additionally requires
exclusion (the offer affects the outcome only through adoption) and no residual
confounding of adoption timing with outcome innovations given the modeled
covariates and unit level. Randomizing the offer does not itself establish
these assumptions about adoption.

**Adoption as a discrete-time hazard on the offer clock.** With $S^Z_{it}$ the
periods since the offer (0 for the control arm), the hazard of first adoption
on the at-risk set $`\{D_{i,t-1} = 0\}`$ is

```math
\lambda_{it} = \Pr(D_{it} = 1 \mid D_{i,t-1} = 0)
= \Phi\!\Big(\mu_D(X_i) + \eta^D_{it} + \beta^D_{S^Z_{it}}\, \nu_D(X_i,t)\, \mathbf{1}\{S^Z_{it} \ge 1\} + u_i\Big),
```

a binary LongBet fitted to the first-adoption indicator with the cells after
adoption masked out; the unit intercept $u_i \sim \mathcal{N}(0, \sigma_u^2)$
is a frailty.

**Composition.** Conditional on a frailty draw, under an offer schedule $z$
the hazard equation gives the
probability of adopting first in period $a$,
$`\pi_a(x, z) = \lambda_a \prod_{s < a}(1 - \lambda_s)`$, and of having adopted
by $t$, $P_t(x, z) = 1 - \prod_{s \le t}(1 - \lambda_s)$; the outcome equation
gives the continuous response $\tau_t(x,s)=\beta^Y_s\nu_Y(x,t)$ to having adopted $s$
periods ago. The offer's effects are

```math
\mathrm{CITT}_Y(x, t) = \sum_{a \le t} \big[\pi_a(x, 1) - \pi_a(x, 0)\big]\, \tau_t(x,\, t - a + 1),
\qquad
\mathrm{CITT}_D(x, t) = P_t(x, 1) - P_t(x, 0),
```

For binary outcomes, substitute the corresponding probability contrast for
$\tau_t$. The two prediction interfaces use different target populations:

- **`predict()`** evaluates every original study unit under "offered from
  launch" and "never offered", using its own fitted intercepts. It averages
  over all study units, or prespecified baseline groups, with equal weights
  unless fixed weights are supplied. Its `reference` reports assigned-arm
  mean differences and normal-approximation Fieller/Anderson–Rubin-style ratio
  confidence sets. Randomization identifies the reference offer effects
  without the model's adoption-ignorability or exclusion assumptions.
  The posterior Wald draws are the **unstabilized** ratio $ITT_Y/ITT_D$;
  negative first stages are retained. `table` keeps summaries with diagnostic
  flags, whereas `wald()` withholds ratio summaries when its diagnostic gate
  fails. Ratio means are not reported, and zero denominators remain undefined.
- **`predict_conditional()`** describes a unit drawn from the modeled
  population at a specified covariate profile, including when that profile
  belongs to a study unit. It integrates the adoption frailty *outside* the
  survival product using Gauss–Hermite quadrature. For a binary outcome it
  also integrates the outcome intercept. By default, `monotonic_first_stage=True`
  clips negative adoption contrasts to zero in postprocessing; it does not
  constrain the fitted hazard model to be monotone. The field called `cace`
  is the **stabilized** draw-wise ratio
  $`\mathrm{CITT}_Y/(\max(\mathrm{CITT}_D,0)+0.02)`$ by default. The positive
  `cace_stabilization` parameter controls this denominator adjustment, which
  changes the estimand and must not be presented as the ordinary Wald ratio.
  With `monotonic_first_stage=False`, the clipping is disabled.

A complier-effect interpretation requires exclusion, relevance, monotonicity,
and appropriate assumptions about adoption histories. An offer can accelerate
adoption among people who would eventually adopt either way, so a dynamic Wald
ratio is not automatically a CACE. Under valid horizon-specific monotonicity,
principal-stratum shares are $P_t(x,0)$ (always-takers),
$P_t(x,1)-P_t(x,0)$ (compliers), and $1-P_t(x,1)$ (never-takers). Clipped
summaries alone do not establish that assumption. Conditional predictions also
provide cumulative lift, breakeven probabilities, and a common-cost allocation
rule under capacity/budget constraints. Costs must use the same units as the
cumulative outcome effect.

### Python

```python
from longbet import LongBetEncourage

fit = LongBetEncourage(num_chains=4, num_burnin=2000, num_sweeps=1000)
fit.fit(y, d, z, x, t=t)                                  # y (N, T); d, z absorbing binary (N, T); x (N, P)

pred = fit.predict(groups=segment)                        # study units, their own intercepts
print(pred.table)                                         # posterior ITT_Y, ITT_D, Wald ratio; diagnostics and support
print(pred.reference)                                     # design-based estimates and confidence sets
print(pred.stability())                                   # ESS and R-hat per quantity, group and week

cond = fit.predict_conditional(x=new_cohort)              # population-marginal, any covariate profiles
cond.citt_y.mean, cond.citt_d.mean, cond.cace.median      # (N_new, horizons); draws in .draws
cond.principal_strata()                                   # compliers, always-takers, never-takers by horizon
cond.breakeven_probability(cost=20.0)                     # Pr(cumulative lift > cost) per unit
cond.knapsack_policy(cost=20.0, capacity=80, budget=1600, hurdle=0.5)   # boolean allocation

fit.outcome_model, fit.adoption_model                     # the two LongBet fits
fit.save("encouragement.npz"); fit = LongBetEncourage.load("encouragement.npz")
```

### R

```r
fit <- longbet_encourage(y, d, z, x, t = t,
                         config = list(num_chains = 4, num_burnin = 2000, num_sweeps = 1000))
pred <- predict(fit, groups = segment)
pred$effects; pred$reference; pred$diagnostics; encouragement_wald(pred)

cond <- predict_conditional(fit, new_x = new_cohort, cost = 20, capacity = 80, budget = 1600, hurdle = 0.5)
cond$citt_y$mean; cond$citt_d$mean; cond$cace$median
cond$strata; cond$breakeven_prob; cond$decision
principal_strata(cond); knapsack_policy(cond, cost = 20, capacity = 80)
saveRDS(fit, "encouragement.rds"); fit <- readRDS("encouragement.rds")
```

## 4. Diagnostics and configuration

Defaults are starting settings, not a guarantee of convergence, causal
identification, or interval coverage. The scalar diagnostic defaults require
rank-normalized split $\widehat R\le1.01$ and both bulk and tail ESS of at least
$400$ at every reported exposure. `ess_ok` and `rhat_ok` are threshold checks;
the scalar `reliable` field is deliberately `None` and its verdict is withheld.
The looser $`\widehat R>1.05`$ or bulk ESS below $100$ criteria identify warnings;
they are not the software's default acceptance thresholds.

### Inspect support and posterior draws

1. Run `diag = pred.stability()` (`att_stability(pred)` in R) and inspect
   `diag.summary` and `diag.by_exposure`. Check the reported effect, not only
   internal factors whose scale is ambiguous. Multi-outcome predictions return
   diagnostics separately for each outcome; encouragement predictions report
   them for each group, horizon, ITT, and Wald ratio.
2. Inspect `n_treated` in the diagnostics and `mean_concurrent_controls` and
   `pct_zero_control_cells` from `pred.att()`/`get_att(pred)`. A small treated
   count means little information; zero concurrent controls means that calendar
   comparison relies on extrapolation. These counts do not establish overlap
   at a covariate profile. Inspect cohort and covariate support separately.
3. Report poor chain agreement even when it occurs only at long exposures.
   Use a substantively defined target population and supported horizon; do not
   silently remove exposures just to obtain passing diagnostics. Credible
   intervals need not widen enough to reveal unsupported extrapolation.
4. Compare multiple chains and any changed model/budget transparently. Longer
   runs or tempering can help, but neither guarantees exploration of all modes.
   Passing ATT diagnostics does not automatically validate subgroup contrasts,
   joint events, or decision probabilities; inspect their draws too.

### Options to evaluate for the scientific setting

These names are Python `LongBetConfig` fields unless a prediction argument is
specified. The R scalar and multi-outcome wrappers expose a subset: they call
the prognostic calendar switch `split_time_ps`, expose `trend_num_features`,
and currently do not expose the trend degree, trend prior scales, horseshoe,
or inter-ensemble switch. The R encouragement wrapper accepts those Python
configuration fields through its `config` list.

| Setting | Option to evaluate | What changes |
|---|---|---|
| Smooth differential trends | Start with the default degree-2 block and `trend_num_features=64` | Fits nonlinear covariate loadings within a finite trend basis, with unrestricted common-period effects. Check whether that restriction is appropriate. |
| Short pre-treatment histories | `trend_degree=1` | Restricts differential trends to linear paths; it can reduce flexibility but does not restore missing identifying information. |
| Sparse trend interactions | `trend_horseshoe=True` | Replaces the interaction ridge prior with a global–local horseshoe. Its global prior scale is derived from `trend_prior_scale` and the interaction count. Common-period effects remain Gaussian. |
| More than 24 baseline columns | Put the covariates intended to explain differential trends first | The trend feature map uses at most the first 24 columns. Horseshoe shrinkage does not remove this cap; the forests still see all supplied covariates. |
| More complex smooth differential trends | `trend_degree=3` | Adds a cubic direction constructed by discrete orthogonalization on the period grid. Assess the added flexibility against the available pre-treatment history. |
| Covariate-specific discontinuous time shocks | An appropriate indicator in `x_tv`, or `split_calendar_mu=True` | Lets the prognostic forest represent a discontinuity. Adjustment must be causally valid; allowing calendar splits can increase overlap between model components and impede mixing. |
| Less smooth exposure trajectories | Refit with `kernel_type="matern32"` or `"ar1"`, and examine `lambda_knl` | Changes the GP prior. The default is squared exponential with `lambda_knl=1.0`; no kernel choice guarantees recovery of abrupt effects. |
| Different exposure shapes across covariate profiles | `split_exposure_trt=True` | Lets the treatment forest also depend on exposure, relaxing the default factorization while introducing additional dependence between the forest and GP. |
| Forecasts beyond observed exposure | Vary `sig_knl` and `lambda_knl` in `model.predict(...)` | Changes only the conditional GP projection, holding fitted posterior draws fixed. This is not a refit under a new prior. `cache_forest_evaluations=True` can reuse evaluations at a memory cost. |
| No persistent unit level beyond recorded covariates | `random_intercept=False` | Imposes a stronger model restriction. It does not turn repeated cross-sections with changing row identities into a longitudinal panel. |
| Poor mixing | Increase `num_burnin`/`num_sweeps`; optionally evaluate `tempering_levels>1` | Changes the computation rather than proving that the model is adequate. Report remaining failures and sensitivity of the estimates. |
