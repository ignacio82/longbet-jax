# LongBet

Bayesian causal forests for panel data, in JAX, with a Python front door and an R
front door that call the same engine. Three models are supported.

## 1. `LongBet`: time-varying heterogeneous treatment effects in a panel

### The model

For unit $i$ in period $t$ with exposure $S_{it}$, the number of periods since
the unit's (absorbing) treatment started and $0$ before it,

$$
Y_{it} = \mu(X_i, X^{\mathrm{tv}}_{it}) + \eta_{it}
       + \beta_{S_{it}}\, \nu(X_i, t, X^{\mathrm{trt,tv}}_{it})\, \mathbf{1}\{S_{it} \ge 1\}
       + \gamma_i + \epsilon_{it},
\qquad \epsilon_{it} \sim \mathcal{N}(0, \sigma^2).
$$

- **Prognostic and treatment forests ($\mu, \nu$).** $\mu$ is a prognostic sum of
  trees over baseline covariates $X_i$ and optional time-varying covariates
  $X^{\mathrm{tv}}_{it}$; $\nu$ is a treatment sum of trees over baseline
  covariates $X_i$ (or a separate moderation matrix `x_trt`), calendar time $t$
  (`split_calendar_trt = True`), and optional time-varying covariates
  $X^{\mathrm{trt,tv}}_{it}$. Each tree has the BART
  prior: a node at depth $d$ splits with probability $\alpha/(1+d)^{\beta}$
  ($0.95/(1+d)^2$ prognostic, $0.25/(1+d)^3$ treatment) and leaf values are
  $\mathcal{N}(0, \tau^2)$ with $\tau = 3/(2\sqrt{m_\mu})$ and
  $1/\sqrt{m_\nu}$ for $m$ trees on the standardized response.
- **Conjugate calendar-time block ($\eta_{it}$, `use_trend_block = True`).**
  Within every adoption cohort $g_i$, calendar time $t$, unit identity $i$, and
  exposure duration $S_{it} = t - g_i + 1$ are exactly collinear. Letting the
  prognostic BART forest split on $t$ forces local grow/prune steps to negotiate
  that ridge against $(\gamma_i, \beta_S)$, trapping chains in distinct step-function
  partitions. By default (`split_calendar_mu = False`, `split_calendar_trt = True`),
  the prognostic forest omits $t$ while smooth covariate-dependent calendar
  trends and common period shocks are carried by the conjugate basis block
  $$
  \eta_{it} = \Phi^{\mathrm{time}}_t c_0 + \sum_{k=1}^{d} \phi_k(t)\, b(X_i)^\top c_k,
  $$
  where $\Phi^{\mathrm{time}} \in \mathbb{R}^{T \times (T-1)}$
  (`trend_period_effects = True`, prior scale `trend_period_scale = 2.0`) is an
  orthonormal zero-sum Discrete Cosine Transform (DCT-II) period basis,
  $\phi(t) \in \mathbb{R}^d$ ($d = 2$, `trend_degree = 2`) is an orthonormal
  polynomial basis in normalized calendar time, and
  $b(X_i) \in \mathbb{R}^{p + F}$ combines standardized linear covariates and
  $F = 64$ column-centered random Fourier features (`trend_num_features = 64`,
  total unit-trend prior scale `trend_prior_scale = 2.0` so each of the
  $d(p + F)$ interaction coefficients has prior standard deviation
  $\sigma_c / \sqrt{d(p + F)}$, with optional regularized horseshoe via
  `trend_horseshoe = True`). Crucially, the coefficient vector $c = (c_0, c_1, \dots, c_d)$
  is drawn **jointly in a single conjugate Gaussian block Gibbs step** with the
  unit intercepts $\gamma_i$ and the exposure trajectory $\beta_S$, eliminating
  the unit--period--exposure collinearity in one orthogonal projection.
- **Exposure trajectory ($\beta_S$).** $\beta_S$ is a Gaussian process over the
  exposure clock,
  $\beta \sim \mathcal{N}\!\big(0,\, K + \sigma_m^2 \mathbf{1}\mathbf{1}^\top\big)$,
  with a squared-exponential (default), Matérn-3/2, Matérn-5/2 or AR(1) kernel
  $K$ of marginal variance $\sigma_k^2$ and lengthscale $\lambda$, and a constant
  mean marginalized into the kernel so projections beyond the fitted horizon
  revert to the estimated common level. By default the treatment forest does
  not split on $S$ (`split_exposure_trt = False`): the trajectory carries the
  whole exposure profile, which keeps the product $\beta_S\nu$ identified.
- **Random intercepts and variances.** $\gamma_i \sim \mathcal{N}(0, \sigma_\gamma^2)$
  is a unit random intercept (`random_intercept = True`),
  $\sigma^2 \sim \mathrm{IG}(2, 1)$ on the standardized scale and
  $\sigma_\gamma^2 \sim \mathrm{IG}(1, 0.1)$.
- **Binary and ordinal outcomes.** Binary outcomes:
  $Y_{it} = \mathbf{1}\{Y^*_{it} > 0\}$ with the same mean surface (including
  $\eta_{it}$) for the latent $Y^*_{it}$ and unit innovation variance (probit).
  Ordinal outcomes with $K$ categories:
  $Y_{it} = k \iff c_{k-1} < Y^*_{it} \le c_k$ with ordered cutpoints under an
  ordered-normal prior, updated by a marginalized Metropolis move.

The reported effect is the contrast between being $S$ periods into treatment
and not being treated at all, $\tau_t(X_i, S) = \beta_S\,\nu(X_i)$ (since
$\mu(X_i) + \eta_{it} + \gamma_i$ cancels between treated and untreated potential
outcomes), and the ATT by exposure averages it over the treated cells at that
exposure. Missing cells (`NaN`) are marginalized, not imputed. Every sweep
updates both forests with GROW, PRUNE, CHANGE (a new rule at a leaf parent, or
at any internal node with its subtree kept) and one data-driven REGROW proposal,
then draws $(\gamma, \beta, c)$ jointly via the bordered Woodbury solver and
updates the variances, followed by a Metropolis move along the exact scale ridge
between $\beta$ and the treatment leaves, and a two-stage conjugate Gaussian
subspace Gibbs move along the inter-ensemble ridge between $(\mu, \gamma)$ and
$(\beta, \nu)$ (`use_inter_ensemble_move = True`). Defaults: 4 chains, 2,000
burn-in and 1,000 retained sweeps (`n_skip = 1`), 20 prognostic and 60 treatment
trees of depth at most 8.

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
model.fit(y=y, x=x, z=z, t=t)                          # outcome="binary" / "ordinal" for other likelihoods

pred = model.predict(x=x, z=z, t=t, summary_only=True)  # any counterfactual schedule z may be passed
att = get_att(pred)                                     # att, intervals, exposure
diag = pred.stability()                                 # bulk/tail ESS, rank R-hat, MCSE, and control support per exposure
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

Outcomes $m = 1, \dots, M$ share the panel, the covariates, the treatment
schedule, and the calendar-time basis $\Phi$. Each has its own forests,
calendar-time trend coefficients $c^{(m)}$, exposure trajectory $\beta^{(m)}$,
and unit intercept $\gamma^{(m)}$, and the within-period innovations are coupled
through a triangular seemingly unrelated regression (SUR):

$$
Y^{(m)}_{it} = \mu_m(X_i) + \eta^{(m)}_{it}
             + \beta^{(m)}_{S_{it}}\, \nu_m(X_i)\, \mathbf{1}\{S_{it} \ge 1\}
             + \gamma^{(m)}_i + \sum_{j < m} \Gamma_{mj}\, \tilde R^{(j)}_{it} + \epsilon^{(m)}_{it},
\qquad \epsilon^{(m)}_{it} \sim \mathcal{N}(0, \sigma_m^2),
$$

where $\tilde R^{(j)}_{it}$ is outcome $j$'s response minus its own mean surface
(its latent response for a binary or ordinal outcome). Binary and ordinal
outcomes are placed first with zero incoming loadings and unit latent variance;
the loadings $\Gamma_{mj}$ have a $\mathcal{N}(0, \sigma_\Gamma^2)$ prior. Every
mean, latent, trend block $c^{(m)}$ and coefficient update conditions on the
full SUR precision, so later outcomes inform earlier means. Draws are aligned
across outcomes, which makes joint event probabilities such as
$\Pr(\tau^{(1)} > 0,\ \tau^{(2)} < 0)$ computable from the same posterior;
effect draws for binary outcomes are probability differences
$\Phi(\mu_0 + \tau) - \Phi(\mu_0)$.

### Python

```python
from longbet import LongBetMulti, LongBetConfig, effect_draws, joint_prob, outcome_correlation

y = {"revenue": revenue_panel, "churn": churn_panel}     # (N, T) arrays, or a list, or an (N, T, M) array
model = LongBetMulti(LongBetConfig(num_chains=4, num_burnin=2000, num_sweeps=1000))
model.fit(y=y, x=x, z=z, t=t, outcome={"revenue": "continuous", "churn": "binary"})

pred = model.predict(x=x, z=z, t=t)                      # LongBetMultiPrediction; pred["revenue"] is a LongBetPrediction
tau_revenue = effect_draws(pred, "revenue")              # (N, T, draws)
tau_churn = effect_draws(pred, "churn")                  # probability differences
p_win_win = joint_prob(pred, {"revenue": lambda e: e > 0, "churn": lambda e: e < 0})   # (N, T)
R = outcome_correlation(model)                           # posterior mean innovation correlation, (M, M)
model["revenue"]                                         # the outcome's own LongBet fit
model.save("multi.npz"); model = LongBetMulti.load("multi.npz")
```

### R

```r
fit <- longbet_multi(y = list(revenue = revenue_mat, churn = churn_mat), x = x, z = z, t = 1:Tn,
                     outcome = c(revenue = "continuous", churn = "binary"),
                     num_chains = 4, num_burnin = 2000, num_sweeps = 1000)
pred <- predict(fit, x = x, z = z, t = 1:Tn)
eff_revenue <- effect_draws(pred, outcome = "revenue")
eff_churn   <- effect_draws(pred, outcome = "churn")
p_win_win   <- joint_prob(pred, list(revenue = function(e) e > 0, churn = function(e) e < 0))
R <- outcome_correlation(fit)
fit["revenue"]                                            # the outcome's own longbet fit
saveRDS(fit, "multi.rds"); fit <- readRDS("multi.rds")
```

## 3. `LongBetEncourage`: a randomized offer with absorbing adoption

### The model

A single randomized encouragement wave $Z_{it}$ (offered from a launch period
onward, permanent control arm) and an absorbing adoption indicator $D_{it}$ are
observed. Two LongBet equations (both equipped with the conjugate calendar-time
block $\eta_{it}$) are fitted and composed.

**Outcome on the adoption clock.** With $S^D_{it}$ the periods since unit $i$
adopted (0 before adoption),

$$
Y_{it} = \mu_Y(X_i) + \eta^Y_{it} + \beta^Y_{S^D_{it}}\, \nu_Y(X_i)\, \mathbf{1}\{S^D_{it} \ge 1\}
       + \gamma^Y_i + \epsilon^Y_{it}.
$$

This is the model of Section 1 with adoption as the treatment; both arms
contribute adoption events, and the equation contains the jump that adoption
causes, its exposure profile and its covariate heterogeneity. A binary outcome
uses the probit form.

**Adoption as a discrete-time hazard on the offer clock.** With $S^Z_{it}$ the
periods since the offer (0 for the control arm), the hazard of first adoption
on the at-risk set $\{D_{i,t-1} = 0\}$ is

$$
\lambda_{it} = \Pr(D_{it} = 1 \mid D_{i,t-1} = 0)
= \Phi\!\Big(\mu_D(X_i) + \eta^D_{it} + \beta^D_{S^Z_{it}}\, \nu_D(X_i)\, \mathbf{1}\{S^Z_{it} \ge 1\} + \eta_i\Big),
$$

a binary LongBet fitted to the first-adoption indicator with the cells after
adoption masked out; the unit intercept $\eta_i \sim \mathcal{N}(0, \sigma_\eta^2)$
is a frailty.

**Composition.** Under an offer schedule $z$, the hazard equation gives the
probability of adopting first in period $a$,
$\pi_a(x, z) = \lambda_a \prod_{s < a}(1 - \lambda_s)$, and of having adopted
by $t$, $P_t(x, z) = 1 - \prod_{s \le t}(1 - \lambda_s)$; the outcome equation
gives the response $\tau_t(x, s) = \beta^Y_s \nu_Y(x)$ to having adopted $s$
periods ago. The offer's effects are

$$
\mathrm{CITT}_Y(x, t) = \sum_{a \le t} \big[\pi_a(x, 1) - \pi_a(x, 0)\big]\, \tau_t(x,\, t - a + 1),
\qquad
\mathrm{CITT}_D(x, t) = P_t(x, 1) - P_t(x, 0),
$$

with the frailty integrated outside the survival product by Gauss-Hermite
quadrature (it makes survival positively dependent across periods), and for a
binary outcome the response on the probability scale with the outcome
intercept integrated the same way. Principal strata follow under monotonicity:
always-takers $P_t(x, 0)$, compliers $\mathrm{CITT}_D$, never-takers
$1 - P_t(x, 1)$; the complier effect is the draw-wise ratio
$\mathrm{CITT}_Y / (\mathrm{CITT}_D + 0.02)$.

`predict()` evaluates every study unit under "offered from the launch period"
and "never offered" with its own fitted intercepts and averages within
prespecified groups, the finite study population that the model-free
design-based reference (assigned-arm mean differences with Fieller and
Anderson-Rubin-style ratio sets) also describes; it reports both, with bulk and
tail ESS and rank $\hat R$ on every reported quantity. `predict_conditional()`
describes a unit with a given covariate profile drawn from the fitted population
(both intercepts integrated out), for study units and new cohorts alike, and
carries the decision quantities: cumulative lift, breakeven probability, and a
knapsack allocation under cost, capacity and budget.

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

## 4. Practitioner's Guide & Diagnostic Workflow for Real Data

The shipped defaults (`use_trend_block = True`, `trend_degree = 2`,
`trend_num_features = 64`, `trend_prior_scale = 2.0`, `trend_period_effects = True`,
`trend_period_scale = 2.0`, `trend_horseshoe = False`, `split_calendar_mu = False`,
`split_calendar_trt = True`, `split_exposure_trt = False`,
`use_inter_ensemble_move = True`) are calibrated so that standard staggered-adoption
panels mix reliably ($\widehat{R}_{\max} \le 1.05$, $\mathrm{ESS}_{\min} > 200$)
without manual tuning—even when untreated trends diverge nonlinearly across
covariates.

### Step 1: Always run `pred.stability()` (`att_stability(pred)` in R)

After fitting on a real dataset, inspect both the summary and per-exposure rows
of `diag = pred.stability()`:

1. **Check concurrent control and cohort support first (`n_treated`, `mean_concurrent_controls`, `pct_zero_control_cells`).**
   In staggered adoption panels, the longest exposures $s$ are observed only for
   the earliest-adopting cohort in the final periods of the panel. If
   `n_treated[s - 1]` is very small (e.g. $< 15$ units) or
   `pct_zero_control_cells` is near $1.0$ (almost all units are already treated
   in those calendar periods), any counterfactual comparison at exposure $s$
   relies on extrapolation. If $\widehat{R}_s \le 1.05$ on well-supported early
   exposures ($s = 1, \dots, S^*$) and only degrades at the tail where
   `n_treated` is tiny, **do not enable prognostic tree splits on time**—simply
   report exposures up to the supported horizon $S^*$ (or inspect the credible
   intervals, which widen automatically to reflect extrapolation uncertainty).
2. **Check global chain agreement (`rhat_max`, `ess_min`).**
   If $\widehat{R}_s > 1.05$ or $\mathrm{ESS}_s < 100$ even at well-supported
   exposures ($s = 1, 2, 3$ with ample `n_treated` and `mean_concurrent_controls`),
   use the decision table below to adjust the model or sampler configuration.

### Step 2: When to keep the defaults vs. adjust options

| Empirical setting / diagnostic symptom | Recommended configuration | Why |
|---|---|---|
| **Standard staggered panel** ($N \ge 100$, $T \in [5, 24]$, $p \le 30$) with parallel or diverging trends | **Keep all defaults** | The DCT period basis + degree-2 polynomial $\times$ 64 RFF block absorbs common period shocks and smooth $X_i \times t$ confounding while keeping $\mu$ free of calendar-time splits. |
| **Ultra-short panel** ($T \le 4$) | `trend_degree=1` | Uses linear unit time slopes $\phi_1(t)\,b(X_i)$ when $T \le 4$ offers too few pre-treatment periods per cohort to separate quadratic curvature from the exposure curve. |
| **High-dimensional baseline covariates** ($p > 30$) where only a few covariates drive differential trends | `trend_horseshoe=True` (optionally `trend_num_features=32`) | Replaces the isotropic ridge prior on the $(p + F)d$ trend interactions with a regularized Carvalho--Polson--Scott horseshoe prior (`trend_horseshoe_tau0=0.1`), shrinking irrelevant covariate-by-time slopes toward zero. |
| **Long panels ($T \ge 25$)** with multi-inflection unit trends | `trend_degree=3` | Adds cubic Legendre time polynomials $\phi_3(t)\,b(X_i)$. Avoid `trend_degree=3` on short panels ($T \le 12$), where cubic cohort-by-covariate slopes are weakly identified. |
| **Sharp discontinuous calendar regime shifts** that affect only a subset of covariate profiles (e.g. a policy shock hitting one industry in period $t_0$) | Pass a time-varying indicator in `x_tv` (preferred), or set `split_calendar_mu=True` | Passing known regime indicators in `x_tv` lets $\mu$ split on them without opening the full calendar clock $t$. Only set `split_calendar_mu=True` if the timing of the interaction shock is unknown, and monitor `pred.stability()`. |
| **Abrupt or step-change treatment dynamics** over exposure $S$ (rather than smooth build-up/decay) | `kernel_type="matern32"` (or `"ar1"`), `lambda_knl=1.0` | The default squared-exponential GP (`lambda_knl=2.0`) favors smooth trajectories; Matérn-3/2 or AR(1) with a shorter lengthscale allows sharper kinks between $S=1$ and $S=2$. |
| **Long-horizon forecasting** beyond observed exposure ($S_{\mathrm{pred}} > S_{\max}$) | Sweep `sig_knl` and `lambda_knl` in `model.predict(..., cache_forest_evaluations=True)` | Audits how much out-of-window forecasts depend on the GP extrapolation prior while holding the in-sample posterior draws fixed and avoiding refitting. |
| **No unit-level serial correlation** (or repeated cross-sections where row index $i$ is not a persistent unit) | `random_intercept=False` | Drops $\gamma_i$ when units are observed only once or residual correlation is negligible. |
| **Residual forest multimodality** ($\widehat{R} > 1.05$ across early exposures $s=1,2$) | `num_burnin=3000, num_sweeps=1500` or `tempering_levels=4` | Increases the MCMC budget or couples a 4-rung replica-exchange ladder per chain to bridge separated BART partitions. |
