# LongBet

Bayesian causal forests for panel data, in JAX, with a Python front door and an R
front door that call the same engine. Three models are supported.

## 1. `LongBet`: time-varying heterogeneous treatment effects in a panel

### The model

For unit $i$ in period $t$ with exposure $S_{it}$, the number of periods since
the unit's (absorbing) treatment started and $0$ before it,

$$
Y_{it} = \mu(X_i, t, X^{\mathrm{tv}}_{it})
       + \beta_{S_{it}}\, \nu(X_i, t, X^{\mathrm{trt,tv}}_{it})\, \mathbf{1}\{S_{it} \ge 1\}
       + \gamma_i + \epsilon_{it},
\qquad \epsilon_{it} \sim \mathcal{N}(0, \sigma^2).
$$

- $\mu$ is a prognostic sum of trees over baseline covariates, calendar time and
  optional time-varying covariates; $\nu$ is a treatment sum of trees over
  baseline covariates, optional time-varying covariates and, if allowed,
  calendar time. Each tree has the BART prior: a node at depth $d$ splits with
  probability $\alpha/(1+d)^{\beta}$ ($0.95/(1+d)^2$ prognostic, $0.25/(1+d)^3$
  treatment) and leaf values are $\mathcal{N}(0, \tau^2)$ with
  $\tau = 3/(2\sqrt{m_\mu})$ and $3/(3\sqrt{m_\nu})$ for $m$ trees on the
  standardized response.
- $\beta_S$ is a Gaussian process over the exposure clock,
  $\beta \sim \mathcal{N}\!\big(0,\, K + \sigma_m^2 \mathbf{1}\mathbf{1}^\top\big)$,
  with a squared-exponential (default), Matérn-3/2, Matérn-5/2 or AR(1) kernel
  $K$ of marginal variance $\sigma_k^2$ and lengthscale $\lambda$, and a constant
  mean marginalized into the kernel so projections beyond the fitted horizon
  revert to the estimated common level. By default the treatment forest does
  not split on $S$: the trajectory carries the whole exposure profile, which
  keeps the product $\beta_S\nu$ identified. `split_exposure_trt = TRUE` lets
  it split on $S$ as well, for panels whose units need genuinely different
  shapes over exposure; read the diagnostics, because that freedom is a ridge
  between the trajectory and the forest.
- $\gamma_i \sim \mathcal{N}(0, \sigma_\gamma^2)$ is a unit random intercept,
  $\sigma^2 \sim \mathrm{IG}(2, 1)$ on the standardized scale and
  $\sigma_\gamma^2 \sim \mathrm{IG}(1, 0.1)$.
- Binary outcomes: $Y_{it} = \mathbf{1}\{Y^*_{it} > 0\}$ with the same mean for
  the latent $Y^*_{it}$ and unit innovation variance (probit). Ordinal outcomes
  with $K$ categories: $Y_{it} = k \iff c_{k-1} < Y^*_{it} \le c_k$ with ordered
  cutpoints under an ordered-normal prior, updated by a marginalized
  Metropolis move.

The reported effect is the contrast between being $S$ periods into treatment
and not being treated at all, $\tau_t(X_i, S) = \beta_S\,\nu(X_i, t)$, and the
ATT by exposure averages it over the treated cells at that exposure. Missing
cells (`NaN`) are marginalized, not imputed. Every sweep updates both forests
with GROW, PRUNE, CHANGE (a new rule at a leaf parent, or at any internal
node with its subtree kept) and one data-driven REGROW proposal, then $\beta$,
$\gamma$ and the variances, plus a Metropolis move along the exact scale ridge
between $\beta$ and the treatment leaves. Chains start from the prior, so
$\hat R$ measures convergence; `tempering_levels = K` runs each chain as a
parallel-tempering ladder for posteriors whose forest modes the local moves
connect too slowly. Defaults: 4 chains, 2,000 burn-in and 1,000 retained
sweeps, 20 prognostic and 60 treatment trees of depth at most 8.

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

model = LongBet(LongBetConfig(num_chains=4, num_burnin=2000, num_sweeps=1000))
model.fit(y=y, x=x, z=z, t=t)                          # outcome="binary" / "ordinal" for other likelihoods

pred = model.predict(x=x, z=z, t=t, summary_only=True)  # any counterfactual schedule z may be passed
att = get_att(pred)                                     # att, intervals, exposure
diag = pred.stability()                                 # bulk/tail ESS, rank R-hat, MCSE per exposure
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

Outcomes $m = 1, \dots, M$ share the panel, the covariates and the treatment
schedule. Each has its own forests, trajectory and unit intercept, and the
within-period innovations are coupled through a triangular seemingly unrelated
regression (SUR):

$$
Y^{(m)}_{it} = \mu_m(X_i, t) + \beta^{(m)}_{S_{it}}\, \nu_m(X_i, t)\, \mathbf{1}\{S_{it} \ge 1\}
             + \gamma^{(m)}_i + \sum_{j < m} \Gamma_{mj}\, \tilde R^{(j)}_{it} + \epsilon^{(m)}_{it},
\qquad \epsilon^{(m)}_{it} \sim \mathcal{N}(0, \sigma_m^2),
$$

where $\tilde R^{(j)}_{it}$ is outcome $j$'s response minus its own mean surface
(its latent response for a binary or ordinal outcome). Binary and ordinal
outcomes are placed first with zero incoming loadings and unit latent variance;
the loadings $\Gamma_{mj}$ have a $\mathcal{N}(0, \sigma_\Gamma^2)$ prior. Every
mean, latent and coefficient update conditions on the full SUR precision, so
later outcomes inform earlier means. Draws are aligned across outcomes, which is
what makes joint event probabilities such as $\Pr(\tau^{(1)} > 0,\ \tau^{(2)} < 0)$
computable from the same posterior; effect draws for binary outcomes are
probability differences $\Phi(\mu_0 + \tau) - \Phi(\mu_0)$.

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
observed. Two LongBet equations are fitted and composed.

**Outcome on the adoption clock.** With $S^D_{it}$ the periods since unit $i$
adopted (0 before adoption),

$$
Y_{it} = \mu_Y(X_i, t) + \beta^Y_{S^D_{it}}\, \nu_Y(X_i, t)\, \mathbf{1}\{S^D_{it} \ge 1\}
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
= \Phi\!\Big(\mu_D(X_i, t) + \beta^D_{S^Z_{it}}\, \nu_D(X_i, t)\, \mathbf{1}\{S^Z_{it} \ge 1\} + \eta_i\Big),
$$

a binary LongBet fitted to the first-adoption indicator with the cells after
adoption masked out; the unit intercept $\eta_i \sim \mathcal{N}(0, \sigma_\eta^2)$
is a frailty.

**Composition.** Under an offer schedule $z$, the hazard equation gives the
probability of adopting first in period $a$,
$\pi_a(x, z) = \lambda_a \prod_{s < a}(1 - \lambda_s)$, and of having adopted
by $t$, $P_t(x, z) = 1 - \prod_{s \le t}(1 - \lambda_s)$; the outcome equation
gives the response $\tau_t(x, s) = \beta^Y_s \nu_Y(x, t)$ to having adopted $s$
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
knapsack allocation under cost, capacity and budget. The model uses the
exclusion restriction and assumes that, given covariates and the unit
intercept, adoption timing is not confounded with the outcome innovations; the
design-based reference needs neither and is the check on the population offer
effects.

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
