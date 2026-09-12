# Count data LongBet: implementation specification

Status: proposed implementation; this is a design and specification document
aligned with the architecture established in `ordinal.md` and `README.md`.
Reviewed against the repository on 2026-09-12. Implement the phases in order and
use the acceptance criteria in section 9 to determine completion.

---

## 1. Objective and design decisions

Add native support for count outcomes ($Y_{it} \in \{0, 1, 2, \dots\}$) under
staggered adoption panels. Count outcomes arise routinely in intervention
analysis—such as customer orders, website conversions, healthcare utilization,
adverse events, and crime reports. These data are characterized by
non-negativity, heteroskedasticity (variance scaling with the mean),
overdispersion, varying baseline exposure, and often substantial zero-inflation.

The goal is to estimate causal intervention effects—specifically the **Incidence
Rate Ratio (IRR)**, the **count-scale ATT** ($\Delta \mathbb{E}[Y]$), and
**extensive/intensive margins**—without resorting to ad-hoc transformations such
as $\log(Y + 1)$ or treating bounded counts as unscaled ordinal categories.

To maintain LongBet's vectorized JAX architecture, the implementation must
preserve exact conditional Gaussianity for tree split integration, the exposure
Gaussian Process $\beta_S$, unit random intercepts $\gamma_i$, and adaptive
coding $b_0, b_1$.

### Design decisions

| Issue | Required decision | Rationale |
|---|---|---|
| **Primary backend** | Implement Negative Binomial with Pólya-Gamma (PG) augmentation (`outcome="negbin"`). | Supplies an exact conditionally Gaussian pseudo-response with heteroskedastic precision weights $\omega_{it}$, preserving all conjugate normal tree and GP updates in `_step.py`. Accommodates overdispersion and includes Poisson in the limit ($r \to \infty$). |
| **Secondary backend** | Two-part Hurdle model (`outcome="hurdle"` or wrapper over `LongBetMulti`). | For heavily zero-inflated panels (e.g. >50% zeros), explicitly separates the extensive margin ($Y > 0$ via binary probit) from the intensive margin ($Y \mid Y > 0$ via continuous log-normal) with joint SUR error correlation. |
| **Exposure / offset** | Add first-class `exposure` / `offset` support in `LongBetConfig` and `_design.py`. | Count rates must scale with varying observation windows, population at risk, or active days ($E_{it}$). Fixed offset $\log(E_{it})$ is absorbed directly into the prognostic baseline. |
| **Direct Poisson BART** | Defer non-conjugate log-link tree samplers. | As derived in Section 10 of `ordinal.md`, tree leaves weighted by exposure $w_{it} = b_Z \beta_S$ cannot be integrated analytically under a Poisson likelihood, destroying `bartz`'s fast vector-tree integration. |
| **Log(Y+1) heuristic** | Explicitly reject as an internal default. | Subject to severe retransformation bias (Jensen's inequality), sign reversals, and dependence on arbitrary shift constants $c$ (Silva & Tenreyro 2006; Chen & Roth 2023). |
| **Bounded count fallback** | Document existing `outcome="ordinal"` with `att_expected_score`. | When counts are strictly bounded to small integers ($Y \le 8$), `outcome="ordinal"` is already operational and should be documented as the immediate zero-code alternative. |

---

## 2. Statistical contract

### 2.1 Notation, rate surface, and identification

Let $n = N \times T$ be total panel cells in unit-major order, $C$ parallel
Markov chains, and $D$ total retained posterior draws. Treatment adoption is
staggered and absorbing, recorded by $A_{it} \in \{0, 1\}$ (`state.z_vec`).
$S_{it} = t - E_i + 1$ denotes time elapsed since adoption ($S_{it} \ge 1$ for
treated cells, $0$ otherwise).

Each cell has an observed non-negative integer count $Y_{it} \in \{0, 1, 2, \dots\}$
and an optional positive exposure baseline $E_{it} > 0$ (defaulting to $1.0$).
Write the fixed exposure log-offset as $o_{it} = \log(E_{it})$.

The latent log-rate predictor $\eta_{it}$ is parameterized as:

$$
\eta_{it} = \alpha m_{it} + b_{A_{it}} \beta_{S_{it}} \nu(X_i, S_{it}, t, X^{\mathrm{trt,tv}}_{it}) + \gamma_i + o_{it},
$$

where:
* $m_{it} = \bar{o} + f_\mu(X_i, t, X^{\mathrm{tv}}_{it})$ is the prognostic forest fit with forest offset $\bar{o}$.
* $\alpha$ is the prognostic scale factor (optionally sampled or fixed at 1).
* $\nu(\cdot)$ is the treatment effect forest moderating the proportional impact across covariates.
* $\beta_{S_{it}}$ is the exposure GP trajectory over time-since-adoption, capturing dynamic post-treatment rate multipliers.
* $b_0, b_1$ are adaptive coding weights. Under standard identification, $b_1$ scales post-treatment lift while $b_0$ enables pre-treatment trend checks.
* $\gamma_i \sim \mathcal{N}(0, \sigma_\gamma^2)$ is the unit random intercept, serving as an unobserved baseline activity multiplier $\exp(\gamma_i)$.

### 2.2 Negative Binomial rate model

Conditioned on $\eta_{it}$ and dispersion parameter $r > 0$, $Y_{it}$ follows a
Negative Binomial distribution:

$$
Y_{it} \mid \eta_{it}, r \sim \mathrm{NB}(r, p_{it}), \qquad p_{it} = \frac{\exp(\eta_{it})}{1 + \exp(\eta_{it})} = \operatorname{expit}(\eta_{it}).
$$

Under this parameterization:
* Mean: $\lambda_{it} = \mathbb{E}[Y_{it} \mid \eta_{it}, r] = r \exp(\eta_{it}) = r E_{it} \exp(\eta^*_{it})$, where $\eta^*_{it} = \eta_{it} - o_{it}$.
* Variance: $\operatorname{Var}(Y_{it} \mid \eta_{it}, r) = \lambda_{it} + \frac{\lambda_{it}^2}{r}$.
* As $r \to \infty$, the Negative Binomial converges continuously to a $\mathrm{Poisson}(\lambda_{it})$ distribution, accommodating equi-dispersed counts as a limiting case.

### 2.3 Pólya-Gamma data augmentation

Following Polson, Scott, and Windle (2013), the Negative Binomial likelihood
contribution for cell $it$ can be written using an integral with respect to a
Pólya-Gamma density $\mathrm{PG}(Y_{it} + r, 0)$:

$$
\frac{(e^{\eta_{it}})^{Y_{it}}}{(1 + e^{\eta_{it}})^{Y_{it} + r}} = 2^{-(Y_{it} + r)} \exp\left( \kappa_{it} \eta_{it} \right) \int_0^\infty \exp\left( -\frac{\omega_{it} \eta_{it}^2}{2} \right) p(\omega_{it}) \, d\omega_{it},
$$

where:

$$
\kappa_{it} = \frac{Y_{it} - r}{2}, \qquad \omega_{it} \sim \mathrm{PG}(Y_{it} + r, 0).
$$

Conditioned on the latent variables $\omega_{it} > 0$, completing the square with
respect to $\eta_{it}$ reveals a Gaussian kernel:

$$
\exp\left( \kappa_{it} \eta_{it} - \frac{\omega_{it} \eta_{it}^2}{2} \right) \propto \exp\left( -\frac{\omega_{it}}{2} \left( \frac{\kappa_{it}}{\omega_{it}} - \eta_{it} \right)^2 \right).
$$

This defines a **conditionally Gaussian working response** $Z_{it}$ with known,
cell-specific observation precision $\omega_{it}$:

$$
Z_{it} \equiv \frac{Y_{it} - r}{2 \omega_{it}} = \eta_{it} + \varepsilon_{it}, \qquad \varepsilon_{it} \mid \omega_{it} \sim \mathcal{N}\left(0, \frac{1}{\omega_{it}}\right).
$$

#### Key architectural consequence
LongBet's existing sampler in [`_step.py`](file:///home/ignacio/longbet-jax/src/longbet/_step.py) already accepts an arbitrary vector of cell precisions `conditional_precision` (developed for SUR). Therefore, conditional on $\omega_{it}$, **every single parameter update in LongBet remains exactly conjugate Gaussian**:
* The prognostic forest $\mu$ updates via `bartz` with precision scale $\omega_{it}$.
* The treatment forest $\nu$ updates via `bartz` with precision scale $\omega_{it} w_{it}^2$.
* The exposure GP $\beta_S$ updates via `sample_beta_gp` with information accumulator $A_s = \sum_{it \in s} \omega_{it} d_{it}^2$.
* The adaptive coding weights $b_0, b_1$ update via univariate conjugate Gaussian conditionals with precisions $\sum \omega_{it} g_{it}^2 + \sigma_b^{-2}$.
* The unit intercepts $\gamma_i$ update via Gaussian conditionals with precisions $\sum_{t} \omega_{it} + \sigma_\gamma^{-2}$.

### 2.4 Dispersion parameter $r$ prior and update

Assign $r$ a proper Gamma prior:

$$
r \sim \mathrm{Gamma}(a_r, b_r), \qquad p(r) \propto r^{a_r - 1} e^{-b_r r}, \quad a_r > 0, \; b_r > 0.
$$

Defaults: $a_r = 2.0, b_r = 0.5$ (prior mean 4.0, prior variance 8.0, placing
diffuse mass over moderate overdispersion while regularizing against numerical
instability).

Because the full conditional for $r$ given $(\eta, \omega, Y)$ does not admit a
standard conjugate form, update $\psi = \log r \in (-\infty, \infty)$ via a
Metropolis-Hastings random walk step:
1. Propose $\psi^* = \psi + \sigma_\psi \cdot \xi$, where $\xi \sim \mathcal{N}(0, 1)$.
2. Compute $r^* = \exp(\psi^*)$.
3. The acceptance probability $\min(1, \exp(\Delta \log p))$ incorporates the
   summed Negative Binomial log-likelihood, the Gamma prior on $r$, and the
   Jacobian $|\frac{dr}{d\psi}| = r$:

$$
\log p(\psi \mid Y, \eta) = \sum_{it: \mathrm{obs}} \left[ \log \Gamma(Y_{it} + e^\psi) - \log \Gamma(e^\psi) + e^\psi \log(1 - p_{it}) \right] + a_r \psi - b_r e^\psi.
$$

---

## 3. Configuration, inputs, and initialization

### 3.1 Configuration in `LongBetConfig`

Extend [`LongBetConfig`](file:///home/ignacio/longbet-jax/src/longbet/_config.py):

```python
# In src/longbet/_config.py:
outcome: Literal["continuous", "binary", "ordinal", "negbin", "hurdle"] = "continuous"
dispersion_prior_shape: float = 2.0
dispersion_prior_rate: float = 0.5
dispersion_proposal_sigma: float = 0.15
sample_dispersion: bool = True
initial_dispersion: float | None = None
```

#### Validation invariants
* When `outcome="negbin"`, `y` must contain non-negative integers ($\ge 0$). Reject negative values, fractions, infinities, and all-`NaN` panels. Accept floating arrays where values equal their floor, with `NaN` denoting missingness.
* `dispersion_prior_shape` and `dispersion_prior_rate` must be strictly positive floats.
* If `initial_dispersion` is specified, require a strictly positive finite float.
* Reject `num_categories` when `outcome="negbin"` (only valid for `"ordinal"`).
* Continuous response standardization is automatically disabled (`standardize=False`, `meany=0.0`, `sdy=1.0`).

### 3.2 Exposure offsets

Add explicit `exposure` support in `LongBet.fit`:

```python
def fit(
    self,
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    t: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    ...
)
```

* `exposure` can be passed as an $(N, T)$ matrix or a flattened $(N \cdot T,)$ array matching `y`.
* All observed entries must be strictly positive ($E_{it} > 0$). Reject zeros, negatives, and nonfinite values.
* Compute $o_{it} = \log(E_{it})$ in float64. When `exposure=None`, $o_{it} = 0.0$.
* The offset array is stored in `LongBetState` and passed to prediction routines.

### 3.3 Offset calculation and initial state

Compute starting parameters in host float64 on observed cells:
1. Mean empirical rate per unit exposure:
   $$
   \bar{\lambda} = \frac{\sum_{it \in \mathrm{obs}} Y_{it}}{\sum_{it \in \mathrm{obs}} E_{it}}.
   $$
2. Empirical variance ratio for initial dispersion $r^{(0)}$:
   $$
   v = \operatorname{Var}_{\mathrm{obs}}(Y_{it} / E_{it}), \qquad r^{(0)} = \max\left(0.5, \frac{\bar{\lambda}^2}{\max(v - \bar{\lambda}, 0.1 \bar{\lambda})}\right).
   $$
   If `initial_dispersion` is set in config, override with that value.
3. Forest base offset:
   Since $\mathbb{E}[Y] = r e^\eta = r E e^{\eta^*}$, set:
   $$
   \bar{o} = \log(\bar{\lambda} / r^{(0)}).
   $$
   This centers the prognostic forest prior around the unadjusted empirical rate.
4. Starting latents:
   Set initial log-rate $\eta_{it}^{(0)} = \bar{o} + o_{it}$.
   Draw $\omega_{it}^{(0)} \sim \mathrm{PG}(Y_{it} + r^{(0)}, \eta_{it}^{(0)})$.
   Compute initial working response $Z_{it}^{(0)} = \frac{Y_{it} - r^{(0)}}{2 \omega_{it}^{(0)}}$ and residual $R^{(0)} = Z_{it}^{(0)} - \eta_{it}^{(0)}$.

### 3.4 State construction and chain overdispersion

Extend [`LongBetState`](file:///home/ignacio/longbet-jax/src/longbet/_state.py):

| Field | Single-chain representation | Chain metadata |
|---|---|---|
| `dispersion` | `Float32[Array, '']` | `chains=CHAIN_AXIS` |
| `dispersion_prior_shape` | Python float | static |
| `dispersion_prior_rate` | Python float | static |
| `dispersion_proposal_sigma`| Python float | static |
| `log_exposure` | `Float32[Array, ' n']` | shared |
| `omega` | `Float32[Array, ' n']` | `chains=CHAIN_AXIS` |
| `z` | `Float32[Array, ' n']` ($Z_{it}$ working response) | `chains=CHAIN_AXIS` |

In `_overdisperse_chains`:
* Disperse initial dispersion across chains: $r_c^{(0)} = r^{(0)} \cdot \exp(0.3 \cdot \mathcal{N}(0, 1))$.
* Compute initial chain-specific working responses $Z_{it, c}^{(0)}$ from newly sampled $\omega_{it, c}^{(0)}$.
* Set `sigma2` strictly to `1.0` (latent scale is fixed by the PG precision weights).

---

## 4. Sampling primitives and sweep integration

Create a new dedicated module: `src/longbet/_count.py`.

### 4.1 Pólya-Gamma sampling primitive

Sampling $\omega \sim \mathrm{PG}(b, c)$ with $b = Y + r$ and $c = \eta$:
Implement a vectorized, jitted JAX kernel using Devroye's alternating series
rejection sampler for $\mathrm{PG}(1, c)$ combined with Windle's gamma-sum
decomposition for general $b > 0$.

```python
# In src/longbet/_count.py:
def sample_polya_gamma(
    key: Key[Array, ''],
    b: Float32[Array, ' n'],
    c: Float32[Array, ' n'],
) -> Float32[Array, ' n']:
    """Draw Pólya-Gamma variates PG(b, c) in parallel on CPU/GPU."""
```

#### Numerical safeguards
* Float32 tail stability: When $|c| > 20$, evaluate $\tanh(c/2) / (2c)$ using
  asymptotics to prevent overflow/underflow in the tilting factor.
* Large counts ($b > 50$): Switch to the saddlepoint Gaussian approximation:
  $$
  \omega \approx \mathcal{N}\left(\frac{b}{2c} \tanh(c/2), \; \frac{b}{4 c^3} (\sinh(c) - c) \operatorname{sech}^2(c/2)\right),
  $$
  which is exact to order $O(1/b)$ and avoids loop non-convergence on high-count
  outliers.
* Zero bounding: Clip returned $\omega_{it} \ge 10^{-12}$ to prevent infinite
  variance in pseudo-residuals.

### 4.2 Dispersion Metropolis-Hastings primitive

```python
# In src/longbet/_count.py:
def update_dispersion(
    key: Key[Array, ''],
    r_current: Float32[Array, ''],
    y: Float32[Array, ' n'],
    eta: Float32[Array, ' n'],
    obs_mask: Bool[Array, ' n'],
    shape_prior: float,
    rate_prior: float,
    proposal_sigma: float,
) -> tuple[Float32[Array, ''], Bool[Array, '']]:
    """Random-walk Metropolis-Hastings move on log(r)."""
```

* Evaluate log-likelihoods in stable JAX code using `jax.lax.lgamma` and
  `jax.scipy.special.logsumexp`.
* Return the accepted $r$ along with a boolean indicator for chain acceptance
  monitoring.

### 4.3 Sweep execution in `longbet_single_step`

Integrate into [`longbet_single_step`](file:///home/ignacio/longbet-jax/src/longbet/_step.py):

```python
# Sequence for outcome="negbin":
# 1. Update latents:
#    omega ~ PG(Y + r, eta)
#    Z = (Y - r) / (2 * omega)
#    R = Z - eta
# 2. Update prognostic forest mu with cell weights omega
# 3. Update prognostic scale alpha
# 4. Update treatment forest nu with cell weights omega * (b_z * beta_S)^2
# 5. Update GP trajectory beta_S with weighted accumulators
# 6. Update adaptive coding weights b0, b1
# 7. Update unit random intercepts gamma_i with precision sum(omega) + 1/sigma_gamma^2
# 8. Update dispersion parameter r via Metropolis-Hastings
# 9. Optional ridge scale step
```

Because $\omega_{it}$ changes every sweep, pass $\omega_{it}$ as
`conditional_precision` into the existing forest, GP, and intercept updates.
`state.sigma2` remains fixed at `1.0`.

---

## 5. Prediction and estimands

### 5.1 Causal estimands for counts

For each retained posterior draw $d = 1, \dots, D$ and cell $it$:
* Counterfactual control log-rate:
  $$
  \eta_{it, 0}^{(d)} = \alpha^{(d)} m_{it}^{(d)} + b_0^{(d)} \beta_0^{(d)} \nu_0^{(d)} + \gamma_i^{(d)} + o_{it}.
  $$
* Counterfactual treated log-rate:
  $$
  \eta_{it, 1}^{(d)} = \alpha^{(d)} m_{it}^{(d)} + b_1^{(d)} \beta_{S_{it}}^{(d)} \nu_{it}^{(d)} + \gamma_i^{(d)} + o_{it}.
  $$
* Counterfactual expected count rates:
  $$
  \lambda_{it, 0}^{(d)} = r^{(d)} \exp\left(\eta_{it, 0}^{(d)}\right), \qquad \lambda_{it, 1}^{(d)} = r^{(d)} \exp\left(\eta_{it, 1}^{(d)}\right).
  $$

From these surfaces, define three distinct identified estimands:

1. **Incidence Rate Ratio (IRR)**:
   $$
   \mathrm{IRR}_{it}^{(d)} = \frac{\lambda_{it, 1}^{(d)}}{\lambda_{it, 0}^{(d)}} = \exp\left( b_1^{(d)} \beta_{S_{it}}^{(d)} \nu_{it}^{(d)} - b_0^{(d)} \beta_0^{(d)} \nu_0^{(d)} \right).
   $$
   This represents the dynamic proportional multiplier induced by the intervention. Under standard identification ($b_0 = 0$), $\mathrm{IRR}_{it} = \exp(b_1 \beta_{S_{it}} \nu_{it})$.

2. **Absolute Count Difference (ATT)**:
   For treated units at exposure horizon $s \in \{1, \dots, S_{\max}\}$:
   $$
   \Delta_{it}^{(d)} = \lambda_{it, 1}^{(d)} - \lambda_{it, 0}^{(d)} = r^{(d)} \exp\left(\eta_{it, 0}^{(d)}\right) \left[ \mathrm{IRR}_{it}^{(d)} - 1 \right].
   $$
   The horizon ATT aggregates over the treated population:
   $$
   \mathrm{ATT}_s^{(d)} = \frac{1}{|\mathcal{T}_s|} \sum_{it \in \mathcal{T}_s} \Delta_{it}^{(d)}.
   $$

3. **Relative Percentage Lift**:
   $$
   \mathrm{Lift}_s^{(d)} = \frac{\sum_{it \in \mathcal{T}_s} \lambda_{it, 1}^{(d)} - \sum_{it \in \mathcal{T}_s} \lambda_{it, 0}^{(d)}}{\sum_{it \in \mathcal{T}_s} \lambda_{it, 0}^{(d)}}.
   $$

### 5.2 Public prediction API

Extend [`LongBetPrediction`](file:///home/ignacio/longbet-jax/src/longbet/_model.py):

```python
class LongBetPrediction:
    # Existing fields...
    att_full: np.ndarray             # (S, D) - count scale ATT
    att_irr_full: np.ndarray | None  # (S, D) - Incidence Rate Ratio ATT
    rate_summary: PosteriorSummary   # (N, T) expected rates under factual regime
    rate_control_summary: PosteriorSummary # (N, T) baseline rates
    dispersion_samples: np.ndarray | None  # (D,) draws of r
```

Expose methods:
* `pred.att(alpha=0.05)`: returns absolute count ATT by exposure time.
* `pred.irr(alpha=0.05)`: returns posterior mean and credible intervals for the Incidence Rate Ratio by exposure horizon.
* `pred.rate_counterfactuals()`: returns posterior summaries of $\lambda_1$ and $\lambda_0$.

### 5.3 Bounded memory and blocked summaries

When `summary_only=True`:
* Accumulate running sums and sum-of-squares of $\Delta_{it}$ and $\log(\mathrm{IRR}_{it})$ across cell blocks.
* Never instantiate the full $N \times T \times D$ float64 array of counterfactual counts.
* Exact quantiles for the aggregate $\mathrm{ATT}_s$ and $\mathrm{IRR}_s$ remain available because `att_full` and `att_irr_full` have shape $(S_{\max}, D)$—which is compact (typically $20 \times 1000$ floats $\approx 160\text{ KB}$).

---

## 6. Two-part Hurdle model via `LongBetMulti`

For panels dominated by zeros (e.g. >50% zero cells), implement `outcome="hurdle"`
by coupling two processes:

$$
Y_{it} = D_{it} \cdot Y^*_{it}, \qquad D_{it} = \mathbb{I}(Y_{it} > 0),
$$

where:
1. **Extensive margin**: $D_{it} \in \{0, 1\}$ is modeled via LongBet's binary probit:
   $$
   D^*_{it} = \alpha_D m_{D, it} + b_{D, A} \beta_{D, S} \nu_D + \gamma_{D, i} + \varepsilon_{D, it}, \qquad D_{it} = \mathbb{I}(D^*_{it} > 0).
   $$
2. **Intensive margin**: $Y^*_{it}$ (defined on cells where $Y_{it} > 0$) is modeled
   on the log scale via LongBet's continuous Gaussian model:
   $$
   \log(Y^*_{it}) = \alpha_Y m_{Y, it} + b_{Y, A} \beta_{Y, S} \nu_Y + \gamma_{Y, i} + \varepsilon_{Y, it}.
   $$

### Triangular SUR coupling
Leverage [`LongBetMulti`](file:///home/ignacio/longbet-jax/src/longbet/_multi_model.py) with `sampler_semantics="full_precision_sur_v1"`:
* Place $D$ first and $\log Y^*$ second in the Cholesky hierarchy.
* $\varepsilon_{Y, it} = \Gamma_{21} \varepsilon_{D, it} + u_{it}$, allowing unobserved propensity to participate to correlate with volume.
* For zero cells ($D_{it} = 0$), $Y^*_{it}$ is treated as missing (`NaN`) and marginalized through the existing SUR conditional.
* The total expected count ATT recombines both margins using the log-normal smearing factor:
  $$
  \mathbb{E}[Y_{it}(a)] = \Phi\left(\eta_{D, it}(a)\right) \cdot \exp\left( \eta_{Y, it}(a) + \frac{\sigma_Y^2}{2} \right).
  $$

---

## 7. Trace storage, persistence, and R interface

### 7.1 Trace and NPZ schema (Format 4)

Extend [`_io.py`](file:///home/ignacio/longbet-jax/src/longbet/_io.py) to format version 4:
* Store `dispersion_samples` array of shape `(num_chains, num_sweeps)`.
* Store `exposure_offset` array of shape `(N, T)`.
* Include `outcome="negbin"` and `dispersion_prior` metadata in the archive JSON header.
* Rehydrate models cleanly: verify that loading an NPZ file into `LongBet.load()` restores identical predictions and ATT estimates.

### 7.2 R front door

Update `R/longbet.R`, `R/predict.longbet.R`, and `R/att.R`:
* Add `outcome = c("continuous", "binary", "ordinal", "negbin", "hurdle")`.
* Add `exposure = NULL` parameter to `longbet()`.
* Add `att_irr()` function in `R/att.R` returning a tidy data frame of horizon IRRs and credible intervals.
* Ensure rehydration via `R/rehydrate.R` preserves count metadata.

---

## 8. Implementation map and delivery order

Implement in five distinct, testable phases:

| Phase | Target files | Key deliverables | Completion gate |
|---|---|---|---|
| **1. Count primitives** | `_config.py`, new `_count.py` | Configuration validation, Pólya-Gamma sampler, dispersion MH step, offset handling. | Unit tests pass on PG distribution, MH acceptance, and input validation. |
| **2. Scalar sampling** | `_state.py`, `_step.py`, `_loop.py` | State extension, overdispersed chain initialization, sweep integration with $\omega_{it}$ precision. | Multi-chain execution passes, dispersion converges on synthetic overdispersed panels. |
| **3. Prediction & persistence** | `_model.py`, `_summary.py`, `_io.py` | Counterfactual rate calculation, IRR and count ATT, bounded memory summaries, NPZ format 4. | Full/summary parity passes; round-trip serialization preserves exact predictions. |
| **4. Hurdle / Multi integration** | `_multi_input.py`, `_multi_model.py` | Multi-outcome validation for count, two-part hurdle wrapper with joint SUR. | Joint extensive/intensive margins pass; zero-marginalization verified. |
| **5. R interface & docs** | `R/longbet.R`, `R/att.R`, `README.md`, `man/` | R wrapper parameters, `att_irr()` helper, package documentation and vignettes. | R test suite passes with exact Python parity on seeds and values. |

---

## 9. Verification and acceptance criteria

### 9.1 Fast unit and distributional tests

Add `tests/test_count_primitives.py`, `tests/test_count_model.py`, and `tests/test_count_predict.py`:

* **Input validation**: Reject negative counts, floating-point non-integers, infinite exposures, and negative offsets. Verify `NaN` missingness handling.
* **Pólya-Gamma moments**: For grid of $(b, c) \in \{1, 5, 20\} \times \{0.0, 1.5, -2.0\}$, generate $10^5$ draws and verify:
  $$
  \mathbb{E}[\omega] = \frac{b}{2c} \tanh(c/2), \qquad \operatorname{Var}(\omega) = \frac{b}{4 c^3} (\sinh(c) - c) \operatorname{sech}^2(c/2),
  $$
  within 3 Monte Carlo standard errors.
* **Dispersion MH recovery**: On fixed synthetic latents $\eta$, verify the MH sampler recovers true $r$ from Gamma prior and NB likelihood.
* **Non-negativity**: Assert all counterfactual predictions $\lambda_{0}, \lambda_{1}$ are strictly positive ($> 0$).
* **Poisson limit**: As $r \to \infty$, verify Negative Binomial sample paths converge to Poisson likelihood results.

### 9.2 Slow statistical validation and benchmark DGP

Create a reproducible benchmark script: `benchmarks/count_validation.py`.

* **DGP Setup**:
  * Panel: $N = 400$ units, $T = 12$ periods, staggered adoption across 4 waves ($t \in \{4, 6, 8, \infty\}$).
  * Covariates: $X_1, X_2 \sim \mathcal{N}(0, 1)$.
  * Baseline log-rate: $\mu(X) = 0.5 \sin(X_1) + 0.3 X_2$.
  * Unit intercepts: $\gamma_i \sim \mathcal{N}(0, 0.4^2)$.
  * Exposure offsets: $E_{it} \sim \mathrm{Uniform}(0.8, 1.5)$.
  * True treatment effect: Proportional lift $\nu(X) = 0.3 + 0.2 X_1$.
  * Exposure trajectory: AR(1) path $\beta_s = 1.0 + 0.8(\beta_{s-1} - 1.0)$, with $\beta_0 = 0$.
  * True dispersion: $r = 3.0$.
  * Response: $Y_{it} \sim \mathrm{NB}(r, p_{it})$.
* **Evaluation criteria**:
  * **IRR Bias**: Average bias on $\mathrm{IRR}_s$ across exposure horizons $< 0.05$.
  * **ATT Coverage**: 95% posterior credible intervals for $\mathrm{ATT}_s$ must cover true counterfactual difference with empirical coverage between 91% and 98% across 10 seeded datasets.
  * **MCMC Diagnostics**: $\hat{R} \le 1.04$ and bulk $\mathrm{ESS} \ge 250$ on all exposure horizon ATTs.
  * **Comparison baseline**: Compare against continuous LongBet on $\log(Y + 1)$ and verify that the Negative Binomial model exhibits lower bias and properly calibrated interval widths.

---

## 10. Evaluation of alternatives and design trade-offs

### 10.1 Comparison of count handling approaches

```
+-----------------------------+-----------------------+--------------------+------------------------+
| Approach                    | Tree Conjugacy        | Physical Bounds    | Zero-Inflation         |
+-----------------------------+-----------------------+--------------------+------------------------+
| Continuous on raw Y         | Exact Gaussian        | Can predict Y < 0  | Assumes homoskedastic  |
| Continuous on log(Y + 1)    | Exact Gaussian        | Strictly Y > -1    | Ad-hoc constant bias   |
| Ordinal Probit (K <= 8)     | Truncated Normal      | Discrete [0, K-1]  | Estimates cutpoints    |
| Direct Poisson BART         | NON-CONJUGATE         | Strictly Y >= 0    | Equi-dispersed only    |
| NegBin + Pólya-Gamma (Ours) | EXACT COND. GAUSSIAN  | Strictly Y >= 0    | Overdispersed + offset |
| Two-Part Hurdle (Ours)      | EXACT SUR CONDITIONAL | Strictly Y >= 0    | Explicit zero margin   |
+-----------------------------+-----------------------+--------------------+------------------------+
```

### 10.2 Summary conclusion

Negative Binomial modeling via Pólya-Gamma data augmentation (`outcome="negbin"`)
is the optimal engineering and statistical choice for extending LongBet. It
seamlessly leverages the existing weighted-precision tree infrastructure in
`bartz` and `_step.py`, respects the non-negative integer support of counts,
eliminates $\log(Y + 1)$ retransformation biases, and provides researchers with
direct posterior inference on both relative rate multipliers (IRR) and natural
count ATTs.
