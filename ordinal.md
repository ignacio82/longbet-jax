# Ordered probit LongBet: implementation specification

Status: proposed implementation; this is not an existing feature. Reviewed
against the repository on 2026-09-09. Implement the phases in order and use the
acceptance criteria below to determine completion. Sections 1–9 specify the
ordered-probit implementation; section 10 evaluates cloglog as an alternative
backend and defines the evidence needed before changing that choice.

## 1. Objective and design decisions

Add `outcome="ordinal"` for ordered categorical panels under staggered adoption.
Retain the prognostic forest, treatment forest, exposure GP, adaptive coding,
and unit random intercepts. Add latent-response augmentation and sampled
cutpoints, then expose category probabilities and treatment effects on category
probabilities and user-defined scores.

Ordered probit supplies a conditionally Gaussian response compatible with the
current forest and parameter updates. The augmentation principle is described
by [Albert and Chib (1993)](https://stat.cmu.edu/~brian/905-2009/all-papers/albert-chib-1993.pdf).
The prior and API below are explicit LongBet design choices.

Required scope includes scalar Python fitting, prediction, bounded-memory
summaries, NPZ persistence, `LongBetMulti` within its existing triangular SUR
model, shared treatment trees, child-model extraction, R wrappers, and tests.
Deliver these in phases, with explicit rejection of unfinished combinations.

Defer covariate-dependent thresholds, unordered multinomial outcomes,
logit links, freely correlated discrete residuals, and accelerated cutpoint
proposals. Evaluate cloglog separately under section 10; it is not a drop-in
replacement for the sampler specified here. Do not add unused proposal-tuning
configuration or expose an unimplemented link option.

| Issue in the initial draft | Required decision |
|---|---|
| Link choice | Implement ordered probit first to preserve current LongBet features; evaluate cloglog with the separate prototype and decision gates in section 10. |
| Offset sign | With the first threshold at zero, use `offset = -Phi_inverse(P(Y=0))`. |
| Offset placement | Existing `mu_fit` includes the forest offset, and `alpha` scales it. Preserve that convention. |
| Binary equivalence | Allow ordinal `K=2`, dispatching the existing binary initialization and sampler. |
| Empty categories | Allow declared categories without observations; use a proper cutpoint prior. |
| Cutpoint updates | Use sequential Gibbs draws with current neighbors. |
| SUR scale | Distinguish fixed structural variance from cell-specific conditional variance; preserve zero incoming loadings for discrete outcomes. |
| Public array axes | Keep cells first and draws last; put category immediately before draws. |
| Persistence | Extend explicit NPZ keys and validators; serialization is not automatic PyTree serialization. |

## 2. Statistical contract

### 2.1 Notation, mean surface, and identification

Let `n=N*T` be cells in unit-major order, `C` chains, `D` total retained draws,
and `K` declared categories. Use `A_it` for treatment and `ell_it` for the
latent response. These are `state.z_vec` and `state.z` in code, respectively.
`state.y` remains observed labels; do not confuse treatment with augmentation.

Write the repository's prognostic fit as

$$m_{it}=o+f_\mu(X_i,t,X^{\mathrm{tv}}_{it}),$$

where `o=model.offset_` is stored in the prognostic forest trace. Preserve

$$\eta_{it}=\alpha m_{it}
 +b_{A_{it}}\beta_{S_{it}}\nu(X_i,S_{it},t,X^{\mathrm{trt,tv}}_{it})+\gamma_i.$$

For a scalar ordinal outcome,

$$\ell_{it}\mid\eta_{it}\sim N(\eta_{it},1),\qquad
Y_{it}=k\iff\theta_k<\ell_{it}\leq\theta_{k+1},\quad k=0,\ldots,K-1.$$

Use one threshold vector per outcome, shared across units and times:

$$\theta_0=-\infty,\quad\theta_1=0,
\quad0<\theta_2<\cdots<\theta_{K-1},\quad\theta_K=+\infty.$$

Fixing `sigma2=1` identifies latent scale; fixing `theta_1=0` identifies the
threshold/mean location. Do not also center latents or move the anchor during
sampling. The fixed forest offset centers the prior; it is not a second sampled
intercept. With `sample_alpha=True`, preserve
`alpha * (offset + forest_sum)` in fitting and prediction.

### 2.2 Proper cutpoint prior

Store the `K-2` free cutpoints. With `s_theta=cutpoint_prior_scale`, default
`5.0`, specify the joint prior

$$p(\theta_2,\ldots,\theta_{K-1})\propto
\exp\left[-\frac{1}{2s_\theta^2}\sum_{j=2}^{K-1}\theta_j^2\right]
\mathbb I(0<\theta_2<\cdots<\theta_{K-1}).$$

This is a proper ordered-normal prior: equivalently, sort `K-2` independent
half-normal draws. Its scale is in latent probit units and is a substantive
prior choice. Sparse categories can be prior-sensitive. There is no cutpoint
prior to evaluate for `K=2`.

Replace the initial draft's unbounded flat prior. An unobserved top category
can leave the last threshold with an infinite upper conditional bound, making
a uniform draw undefined. The proper prior gives a valid threshold conditional
for empty categories. This is not a proof of posterior propriety for every
other parameter/prior in the complete model.

## 3. Configuration, inputs, and initialization

### 3.1 Scalar API

Extend `LongBetConfig` in `src/longbet/_config.py`:

```python
outcome: Literal["continuous", "binary", "ordinal"] = "continuous"
num_categories: int | None = None
cutpoint_prior_scale: float = 5.0
```

Validate before compilation:

- `outcome` must be one of the three values.
- For ordinal, `num_categories` is an integer `>=2`, excluding booleans.
  For scalar nonordinal configurations require `None`.
- `cutpoint_prior_scale` is finite and positive, excluding booleans. It is
  inactive for nonordinal outcomes and `K=2`.
- Observed `y` values are integers in `[0,K-1]`. Accept integer-valued floating
  arrays so `NaN` can represent missingness. Reject fractions, negative or
  out-of-range labels, strings, and either infinity.
- Require at least one observed cell. Missing categories, including first or
  last, and panels containing only one observed category are valid.
- Check infinities before constructing the mask. Only `NaN` means missing.
  Replace missing labels with zero before casting or threshold indexing; the
  mask excludes these placeholders from all likelihood calculations.
- Do not infer `K` or recode categories: a five-level scale remains `K=5` if
  the training sample contains only labels `0,2,4`.
- Ordinal uses `meany=0`, `sdy=1` regardless of `standardize`. Preserve existing
  binary and continuous behavior.

### 3.2 Offset and starting thresholds

Compute in host float64 on observed cells only:

$$p_j=\frac{\#\{Y_{it}\leq j:\mathrm{observed}\}}{n_{\mathrm{obs}}},
\qquad q_j=\Phi^{-1}(\mathrm{clip}(p_j,10^{-4},1-10^{-4})).$$

Set `offset_=-q_0`, since `P(Y=0 | eta=offset_)=Phi(-offset_)`.
If 80% of observations are category zero, the offset is approximately
`-0.841621`, giving `P(Y=0)=0.8` in the zero-forest initial mean.

Initialize in ascending order, for `j=2,...,K-1`:

$$\theta_j^{(0)}=\max(q_{j-1}-q_0,\theta_{j-1}^{(0)}+0.1),
\qquad\theta_1^{(0)}=0.$$

The `0.1` gap repairs duplicate empirical quantiles at initialization only; it
is not a posterior gap constraint. Convert to float32 and check finite strict
ordering. For ordinal `K=2`, use the existing binary rate calculation directly:
`Phi_inverse(clip(mean(y),1e-4,1-1e-4))`. It equals `-q_0` mathematically and
avoids numerical differences in compatibility tests.

### 3.3 State construction and chains

The current initializer is `init_longbet` in `src/longbet/_state.py`, not
`_init_state`. Add fields using `bartz._jaxext.field` conventions:

| Field | Single-chain representation | Chain metadata |
|---|---|---|
| `cutpoints` | float32 `(K-2,)` for ordinal; `(0,)` otherwise | `chains=CHAIN_AXIS` |
| `num_categories` | Python int: ordinal `K`, binary `2`, continuous `0` | static |
| `cutpoint_prior_scale` | Python float | static |
| inherited `z` | `(n,)` for discrete; existing representation for continuous | preserve inherited metadata |
| inherited `y` | `(n,)`, sanitized observed category codes for ordinal | shared |

Keep Equinox required/default field ordering valid. Do not replicate shared
`X`, `y`, masks, or indices across chains. Empty cutpoint arrays keep one
consistent array representation for nonordinal outcomes.

For `K>2`, initialize Gaussian `bartz` forest machinery with a finite latent
working response; do not pass category labels through its binary branch.
Explicitly set LongBet's observed labels, augmented `z`, and full residual
consistently after construction. Set `binary_indices=None`. Keep `mu_fit`
inclusive of the offset and both forest variance draws disabled. The forest
step views must receive `z=None, binary_indices=None` to avoid reaugmentation.

Use `broadcast_to_chains`, rather than manually tiling inside the single-chain
initializer. Extend `_overdisperse_chains` so ordinal `sigma2` stays exactly one.
For `K>2`, disperse starting positive threshold gaps independently per chain by
multiplying by `exp(0.25 * Normal(0,1))`, then cumulatively sum from zero.
After parameter/threshold dispersion, draw category-consistent latents from
each chain's actual mean and reconstruct its residual. Keep missing latents
and residuals zero. Preserve binary initialization for `K=2`.

Use independent initialization subkeys derived from the existing fit key/seed.
No global RNG or hidden fixed seed. When a private initializer has no key, a
deterministic interior latent is acceptable until the first mandatory latent
update; its residual must still match the initialized mean.

## 4. Sampling primitives and sweep integration

Put reusable ordinal primitives in a new `src/longbet/_ordinal.py`, with host
validation/preparation shared by scalar and multi-outcome callers. Suggested
private interfaces, each sampler operating on one chain:

```python
sample_ordinal_latents(key, labels, mean, sd, cutpoints, obs_mask, old_z)
sample_cutpoints(key, z, labels, cutpoints, obs_mask, prior_scale)
category_probabilities(mean_draws, cutpoint_draws)
```

Cell vectors have length `n`; `sd` may broadcast. Free cutpoints have length
`K-2`, which determines static `K`. Keep host validation outside jitted kernels.
For prediction, `category_probabilities` takes means `(D,B)` and free
cutpoints `(D,K-2)` and returns `(D,B,K)` for a block of `B` cells; broadcast
thresholds across cells, never across the draw axis. The caller transposes
into public layouts. Apply the same formula category-by-category if needed
to avoid allocating all categories simultaneously.

### 4.1 Latent-response update

At scalar sweep boundaries, `R = ell - eta` on observed cells. Reconstruct
`mean = state.z - R`. Use `sd=1` for scalar ordinal likelihoods. When SUR
supplies `conditional_precision`, use `sd=rsqrt(conditional_precision)`; the
supplied residual reconstructs the conditional mean (section 6).

Construct `full = [-inf, 0, *cutpoints, inf]`. For observed category `k`, draw

$$\ell_{it}^{\mathrm{new}}\sim\mathrm{TN}
(\mathrm{mean}_{it},\mathrm{sd}_{it}^2;\theta_k,\theta_{k+1}).$$

Standardized bounds are `(full[k]-mean)/sd` and `(full[k+1]-mean)/sd`;
rescale the sample as `mean + sd * draw`. Update

```python
R = where(obs_mask, R + new_z - old_z, 0.0)
```

Leave missing `z` entries at zero. Supply benign means, bounds, and scales for
missing cells before sampling, not only in a final `where`: masked invalid
arithmetic must not enter JAX branches. Missing cells contribute neither
threshold bounds nor sufficient statistics.

### 4.2 Sequential cutpoint Gibbs update

After refreshing latents, for each free threshold `j=2,...,K-1`, compute

$$L_j=\max\left(\theta_{j-1},\max_{\mathrm{obs},Y=j-1}\ell\right),
\qquad
U_j=\min\left(\theta_{j+1},\min_{\mathrm{obs},Y=j}\ell\right).$$

Use `max(empty)=-inf` and `min(empty)=+inf`. The exact conditional under the
specified joint prior is

$$\theta_j\mid\ell,Y,\theta_{-j}\sim
\mathrm{TN}(0,s_\theta^2;L_j,U_j).$$

Conditional on latents, the likelihood only restricts admissible thresholds;
the remaining density is the normal prior. A uniform draw targets a different
prior. Precompute masked category minima/maxima with fixed-size segment or
scatter reductions. Use `lax.fori_loop` or `lax.scan` over thresholds, inserting
each new value before updating its right neighbor. The left neighbor is the
latest draw; the right neighbor is its current value. This matters especially
for empty intermediate categories. Parallelize cells/reductions and chains,
not dependent neighboring cutpoint draws. For `K=2`, this is a no-op.

Remove the earlier Cowles MH formula involving `P(Y | latent, theta)`: that
likelihood is an indicator. An accelerated move needs a correctly derived
joint or marginalized transition and compatible latent refresh. It is deferred;
[Cowles (1996)](https://link.springer.com/article/10.1007/BF00162520) is background
for future work, not a second sampler required for this feature.

### 4.3 Numerical requirements

The [JAX API](https://docs.jax.dev/en/latest/_autosummary/jax.random.truncated_normal.html)
accepts broadcast bounds for a standard normal. Test the installed implementation
in float32 before relying on it for far-tail draws. Naive CDF interpolation can
collapse when both endpoint probabilities round together.

Require infinite endpoints, narrow finite intervals, and intervals such as
`[8,9]`, `[12,inf]` and their reflections. The primitive must work under `jit`
and `vmap`, without SciPy or host transfers inside a sweep. If the stock
primitive fails distributional checks, implement a stable rejection sampler.
One concrete construction for a standard normal on `(a,b)` is:

1. If `b<=0`, reflect to `(-b,-a)` and negate the result.
2. If `a>=0`, let `lambda=(a+sqrt(a*a+4))/2`. Propose from an exponential of
   rate `lambda` truncated to `(a,b)`:
   `x=a-log1p(-u*(-expm1(-lambda*(b-a))))/lambda`, taking the inner mass as
   one when `b=inf`. Accept when `log(v)<=-0.5*(x-lambda)**2`, with independent
   uniforms `u,v` in `(0,1)`.
3. For intervals crossing zero of width at most two, propose uniformly on
   `(a,b)` and accept when `log(v)<=-0.5*x*x`.
4. For wider intervals crossing zero, reject ordinary normal draws outside
   the interval.

Use a JAX loop with a per-element pending mask, retaining accepted values and
using fresh subkeys. Verify the distribution with an independent SciPy oracle.
These formulas specify a possible implementation; they do not describe JAX's
current internal algorithm. Vectorized branches need benign inputs even for
inactive elements. `K=2` retains the legacy binary primitive for compatibility;
compare the general ordinal primitive to it in distribution, not draw by draw.

Floating arithmetic can round a rescaled draw onto an endpoint. Use `nextafter`
only to correct endpoint rounding. Do not clip extreme means, impose posterior
gap floors, or replace tail draws by a boundary constant. Check `L<U` and that
an interior value is representable. Invalid intervals or nonfinite results must
surface as numerical failures, for example a checked-kernel error propagated
at a host batch boundary. Do not silently sort, jitter, or accept them. Avoid
globally enabling float64 to work around a local sampling issue.

### 4.4 Full sweep and compatibility

For ordinal `K>2`, extend `longbet_single_step` in `_step.py`:

```text
latent -> cutpoints -> mu -> alpha -> nu -> beta -> b0,b1
       -> gamma -> sigma_gamma2 -> fixed sigma2 -> ridge
```

Reuse downstream weighted Gaussian conditionals. Drawing a threshold does not
change `R`, since neither latents nor means change. Include new latents and
cutpoints in the returned state. Audit all binary-only variance guards in
initialization, overdispersion, and end-of-sweep updates: ordinal structural
`sigma2` and its saved inverse must stay exactly one. Preserve dynamic
leaf-precision cache refreshes, `_load`/`_read` units, and ridge handling.

Keep the existing `random.split(key,10)` schedule for legacy steps. Allocate
new ordinal-only substreams with documented distinct `random.fold_in` tags;
increasing the split count changes every legacy draw. For ordinal `K=2`,
dispatch the existing binary latent path and skip cutpoint work, permitting
identical seeded sampling under matched configurations.

## 5. Prediction and estimands

### 5.1 Draw-wise probabilities and counterfactuals

The existing class is `LongBetPrediction` (singular) in `_model.py`. Preserve
`tauhats`, `muhats0`, `yhats`, their summaries, `att()`, and `stability()` on the
latent scale. New ordinal fields/methods expose observed-scale results without
changing existing consumers. Probabilities integrate the unit-variance
observation noise; they are not sampled category labels.

For each posterior draw `d`, using that draw's thresholds and mean,

$$p_{itk}^{(d)}(\eta)=\Phi(\theta_{k+1}^{(d)}-\eta_{it}^{(d)})
 -\Phi(\theta_k^{(d)}-\eta_{it}^{(d)}).$$

The paired counterfactual means are

$$\eta_{it}^{0,(d)}=\alpha^{(d)}m_{it}^{(d)}
 +b_0^{(d)}\beta_0^{(d)}\nu^{(d)}(X_i,0,t)+\gamma_i^{(d)},$$

$$\eta_{it}^{1,(d)}=\alpha^{(d)}m_{it}^{(d)}
 +b_1^{(d)}\beta_{S_{it}}^{(d)}\nu^{(d)}(X_i,S_{it},t)+\gamma_i^{(d)}.$$

Time-varying covariates are held at the supplied values in both expressions,
as in current prediction. Evaluate the treatment forest at factual exposure
and at `S=0`, exactly as `_predict_impl` currently does. `eta1=mu0+tau`;
factual mean is `yhats`. Do not add the offset a second time. For untreated
cells, retain the existing `S=0` prediction convention; ATT only uses treated
cells with positive exposure, not a newly invented treatment schedule.

Compute `delta_p = p(eta1)-p(eta0)` within the same draw. For exposure `s`,

$$\mathrm{ATT}_{s,k}^{(d)}=
\frac{1}{n_s}\sum_{\{(i,t):A_{it}=1,S_{it}=s\}}
\left[p_{itk}^{(d)}(\eta^1)-p_{itk}^{(d)}(\eta^0)\right].$$

Use the prediction panel's treated-cell membership and `att_counts`, including
cells whose outcomes were missing during fitting. This matches existing ATT
aggregation; it does not estimate an observed-outcomes-only effect. If `n_s=0`,
return `NaN` draws/summaries for that exposure. Preserve the existing exposure
axis, including its minimum length of one when no treated cells exist.

Never apply a CDF to posterior means, average cutpoints first, transform latent
ATT directly, or subtract independently summarized credible bounds. Nonlinear
transformation, paired differencing, and cell averaging all precede posterior
summarization. Per draw, probabilities sum to one and category effects sum to
zero. Use float64 host probability calculations with stable CDF/survival or
log-CDF differences in tails; clip only negligible rounding excursions.

### 5.2 Public API and shapes

For ordinal predictions, always compute category summaries and category ATT
draws during the blocked prediction pass. Full per-cell probability draws are
retained only when `summary_only=False`. Add these attributes:

| Attribute | Shape / meaning |
|---|---|
| `num_categories`, `categories` | `K`, integer labels `arange(K)` |
| `cutpoints_samples` | `(D,K-2)`, chain-major; free thresholds only |
| `prob_y`, `prob_mu0`, `prob_tau` | `(N,T,K,D)`, or `None` in summary mode |
| `prob_y_summary`, `prob_mu0_summary`, `prob_tau_summary` | `PosteriorSummary`, each component `(N,T,K)` |
| `att_prob_full` | `(S_out,K,D)`, always retained |

For nonordinal predictions, new attributes are `None`; ordinal-only methods
raise a clear `ValueError`. Ordinal `K=2` exposes both category probabilities,
with category-one probability equal to `Phi(eta)`.

Specify the new methods exactly:

```python
pred.predict_probabilities(arm="factual", summary=True)
pred.att_probabilities(alpha=0.05)
pred.att_expected_score(weights=None, alpha=0.05)
```

`arm` is `"factual"`, `"control"`, or `"effect"`, selecting the `y`, `mu0`, or
`tau` probability fields. `summary=True` returns a `PosteriorSummary` with the
interval level chosen at `model.predict`; `summary=False` returns full draws,
or raises if they were discarded. The effect is treated-minus-control and
is not itself a probability distribution. There is no ambiguous `scale` flag.

`att_probabilities` returns the existing ATT-style dictionary keys:
`att` `(S_out,K)`, `intervals` `(2,S_out,K)`, `att_full` `(S_out,K,D)`,
`exposure` `(S_out,)`, plus `categories` `(K,)`. Summarize along the final axis.

`att_expected_score` uses finite numeric weights of length `K` (default
`arange(K)`), and first forms `score_att[s,d] = sum_k weights[k]*att_prob_full[s,k,d]`.
Return `att` `(S_out,)`, `intervals` `(2,S_out)`, `att_full` `(S_out,D)`,
`exposure`, and the chosen `weights`. Do not require increasing weights: an
indicator score can request an exceedance probability effect. Ordinal ordering
does not imply equal spacing; default rank scores are a reporting convention.
No per-cell score API is required in this version.

Example of the intended interface (to become a tested example):

```python
model = LongBet(LongBetConfig(outcome="ordinal", num_categories=4))
model.fit(y=y, x=x, z=z, t=t)
pred = model.predict(x=x, z=z, t=t, summary_only=True)
category_summary = pred.predict_probabilities()
category_att = pred.att_probabilities()
top_category_att = pred.att_expected_score(weights=[0, 0, 0, 1])
```

Keep current unit-intercept prediction semantics: rows match fitted units by
position when counts agree; otherwise current code uses zero intercepts. These
probabilities are conditional on that choice, not marginal predictions for a
new random unit. Explain this limitation in ordinal examples; use
`random_intercept=False` in held-out-new-unit validation unless unit identity
handling is separately implemented. Do not broaden this feature into a unit-ID
API redesign.

### 5.3 Bounded memory and diagnostics

Implement probability work inside `_predict_impl`'s cell loop. Given a block
of `B` cells, transform all `D` paired draws before reducing. Either process
categories individually or size the block to account for `K`, simultaneous
arm buffers, and float64 temporaries. Retain exact per-cell quantiles by keeping
all draws for each cell; streaming moments alone cannot recover them.

Accumulate category ATT sums in float64 `(S_out,K,D)` arrays and divide by the
same counts as latent ATT. Desired additional memory in summary mode is
`O(B*D*K + N*T*K + S_out*K*D + D*K)`, with the working-block byte budget
independent of panel length. Never build a full `(D,N*T,K)` or new full
`(D,N*T)` buffer in this path. The existing opt-in forest-evaluation cache
still deliberately retains full forest evaluations; document that exception
and test bounded memory with caching disabled.

Keep latent `pred.stability()` behavior. Validate category and score effects
separately by reshaping their retained ATT draws back to `(C,draws_per_chain,S)`
per category/score and calling existing diagnostics. Diagnose free cutpoints
too; report nonfinite diagnostics and empty exposure groups explicitly. Do not
claim good latent ATT diagnostics guarantee probability-scale convergence.

## 6. Multiple outcomes and SUR

### 6.1 Per-outcome category metadata

Extend `LongBetMulti.fit` with `num_categories=None`. Accept an integer
broadcast to ordinal outcomes, a sequence in user order, or a mapping whose
keys exactly match all outcome names. Sequence/mapping entries for nonordinal
outcomes must be `None`; every ordinal entry must be an integer `>=2`.
When the argument is omitted, use `config.num_categories` for ordinal children
and `None` for other children. Missing ordinal counts are errors. Reject an
explicit category-count argument if there are no ordinal outcomes.

Store resolved counts in user and internal order in `NormalizedMultiInput`.
Use the shared host preparation helper for offsets, validation, and starting
thresholds. Construct each child config with both `outcome=child_type` and
`num_categories=child_K_or_None`, including extracted/loaded child models.
The prior scale is shared configuration for this version. Different children
can have different `K`; keep their cutpoint states/traces in tuples, not a
padded array with fake categories.

Internal order is a **stable partition into discrete then continuous**:
preserve caller order within the union of binary/ordinal outcomes, then within
continuous outcomes. For example, user order
`(continuous_A, ordinal_B, binary_C, ordinal_D)` becomes `(B,C,D,A)`.
Existing binary/continuous models keep their current ordering. User-facing
models, predictions, names, and category metadata remain in user order.

### 6.2 Preserve the identified triangular likelihood

The existing SUR module uses `B=I-Gamma`, innovation variances `v`, and
precision `Q=B.T @ diag(1/v) @ B`. For every discrete outcome require
`v_m=1` and the entire incoming row `Gamma[m,:]=0`. Continuous rows can load
on preceding discrete or continuous residuals. Consequently each discrete
marginal latent variance is one. Allowing incoming discrete loadings while
only fixing innovation variance would change that scale and is out of scope.

This model does not introduce free residual correlations between two discrete
outcomes. Joint posterior dependence through shared forests or continuous
outcomes must not be described as such a residual covariance parameter.

Use `_sur.conditional_residual` as `_multi_step.multi_single_step` already does.
For an observed cell its returned pseudo-residual is `(Q*r)_m/Q_mm` and
precision is `Q_mm`, using only supported observed likelihood rows. Thus the
latent conditional is

$$\ell_m\mid\ell_{-m},Y_m,\ldots\sim
\mathrm{TN}(\ell_m-\widetilde R_m,Q_{mm}^{-1};
\theta_{Y_m},\theta_{Y_m+1}).$$

A downstream continuous observation can make this variance smaller than one.
Passing `sd=1` here is incorrect. After child updates, restore raw residuals
with the existing offset correction. Cutpoint bounds use the actual ordinal
latents and observation mask, not pseudo-responses or innovation residuals.

Keep `continuous_mask` literally continuous. Only continuous rows sample
incoming loadings and innovation variances. Preserve the proper variance-prior
validation for coupled continuous outcomes and the existing missingness rule:
an observed downstream continuous response requires all predecessor outcomes
observed at that cell. Missing discrete patterns need not be nested among
themselves. Do not add unobserved latent imputations to bypass this rule.

Extend shared-tree integration using the existing full conditional precisions,
raw-residual conventions, and common draw alignment. Preserve equality with
independent scalar sweeps when both SUR and sharing are off, under explicitly
matched per-outcome keys. Test ordinal children in both coupling modes.

Child ordinal predictions use the same unit-variance marginal category formula
as scalar predictions; the fitting-time conditional variance is not the
prediction variance. Do not add fitted SUR residual/loading offsets to the
counterfactual mean surface.

`effect_draws`, `effect_draws_from_arrays`, and `joint_prob` currently assume
one scalar effect per outcome. Reject ordinal use in those existing interfaces
with guidance to select a category/score via the new methods; do not silently
return a latent or arbitrarily chosen category effect. Generalized joint
category/score event APIs are deferred. Keep their existing continuous/binary
behavior and `outcome_correlation`'s existing covariance meaning.
For a multi prediction, reject only when a requested outcome is ordinal;
continuous/binary selections from a mixed prediction remain supported.

## 7. Trace storage, archives, and R

### 7.1 Trace and schema changes

`LongBetTrace` lives in `_loop.py`. Add optional `cutpoints`, default `None` for
nonordinal traces, with ordinal shape `(n_save,K-2)` or `(C,n_save,K-2)`.
Ordinal `K=2` has an empty final dimension, not a missing field. State cutpoints
remain arrays even for nonordinal states (section 3); the trace's optional
field preserves compatibility with callers constructing legacy traces.

Add buffers, carry fields, writes via `_set_param`, and final trace assembly in
both `_loop.py` and `_multi_loop.py`. Save cutpoints on exactly the same sweeps
as forests/scalars after burn-in and thinning. Flatten chain-major alongside
other parameters; preserve empty dimensions without `reshape(-1,0)` ambiguity.
Do not retain per-sweep latents, which would add `O(n*D)` storage.

`_io.py` uses `_PARAM_KEYS`, `_FOREST_KEYS`, JSON metadata, and explicit arrays.
Keep existing parameter keys; add conditional cutpoint save/load handling
instead of blindly extending loops that assume scalar parameter shapes.
Use `cutpoints` in scalar archives and `outcome_{m}_cutpoints` internally in
multi archives. Persist resolved `K`, prior scale, anchor convention
`"first_finite_zero"`, and `ordinal_schema_version=1`. Config plus metadata
must suffice for prediction after load; never recompute thresholds from data.

For multi archives containing ordinal outcomes, use `format_version=3` and
extend `_multi_io._validate_archive` and its loader. Keep existing format 1/2
support for compatible nonordinal archives, subject to existing version checks.
Maintain current sampler-semantics/shared-topology metadata and add the ordinal
schema metadata; do not reinterpret old draws as a different sampler.

Validate counts, shapes, draw/chain alignment, finite strictly ordered positive
free cutpoints, unit discrete variances, zero incoming discrete loading rows,
ordering permutations, and per-outcome metadata agreement. Reject missing or
malformed cutpoints on ordinal archives, including an absent `K=2` empty array.
Compatible old nonordinal archives omit the new keys and load with `None`
cutpoint traces/default config values. Retain `PRECISION_CACHE_VERSION` checks;
this feature must not make previously rejected stale archives loadable.

Round-trip parent and extracted child models, with and without shared trees.
Loaded predictions must agree for the same inputs and projection key. Existing
save/load stores posterior prediction state, not resumable MCMC state; do not
promise resume support as part of this change.

### 7.2 R interface

Update `R/longbet.R` and `R/multi.R` to accept ordinal outcome types,
`num_categories`, and `cutpoint_prior_scale`. For multi fits, use an integer
broadcast or a list in user order (optionally named by every outcome), with
`NULL` for nonordinal entries. Validate integrality before `as.integer` so
fractional categories/counts cannot silently truncate. Require explicit numeric
`0,...,K-1` labels; do not silently use factor level numbers as category codes.

Extend `predict.longbet` and multi prediction wrapping with the new probability
fields, metadata, and summaries, retaining dimensions when `K=2`, one cell, or
one draw. Preserve default latent fields and existing signatures. Add exported
wrappers `predict_probabilities(object, arm="factual", summary=TRUE)`,
`att_probabilities(object, alpha=0.05)`, and
`att_expected_score(object, weights=NULL, alpha=0.05)` matching Python shapes
and return values. Call Python methods; do not reimplement CDFs or ATT arithmetic
in R. No `type="score"` prediction mode is needed for an ATT-only score API.

Update printing to distinguish latent predictions from ordinal probabilities.
Update roxygen, `NAMESPACE`, and relevant `man/*.Rd` files. Fit objects must
continue to round-trip via `raw_model`, `saveRDS/readRDS`, and
`R/rehydrate.R`, including after a cached Python handle is invalidated.

## 8. Implementation map and delivery order

Read the named functions before editing; this table supplements their existing
contracts, rather than replacing their numerical safeguards.

| Phase | Files and work | Completion gate |
|---|---|---|
| 1. Model primitives | `_config.py`, new `_ordinal.py`; host validation, initialization, stable interval sampler, sequential thresholds | Input, analytic probability, and distributional primitive checks pass. |
| 2. Scalar sampling | `_state.py`: `init_longbet`, chain metadata/overdispersion; `_step.py`: augmentation, variance guards, return state; `_loop.py`: trace buffers | Multi-chain/missingness invariants, `K=2` equivalence, aligned traces pass. |
| 3. Prediction and persistence | `_model.py`: `LongBetPrediction`, `_predict_impl`, save/load metadata; `_summary.py`; `_io.py` | Full/summary parity, ATT/score identities, memory bounds, scalar round-trip pass. |
| 4. Joint fitting | `_multi_input.py`, `_multi_state.py`, `_multi_step.py`, `_multi_loop.py`, `_multi_model.py`, `_multi_io.py`; audit `_sur.py`, `_shared_forest.py` | Heterogeneous `K`, conditional-normal oracle, ordering, sharing, multi/child persistence pass. |
| 5. R and documentation | `R/longbet.R`, `R/predict.longbet.R`, `R/multi.R`, `R/att.R`, `R/rehydrate.R`, `NAMESPACE`, `man/`, `README.md`; new example if useful | R/Python parity and saved-fit rehydration pass; slow benchmark results documented. |

During phases 1–3, explicitly reject ordinal multi-outcome requests before
normalization can fall through to a binary/continuous branch. Remove that guard
only once phase 4 passes. Audit all `LongBetState`/`LongBetTrace` construction
sites and all tests branching on binary outcome type. This document's full
ordered-probit scope is complete only after phase 5; mark unfinished work
honestly. The cloglog evaluation in section 10 has separate completion gates
and does not silently expand these phases into a second production sampler.

## 9. Verification and acceptance criteria

### 9.1 Fast contracts and distributional tests

Add focused files such as `tests/test_ordinal_primitives.py`,
`tests/test_ordinal_model.py`, `tests/test_ordinal_predict.py`, and
`tests/test_ordinal_multi.py`. Extend existing fixtures where that avoids
duplicating chain, archive, and R interface tests.

| Area | Required checks |
|---|---|
| Inputs | `K=2,3,5`; invalid type/count/prior/labels; NaN versus infinity; missing first/interior/last categories; only one observed category; all-missing rejection; no recoding. |
| Initialization | The 80%-category-zero sign example; finite ordered thresholds despite duplicate empirical CDF values; no standardization; valid latents after chain dispersion for `K>2` (after first augmentation for the legacy `K=2` path); `mu_fit`/offset agrees with prediction, including sampled alpha. |
| Latent primitive | Category membership and finite samples for varying means/scales, two-sided and one-sided bounds, narrow intervals, `[8,9]`, `[12,inf]`, and negative reflections under `jit`/`vmap`. Compare means/variances or CDF quantiles to `scipy.stats.truncnorm` with Monte Carlo-aware tolerances; finiteness alone is insufficient. |
| Threshold conditional | For fixed synthetic latents/neighbors compare draws to the specified truncated-normal conditional, not a uniform distribution. Check anchor, strict ordering and all observed-category inequalities after each sweep. Include empty adjacent categories and an empty top category. |
| Residuals and variance | Reconstruct `where(mask,z-alpha*mu_fit-b_A*beta_S*nu_fit-gamma,0)` after a complete scalar sweep. Test missing placeholders do not change draws on observed cells with fixed keys. Assert exact unit ordinal variance in initial, dispersed, final, and saved states. |
| Chain axes and RNG | Extend `test_chain_axis.py`: vmapped chains match independent single-chain transitions with identical per-chain keys and differing valid cutpoints. Shared inputs have no chain axis. Adding ordinal code does not change legacy seeded transitions. |
| Binary reduction | Public ordinal `K=2` and binary fits have identical latent/parameter/forest draws under matched keys and settings (excluding ordinal metadata). Use absent-class, missing-cell, and multi-chain cases. General interval primitive versus binary is distributional equivalence only. |
| Prediction algebra | Check hand-computed `K=3` probabilities; extreme means; paired threshold/forest draws; both counterfactual forest inputs; sum-to-one probabilities and sum-to-zero category effects. Positive latent shifts increase expected rank, without requiring every category effect to be positive. |
| Shapes and summaries | Check all documented shapes, chain-major alignment, `K=2` empty thresholds, and unsupported method errors. Full and summary modes agree on means/SDs/bounds/ATT across block sizes including a ragged final block; exact quantiles match reduction of full draws. |
| Aggregation | Category ATT matches direct treated-cell averaging; empty exposures are NaN. Custom score draws equal weighted category ATT draws; constant weights give zero. Test missing training outcomes do not alter prediction membership rules. |
| Memory | Instrument allocations/block transforms in summary mode with cache off; no full panel-by-draw probability or extra latent buffers. Fixed block byte budget accounts for `K` and float64, while required summary and ATT outputs may grow. |
| Scalar persistence | Identical predictions after round-trip, including custom-score ATT from retained category ATT; corruption tests; compatible old nonordinal loading; stale precision-cache rejection preserved. |

Use independent seeds/sample sizes and tolerances justified by Monte Carlo
error for distribution checks. Avoid requiring a single stochastic recovery
run to hit a truth value or an arbitrary correlation threshold in fast CI.

### 9.2 Joint and R tests

- Extend `tests/test_full_sur.py`'s conditional-normal oracle to ordinal
  intervals with a nonzero downstream continuous loading. Check conditional
  latent mean and variance against the analytic truncated normal; the test
  must fail if the sampler uses variance one.
- Use mixed user order, at least two ordinal outcomes with different `K`,
  binary and continuous outcomes, allowed/rejected missingness patterns, and
  proper continuous variance priors. Check discrete loading rows and variances.
- Test SUR-off/shared-off scalar equivalence with matched internal keys, and
  ordinal shared-tree sampling with residual and topology invariants.
- Extend `test_multi_io.py` and related saving/loading tests for format 3,
  per-outcome category metadata, shared parent and extracted child round-trips,
  and malformed shape/order/cutpoint rejection.
- Extend `tests/testthat/` for argument validation, category-axis preservation,
  exact Python/R probability/ATT/score parity, and saved-fit rehydration.

### 9.3 Slow statistical validation

Use a reproducible ordinal panel DGP with `N=300`, `T=10`, three adoption
waves plus never-treated units, `K=4`, and full thresholds
`[-inf,0,1.2,2.5,inf]`. Use `t=1,...,10`, independent standard-normal `x1,x2`,
and four groups of 75 units adopting at times 3, 5, 7, and never. Generate with
`mu(x)=sin(x1)+x2`, `nu(x)=1+0.5*x1`, `alpha=1`, and `b0=b1=1`. Let `beta_0=0`
and generate `beta_s=0.8+0.7*(beta_(s-1)-0.8)+e_s`, where
`e_s ~ N(0,0.05**2)` independently; save the realized trajectory. Add independent
unit-variance normal cell errors and threshold the resulting latent panel.
Derive exposure with the same first-treated-period-equals-one convention as
`derive_exposure`. Include the true control `beta_0*nu(x,0)` term when
computing ground-truth effects, even though it is zero in this initial DGP.

Use data seeds `20260909,...,20260918` and fit seeds offset by 1000. Start with
four chains, 2,000 burn-in iterations, 1,000 retained draws per chain, `n_skip=1`,
`adaptive_coding=False`, and `kernel_type="ar1"`; record all remaining config
values. Inspect category counts and overlap, and report weakly supported
categories rather than silently dropping them. Run one dataset first to check
the benchmark, then all ten for the initial coverage report. Ten datasets give
only a coarse coverage estimate; include its binomial uncertainty.

Start with no unit intercept for held-out-unit probability calibration, then
add a separate panel with unit intercepts and missing outcomes to test those
features. Save DGP parameters, seeds, fit settings, and evaluation membership.
Measure category Brier score/log loss against an empirical-frequency baseline,
category/score ATT error and interval coverage, threshold recovery, rank R-hat,
bulk/tail ESS, MCSE, and runtime. Diagnose cutpoints and reported effects, not
unidentified GP/forest factors individually.

Target `R-hat<1.05` and bulk/tail `ESS>200` for supported cutpoints and reported
ATTs as benchmark goals, recording failures. Assess nominal coverage over
multiple seeded datasets; one missed 95% interval is not by itself a sampler
bug. Record the actual run length and inspect mixing rather than promising
that default burn-in guarantees recovery. If sequential threshold Gibbs is
too slow, report the limitation and design acceleration as separate work.

Mark recovery/coverage checks `pytest.mark.slow`. Run the relevant fast suite,
then the slow ordinal validation and R suite in a configured environment.
Synchronize JAX arrays when timing. Document commands, versions, settings, and
any unavailable device/R checks; do not present a compile-only smoke test as
statistical validation.

## 10. Cloglog alternative: feasibility, prototype, and decision gates

### 10.1 Recommendation and motivation

Cloglog is a credible alternative for ordinal LongBet. Keep ordered probit as
the first production implementation when preserving the GP-weighted treatment
forest, Gaussian unit intercepts, adaptive coding, and joint SUR is the priority.
Evaluate cloglog as a separate backend prototype before replacing that plan.
This recommendation concerns compatibility with the current code, not a claim
that probit is statistically superior for ordinal outcomes.

Alam and Linero's construction uses positive hazard increments to obtain ordered
cutpoints automatically. Truncated-exponential augmentation and log-gamma
priors yield conjugate updates for additive BART leaves and hazard increments,
avoiding direct sampling within narrow ordered-cutpoint intervals.
[Paper, sections 3.1–3.3 and appendix S.1](https://arxiv.org/html/2502.00606v1#S3.SS1)

StochTree demonstrates ordinal BART with
`OutcomeModel(outcome="ordinal", link="cloglog")`, including posterior category
probabilities. The link is asymmetric. That can matter for skewed category
responses, but neither skewed counts alone nor an existing BART implementation
establishes better causal estimates for LongBet.
[StochTree ordinal vignette](https://stochtree.ai/vignettes/ordinal-outcome.html)

The paper's ordinal application compares cloglog BART with linear cumulative
link models, including linear probit. It is not a matched comparison against
ordered-probit LongBet, and cannot establish superiority for this panel model.
[Paper, section 5.2](https://arxiv.org/html/2502.00606v1#S5.SS2)

### 10.2 Probability construction and parameter meanings

For implementation, translate the paper's category labels to this repository's
`0,...,K-1`. Denote log-hazard increments by `a_j`, rather than `gamma`, which
already means unit intercepts in LongBet. For `k=0,...,K-2`, define

$$H_k=\sum_{j=0}^{k}e^{a_j},\qquad
c_k=\log H_k,\qquad
P(Y>k\mid r)=\exp(-H_k e^r).$$

Set `H_-1=0`, and define the last-category survival as zero. Category
probabilities are consecutive survival differences; do not materialize
`inf * exp(r)` to represent the terminal category. These equations express
the proportional-hazards ordinal model in the paper's predictor convention.
[Paper, section 3.1](https://arxiv.org/html/2502.00606v1#S3.SS1)

Consequences for our implementation:

- Increasing `r` shifts mass toward lower categories. Preserve LongBet's
  positive-means-higher interpretation by explicitly defining `r=-eta`, or
  explicitly adopt and document the reversed convention. Do not silently
  reuse probit effect signs. Include a direction-of-effect test.
- `a_j` are unconstrained log increments, not ordered cumulative cutpoints.
  Use all preceding increments when computing `H_k`, including for `K>3`.
  Use `logsumexp`, `expm1`, and stable survival differences in numerical code.
- Fix an identification convention before sampling; the prototype can fix
  `a_0=0`, hence `c_0=0`, and sample the remaining increments with proper priors
  on `exp(a_j)`. Specify their shape/rate and calibrate induced category priors.
  Section 2's ordered-normal cutpoint prior does not transfer automatically.
- The basic model uses one predictor across category thresholds. A predictor
  that varies by category would be a separate model extension with additional
  computation and estimand choices; it is outside the initial prototype.
- The ordinal `K=2` equality with existing binary probit is specific to the
  probit backend. Cloglog must have its own binary-limit probability tests.
  It does not inherit a Gaussian latent error with variance one.

### 10.3 Why the current LongBet sampler cannot be reused unchanged

The following is an algebraic assessment of the paper's augmented likelihood
combined with this repository's model. Conditional on augmentation, write the
predictor-dependent log likelihood as

$$\log L(r)=\sum_i\{d_i r_i-q_i e^{r_i}\}+\mathrm{constant},$$

where `d_i` indicates a nonterminal observed category and `q_i` is a
nonnegative augmented exposure term. For an ordinary additive leaf `v`,
`r_i=r_{-i}+v`, giving the log-gamma-compatible kernel `A*v-B*exp(v)`.

LongBet instead multiplies treatment leaves by
`w_it=b_A*beta_S`, as implemented in `_step.py`'s treatment-forest update.
For `r_i=r_{-i}+w_i*v`, the leaf likelihood becomes

$$v\sum_i d_iw_i-\sum_i q_i e^{r_{-i}}e^{w_i v}.$$

When weights differ within a leaf, these exponential terms do not reduce to
`B*exp(v)`. Weights can also be negative. Merely replacing normal leaf priors
with log-gamma priors therefore does not supply a valid treatment-tree Gibbs
update. Reversing the predictor sign changes the weights' signs but does not
remove this problem.

| Component | Work needed for a cloglog backend |
|---|---|
| Prognostic forest | Implement log-gamma likelihood/prior and integrated tree scores for additive contributions; initially fix `alpha=1`. Sampling alpha or changing its scaling requires an additional derivation. |
| GP-weighted treatment forest | Derive nonconjugate leaf and tree-structure transitions, or explicitly replace the treatment parameterization. Existing Gaussian weighted-tree scores are invalid for this likelihood. |
| GP trajectory and adaptive coding | Gaussian priors can remain, but the normal conditionals in `_step.py` must be replaced by valid nonconjugate updates. |
| Unit intercepts | Gaussian priors can remain with new updates. A log-gamma intercept prior is another option, but changes the model and must be labeled accordingly. |
| Ridge moves | Recheck likelihood invariance and prior/Jacobian terms under the chosen leaf priors; do not reuse a Gaussian-prior acceptance ratio. |
| SUR and shared trees | Derive and validate a non-Gaussian joint model and its transitions before enabling coupling or sharing. `_sur.py`'s Gaussian conditional precision is not a cloglog likelihood. |
| Prediction and storage | Reuse paired-arm ATT, score aggregation, block summaries, and chain alignment. Change probability transforms, parameter meanings, and link-specific archive metadata. |

Nonconjugacy does not make the model impossible. It means the paper's
computational advantage must be measured again after adding LongBet's
structure. A different tree likelihood kernel is substantial work; it need
not require replacing unrelated `bartz` design, traversal, or storage code.

### 10.4 Prototype sequence and evaluation

Keep experimental code separate from the production probit path. The following
gates define the alternative's evaluation, not an additional promise of full
cloglog support in phases 1–5.

1. **Validate basic ordinal cloglog BART.** Reproduce the scalar additive
   likelihood using the paper and StochTree as references. Initially omit GP
   multipliers, adaptive coding, unit intercepts, SUR, and shared trees. Test
   arbitrary `K`, empty categories, identification, stable probabilities, and
   known conditional distributions. Cross-engine agreement means comparable
   posterior results with matched models/priors, not identical random draws.
2. **Test the LongBet bottleneck.** First test a fixed tree with externally
   specified, varying positive and negative treatment weights against numerical
   integration on a small problem. Then implement valid tree-structure moves
   and add the exposure GP. Add Gaussian unit intercepts and adaptive coding
   one at a time. Track which model features and priors differ at every stage.
   A benchmark that removes the GP-weighted structure is evidence for a
   simplified ordinal model, not for full LongBet.
3. **Compare matched panels under both links.** Extend section 9.3 with both
   probit- and cloglog-generated outcomes. Use the same covariates, rollout,
   exposure trajectories, missingness, and held-out units where applicable.
   Calibrate intercepts/cutpoints and effect scales to comparable category
   frequencies and probability effects; equal numeric latent coefficients do
   not imply equal effects under different links. Derive each dataset's true
   category/score ATT from its generating probabilities.
4. **Record statistical and computational results.** Compare category Brier
   score, log loss, probability/score ATT bias and interval coverage, R-hat,
   bulk/tail ESS, MCSE, and effective samples per second. Include sparse and
   asymmetric category distributions. Report compilation separately from
   synchronized sampling time, peak memory, versions, device, seeds, priors,
   and all retained model simplifications. Compare under equal wall-clock
   budgets as well as checking adequate convergence.

### 10.5 Decision and API consequences

Retain ordered probit if the intended deliverable remains full LongBet and the
cloglog prototype has no demonstrated advantage under comparable model
structure. Consider prioritizing cloglog if category calibration, effect
coverage, or sampling efficiency improves enough to justify its additional
sampler work, or if a simplified ordinal model is explicitly the new scope.
Prediction accuracy alone does not establish calibrated causal uncertainty.

Produce a benchmark report with reproducible settings, results, limitations,
and a recommendation before replacing the main implementation plan. No result
should be inferred from the paper's cross-sectional examples alone. If cloglog
is selected, revise the model/sampler specification around the final design;
do not replace only the CDF in sections 4–6.

For a future production backend, separate outcome kind from link, for example
`outcome="ordinal", ordinal_link="cloglog"`, keeping `"probit"` the default.
Add that configuration only when supported, persist the link and its parameter
schema in scalar/multi archives, and reject unsupported combinations explicitly.
Reuse the public category/score summaries while documenting that a predictor
contrast has a different meaning under each link. Keep link-specific priors,
binary-limit tests, and covariance assumptions distinct.
