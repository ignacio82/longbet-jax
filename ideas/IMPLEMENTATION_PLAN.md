# Continuous-treatment LongBet (`LongBetDose`): implementation and validation specification

Status: proposed; nothing in `src/` implements it. Revised 2026-09-17 against commit
`bfe845d` plus the uncommitted count work in the tree. Implement the numbered phases of
section 11 and record evidence for every gate in
`benchmarks/continuous_dose/implementation_status.md`.

Companion files, all in `ideas/`:

| File | What it is |
|---|---|
| `dose_reference.py` | Dense float64 NumPy oracle for every closed-form quantity in this document, with self-checks against brute force. **Verified**: `python ideas/dose_reference.py` passes. It becomes `tests/_dose_reference.py`. |
| `dose_engine_prototype.py` | Feasibility prototype of the hardest piece (section 5.2). **Verified** against the exact enumerated posterior; `--mutants` shows the test can fail. Not production code. |
| `BENCHMARK_PROTOCOL_TEMPLATE.yaml` | The protocol that must be frozen before the comparative study (section 10). |
| `evidence/` | The scripts behind appendix A, runnable from the repository root. |

Public support ships only when three gates pass, each decided independently:

1. **Sampler exactness** (section 8): the implemented kernel targets the documented
   posterior. Evidence: algebra parity with the oracle, exact enumerated tree posteriors,
   `bartz` equivalence in the nested cases, and a Geweke joint-distribution test whose
   mutation arms are shown to fail.
2. **Convergence** (section 9): dispersed chains agree and reach the package standard
   (rank R-hat at most 1.01, bulk and tail ESS at least 400) on a fixed list of reported
   quantities in every prescribed scenario.
3. **Comparative usefulness** (section 10): in a predeclared, frozen study, `LongBetDose`
   beats a credible reference on a prespecified loss in at least one scenario family
   without under-covering. All losses and ties are reported.

A study that fails gate 3 is still a legitimate result. Do not lower thresholds, change
confirmatory seeds, drop difficult replications, tune a competitor with oracle knowledge,
or relabel a modified estimator as the original. Do not claim that diagnostics prove
convergence or that a finite simulation proves general superiority.

## 0. Assessment

### 0.1 Is this a good idea for LongBet? Yes, with the architecture below

- **There is a real gap.** The reference implementation of continuous-treatment DiD
  (`contdid` 0.1.1, alpha, GitHub only) documents that it does not support covariates,
  unbalanced panels or time-varying doses, and offers its data-driven dose curve only
  for two periods without staggering [R5]. Heterogeneous effects, covariate adjustment,
  staggered adoption with a nonparametric dose curve and full posterior uncertainty are
  exactly what a forest model adds.
- **Users will otherwise improvise.** Today the only route is `x_trt = cbind(x, dose)`,
  which makes the effect piecewise constant in dose, offers no derivative, labels no
  estimand, and inherits the problem in the next bullet.
- **The first-difference likelihood is the right foundation, and this was measured.**
  The current level model with a random intercept is not invariant to selection on unit
  *levels*, the situation DiD exists for. With parallel trends holding exactly, a true
  effect of 0.5 and treated units 2 SD higher in level, plain DiD is unaffected while
  current `LongBet` moves from 0.62/0.67 to **0.90/0.90** at `T = 2` (95% interval
  0.68 to 1.11, excluding the truth) and from 0.50/0.36 to 0.58/0.44 at `T = 6`. The
  shift matches the closed form
  $a(1-\theta)^2/\{1-\theta p(2-\theta)\}$ with
  $\theta = 1-\sqrt{\sigma^2/(\sigma^2+T\sigma_\gamma^2)}$ (0.22 and 0.08; appendix A.1).
  Higher-dose units plausibly differ in level, so a dose model on levels would bias the
  *slope* of the dose response. The difference likelihood of section 4.1 is exactly
  invariant: it equals the level likelihood with a flat prior on every unit intercept,
  constants included (oracle check 9).
- **The cost is a new forest engine, and it has been de-risked.** Leaves that hold a
  dose-by-exposure coefficient surface cannot use `bartz`'s leaf algebra. They *can* use
  `bartz`'s proposals, transition/prior ratios and heap bookkeeping unchanged. The
  prototype does this and reproduces the exact posterior over tree structures: total
  variation 0.010 (one tree), 0.008 (with the minimum-units veto) and 0.012 (two trees,
  484 forests, leaves integrated jointly), against 0.136 and 0.369 for two deliberately
  broken kernels (appendix A.3).

The binding risk is **mixing**, not algebra. This repository's own audits
(`benchmarks/tree_moves_review.md`, `tempering_plan.md`, `count/implementation_status.md`)
show forests freezing in chain-specific decompositions and failing the R-hat 1.01 / ESS
400 gate even when every conditional is exact. Section 9 plans for that from the start.

### 0.2 Decisions taken, and what they replace

The left column is the earlier draft of this specification, kept here because each entry
records a design decision and its reason, not because that draft still exists.

| Earlier draft | This revision |
|---|---|
| Priors, bases, covariance prior and tree prior left to the implementer ("approximately six columns", "a documented kernel") | Every modeling choice pinned with a default and a normalization (sections 4.4 to 4.7); defaults sanity-checked in closed form (appendix A.4) |
| "Implement GROW and PRUNE" from scratch for both forests; `bartz` treated as unusable | One vector-leaf engine for both forests that reuses `bartz`'s `_propose_moves`, `complete_ratio` and `apply_*` internals, so the tree prior is `bartz`'s by construction (section 5.2); feasibility proven |
| A second, full CPU reference *sampler* | A small dense **oracle** for the closed forms plus exact enumerated tree posteriors; no second sampler to keep in sync |
| Enumerated-tree validation described in prose | Executable oracle, fixture, thresholds, measured power and runtime (section 8.3) |
| No use of the repository's strongest tool | Geweke joint-distribution test with closed-form marginals that hold for *any* tree prior (section 8.5) |
| No nested-model check | `bartz` scalar and multivariate BART as independent oracles for the complete kernel (section 8.4) |
| R-hat/ESS gate on every point of dense grids | A fixed list of about 45 gated quantities, dense grids reported; escalation ladder and stop rules (section 9) |
| 200-replication SBC as a release requirement | Geweke is the primary joint test; SBC is a 100-replication scheduled study with a mutation arm (section 8.7) |
| Tests under `tests/dose/` | `tests/test_dose_*.py`: `.Rbuildignore` only excludes `^tests/test_.*\.py$`, and CI runs every `slow` test, so long studies live in `benchmarks/` |
| No release control, no working agreements | Preview flag as for `negbin`; worktree, status file and hazards learned in this repository (sections 2 and appendix B) |
| Widths of float types, memory and cost left open | Engine in float64 with whitened coordinates; trace memory formula and guard; factored statistics (sections 5.1, 5.5, 6.5) |

## 1. Deliverable and scope

Implement `LongBetDose` in Python and R: a Gaussian panel model for an absorbing
treatment with a fixed positive dose, fitted on first differences, with a prognostic
forest, a treatment forest whose leaves hold a smooth dose-by-exposure surface, and a
within-unit error covariance. Deliver the estimands of section 3, diagnostics, archives,
the verification evidence of sections 8 to 10 and documentation. Preserve the behavior
and random-key schedules of `LongBet`, `LongBetMulti` and `LongBetEncourage`.

Decisions for the first release:

| Area | Required behavior |
|---|---|
| Public API | New classes `LongBetDose`, `LongBetDoseConfig`. Do not overload `outcome=` (that names the response distribution) and do not relax `_check_absorbing()`. |
| Response | Continuous, Gaussian errors. Binary, ordinal, count and multiple outcomes are rejected. |
| Panel | Balanced, regularly spaced, no missing cells, `T >= 2`. `T = 2` is the same code path with `q = 1`. At most 20 periods until memory and time scaling are measured (section 6.5). |
| Treatment | Absorbing adoption at `G_i` in `2..T` or never; dose `D_i > 0` fixed after adoption; `D_i = 0` for never treated. A never-treated group is required. Treatment in period 1, reversals, time-varying dose and negative dose are rejected. |
| Covariates | Baseline (time-invariant, pre-treatment) only. |
| Likelihood | One `q = T - 1` vector of first differences per unit, multivariate normal with a common covariance `V` (section 4.1). No unit intercept: it cancels. Never stack cohort-specific differences as if they were independent data. |
| Covariance | `unstructured` (default, inverse Wishart), `difference_iid`, `fixed` (tests). |
| Treatment surface | Forest with vector leaves: cubic B-spline in dose times an exposure grid with a Gaussian-process prior (sections 4.4, 4.5). Trees split on baseline covariates, optionally on cohort, never on dose or exposure. No multiplicative trajectory, hence no scale ridge and no ridge move. |
| Prognostic surface | Forest with a `q`-vector per leaf, splitting on baseline covariates only. Adding dose or cohort there would change the parallel-trends restriction. |
| Engine | One vector-leaf engine for both forests, built on `bartz` 0.12.1 internals for proposals and bookkeeping (section 5). float64. |
| Sweep moves | GROW/PRUNE and leaf-parent CHANGE (phase 3); internal CHANGE (phase 6); REGROW and tempering only if the mixing pilot shows they are needed (section 9.4). |
| Hyperparameters | Fixed at documented defaults. Hyperpriors are deferred (section 12.3). |
| Estimands | Separate, labeled objects for effects on recipients at their own dose (parallel trends) and for cross-dose curves, derivatives and contrasts (strong parallel trends, declared by the user) (section 3). |
| Extrapolation | Doses outside the fitted knot range are rejected. Holes in dose support and unobserved cohort-exposure cells are flagged on every result. |
| Release control | Until phase 10, `LongBetDose.fit` raises unless a private flag is set (`longbet._dose_config.dose_enabled()`, R `longbet_enable_dose_preview()`), as `negbin` does. Flip the default only in the commit that records passing evidence for all three gates. |

Rejected inputs raise errors that name the offending unit or period. Not-yet-treated
controls without a never-treated group, missing cells, irregular grids and time-varying
covariates are deferred (section 12.3).

## 2. Working agreements

### 2.1 Workspace

- The main working tree holds **uncommitted count work** and count protocols are running
  from it (`benchmarks/count/protocol_reduced.py`). Do not stash, reset, reformat or
  commit those changes, and do not edit shared modules in that tree while those jobs run.
- Work in a separate worktree and environment:
  `git worktree add ../longbet-jax-dose -b feature/continuous-dose`, then a fresh
  virtualenv there with `pip install -e ".[dev]"`. Ask the maintainer whether to branch
  from `main` or to wait until the count work is committed; both touch `__init__.py`,
  `_io.py`, the contract YAMLs, `NAMESPACE`, `.Rbuildignore` and `README.md`.
- The host is shared (16 cores, 28 GiB, CPU only when this was written). Run long jobs
  with `nice -n 19`, cap threads
  (`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"`), and
  check `free -g` first. `/tmp` is a 15 GiB `tmpfs`: every byte written there is resident
  memory. Write benchmark output to disk-backed storage.
- Pin `bartz==0.12.1` (already pinned). The engine imports private `bartz` modules, which
  this repository already does in `_change_move.py`, `_regrow_move.py` and `_scales.py`.
  `tests/test_dose_bartz_api.py` (section 8.4) catches upstream drift on a version bump.

### 2.2 Evidence

- Keep `benchmarks/continuous_dose/implementation_status.md` in the format of
  `benchmarks/count/implementation_status.md`: a phase table, then every command run with
  its result. Nothing is inferred from a file existing; quote only what a completed
  command printed. Record failures and lost runs too.
- Before editing, record the starting commit, `pip freeze`, the R `sessionInfo()` if R is
  used, the device, and the result of `pytest -m "not slow" -q` on the untouched tree.
- Measure before launching anything long: time 50 iterations, extrapolate, and write the
  estimate in the status file. A harness that calls an un-jitted sweep in a Python loop
  leaked 26 MB per iteration in this repository; jit the loop body and watch memory for
  the first minute.

### 2.3 Repository rules that apply here

From `CONTRIBUTING.md`, all binding: one engine and two front doors (nothing statistical
is implemented in R); adding an option means editing the config, **both** contract YAMLs
(byte identical) and the R front door; chains start overdispersed; sweeps are jitted at
the boundary; `predict` rebuilds the design from a persisted spec; when touching a
conditional, run the Geweke test; diagnose every reported effect; plotting stays
optional. Match the surrounding code's naming, comment density and license headers.

## 3. Causal contract

### 3.1 Notation

Units `i = 1..N`, periods `t = 1..T`, baseline covariates `X_i`, adoption period `G_i` in
`{2..T}` or never (`inf` in Python; R maps `Inf`, `NA` and `0` to never treated and
documents it), dose `D_i`.

$$
Z_{it}=1\{t\ge G_i\},\qquad S_{it}=Z_{it}(t-G_i+1),\qquad A_{it}=Z_{it}D_i .
$$

`D_i` is stored for an eventually treated unit in its pre-treatment rows too; it is a
unit attribute, distinct from the active dose `A_it`. `exposure` always means the
treatment clock `S`, as elsewhere in the package.

### 3.2 Assumptions

Identification (the user's responsibility; the software records, never certifies):

- **A1 Conditional parallel trends.**
  $E[Y_{it}(0)-Y_{i,t-1}(0)\mid X_i,G_i,D_i]=m_t(X_i)$ for `t = 2..T`.
- **A2** No anticipation, consistency, no interference.
- **A3 Overlap.** Untreated units exist at every covariate profile and period used;
  positive doses are observed around every dose at which a curve is reported.
- **A4** Covariates are measured before treatment.
- **A5 Conditional strong parallel trends** (only for cross-dose statements):
  $E[Y_{it}(d)-Y_{it}(0)\mid X_i,G_i,D_i=d']$ does not depend on $d'$. Under A1 to A4
  alone, comparing recipients of different doses mixes the causal response with
  selection on gains [R4].

Estimation assumptions, kept separate from the above: Gaussian errors, a covariance
common to all units, the forest and smoothness priors.

### 3.3 Estimands

With posterior draws of the unit surface $f_i(s,d)$ (section 4.2):

| Method | Definition | Needs |
|---|---|---|
| `att(by=...)` | $\sum_{(i,t)\in\mathcal C} w_{it}\,f_i(S_{it},D_i)$ over treated cells, by exposure, cohort, dose bin or overall. Each unit is evaluated at its **own** dose. A dose-bin average is not a pointwise effect at the bin midpoint. | A1 to A4 |
| `local_att_curve(dose_grid, bandwidth)` | $\theta_h(d;s)=\sum_i w_i(d;h)\,f_i(s,D_i)$, $w_i\propto K\{(D_i-d)/h\}$ over units whose cohort reaches exposure `s`: a kernel-smoothed $\mathrm{ATT}(D\mid D)$ around `d`. The weights change with `d`, so its slope contains a composition term and is **not** a causal response. Persist kernel, bandwidth, indices and weights; report two bandwidths. | A1 to A4 |
| `dose_response(dose_grid, ...)` | $\mu_\pi(d;s)=\sum_i \pi_i\,f_i(s,d)$ for weights $\pi$ over a **fixed** target population, the same at every dose. Default: treated units whose cohort reaches `s`, equal weights. | A5 to be causal; otherwise labeled descriptive |
| `acr(dose_grid, ...)` | $\partial\mu_\pi(d;s)/\partial d$, analytic (section 4.4). | A5 |
| `contrast(d_low, d_high, ...)` | $\mu_\pi(d_{high};s)-\mu_\pi(d_{low};s)$, same population on both sides. | A5 |
| `unit_effects()` | Per treated cell summaries of $f_i(S_{it},D_i)$, blocked as in `LongBet.predict`. | A1 to A4 |
| `pretrend_residuals(by="dose_bin")` | Posterior mean residual of pre-treatment differences by eventual dose bin and period. Descriptive; it cannot validate post-treatment trends. | none |

`dose_response`, `acr` and `contrast` take
`identification="parallel_trends" | "strong_parallel_trends"`. With the default
`"parallel_trends"` they still compute, but the result is labeled
`causal_interpretation="descriptive: requires strong parallel trends"` and `acr` warns.
More data never upgrades the label.

Every result carries metadata: estimand name, identification declaration, target
population and a hash of its weights, cohorts and exposures used, dose scale, support
flags (section 3.4), response scale, finite-population interpretation, draw layout
`(chain, draw, ...)`. Axis meaning is never inferred from array lengths.

Finite-population targets are the default. A superpopulation target needs uncertainty
about the covariate distribution (for example a Bayesian bootstrap over units); that
layer is deferred, and the benchmark truth must match whichever target is reported.

### 3.4 Support

- `exposure_support[g, s]` is true when `g + s - 1 <= T`. Cells beyond it are
  extrapolations through the exposure prior; aggregate only supported cells by default.
- `dose_support(d)` is the number of treated units within `+/- dose_support_halfwidth`
  (default 5% of the dose range). Points below `min_dose_support` (default 5 units) are
  flagged `supported=False` and returned as `NaN` in summary tables unless
  `allow_unsupported=True`. A hole inside the marginal dose range is flagged this way.

## 4. Statistical model

### 4.1 Scaling and the difference likelihood

Let `q = T - 1` and, once per unit,

$$
v_i=(Y_{i2}-Y_{i1},\ldots,Y_{iT}-Y_{i,T-1})^\top .
$$

Difference `j` pairs periods `j` and `j + 1`. With `standardize=True` (default) compute,
over the cells that are untreated in both periods (`Z_{i,j+1} = 0`),
$c_j=\mathrm{mean}(v_{ij})$ and one pooled
$s_y=\{\mathrm{mean}(v_{ij}-c_j)^2\}^{1/2}$, and fit $\tilde v_{ij}=(v_{ij}-c_j)/s_y$.
Effects and derivatives are multiplied by $s_y$ on output, covariances by $s_y^2$; no
centering constant is ever added to an effect. `standardize=False` sets `c = 0`,
`s_y = 1` and is required by the Geweke and SBC tests, whose prior simulators know
nothing about data-dependent scaling.

$$
\tilde v_i\sim N_q(m_i+\Delta F_i,\;V),\qquad m_i=\sum_{j=1}^{M_m}a_{j,\ell_j(X_i)},
$$

with $a\in\mathbb R^q$ a prognostic leaf. A time-invariant unit intercept cancels in
$v_i$; do not add one back.

### 4.2 Treatment contribution and the cohort designs

The level effect is $F_{it}=Z_{it}\,f_i(S_{it},D_i)$ with $F_{i1}=0$, and

$$
f_i(s,d)=\sum_{j=1}^{M_f}\tilde h(d)^\top U_{j,\ell_j(X_i,G_i)}\,\tilde r(s),
$$

where a treatment leaf holds a **whitened** coefficient matrix $U\in\mathbb R^{K\times L}$
(dose index first), $\tilde h$ and $\tilde r$ are the whitened bases of section 4.5, and
`L` is the largest observed exposure. The model needs $\Delta F_{ij}=F_{i,j+1}-F_{ij}$,
the **change** in the level effect. Reporting must rebuild `F`; returning $\Delta F$ as
an effect is a bug the tests look for.

For cohort `g` let $E_g\in\mathbb R^{q\times L}$ have, in row `j`, `+1` in column
$s=j+2-g$ when $j\ge g-1$ and `-1` in column $s-1$ when $j\ge g$
(`dose_reference.exposure_design`). Then $\Delta F_i=\tilde E_{g_i}U^\top\tilde h(D_i)$
with $\tilde E_g=E_gL_s$, and with the row-major flattening $\theta=\mathrm{vec}_r(U)$,

$$
\Delta F_i=H_i\theta,\qquad H_i=\tilde h(D_i)^\top\otimes\tilde E_{g_i}\in\mathbb R^{q\times KL}.
$$

Never-treated units have $H_i=0$ exactly. The cumulative sum of $E_g f$ is the level
effect at every period (oracle check 1).

The prognostic forest is the same object with `K = 1`, $\tilde h\equiv1$, a single
pseudo-cohort with $\tilde E=L_m$ (section 4.5), `L = q` and every unit contributing.
One engine serves both forests.

### 4.3 Within-unit covariance

Even independent level errors give correlated differences: with $A$ the `q x T`
difference matrix, $AA^\top$ is tridiagonal with 2 on the diagonal and -1 beside it. Let
$\bar\Omega=AA^\top/2$ (unit diagonal).

| `residual_covariance` | Model and conditional |
|---|---|
| `"unstructured"` (default) | $V\sim\mathrm{IW}(\nu_0,S_0)$, density $\propto\lvert V\rvert^{-(\nu_0+q+1)/2}\exp\{-\mathrm{tr}(S_0V^{-1})/2\}$, which is `scipy.stats.invwishart(df=nu0, scale=S0)`. Defaults $\nu_0=q+3$, $S_0=(\nu_0-q-1)\bar\Omega=2\bar\Omega$, so $E[V]=\bar\Omega$: each standardized difference has prior mean variance 1, and at `q = 1` this is exactly the package's `IG(2, 1)`. Conditional: $\mathrm{IW}(\nu_0+N,\;S_0+\sum_ie_ie_i^\top)$. Draw the precision with `bartz.mcmcstep._step._sample_wishart_bartlett(key, df, scale_inv)`, whose third argument is the inverse-Wishart scale. |
| `"difference_iid"` | $V=\sigma^2\bar\Omega$, $\sigma^2\sim\mathrm{IG}(2,1)$; conditional $\mathrm{IG}(2+Nq/2,\;1+\tfrac12\sum_ie_i^\top\bar\Omega^{-1}e_i)$. Equivalent to unit fixed effects with independent level errors. |
| `"fixed"` | `V` supplied; tests only. |

The covariance likelihood contains $-\tfrac N2\log\lvert V\rvert$. Whenever `V` changes,
every cached $Q_g=\tilde E_g^\top P\tilde E_g$ (section 5.1) is stale. `unstructured` is
robust to serial correlation, not to heteroskedasticity across units or heavy tails;
section 10 stress-tests both.

### 4.4 Dose basis

- Clamped cubic B-splines, `K = dose_basis_dim = 6` columns, hence two interior knots at
  the 1/3 and 2/3 quantiles of the **training** positive doses; boundary knots at their
  minimum and maximum. Persist knots, degree and bounds; `predict` replays them and
  rejects doses outside them (`extrapolate=False`; the oracle raises on `NaN`).
- A B-spline basis already spans the constants. Do not append an intercept. Test rank,
  partition of unity and conditioning.
- The treatment contribution is exactly zero when `Z = 0`. The positive-dose curve is
  **not** forced toward zero as `d -> 0`: an extensive-margin jump is allowed. A
  continuous-through-zero variant is deferred.
- Derivatives are analytic in the dose variable of the original units,
  $\partial f/\partial d=\tilde h'(d)^\top U\tilde r(s)$ with
  $\tilde h'=L_d^\top h'$ from `BSpline.derivative` (oracle `bspline_basis(deriv=1)`).
  Multiply by $s_y$ only. Never differentiate tree routing.

### 4.5 Priors, normalization and whitening

Physical coefficients $W=\mathrm{unvec}(\,(L_d\otimes L_s)\,\theta)$ have the prior
$\mathrm{vec}_r(W)\sim N\{0,\ (\tau_f^2/M_f)\,(C_d\otimes K_s)\}$. Row-major flattening
pairs with $C_d\otimes K_s$; the column-major convention would reverse the order (oracle
check 3 tests this). The engine never sees $C$: it works with

$$
\tilde h(d)=L_d^\top h(d),\qquad \tilde r(s)=L_s^\top e_s,\qquad
\theta\sim N(0,\ \kappa^{-1}I),\quad \kappa=M/\tau^2 .
$$

- **Dose smoothness.** $C_d=c\,(I_K+\lambda_dD_2^\top D_2)^{-1}$ with $D_2$ the second
  difference matrix, `dose_penalty` $\lambda_d=4$, and $c$ set so that
  $\mathrm{mean}_i\,h(D_i)^\top C_dh(D_i)=1$ over treated training units. Constant and
  linear dose effects are penalized only by the ridge.
- **Exposure kernel.** `longbet._gp.build_kernel_matrix(s = 1..L, sig_knl, lambda_knl,
  kernel_type, sigma_m, gp_constant_mean)` with the package defaults (`se`, lengthscale
  1, constant mean on), divided by its mean diagonal. Adjacent exposures then correlate
  about 0.8 a priori. Build and factor in float64 NumPy at initialization, as `_gp.py`
  does; never form an inverse kernel.
- **Prognostic leaves.** $a=L_m\tilde u$, $L_mL_m^\top=\Sigma_m=(1-\rho_m)I_q+\rho_mJ_q$,
  `rho_pr = 0.5`: half of a covariate-driven trend is persistent a priori. Common period
  shocks are removed by $c_j$.
- **Scales.** `tau_trt = 1.0`, `tau_pr = 1.5` in standardized units, the package's
  `3/(3 sqrt(m))` and `3/(2 sqrt(m))` summed over trees.

With these normalizations the prior variance of $f_i(s,d)$ averages $\tau_f^2$. In the
closed-form root-only model (appendix A.4) the defaults cover at 0.96 for smooth curves
and 0.94 for a full sine period; $\lambda_d=16$ over-smooths the sine (0.88). Candidates
for development tuning: `K` in {5, 6, 8}, $\lambda_d$ in {1, 4, 16}, chosen on
development replications only and frozen before confirmation.

### 4.6 Forests and the tree prior

The tree prior is `bartz`'s, unchanged: a node at depth `d` (root 0) is nonterminal with
probability $\alpha/(1+d)^\beta$ **if a rule is available**, the variable is uniform
over variables with an available cut given the ancestors, the cut uniform over that
variable's available cuts, and the last heap level cannot split. Binned predictors take
values `0..max_split[v]`; rule `(v, c)`, `c` in `1..max_split[v]`, sends `x_v >= c`
right. This reading was checked against `bartz` itself (appendix A.2).

Count thresholds are **not** part of the prior. `min_units_per_leaf` vetoes moves, which
restricts the support to trees whose leaves all hold enough *contributing* units
(treated units for the treatment forest, all units for the prognostic one) and
renormalizes; the oracle implements exactly this. Count units, never panel rows. Prior
simulators (Geweke tree sizes, SBC) must run with the thresholds off.

### 4.7 Defaults

| `LongBetDoseConfig` field | Default | Note |
|---|---|---|
| `num_chains`, `num_burnin`, `num_sweeps`, `n_skip` | 4, 2000, 1000, 1 | as `LongBetConfig` |
| `num_trees_pr`, `num_trees_trt` | 30, 50 | development defaults; freeze after section 9.4 |
| `max_depth_pr`, `max_depth_trt` | 5, 4 | heaps of 32 and 16 nodes; bounds trace memory |
| `alpha_split_pr`, `beta_split_pr` | 0.95, 2.0 | package defaults |
| `alpha_split_trt`, `beta_split_trt` | 0.25, 3.0 | package defaults |
| `min_units_per_leaf_pr`, `min_units_per_leaf_trt` | 10, 10 | units, not rows |
| `num_cutpoints` | 100 | via `_design.quantile_block` |
| `dose_basis_dim`, `dose_degree`, `dose_penalty` | 6, 3, 4.0 | section 4.4 |
| `sig_knl`, `lambda_knl`, `kernel_type`, `sigma_m`, `gp_constant_mean` | package defaults | exposure kernel |
| `tau_pr`, `tau_trt`, `rho_pr` | 1.5, 1.0, 0.5 | section 4.5 |
| `residual_covariance`, `cov_prior_df_extra` | `"unstructured"`, 3 | $\nu_0=q+$ `cov_prior_df_extra` |
| `split_cohort_trt` | `False` | cohort as an `integer_grid_block` when on |
| `standardize` | `True` | |
| `dose_support_halfwidth`, `min_dose_support` | 0.05, 5 | section 3.4 |
| `init_prior_sweeps` | 0 | sweeps with `P = 0` to disperse forests; used in section 9 |
| `max_trace_gib` | 2.0 | section 6.5 |
| `random_seed`, `device`, `inner_loop_length` | 0, `"auto"`, `None` | as `LongBetConfig` |

Validate every field in `__post_init__` in the style of `LongBetConfig` (type, range,
message naming the field and value).

## 5. Sampler

### 5.1 Sufficient statistics, leaf posterior and integrated likelihood

For one tree, let $r_i^+$ be the full residual with that tree's contribution added back
and $P=V^{-1}$. A node containing the unit set $\mathcal N$ has

$$
A=\sum_{i\in\mathcal N}H_i^\top PH_i=\sum_gG_{\mathcal N,g}\otimes Q_g,\qquad
b=\sum_{i\in\mathcal N}\tilde h_i\otimes(\tilde E_{g_i}^\top Pr_i^+),
$$

$$
G_{\mathcal N,g}=\sum_{i\in\mathcal N,\,g_i=g}\tilde h_i\tilde h_i^\top,\qquad
Q_g=\tilde E_g^\top P\tilde E_g .
$$

Use this factored form (`dose_reference.node_stats_factored`); never materialize the
`p x p` information matrix per unit (`p = KL`), which the prototype does only for
brevity. `Q_g` is computed once per sweep and again after `V` changes.

$$
\theta\mid\cdot\sim N(\Lambda^{-1}b,\Lambda^{-1}),\quad\Lambda=\kappa I+A,\qquad
\log\mathrm{ml}=\tfrac p2\log\kappa-\tfrac12\log\lvert\Lambda\rvert+\tfrac12b^\top\Lambda^{-1}b .
$$

`log ml` is the log ratio between the leaf-integrated likelihood and the likelihood with
the leaf set to zero, so it is 0 for an empty node and additive over leaves; it equals a
brute-force ratio of multivariate normal densities to `1e-8` (oracle check 5). Keep the
determinant: the mutation arm without it is detected (appendix A.3). Use Cholesky solves,
$\theta=\Lambda^{-1}b+L_c^{-\top}z$ with $\Lambda=L_cL_c^\top$; `solve(L_c, z)` has the
wrong covariance whenever $\Lambda$ is far from isotropic (oracle check 6).

### 5.2 GROW and PRUNE on `bartz` internals

Follow `ideas/dose_engine_prototype.py::sweep`. Per forest and sweep:

1. `moves = bartz.mcmcstep._moves._propose_moves(keys, var_tree, split_tree,
   affluence_tree, max_split, blocked_vars, p_nonterminal[:half], None)`. `bartz`
   chooses the leaf to grow with probability proportional to `p_nonterminal`, not
   uniformly; that is inside `partial_ratio`. `p_nonterminal` is indexed by heap node.
2. `grown = apply_grow_to_indices(moves, leaf_indices, X)`: grows are pre-applied.
3. Count contributing units in the left and right nodes of every move (the parent is
   their sum). Set `lrt_affluent = (lrt_nodes < half) & lrt_growable` and, when
   `min_units_per_leaf` is set, `allowed &= all(counts[:2] >= min_units_per_leaf)`.
4. `moves = complete_ratio(moves, p_nonterminal)`.
5. **Copy the parent's coefficients into both children of every GROW** (`bartz`'s
   `adapt_leaf_trees_to_grow_indices`). Unused heap slots hold prior draws, not their
   parent's value; without this step the residual is silently corrupted (the prototype's
   first version violated the residual invariant by 7).
6. Sequentially over trees (`lax.scan`), with the running residual:
   `log_lk = ml(left) + ml(right) - ml(parent)`;
   `log_ratio = log_trans_prior_ratio + log_lk`, **negated for PRUNE**;
   `acc = allowed & (logu <= log_ratio)`; `to_prune = acc ^ grow`; move the units of the
   two children back to the parent where `to_prune`; redraw **every** leaf of the tree
   from section 5.1 given the final membership; where `to_prune`, mirror the parent's
   new coefficients into the two children; update the residual with the new fit.
7. `split_tree = apply_moves_to_split_trees(...)`,
   `affluence_tree = apply_moves_to_affluence_trees(...)`, `var_tree = moves.var_tree`.
   `var_tree` keeps stale entries at leaves: always read rules through `split_tree > 0`.

Unlike `bartz`, apply prunes to `leaf_indices` immediately (no deferred `to_prune`
state); the engine is not performance-bound there, and every later move then sees exact
memberships.

### 5.3 CHANGE moves

- **Leaf-parent CHANGE** (phase 3). Reuse `longbet._change_move._propose` unchanged (it
  depends only on the tree arrays). Accept with
  `logu <= sum_children(ml_new - ml_old) + sum_children(log1p(-pnt*g_new) - log1p(-pnt*g_old))`,
  veto on `min_units_per_leaf`, redraw the two children, update the residual.
- **Internal CHANGE** (phase 6). Reuse `_change_internal_move._propose` and
  `_change_internal_move.tree_log_prior`; re-route the subtree, sum `ml` over its leaves
  before and after, veto on any under-filled leaf or inadmissible descendant rule.
- Both keep the stationary law of section 8.3 unchanged; the enumerated tests run with
  each move switched on.

### 5.4 Sweep, state, initialization and chains

Order: prognostic forest (GROW/PRUNE, then CHANGE), treatment forest (same), covariance,
log likelihood for the trace. Each block conditions on the latest values of the others.

- **State** (an `eqx.Module`; declare chain axes with `bartz._jaxext.field(chains=CHAIN_AXIS)`
  so `longbet._state.chain_filter_spec` partitions it). Shared: binned `X`, `max_split`
  per forest, `h_w (N, K)`, `cohort_idx (N,)`, `E_w (C, q, L)`, `contributes (N,)`,
  `v (N, q)`, `L_m`. Per chain: for each forest `var_tree`, `split_tree`,
  `affluence_tree`, `leaf_indices (M, N)`, `coef (M, 2^depth, p)`; residual `R (N, q)`;
  `P` and `V`; `sigma2`; `temperature`; move counters; `log_likelihood`.
- **Invariant**, checked after every sweep in tests to `1e-9`:
  `R == v - prognostic_fit - treatment_increment_fit`, recomputed from trees and bases.
- **Initialization.** Forests are roots with zero coefficients, `R = v`. Each chain's `V`
  is drawn from its prior; `init_prior_sweeps > 0` additionally runs that many sweeps
  with `P = 0`, which samples forests from the prior.
- **Chains.** `jax.vmap` over the per-chain partition only, as `longbet_step` does.
- **Keys.** One `random.split` for the fixed blocks; every move added later takes
  `random.fold_in(key, tag)` with a new tag, so existing streams never shift.
- **Loop and trace.** Copy the structure of `_loop.py`: preallocated traces,
  `lax.while_loop`, traced loop bound, `inner_loop_length`, host callback. Retain per
  draw both forests' `var_tree`, `split_tree`, `coef`; `V`; `sigma2`; `log_likelihood`;
  per-sweep acceptance counters (burn-in included).

### 5.5 Numerics

- The engine's statistics, factorizations, residual and leaf draws are **float64** in the
  first implementation: parity with the oracle at `1e-9` is the cheapest bug detector
  there is, the residual is a running sum over thousands of sweeps, and at unit level the
  cost is negligible. Run the sweep under `enable_x64(True)`; the prototype exercised the
  `bartz` helpers and `_sample_wishart_bartlett` under x64. Store traces in float32.
  float32 is probably viable later: on a realistic node (`p = 54`, 600 units, whitened,
  $\kappa=50$) $\Lambda$ has condition number 26 and float32 moves the GROW ratio by at
  most 0.001 nats (appendix A.5). Validate it against the float64 engine before adopting
  it, and again at large `N`, where the condition number grows with the node size.
- Existing sweeps carry `@enable_x64(False)`; a global setting does not override them.
- Whitened coordinates keep $\Lambda\succeq\kappa I$. Never form $C^{-1}$ or $K_s^{-1}$.
- Static shapes: redraw all `2^depth` heap slots (empty slots draw from the prior and
  are never read). Padded or non-contributing units never enter a count, a statistic, the
  covariance update or a support report.
- `bartz` in float32 never accepts a GROW at `error_cov_inv = 1e-12`. Make `bartz`-side
  likelihoods flat with zero responses and precision `1e-4` (appendix A.2).

### 5.6 Tempering (only if section 9.4 asks for it)

Replace `P` by $\beta P$ in both forests; draw
$V\sim\mathrm{IW}(\nu_0+\beta N,\ S_0+\beta\sum e_ie_i^\top)$; swap on the untempered
energy $-\tfrac N2\log\lvert V\rvert-\tfrac12\sum e_i^\top Pe_i$. Tempering integrates the
**powered** likelihood; raising an integrated likelihood to a power is a different
calculation. Reuse `_tempering.ladder_temperatures` and the exchange bookkeeping. Ladders
are independent; diagnostics read cold replicas only.

## 6. Public interface

### 6.1 Python

```python
from longbet import LongBetDose, LongBetDoseConfig
from longbet._dose_config import dose_enabled

with dose_enabled():                                     # preview flag, section 1
    model = LongBetDose(LongBetDoseConfig(num_chains=4, random_seed=20260917))
    model.fit(y=y, x=x, dose=dose, adoption_time=g, t=t)   # or z=z instead of adoption_time

att = model.att(by="exposure")                           # own-dose ATT, parallel trends only
local = model.local_att_curve(dose_grid=d_grid, bandwidth=0.1, exposure=3)
curve = model.dose_response(dose_grid=d_grid, exposure=3,
                            identification="strong_parallel_trends")
slope = model.acr(dose_grid=d_grid, exposure=3, identification="strong_parallel_trends")
diff = model.contrast(0.3, 0.8, exposure=3, identification="strong_parallel_trends")
curve.mean, curve.lower, curve.upper, curve.draws        # draws: (chain, draw, dose)
curve.simultaneous_band(level=0.95)                      # max-|z| band over the grid
curve.meta; curve.support; curve.stability()             # ESS, R-hat, MCSE per reported number
model.diagnostics_                                       # acceptance by move and forest, V and log-lik ESS/R-hat
model.save("dose.npz"); model = LongBetDose.load("dose.npz")
```

Inputs: `y (N, T)`; `x (N, P)`; optional `x_trt (N, P_trt)`; `dose (N,)`;
exactly one of `adoption_time (N,)` (values of `t`, `inf` for never) or an absorbing
`z (N, T)`; `t (T,)` equally spaced. Validation (each with its own test and message):
shapes, finiteness, balanced and regular grid, `T >= 2`, no adoption in period 1,
absorbing `z`, dose positive if and only if ever treated, at least
`min_units_per_leaf_pr` never-treated units, at least `2 * min_units_per_leaf_trt`
treated units, `K` not larger than the number of distinct positive doses, the memory
guard of section 6.5.

Predictions reuse the blocked pattern of `_model._predict_impl`. The per-draw unit
coefficient $U_i=\sum_jU_{j,\ell_j(x_i,g_i)}$ is never stored for all units: accumulate
$\bar U=\sum_i\pi_iU_i$ per target population (a `(chain, draw, K, L)` array). Any dose
grid, exposure, derivative or contrast is then a contraction of $\bar U$; these compact
functional draws are saved even when unit-level draws are not.

`stability()` generalizes `_diagnostics.att_stability` to named quantities
`(chain, draw, n)` with `compute_ess`, `compute_rhat` and mean-ESS MCSE; tail ESS at 0.025
and 0.975; exact structural zeros tagged `structural_constant`, any other zero-variance
trace a failure; `reliable=None` and the "verdict withheld" wording preserved.

### 6.2 R

`R/dose.R`: `longbet_dose(y, x, dose, adoption_time = NULL, z = NULL, t = NULL, ...)`
marshals to Python; `dose_att()`, `dose_local_att()`, `dose_response()`, `dose_acr()`,
`dose_contrast()`, `dose_stability()`, `print`/`summary` methods and
`longbet_enable_dose_preview()`. Nothing statistical in R. Extend
`rehydrate_jax_model()` for class `longbet_dose`. Regenerate man pages with roxygen (a
missing ownership marker once made roxygen skip pages silently). `R CMD check --no-manual`
must stay at its current status; add `^ideas$` to `.Rbuildignore`.

### 6.3 Contract

Add a `dose_api` section to `contract/longbet-api.yaml` (arguments, R names, defaults,
return fields), copy it byte for byte to `inst/contract/`, and extend
`tests/test_contract.py` and `tests/testthat/test-contract.R` in both directions, as
`multi_api` does.

### 6.4 Archives

`.npz` with `allow_pickle=False`, metadata `model_kind="dose"`, `dose_schema_version=1`,
the design spec (`Design.to_dict()`), knots, bounds, scaling constants, cohort table,
whitening factors, prior constants and the `bartz` version. `LongBetDose.load` rejects any
other `model_kind`; add the symmetric rejection of `"dose"` to `_io.load_npz` and
`load_multi_npz`. Test round trips (every estimand identical) and a corruption matrix as
`tests/test_count_io.py` does.

### 6.5 Memory

Trace bytes are
`4 * chains * draws * (M_pr * 2^depth_pr * q + M_trt * 2^depth_trt * K * L)` plus small
terms. At the defaults: 0.09 GiB at `T = 2`, 0.80 GiB at `T = 12` (`q = 11`, `L = 9`),
1.63 GiB at `T = 20` and 2.49 GiB at `T = 30` — the 20-period cap and the 2 GiB guard are
the same decision. Estimate before allocating and raise an error that names the terms and
the remedies (`n_skip`, depth, trees, draws) above `max_trace_gib`. Report measured peak memory and time per
sweep in the status file at `N` in {1e3, 1e4, 1e5}.

## 7. Code layout and reuse

```text
src/longbet/_dose_config.py      config, validation, preview flag
src/longbet/_dose_input.py       panel validation, cohorts, exposure, differencing, scaling
src/longbet/_dose_basis.py       B-spline basis, priors, whitening, E_g, persistence
src/longbet/_vleaf.py            factored statistics, log ml, leaf draws, fits
src/longbet/_vleaf_moves.py      GROW/PRUNE on bartz internals, CHANGE, internal CHANGE
src/longbet/_dose_state.py       state, initialization, chain partition
src/longbet/_dose_step.py        one sweep: two forests, covariance, log likelihood
src/longbet/_dose_loop.py        driver and traces
src/longbet/_dose_estimands.py   ATT, curves, derivative, contrasts, support, bands
src/longbet/_dose_model.py       LongBetDose, result containers, diagnostics
src/longbet/_dose_io.py          archives
R/dose.R
tests/_dose_reference.py         moved from ideas/, unchanged
tests/test_dose_*.py             section 8
tests/testthat/test-dose.R
benchmarks/continuous_dose/      status file, convergence, SBC, comparison (sections 9, 10)
docs/continuous_treatment.md
```

Reuse, do not duplicate: `_design.quantile_block`, `integer_grid_block`, `Design`;
`_gp.build_kernel_matrix`, `kernel_cholesky`; `_state.chain_filter_spec`;
`_change_move._propose`, `_change_internal_move._propose` and `tree_log_prior`;
`_diagnostics.compute_ess`, `compute_rhat`; `_summary.BlockAccumulator`,
`choose_block_size`; `_model.resolve_device`; `_tempering.ladder_temperatures`;
`bartz.grove.traverse_forest` for routing new units.

## 8. Verification for gate 1

Conventions. Deterministic algebra: relative error at most `1e-9` in float64 on
well-conditioned fixtures. Stochastic comparisons: z-scores with a Monte Carlo standard
error from the **effective** sample size, `|z| <= 4`. Every comparative test asserts that
it is **not vacuous** (the arms differ, the mutation changes the answer). Build each
mutation arm from a **new function object**: `jax.jit` caches traces by the identity of
the wrapped function, so re-wrapping the same function silently reuses the unpatched
trace (reproduced while writing this plan: three arms, identical to every digit; see
`run_mutation_arms`). Budgets: all fast dose tests under 2 minutes, all `slow` ones
under 15 minutes on 4 cores; CI runs both. Anything longer is a benchmark script.

### 8.1 Deterministic: `test_dose_inputs.py`, `test_dose_design.py`, `test_dose_basis.py`

- One test per rejection of section 6.1; consistency of `adoption_time`, `z`, `S`, `A`.
- `E_g`: rows before adoption are zero; cumulative sums equal level effects at every
  horizon and cohort; never-treated designs are exactly zero.
- Basis: partition of unity, rank `K`, derivative against central differences (`1e-5`),
  right boundary included, extrapolation rejected, persistence through save/load.
- Whitening: `kron(L_d, L_s)` factors `kron(C_d, K_s)`; physical and whitened surfaces
  agree; both normalizations equal 1.
- Invariances, on a complete small fit with a fixed key: adding a unit-specific constant
  to every period leaves all draws **bitwise** identical; `y -> a*y + b` with
  `standardize=True` leaves standardized draws bitwise identical and scales effects by
  `a`; permuting units permutes unit-level output (`1e-10`).
- Estimands: weights sum to one; the population is identical on both sides of a
  contrast; a dose-bin ATT differs from a point on the standardized curve on a fixture
  built to separate them; support flags fire for a hole in the dose range and for
  unobserved cohort-exposure cells; `acr` equals a finite difference of `dose_response`.

### 8.2 Algebra: `test_dose_leaf_algebra.py`, `test_dose_fixed_tree_gibbs.py`

- Engine against oracle: factored statistics, `log ml`, posterior mean and covariance,
  empty nodes, non-contributing units, several cohorts, non-diagonal `V`.
- Leaf draws: sample covariance against $\Lambda^{-1}$ on an informative node, where the
  wrong Cholesky orientation fails (oracle check 6 gives the fixture).
- Covariance conditionals against `scipy.stats.invwishart` moments and the inverse-gamma
  form; determinant and scale transformations under `standardize`.
- Fixed trees: with moves disabled and arbitrary partitions installed in both forests,
  long-run means and covariances of all coefficients match
  `dose_reference.joint_gaussian_posterior` (`|z| <= 4`, relative covariance error at
  most 0.1). This catches wrong backfitting that per-leaf checks miss.
- Two periods, one stump per forest: the same comparison, against ordinary Bayesian
  linear regression.

### 8.3 Exact tree space: `test_dose_tree_posterior.py` (`slow`)

Fixture: `dose_reference._fixture(n=14, T=4, K=4)`, two cohorts, three non-contributing
units, predictors with `max_split = [2, 1]`, depth 3 (22 trees), signal in the data, fixed
`V`. Compare the visited structures with `exact_tree_posterior` and
`exact_forest_posterior`:

| Case | Sweeps | Pass if |
|---|---|---|
| one tree | 60,000 | TV at most 0.03; `\|z\| <= 4` for every tree with exact mass at least 0.01; posterior-mean fit within `1e-2` of the exact mixture |
| one tree, `min_units_per_leaf = 3` | 60,000 | same; vetoed trees never visited |
| two trees, leaves integrated jointly (484 forests) | 200,000 | TV at most 0.05 |
| flat likelihood | 60,000 | TV to the prior at most 0.03 |
| each of the above with CHANGE, then internal CHANGE, switched on | | unchanged targets |
| mutation arms (no determinant; PRUNE ratio not negated) | 60,000 | TV at least 0.08 **and** at least 5 times the correct arm |

Measured with the prototype: 0.010, 0.008, 0.012 for the correct kernels, 0.136 and 0.369
for two mutants; 6 s and 26 s of wall time. Also assert that the JAX
`_change_internal_move.tree_log_prior` equals `dose_reference.tree_log_prior` on all 22
trees.

### 8.4 `bartz` equivalence: `test_dose_bartz_equivalence.py` (`slow`), `test_dose_bartz_api.py`

The engine nests two models that `bartz` samples with a mature, independent
implementation. On `N = 200`, same binned `X`, tree prior and fixed error precision, four
chains of 5,000 draws each, GROW/PRUNE only:

- `q = 1`, `K = 1`, all units contributing: equals scalar BART with leaf precision
  $\kappa$.
- `q = 3`, identity design: equals multivariate BART with
  `leaf_prior_cov_inv = kappa * inv(Sigma_m)` and a dense fixed precision. Confirmed
  runnable: `bartz` accepts a dense `(k, k)` leaf prior precision together with a frozen
  dense `Wishart` value, and its `leaf_tree` is then `(trees, k, tree_size)` (appendix A.6).

Compare unit-level posterior means and standard deviations (`|z| <= 4`, non-vacuity: the
fits are not constant) and the distribution of leaves per tree (chi-square, `p > 0.001`).
`test_dose_bartz_api.py` asserts the signatures and return fields of every private
`bartz` symbol the engine imports.

### 8.5 Geweke: `test_dose_geweke.py` (`slow`)

Successive-conditional simulator as in `tests/test_geweke.py`: draw data given
parameters, run one full sweep, repeat; `standardize=False`, thresholds off, small panel
(`N = 12`, `T = 4`, two cohorts, `K = 4`, 2 + 2 trees), jitted loop body. Closed-form
marginals that hold whatever the tree prior does:

- $\sum_jU_{j,\ell_j(x_0)}\sim N(0,\tau_f^2I_p)$ at a fixed covariate point, and the
  prognostic analogue $N(0,\tau_m^2I_q)$ in whitened coordinates: means, variances, and
  one cross-moment each.
- $P=V^{-1}\sim\mathrm{Wishart}(\nu_0,S_0^{-1})$: $E[P]=\nu_0S_0^{-1}$ and the mean of
  $\log\lvert V\rvert$; for `difference_iid` the inverse-gamma moments.
- Leaves per tree against `dose_reference.sample_tree_prior`.

All `|z| <= 4` with ESS-based standard errors. Mutation arms, each proven to move a
statistic by more than 6 standard errors of the **correct** arm: prior scale matrix
omitted from the covariance conditional; determinant dropped; stale `Q_g` after the
covariance update.

### 8.6 Recovery smoke tests: `test_dose_recovery.py` (`slow`)

Loose, single-seed guards, not evidence of performance: two-period
`F = 1 - exp(-3 D)` with `N = 500` (integrated RMSE at most 0.25, at least 80% of 41 grid
points covered); `T = 8` staggered with a nonseparable surface (ATT by exposure inside
its intervals at 90% of exposures, reported effects are levels and not increments); a
null effect; a two-period binary-dose panel where `LongBetDose` with `K = 1` and
`LongBet` agree within posterior uncertainty **without** level selection and disagree
with it (the experiment of appendix A.1 as a regression test).

### 8.7 SBC: `benchmarks/continuous_dose/sbc.py`

A computation check under the generative model, not frequentist coverage [R7, R8]. Fixed
design (`N = 80`, `T = 4`), full prior simulator (trees from `sample_tree_prior`,
coefficients, covariance), `standardize=False`, thresholds off, 100 replications,
ranks from 99 thinned draws per fit with randomized ties. Quantities: `f` at three fixed
`(x, s, d)`, one derivative, one contrast, one prognostic value, $\log\lvert V\rvert$,
$\mathrm{tr}V$ and the joint log likelihood. Decide with simultaneous rank-ECDF envelopes;
one unadjusted `p < 0.05` among many is not a failure. A prior-only mutant must be
flagged. Measure one fit before launching.

### 8.8 Existing models

`pytest -m "not slow"` and `-m slow` on the whole suite pass unchanged, old archives load,
and `LongBet.load` refuses a dose archive.

## 9. Convergence protocol for gate 2

### 9.1 Gated quantities

Per fit, fixed before any run: `att` by exposure and overall; `dose_response` at the
10/30/50/70/90% dose quantiles at the first, middle and last supported exposures; `acr`
at the three interior quantiles at the same exposures; two scenario-defined subgroup
contrasts; $\log\lvert V\rvert$, $\mathrm{tr}V$, log likelihood, mean leaves per tree in
each forest. About 45 numbers. Dense grids are reported with the same diagnostics but do
not gate: neighboring grid points are near-duplicates, and the maximum of hundreds of
R-hat values computed from 4,000 draws exceeds 1.01 by chance far more often than any
single one does.

### 9.2 Per-fit gate

All gated quantities finite; rank R-hat at most 1.01; bulk ESS at least 400; tail ESS at
0.025 and 0.975 at least 400; mean MCSE at most 0.05 posterior SD (mean ESS, not bulk);
no unexplained constant trace. Report acceptance by move and forest, per-chain means and
burn-in traces.

### 9.3 Scenarios and decision

| Scenario | Design |
|---|---|
| C1 | two periods, `N = 1000`, 20 covariates, homogeneous dose curve (D2) |
| C2 | two periods, `N = 1000`, dose-by-covariate heterogeneity with a step (D3) |
| C3 | `T = 12`, cohorts 4, 7, 10, nonseparable dynamics (D4) |
| C4 | `N = 200`, `T = 6`, weak signal |

Three data seeds each. Escalation: E0 defaults; E1 `2 x` burn-in and draws; E2 `4 x`; E3
the moves or tempering chosen in section 9.4 at E1. A failed diagnostic triggers the next
level, never a new seed. Gate 2 passes when every scenario passes on all three seeds at
one common level; that level becomes the documented default budget, and "defaults
insufficient" is reported if it is not E0. Also run C2 and C3 once with
`init_prior_sweeps = 25` and eight chains; the gate must still pass.

### 9.4 Mixing pilot (phase 6, before anything else in this section)

Run C2 and C3 at E0 twice: with GROW/PRUNE and leaf-parent CHANGE only, and with internal
CHANGE added. Record R-hat and ESS of the gated quantities, per-chain means, acceptance
by move, and the decoded splits of disagreeing chains. Then, in this order and only as
far as needed: more treatment trees (the package's measured fix), REGROW restricted to
`p <= 16` or a coarse cut grid (it costs one `p x p` factorization per candidate cut),
tempering (section 5.6). Every added move reruns sections 8.3 and 8.5. Freeze
`num_trees_*` and the move set here, before section 9.3 and long before section 10.

## 10. Comparative study for gate 3

### 10.1 References

| Method | Notes |
|---|---|
| Native `contdid`, CCK dose estimator | Two periods, no staggering: the primary reference. Pin the GitHub commit and `npiv`; record `sessionInfo()`. Set every argument explicitly, because the documentation disagrees with itself about defaults: `target_parameter`, `aggregation="dose"`, `treatment_type="continuous"`, `dose_est_method="cck"`, `control_group="nevertreated"`, `dvals` = the common grid, `bstrap=TRUE`, `cband`, `biters >= 999`, `gname = 0` for never treated. Reproduce the package's own `simulate_contdid_data()` example first. It has no covariates. |
| Tuned dose-spline DiD | `contdid` parametric B-spline or an equivalent; degree and knots tuned on development data only. |
| Bayesian dose spline | This engine with `max_depth_trt = 1`: same likelihood and covariance, no covariate splits. Isolates what the forests add. |
| Covariate-adjusted comparator | Cross-fitted untreated-trend residualization followed by the dose estimator, for randomized-dose designs. Named as such, never as the published estimator. |
| Panel adapter | Cohort-period long differences into the validated two-period estimator, common explicit weights, units resampled jointly. Validated to reduce to native CCK with two periods; labeled an adapter. |
| `LongBetDose` and ablations | full; no cohort pooling; `difference_iid`. |

Every method gets the same units, outcomes, covariates where supported, controls, dose
region, grid and weights. A method that cannot produce a target is `not_applicable`, not
a loss. Compare pointwise with pointwise and simultaneous with simultaneous bands.

### 10.2 Data-generating processes

A pure `generate(seed, config)` returns observed data and a separate truth object;
fitters never see truth; the evaluator computes losses after predictions are saved.
Levels follow $Y_{it}=\alpha_i+\lambda_t+t\,q(X_i)+F_{it}+e_{it}$ with $\alpha_i$ free to
correlate with covariates **and with dose** (level selection, which the difference
likelihood must absorb). Positive doses on `[0.1, 1.0]`; primary grid: 41 points on
`[0.15, 0.95]`; boundaries reported separately.

| Family | Definition | Role |
|---|---|---|
| D0 | `F = 0`; `F = 0.8 Z D`; with and without covariates | sanity, no invented benefit |
| D1 | two periods, `F = Z (1 - exp(-3D))`; a cubic favorable to splines | control; need not win |
| D2 | two periods, `N` in {500, 1000}, 20 uniform covariates, 40% never treated, dose independent of `X`, $q(x)=c\{\sin(\pi x_1x_2)+2(x_3-0.5)^2+x_4+0.5x_5\}$ centered by an independent population mean, `F = Z (1 - exp(-3D))` | **primary** |
| D3 | as D2 with $F=Z\{a(X)(1-e^{-3D})+b(X)D^2\}$, $a=0.6+0.8\cdot1\{x_1>0.5,x_2>0.5\}$, $b=0.2(x_3-0.5)$; and a smooth logistic version. Marginal curve $0.8(1-e^{-3d})$ | **primary** |
| D4 | `T = 12`, cohorts 4, 7, 10, 40% never treated, $F=Z\{a(X)(1-e^{-Ds/2})+b(X)D(1-e^{-s/3})\}$ | secondary |
| D5 | treatment and dose depend on `X` with strict overlap; untreated trends conditionally common | secondary |
| D6 | two periods, $E[\theta\mid D=d]=1+d$, effect $\theta d$: $\mathrm{ATT}(d\mid d)=d+d^2$, slope $1+2d$, causal response $1+d$. The API must not call the slope causal, at any sample size | identification guard |
| D7 | thin overlap, a dose hole, heteroskedastic units, serially correlated and heavy-tailed errors, a trend violation | stress; not all truths identified |

Development grid for D2 and D3: `c` in {0.5, 1, 2}, level-error SD in {0.5, 1}; freeze
two cells per family in the protocol.

### 10.3 Metrics, protocol and decision

Primary loss per replication: weighted integrated squared error of the matched curve on
the common grid. Report its mean and root, bias, pointwise RMSE, 95% coverage, width and
simultaneous coverage (Bayesian band: posterior quantile of the maximum standardized
deviation over exactly the dimensions named in the claim). Secondary: contrasts, subgroup
effects, fit and diagnostic pass rates, compile time, synchronized run time, memory.

1. 30 to 50 development replications per family for debugging and tuning.
2. Freeze `protocol.yaml` from `ideas/BENCHMARK_PROTOCOL_TEMPLATE.yaml`: DGP hashes,
   seed namespaces (data, MCMC and bootstrap streams separate, from
   `SeedSequence([family, cell, replication, stream])`), estimands, grids, weights,
   reference versions, tuning, escalation, thresholds, failure handling, multiplicity.
   The `confirm` phase refuses an unfrozen protocol or unresolved required fields.
3. 500 confirmation replications in each primary cell, 200 in each nearby variant;
   measure one replication of every method first and record the projected cost.

A scenario advantage requires all of: $\bar L_{\text{LongBetDose}}/\bar L_{\text{ref}}\le0.80$;
a paired-replication bootstrap upper bound below 1, familywise adjusted (one-sided 97.5%
each for two primary claims); lower 95% bound on 95% interval coverage at least 0.90 at
the key targets; coverage not worse than the reference's by more than 0.02 (lower bound);
at least 95% of fits passing section 9.2 within the declared cap, every failure counted;
the advantage present in the nearby variants. Failures are never dropped: report
conditional-on-success metrics as such, next to a prespecified capped-loss analysis. Final
status, with separate computational, statistical and comparative fields:
`VALIDATED_WITH_SCENARIO_ADVANTAGE`, `VALIDATED_NO_CONFIRMED_ADVANTAGE`,
`VALIDATION_FAILED` or `COMPARISON_INCOMPLETE`.

## 11. Phases and definition of done

| Phase | Work | Done when |
|---|---|---|
| 0 | Worktree, environment, baseline test run, status file, preview flag, move the oracle to `tests/`, rerun oracle and prototype | both scripts pass in the new environment; baseline recorded |
| 1 | `_dose_config`, `_dose_input`, `_dose_basis`; contract entries | section 8.1 (non-fit parts) green; contract tests green in both directions |
| 2 | `_vleaf`: factored statistics, `log ml`, draws, fits | section 8.2 algebra green |
| 3 | `_vleaf_moves`: GROW/PRUNE, leaf-parent CHANGE | sections 8.3 and 8.4 green, mutation arms detected |
| 4 | State, sweep, covariance, loop, chains | residual invariant, fixed-tree Gibbs and Geweke (8.5) green; a two-period fit runs end to end |
| 5 | `LongBetDose`, estimands, support, diagnostics, archives, memory guard | 8.1 invariances, 8.6, archive and corruption tests green |
| 6 | Internal CHANGE; **mixing pilot** (9.4); then only the moves it justifies | pilot report in the status file; every added move passes 8.3 and 8.5; trees and move set frozen |
| 7 | R front door, contract, docs, README section marked "implemented, not released" | `R CMD check` status unchanged; R tests green with and without Python |
| 8 | SBC (8.7) and convergence protocol (9) | **gates 1 and 2** decided and recorded |
| 9 | Adapters, DGPs, harness, development runs, frozen protocol, confirmation | **gate 3** decided; report names scenario, target, loss ratio with uncertainty, coverage, run time and failures |
| 10 | Flip the preview flag only if all three gates passed; otherwise ship nothing public and keep the status honest | release commit cites the evidence |

Each phase ends with the full fast suite green and a status-file entry. Phases 1 to 4
contain no performance claims.

## 12. Risks, stop rules and deferred work

### 12.1 Risks

| Risk | Signal | Response |
|---|---|---|
| Forest multimodality defeats gate 2 | chains internally healthy, mutually offset | section 9.4 ladder; report honestly, as `count` does |
| Large leaves make splits expensive (each split adds `p` coefficients), so heterogeneity is under-detected | D3 subgroup contrasts shrunk toward the mean | compare with a reduced-rank exposure basis; report, do not retune on confirmation data |
| `bartz` internals change | `test_dose_bartz_api.py` fails on a bump | stay pinned; port deliberately |
| Common `V` wrong under heteroskedastic units | D7 under-coverage | document; group-specific covariance is deferred work |
| Trace memory at long `T` | guard fires | depth, thinning, reduced-rank exposure basis |

### 12.2 Stop rules

Stop and report rather than patch around: a Geweke or enumerated-posterior failure that
is not explained within a day; a residual-invariant violation; any need to change a
threshold, seed or DGP after confirmation has started; gate 2 still failing at E3.

### 12.3 Deferred

Hyperpriors on $\tau_f$, $\lambda_d$ and the lengthscale (each needs its own conditional,
Jacobian and tests); continuous-through-zero curves; reduced-rank exposure bases;
superpopulation standardization; not-yet-treated-only designs; missing cells and
unbalanced panels (pattern-specific precisions); time-varying dose and covariates;
non-Gaussian outcomes; multiple outcomes; float32; a fixed-effects option for the
existing level models, which appendix A.1 suggests is worth its own proposal.

## References

- **[R1]** Repository snapshot: `ignacio82/longbet-jax` at `bfe845d`.
- **[R2]** Wang, Martinez and Hahn, *LongBet: Heterogeneous Treatment Effect Estimation in
  Panel Data*, arXiv:2406.02530.
- **[R3]** Starling, Murray, Carvalho, Bukowski and Scott (2020), *BART with targeted
  smoothing*, Annals of Applied Statistics; Starling et al. (2021), *Targeted Smooth
  Bayesian Causal Forests*. Leaves that are smooth functions of a target variable: the
  construction of section 4.2. Deshpande et al., *VCBART*, for varying-coefficient forests.
- **[R4]** Callaway, Goodman-Bacon and Sant'Anna, *Difference-in-Differences with a
  Continuous Treatment*, `https://psantanna.com/files/CGBS_v4.pdf` (draft of 2025-12-31;
  store a checksum).
- **[R5]** `contdid` documentation, `https://bcallaway11.github.io/contdid/` and
  `.../reference/cont_did.html`, read 2026-09-17: version 0.1.1, alpha, GitHub only; "not
  currently supported": covariates beyond `~1`, discrete treatments, unbalanced panels,
  time-varying doses, `aggregation = "none"`; the CCK estimator "for the case with two
  periods and no staggered adoption". `npiv`: `https://github.com/JeffreyRacine/npiv`.
- **[R6]** Vehtari et al., *Rank-normalization, folding, and localization*, arXiv:1903.08008.
- **[R7]** Talts et al., *Validating Bayesian Inference Algorithms with Simulation-Based
  Calibration*, arXiv:1804.06788. **[R8]** Modrak et al., arXiv:2211.02383.
- **[R9]** Geweke (2004), *Getting it right: joint distribution tests of posterior
  simulators*, JASA.
- **[R10]** `benchmarks/tree_moves_review.md`, `tempering_plan.md`, `slow_mode_review.md`,
  `count/implementation_status.md`: this repository's record of what did and did not fix
  forest mixing.

## Appendix A. Evidence gathered for this revision (2026-09-17, CPU)

The scripts are in `ideas/evidence/` (see its README); A.3 is
`ideas/dose_engine_prototype.py`. None of this is evidence about `LongBetDose` itself,
which does not exist yet.

**A.1 Level selection and the current level model.** $Y_{it}=\alpha_i+0.1t+0.5Z_{it}+e_{it}$,
$\alpha_i=a\,G_i+u_i$, `N = 200`, half treated from the middle period, SDs 1 and 0.5,
noise covariates, `LongBet` with 2 chains of 400 + 300 sweeps and 20 + 20 trees.

| `T` | `a` | DiD (seeds 1, 2) | `LongBet` mean ATT | 95% interval, seed 1 |
|---|---|---|---|---|
| 2 | 0 | 0.655, 0.661 | 0.624, 0.666 | 0.43 to 0.84 |
| 2 | 2 | 0.655, 0.661 | **0.896, 0.902** | 0.68 to 1.11 |
| 6 | 0 | 0.497, 0.382 | 0.497, 0.358 | |
| 6 | 2 | 0.497, 0.382 | **0.578, 0.442** | |

Predicted shift (`dose_reference.random_intercept_level_bias`, $\sigma^2=0.25$,
$\sigma_\gamma^2=2$, half of the periods treated): 0.22 and 0.08. Observed: 0.24 and 0.08.

**A.2 The oracle's tree prior is `bartz`'s.** One `bartz` tree, zero response, error
precision `1e-4`, 198,000 sweeps on the 22-tree space: maximum absolute frequency
difference 0.0019, total variation 0.0038. At precision `1e-12` `bartz` accepted no GROW
in 300 proposals (float32), hence the advice in section 5.5.

**A.3 Prototype against exact posteriors.** Section 8.3's table. The first version failed
with a residual-invariant violation of 7 until step 5 of section 5.2 was added and
non-contributing designs were zeroed: the tests of section 8 find exactly this kind of
defect.

**A.4 Prior defaults, closed form.** Root-only two-period model, `N = 500`, 40% never
treated, unit noise, variance fixed at its truth, 200 replications, `tau_trt = 1`:

| Curve | `K`, $\lambda_d$ | RMSE | Pointwise coverage |
|---|---|---|---|
| `1 - exp(-3d)` | 6, 1 / 6, 4 / 6, 16 / 8, 4 | 0.128 / 0.121 / 0.114 / 0.132 | 0.961 / 0.965 / 0.967 / 0.965 |
| `0.5 sin(2 pi d)` | 6, 1 / 6, 4 / 6, 16 / 8, 4 | 0.132 / 0.133 / 0.150 / 0.134 | 0.958 / 0.938 / 0.876 / 0.962 |

**A.5 float32.** `T = 12`, cohorts 4, 7, 10, `K = 6` (`p = 54`), 600 treated units, a
subgroup effect, `difference_iid` covariance, $\kappa=50$, statistics accumulated and
factored in each precision, 20 replications: condition number of $\Lambda$ 26; GROW log
likelihood ratio about 3,100; absolute difference between float32 and float64 at most
0.001 nats.

**A.6 The multivariate `bartz` oracle exists.** `init` with `offset=zeros(3)`,
`leaf_prior_cov_inv` a dense `3 x 3` precision and a `Wishart` whose `nu` and `rate` are
then set to `None` runs 30 steps with the error precision held exactly fixed, `leaf_tree`
of shape `(trees, 3, 16)` and `prec_tree = None`. Section 8.4's second equivalence is
therefore constructible. (`ideas/evidence/mv_bart_check.py`.)

## Appendix B. Hazards checklist

1. A mutation arm that re-wraps the same function in `jax.jit` is vacuous. New function
   object, and assert the arm changed the answer.
2. Copy parent coefficients into the children of a pre-applied GROW (section 5.2, step 5).
3. Non-contributing units have exactly zero design and are excluded from counts.
4. `var_tree` is stale at leaves; read rules through `split_tree > 0`.
5. PRUNE negates the whole log ratio, transition and prior part included.
6. `bartz` grows a leaf with probability proportional to `p_nonterminal`; do not
   re-derive the transition ratio, call `complete_ratio`.
7. Recompute `Q_g` after every covariance draw.
8. Report levels `F`, never increments; rebuild them by cumulative sums.
9. `standardize=False` and thresholds off in Geweke, SBC and prior-recovery tests.
10. `bartz.mcmcstep.step` donates buffers: copy inputs, jit at the boundary.
11. An un-jitted sweep in a Python loop leaks compiled executables. Jit the loop body.
12. `/tmp` is memory. Long jobs: `nice`, capped threads, measured first.
13. Both contract YAMLs byte identical; man pages regenerated; `^ideas$` in
    `.Rbuildignore`; Python tests named `tests/test_dose_*.py`.
14. Do not touch the uncommitted count work or the jobs running from the main tree.
