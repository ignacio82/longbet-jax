# Convert only when reticulate has not already converted the return value.
.encouragement_as_r <- function(value) {
  if (inherits(value, "python.builtin.object")) reticulate::py_to_r(value) else value
}

#' Validate a single-wave encouragement experiment
#'
#' Supports individual-unit complete randomization, a single encouragement wave,
#' and a permanent control arm. Validation checks arrays, not whether assignment
#' was actually randomized or whether exclusion and monotonicity hold. Blocked,
#' unequal-probability, and cluster assignment require different estimators.
#'
#' @param z Binary encouragement indicator, `[N x T]`, absorbing.
#' @param d Binary actual adoption indicator, `[N x T]`, absorbing and complete.
#' @param t Strictly increasing calendar times with whole-unit gaps; defaults
#'   to `1:ncol(z)`. Uneven gaps are allowed.
#' @return A list of dimensions, arm counts, encouragement period and zero-based
#'   column index, `has_pre_periods`, and `perfect_compliance`.
#' @details Perfect compliance, adoption before encouragement, negative sample
#'   first stages, and experiments without baseline periods are accepted.
#' @export
validate_encouragement <- function(z, d, t = NULL) {
  lb <- longbet_py()
  .encouragement_as_r(lb$validate_encouragement(
    z = reticulate::r_to_py(as.matrix(z)),
    d = reticulate::r_to_py(as.matrix(d)),
    t = if (is.null(t)) NULL else reticulate::r_to_py(as.numeric(t))
  ))
}

#' Describe adoption in assigned encouragement arms
#'
#' Includes baseline periods and preserves signed first-stage estimates. All
#' statistics delegate to the Python engine.
#'
#' @param z,d,t As in [validate_encouragement()].
#' @param alpha Significance level; intervals have level `1-alpha`.
#' @return A data frame with one row per calendar period, arm counts and take-up
#'   rates, first-stage differences and uncertainty, and observed adoption lags.
#' @details The standard error uses separate arm sample variances divided by
#'   independent unit counts. Intervals are pointwise normal confidence intervals
#'   under individual-unit complete randomization, not exact or Bayesian intervals.
#'   If an arm has fewer than two units, uncertainty is unavailable.
#'
#'   Lag summaries describe encouraged units observed adopted by each period.
#'   `share_immediate` is the fraction adopting at encouragement;
#'   `median_observed_lag` is adoption time minus encouragement time;
#'   `share_pre_encouragement_adoption` is the fraction adopting before the nudge.
#'   Undefined summaries are missing. None identifies compliance types or verifies
#'   counterfactual timing assumptions. `horizon` uses the model exposure clock;
#'   `period_index` is a zero-based column index.
#' @export
encouragement_summary <- function(z, d, t = NULL, alpha = 0.05) {
  lb <- longbet_py()
  result <- lb$encouragement_summary(
    z = reticulate::r_to_py(as.matrix(z)),
    d = reticulate::r_to_py(as.matrix(d)),
    t = if (is.null(t)) NULL else reticulate::r_to_py(as.numeric(t)),
    alpha = alpha
  )
  result <- as.data.frame(.encouragement_as_r(result))
  # pandas' index attribute carries a live Python handle through reticulate.
  # These results are ordinary tables and must survive saveRDS independently.
  attr(result, "pandas.index") <- NULL
  rownames(result) <- NULL
  result
}

#' Reference encouragement effects and Wald confidence sets
#'
#' Computes assigned-arm differences in means on the outcome and adoption scales,
#' their joint Neyman covariance estimate, and analytic normal-test inversion.
#' Assumes individual-unit complete randomization with one wave and a permanent
#' control arm. Every statistic is computed by the Python engine.
#'
#' @param y Complete finite numeric outcome panel, `[N x T]`. Binary outcomes
#'   yield risk differences without a latent-scale transformation.
#' @param d,z,t As in [validate_encouragement()].
#' @param alpha Significance level; confidence sets have level `1-alpha`.
#' @return A data frame with one row per observed post-encouragement period:
#'   `itt_y`, `itt_d`, their SEs and bounds, `itt_y_d_cov`, `wald`, and a confidence
#'   set in `wald_lower_1`, `wald_upper_1`, `wald_lower_2`, `wald_upper_2`.
#'   `wald_set_type` describes its form. Infinite endpoints remain infinite and
#'   unused components are missing. `wald_reason` identifies unavailable inference.
#' @details The covariance accounts for both outcomes measured on the same units.
#'   The confidence set inverts the studentized assigned-arm contrast of `Y-w*D`
#'   using a normal critical value. This is asymptotic Fieller/Anderson--Rubin-style
#'   inference, not an exact permutation procedure or Bayesian credible interval.
#'   Adequate arm sizes and moment conditions are still required. Intervals are
#'   pointwise, not simultaneous across horizons. No first-stage cutoff is used.
#'
#'   Sets may be bounded, disjoint, half-line, all-real, singleton, empty, or
#'   unavailable. Arms with fewer than two units retain point estimates but have
#'   unavailable uncertainty. Missing data are rejected rather than dropped.
#'
#'   The Wald ratio is not automatically CACE: that interpretation also needs
#'   exclusion, monotonicity, relevance, and appropriate adoption-history
#'   assumptions. Negative first stages are retained. Randomization alone does
#'   not establish those additional assumptions.
#' @export
encouragement_effects <- function(y, d, z, t = NULL, alpha = 0.05) {
  lb <- longbet_py()
  result <- lb$encouragement_effects(
    y = reticulate::r_to_py(as.matrix(y)),
    d = reticulate::r_to_py(as.matrix(d)),
    z = reticulate::r_to_py(as.matrix(z)),
    t = if (is.null(t)) NULL else reticulate::r_to_py(as.numeric(t)),
    alpha = alpha
  )
  result <- as.data.frame(.encouragement_as_r(result))
  attr(result, "pandas.index") <- NULL
  rownames(result) <- NULL
  result
}

#' Plot adoption by assigned encouragement arm
#'
#' Uses the Python-produced summary table; requires the optional ggplot2 package.
#' @param z,d,t As in [validate_encouragement()].
#' @return A ggplot with assigned-arm adoption curves, their gap shaded, and the
#'   encouragement time marked. Baseline periods are included.
#' @export
plot_encouragement <- function(z, d, t = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("plot_encouragement() needs ggplot2; use encouragement_summary() ",
         "to obtain the table without a plotting dependency.", call. = FALSE)
  }
  df <- encouragement_summary(z, d, t)
  start <- df$period[which(df$post_encouragement)[1]]
  ggplot2::ggplot(df, ggplot2::aes(x = .data$period)) +
    ggplot2::geom_ribbon(ggplot2::aes(
      ymin = pmin(.data$takeup_control, .data$takeup_encouraged),
      ymax = pmax(.data$takeup_control, .data$takeup_encouraged)), alpha = 0.15) +
    ggplot2::geom_line(ggplot2::aes(y = .data$takeup_encouraged,
                                  colour = "Assigned encouragement")) +
    ggplot2::geom_line(ggplot2::aes(y = .data$takeup_control,
                                  colour = "Assigned control")) +
    ggplot2::geom_vline(xintercept = start, linetype = "dashed", colour = "grey") +
    ggplot2::coord_cartesian(ylim = c(0, 1)) +
    ggplot2::labs(x = "Period", y = "Adoption rate", colour = NULL) +
    ggplot2::theme_minimal()
}

#' Evaluate randomization-based Anderson-Rubin test on a grid
#'
#' Inverts an exact finite-sample randomization test of H_0: beta = beta_0.
#'
#' @param y Outcome panel, [N x T].
#' @param d Binary actual adoption panel, [N x T].
#' @param z Binary encouragement panel, [N x T].
#' @param beta Vector of candidate effect values to test.
#' @param t Optional calendar periods, [T].
#' @param alpha Significance level (default 0.05).
#' @param method "monte_carlo", "exact", or "auto".
#' @param permutations Number of Monte Carlo permutations (default 1000).
#' @param seed Random seed.
#' @return A list with `$table` and `$metadata`.
#' @export
randomization_ar <- function(y, d, z, beta, t = NULL, alpha = 0.05,
                             method = "monte_carlo", permutations = 1000L, seed = 0L) {
  lb <- longbet_py()
  result <- lb$randomization_ar(
    y = reticulate::r_to_py(as.matrix(y)),
    d = reticulate::r_to_py(as.matrix(d)),
    z = reticulate::r_to_py(as.matrix(z)),
    t = if (is.null(t)) NULL else reticulate::r_to_py(as.numeric(t)),
    beta = reticulate::r_to_py(as.numeric(beta)),
    alpha = alpha,
    method = method,
    permutations = as.integer(permutations),
    seed = as.integer(seed)
  )
  tbl <- as.data.frame(.encouragement_as_r(result$table))
  attr(tbl, "pandas.index") <- NULL
  rownames(tbl) <- NULL
  list(table = tbl, metadata = .encouragement_as_r(result$metadata))
}

#' Sharp nonparametric identification bounds on principal-stratum causal effects
#'
#' Computes Manski and Balke-Pearl bounds with direct-effect sensitivity limits.
#'
#' @param y Binary outcome vector, [N].
#' @param d Binary adoption vector, [N].
#' @param z Binary encouragement vector, [N].
#' @param x Optional baseline covariates, [N x P].
#' @param delta Maximum direct effect on outcome probabilities (default 1.0).
#' @param draws Number of posterior draws (default 1000).
#' @param chains Number of chains (default 4).
#' @param seed Random seed.
#' @return A list with `$table` and `$metadata`.
#' @export
encouragement_bounds <- function(y, d, z, x = NULL, delta = 1.0,
                                 draws = 1000L, chains = 4L, seed = 0L) {
  lb <- longbet_py()
  result <- lb$encouragement_bounds(
    y = reticulate::r_to_py(as.numeric(y)),
    d = reticulate::r_to_py(as.numeric(d)),
    z = reticulate::r_to_py(as.numeric(z)),
    x = if (is.null(x)) NULL else reticulate::r_to_py(as.matrix(x)),
    delta = delta,
    draws = as.integer(draws),
    chains = as.integer(chains),
    seed = as.integer(seed)
  )
  tbl <- as.data.frame(.encouragement_as_r(result$table))
  attr(tbl, "pandas.index") <- NULL
  rownames(tbl) <- NULL
  list(table = tbl, metadata = .encouragement_as_r(result$metadata))
}

#' Experimental discrete-time hazard adoption forest
#'
#' Models adoption hazard on the active risk set with a relevance indicator.
#' One seeded random stump basis is fixed and shared across chains. Probit
#' utilities and a collapsed relevance/effect block target the posterior
#' conditional on this basis. Calibration is unestablished, and a relevance
#' probability does not establish IV identification or eliminate weak-IV bias.
#'
#' @param d Binary absorbing adoption panel, [N x T].
#' @param z Binary encouragement panel, [N x T].
#' @param x Optional baseline covariates, [N x P].
#' @param t Optional calendar periods, [T].
#' @param trees Number of trees in each ensemble (default 4).
#' @param chains Number of MCMC chains (default 2).
#' @param burnin Number of burnin sweeps (default 200).
#' @param draws Number of retained sweeps (default 300).
#' @param seed Random seed.
#' @return A list with `$table`, `$metadata`, and `$draws`. Contrast draws retain
#'   `(chain, retained_draw, horizon)` axes; relevance draws have the first two
#'   axes. Raw traces permit inspection of convergence and Monte Carlo error.
#' @export
hazard_adoption_effects <- function(d, z, x = NULL, t = NULL,
                                    trees = 4L, chains = 2L,
                                    burnin = 200L, draws = 300L,
                                    seed = 42L) {
  for (name in c("trees", "chains", "burnin", "draws", "seed")) {
    value <- get(name)
    minimum <- if (name %in% c("burnin", "seed")) 0 else 1
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
        value < minimum || value != trunc(value) || value > .Machine$integer.max) {
      stop(sprintf("%s must be an integer >= %d.", name, minimum), call. = FALSE)
    }
  }
  lb <- longbet_py()
  cfg <- lb$HazardConfig(trees = as.integer(trees))
  res <- lb$hazard_adoption_effects(
    d = reticulate::r_to_py(as.matrix(d)),
    z = reticulate::r_to_py(as.matrix(z)),
    x = if (is.null(x)) NULL else reticulate::r_to_py(as.matrix(x)),
    t = if (is.null(t)) NULL else reticulate::r_to_py(as.numeric(t)),
    config = cfg,
    chains = as.integer(chains),
    burnin = as.integer(burnin),
    draws = as.integer(draws),
    seed = as.integer(seed)
  )
  tbl <- as.data.frame(.encouragement_as_r(res$table))
  attr(tbl, "pandas.index") <- NULL
  rownames(tbl) <- NULL
  list(
    table = tbl,
    metadata = .encouragement_plain(res$metadata),
    draws = .encouragement_plain(res$draws)
  )
}

#' Structural treatment-clock duration deconvolution for longitudinal IV
#'
#' Conventional panel 2SLS for a common response to cumulative exposure.
#' Requires exclusion through treatment history, additive unit/time effects,
#' and full rank of instrumented duration regressors. Confidence intervals use
#' unit-cluster CR1 covariance and t(N-1) critical values; they require many
#' independent units and strong instruments and are not weak-IV-robust.
#'
#' @param y Outcome panel, [N x T].
#' @param d Binary absorbing adoption panel, [N x T].
#' @param z Binary absorbing encouragement panel, [N x T], with a common onset
#'   among encouraged units and at least one never-encouraged unit.
#' @param t Optional strictly increasing, equally spaced periods, [T]. Duration
#'   counts observed periods, independently of the numerical units of t.
#' @param model_type "linear", "stepwise", or "quadratic" (default "linear").
#'   The legacy name "spline" aliases a quadratic polynomial without knots.
#' @param max_duration Optional maximum duration to evaluate. For "stepwise",
#'   also imposes a constant response beyond this duration.
#' @param alpha Significance level (default 0.05).
#' @return A list with the duration-response estimates, unit-cluster covariance,
#'   instrument and regressor ranks, and first-stage diagnostics.
#'   `$first_stage_f` is a cluster Wald statistic divided by instrument rank for
#'   a scalar endogenous regressor. It is unavailable (NaN) for multiple
#'   regressors or singular cluster covariance; see `$first_stage_status` and
#'   `$first_stage_diagnostics`. It does not use Stock--Yogo critical values.
#' @export
duration_deconvolution_effects <- function(y, d, z, t = NULL,
                                           model_type = "linear",
                                           max_duration = NULL,
                                           alpha = 0.05) {
  lb <- longbet_py()
  res <- lb$duration_deconvolution_effects(
    y = reticulate::r_to_py(as.matrix(y)),
    d = reticulate::r_to_py(as.matrix(d)),
    z = reticulate::r_to_py(as.matrix(z)),
    t = if (is.null(t)) NULL else reticulate::r_to_py(as.numeric(t)),
    model_type = model_type,
    max_duration = if (is.null(max_duration)) NULL else as.numeric(max_duration),
    alpha = alpha
  )
  tbl <- as.data.frame(.encouragement_as_r(res$summary_table))
  attr(tbl, "pandas.index") <- NULL
  rownames(tbl) <- NULL
  first_stage <- as.data.frame(.encouragement_as_r(res$first_stage_diagnostics))
  attr(first_stage, "pandas.index") <- NULL
  rownames(first_stage) <- NULL
  list(
    summary_table = tbl,
    durations = as.numeric(res$durations),
    effects = as.numeric(res$effects),
    se = as.numeric(res$se),
    ci_lower = as.numeric(res$ci_lower),
    ci_upper = as.numeric(res$ci_upper),
    first_stage_f = as.numeric(res$first_stage_f),
    first_stage_status = as.character(res$first_stage_status),
    first_stage_diagnostics = first_stage,
    coefficients = as.numeric(res$coefficients),
    coefficient_covariance = as.matrix(.encouragement_as_r(res$coefficient_covariance)),
    instrument_rank = as.integer(res$instrument_rank),
    regressor_rank = as.integer(res$regressor_rank),
    residual_df = as.integer(res$residual_df),
    n_clusters = as.integer(res$n_clusters),
    inference_method = as.character(res$inference_method)
  )
}

#' Direct joint Gaussian forest for encouragement panels
#'
#' Fits Gaussian reduced forms with smooth stump leaves and exact conditional
#' Gibbs updates. Returns the same serializable R model as `longbet_encourage()`.
#'
#' @param y Outcome panel, [N x T].
#' @param d Binary adoption panel, [N x T].
#' @param z Binary encouragement panel, [N x T].
#' @param x Baseline covariates, [N x P].
#' @param t Optional calendar periods, [T].
#' @param num_trees Number of trees in each of the baseline and effect ensembles.
#' @param num_sweeps Number of retained draws per chain (default 200).
#' @param num_burnin Number of burnin sweeps (default 50).
#' @param correlated_intercepts Logical; learn cross-equation intercept covariance (default TRUE).
#' @param seed Random seed.
#' @param num_chains Number of independent MCMC chains.
#' @param n_skip Number of sweeps per retained draw after burn-in.
#' @return A serializable `longbet_encourage` model. Use `predict()` to obtain
#'   ITTs, posterior ratios, separate reference confidence sets and diagnostics.
#' @details Sampling options are forwarded unchanged, with integer validation.
#'   Both equations use Gaussian working likelihoods, with an LPM first stage.
#'   The default short run does not guarantee convergence or interval coverage;
#'   inspect `predict(fit)$diagnostics`. For other forest or prior options, use
#'   `longbet_encourage(engine = "direct_smooth", direct_config = list(...))`.
#' @export
longbet_direct_smooth <- function(y, d, z, x, t = NULL,
                                  num_trees = 6L, num_sweeps = 200L,
                                  num_burnin = 50L, correlated_intercepts = TRUE,
                                  seed = 42L, num_chains = 4L, n_skip = 1L) {
  longbet_encourage(
    y, d, z, x, t = t, first_stage = "lpm", engine = "direct_smooth",
    config = list(num_sweeps = num_sweeps, num_burnin = num_burnin,
                  random_seed = seed, num_chains = num_chains, n_skip = n_skip),
    direct_config = list(baseline_trees = num_trees, effect_trees = num_trees,
                         correlated_intercepts = correlated_intercepts)
  )
}
