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

#' Discrete-time hazard adoption model with spike-and-slab relevance
#'
#' Models adoption hazard on the active risk set with exact spike-and-slab relevance indicator.
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
#' @return A list with `$table` and `$metadata`.
#' @export
hazard_adoption_effects <- function(d, z, x = NULL, t = NULL,
                                    trees = 4L, chains = 2L,
                                    burnin = 200L, draws = 300L,
                                    seed = 42L) {
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
    metadata = .encouragement_as_r(res$metadata)
  )
}

#' Structural treatment-clock duration deconvolution for longitudinal IV
#'
#' Estimates cumulative treatment exposure duration effects under adoption acceleration.
#'
#' @param y Outcome panel, [N x T].
#' @param d Binary absorbing adoption panel, [N x T].
#' @param z Binary encouragement panel, [N x T].
#' @param t Optional calendar periods, [T].
#' @param model_type "linear", "stepwise", or "spline" (default "linear").
#' @param max_duration Optional maximum duration to model.
#' @param alpha Significance level (default 0.05).
#' @return A list with `$summary_table`, `$durations`, `$effects`, `$se`, `$ci_lower`, `$ci_upper`, `$first_stage_f`.
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
    max_duration = if (is.null(max_duration)) NULL else as.integer(max_duration),
    alpha = alpha
  )
  tbl <- as.data.frame(.encouragement_as_r(res$summary_table))
  attr(tbl, "pandas.index") <- NULL
  rownames(tbl) <- NULL
  list(
    summary_table = tbl,
    durations = as.numeric(res$durations),
    effects = as.numeric(res$effects),
    se = as.numeric(res$se),
    ci_lower = as.numeric(res$ci_lower),
    ci_upper = as.numeric(res$ci_upper),
    first_stage_f = as.numeric(res$first_stage_f)
  )
}

#' Direct joint Gaussian forest for encouragement panels
#'
#' Fits joint Gaussian forest with exact Gibbs sampling and correlated intercepts.
#'
#' @param y Outcome panel, [N x T].
#' @param d Binary adoption panel, [N x T].
#' @param z Binary encouragement panel, [N x T].
#' @param x Baseline covariates, [N x P].
#' @param t Optional calendar periods, [T].
#' @param num_trees Number of trees (default 30).
#' @param num_sweeps Number of sweeps (default 200).
#' @param num_burnin Number of burnin sweeps (default 50).
#' @param correlated_intercepts Logical; learn cross-equation intercept covariance (default TRUE).
#' @param seed Random seed.
#' @return A fitted model object.
#' @export
longbet_direct_smooth <- function(y, d, z, x, t = NULL,
                                  num_trees = 30L, num_sweeps = 200L,
                                  num_burnin = 50L, correlated_intercepts = TRUE,
                                  seed = 42L) {
  lb <- longbet_py()
  cfg <- lb$DirectSmoothConfig(
    num_trees = as.integer(num_trees),
    num_sweeps = as.integer(num_sweeps),
    num_burnin = as.integer(num_burnin),
    correlated_intercepts = correlated_intercepts,
    seed = as.integer(seed)
  )
  model <- lb$LongBetDirectSmooth(cfg)
  model$fit(
    y = reticulate::r_to_py(as.matrix(y)),
    d = reticulate::r_to_py(as.matrix(d)),
    z = reticulate::r_to_py(as.matrix(z)),
    x = reticulate::r_to_py(as.matrix(x)),
    t = if (is.null(t)) NULL else reticulate::r_to_py(as.numeric(t))
  )
  model
}
