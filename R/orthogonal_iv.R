#' Direct Bayesian Orthogonal IV (Two-Stage BCF / R-Learner) for panel data
#'
#' Fits a direct Bayesian Causal Forest on Neyman-orthogonalized pseudo-outcomes
#' to estimate the Complier Average Causal Effect (CACE) without denominator division.
#'
#' @param y Outcome panel matrix, `[N x T]`.
#' @param d Actual adoption takeup panel matrix, `[N x T]`.
#' @param z Randomized encouragement instrument panel matrix, `[N x T]`.
#' @param x Baseline covariates matrix, `[N x P]`.
#' @param t Optional calendar vector, length `T`.
#' @param config Optional list of hyperparameters for the structural treatment forest.
#' @param first_stage_config Optional hyperparameters for first-stage compliance forest.
#' @param prognostic_config Optional hyperparameters for prognostic outcome forest.
#' @param min_compliance Minimum denominator threshold for first-stage compliance (default 0.02).
#' @param monotonic_first_stage If TRUE (default), enforces non-negative compliance.
#' @param ... Reserved.
#' @return A `longbet_orthogonal_iv` model object.
#' @export
longbet_orthogonal_iv <- function(y, d, z, x, t = NULL, config = NULL,
                                  first_stage_config = NULL, prognostic_config = NULL,
                                  min_compliance = 0.02, monotonic_first_stage = TRUE, ...) {
  longbet_py()
  lb <- reticulate::import("longbet", convert = FALSE)

  cfg_py <- if (is.null(config)) NULL else do.call(lb$LongBetConfig, .encouragement_config_args(config, lb))
  f_cfg_py <- if (is.null(first_stage_config)) NULL else do.call(lb$LongBetConfig, .encouragement_config_args(first_stage_config, lb))
  p_cfg_py <- if (is.null(prognostic_config)) NULL else do.call(lb$LongBetConfig, .encouragement_config_args(prognostic_config, lb))

  model <- lb$LongBetOrthogonalIV(
    config = cfg_py,
    first_stage_config = f_cfg_py,
    prognostic_config = p_cfg_py,
    min_compliance = as.numeric(min_compliance),
    monotonic_first_stage = as.logical(monotonic_first_stage)
  )

  t_py <- if (is.null(t) || length(t) == 0) NULL else reticulate::r_to_py(as.numeric(t))
  model$fit(
    y = reticulate::r_to_py(as.matrix(y)),
    d = reticulate::r_to_py(as.matrix(d)),
    z = reticulate::r_to_py(as.matrix(z)),
    x = reticulate::r_to_py(as.matrix(x)),
    t = t_py
  )

  structure(list(
    py_model = model,
    n_units = nrow(y),
    n_periods = ncol(y),
    min_compliance = min_compliance,
    monotonic_first_stage = monotonic_first_stage
  ), class = "longbet_orthogonal_iv")
}

#' @export
predict_conditional.longbet_orthogonal_iv <- function(object, new_x = NULL, new_z = NULL, t = NULL,
                                                      cost = NULL, hurdle = 0.50, alpha = 0.05,
                                                      budget = NULL, capacity = NULL,
                                                      ranking_metric = "expected_net_value", ...) {
  x_py <- if (is.null(new_x)) NULL else reticulate::r_to_py(as.matrix(new_x))
  z_py <- if (is.null(new_z)) NULL else reticulate::r_to_py(as.matrix(new_z))
  t_py <- if (is.null(t) || length(t) == 0) NULL else reticulate::r_to_py(as.numeric(t))

  pred <- object$py_model$predict_conditional(x = x_py, z = z_py, t = t_py, alpha = alpha)

  citt_y_draws <- .encouragement_plain(pred$citt_y$draws)
  citt_y_mean <- .encouragement_plain(pred$citt_y$mean)
  citt_y_median <- .encouragement_plain(pred$citt_y$median)
  citt_y_lower <- .encouragement_plain(pred$citt_y$lower)
  citt_y_upper <- .encouragement_plain(pred$citt_y$upper)
  citt_y_sd <- .encouragement_plain(pred$citt_y$sd)

  citt_d_draws <- .encouragement_plain(pred$citt_d$draws)
  citt_d_mean <- .encouragement_plain(pred$citt_d$mean)
  citt_d_median <- .encouragement_plain(pred$citt_d$median)
  citt_d_lower <- .encouragement_plain(pred$citt_d$lower)
  citt_d_upper <- .encouragement_plain(pred$citt_d$upper)
  citt_d_sd <- .encouragement_plain(pred$citt_d$sd)

  cace_draws <- .encouragement_plain(pred$cace$draws)
  cace_mean <- .encouragement_plain(pred$cace$mean)
  cace_median <- .encouragement_plain(pred$cace$median)
  cace_lower <- .encouragement_plain(pred$cace$lower)
  cace_upper <- .encouragement_plain(pred$cace$upper)
  cace_sd <- .encouragement_plain(pred$cace$sd)

  periods <- unlist(.encouragement_plain(pred$periods), use.names = FALSE)
  horizons <- unlist(.encouragement_plain(pred$horizons), use.names = FALSE)

  cum_draws <- apply(citt_y_draws, c(1, 3), sum)
  cum_mean <- rowMeans(cum_draws)
  cum_median <- apply(cum_draws, 1, stats::median)
  cum_lower <- apply(cum_draws, 1, stats::quantile, probs = alpha / 2)
  cum_upper <- apply(cum_draws, 1, stats::quantile, probs = 1 - alpha / 2)

  breakeven_prob <- if (!is.null(cost)) rowMeans(cum_draws > cost) else NULL

  decision <- if (!is.null(cost)) {
    if (!is.null(budget) || !is.null(capacity)) {
      as.logical(.encouragement_plain(pred$knapsack_policy(
        cost = as.numeric(cost),
        budget = if (!is.null(budget)) as.numeric(budget) else NULL,
        capacity = if (!is.null(capacity)) as.integer(capacity) else NULL,
        hurdle = as.numeric(hurdle),
        ranking_metric = ranking_metric
      )))
    } else {
      (breakeven_prob >= hurdle)
    }
  } else NULL

  policy_val <- if (!is.null(decision) && !is.null(cost)) {
    mean(ifelse(decision, cum_mean - as.numeric(cost), 0.0))
  } else NULL

  strata_df <- tryCatch({
    as.data.frame(.encouragement_plain(pred$principal_strata()))
  }, error = function(e) NULL)

  structure(list(
    citt_y = list(mean = citt_y_mean, median = citt_y_median, lower = citt_y_lower,
                  upper = citt_y_upper, sd = citt_y_sd, draws = citt_y_draws),
    citt_d = list(mean = citt_d_mean, median = citt_d_median, lower = citt_d_lower,
                  upper = citt_d_upper, sd = citt_d_sd, draws = citt_d_draws),
    cace = list(mean = cace_mean, median = cace_median, lower = cace_lower,
                upper = cace_upper, sd = cace_sd, draws = cace_draws),
    cumulative_lift = list(draws = cum_draws, mean = cum_mean, median = cum_median,
                           lower = cum_lower, upper = cum_upper),
    breakeven_prob = breakeven_prob,
    decision = decision,
    policy_value = policy_val,
    cost = cost,
    hurdle = hurdle,
    budget = budget,
    capacity = capacity,
    ranking_metric = ranking_metric,
    periods = periods,
    horizons = horizons,
    alpha = alpha,
    strata = strata_df
  ), class = "longbet_conditional_pred")
}

#' @export
predict.longbet_orthogonal_iv <- function(object, new_x = NULL, new_z = NULL, t = NULL,
                                          cost = NULL, hurdle = 0.50, alpha = 0.05,
                                          budget = NULL, capacity = NULL,
                                          ranking_metric = "expected_net_value", ...) {
  predict_conditional(object, new_x = new_x, new_z = new_z, t = t,
                      cost = cost, hurdle = hurdle, alpha = alpha,
                      budget = budget, capacity = capacity,
                      ranking_metric = ranking_metric, ...)
}

#' @export
print.longbet_orthogonal_iv <- function(x, ...) {
  cat(sprintf("LongBet Direct Orthogonal IV (Two-Stage BCF)\n"))
  cat(sprintf("  %d study units, %d periods; min_compliance=%.3f, monotonic=%s\n",
              x$n_units, x$n_periods, x$min_compliance, x$monotonic_first_stage))
  invisible(x)
}
