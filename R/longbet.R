#' LongBet: Bayesian ensemble trees for causal inference on longitudinal data
#'
#' Fits time-varying heterogeneous treatment effects on panel data with separate
#' prognostic and treatment forests, a shared Gaussian-process trajectory over
#' the exposure index, and unit-level random intercepts. The sampler is the JAX
#' engine in the `longbet-jax` Python package; this function marshals data in
#' and results out.
#' Fits made before the 2026-09-07 dynamic leaf-precision cache repair must be
#' refitted from the original data. Loading old posterior draws cannot repair
#' their uncertainty; archive loaders reject missing sampler provenance.
#'
#' @section Estimand:
#' The reported effect is the contrast between being `S` periods into treatment
#' and not being treated at all,
#' \deqn{\tau_t(X_i, S) = b_1 \beta_S \nu(X_i, S, t) - b_0 \beta_0 \nu(X_i, 0, t),}
#' so the treatment forest is evaluated on both the factual exposure and a copy
#' of the design with \eqn{S = 0}. Under control the exposure index is 0, not
#' \eqn{S}, so both the multiplier and the forest's input change.
#'
#' @section Sampling defaults:
#' The defaults use 2,000 burn-in iterations and 250 retained draws, thinned by
#' 2, across four chains: 2,500 iterations per chain and 1,000 retained draws
#' overall. These are starting settings, not a convergence guarantee. Check
#' [att_stability()] and the subgroup or probability-scale effects used in
#' decisions. More burn-in can remove an initial transient, but cannot by itself
#' repair persistent poor mixing; the retained chains also need adequate
#' effective sample size.
#'
#' @section Categorical covariates:
#' Every column is treated as ordered numeric; there is no `pcat` argument.
#' One-hot encode unordered categorical variables with more than two levels
#' before calling this function. This is a real modelling difference from the
#' reference C++ implementation, not a detail.
#'
#' @param y Outcome panel, `[N x T]`. `NA` marks an unobserved cell; unbalanced
#'   panels are marginalized, not imputed.
#' @param x Covariates for the prognostic forest, `[N x P]`.
#' @param z Binary treatment indicator, `[N x T]`. Must be absorbing.
#' @param t Calendar time vector of length `T`. Defaults to `1:ncol(y)`.
#' @param x_trt Covariates for the treatment forest, `[N x P_trt]`. Defaults to
#'   `x`, in which case both forests share one block of columns.
#' @param x_tv Time-varying prognostic covariates, `[N x T x P]`.
#' @param x_trt_tv Time-varying treatment covariates, `[N x T x P]`.
#' @param ps Propensity score, length `N` or `[N x T]`. Shown to the prognostic
#'   forest only. For staggered adoption the principled analogue is a timing or
#'   hazard score rather than one scalar per unit.
#' @param num_sweeps Posterior draws saved per chain, after burn-in.
#' @param num_burnin Initial burn-in iterations discarded. This does not
#'   guarantee convergence or sufficient effective sample size.
#' @param n_skip Thinning interval; 1 saves every post-burn-in iteration.
#'   At fixed retained draw count, larger values add sampling work while keeping
#'   prediction cost and trace storage approximately fixed.
#' @param num_chains Parallel MCMC chains. At least 2 are needed for R-hat.
#' @param inner_loop_length Sweeps per dispatch; bounds compile time.
#' @param num_trees_pr,num_trees_trt Trees in each forest.
#' @param min_points_per_leaf_pr,min_points_per_leaf_trt Minimum leaf sizes.
#' @param max_depth_pr,max_depth_trt Maximum tree depths.
#' @param alpha_split_pr,beta_split_pr Tree-depth prior, prognostic forest.
#' @param alpha_split_trt,beta_split_trt Tree-depth prior, treatment forest.
#' @param num_cutpoints Maximum quantile bins per continuous covariate.
#' @param sig_knl Kernel standard deviation for the exposure trajectory.
#' @param lambda_knl Kernel lengthscale over the exposure index.
#' @param kernel_type One of `"se"`, `"matern32"`, `"matern52"`, `"ar1"`.
#' @param gp_jitter Relative diagonal jitter on the kernel.
#' @param sigma_m Prior standard deviation of the marginalized constant GP mean.
#' @param gp_constant_mean Whether the projection reverts to an estimated common
#'   level rather than to zero.
#' @param split_time_ps Whether the *prognostic* forest may split on calendar time.
#' @param split_time_trt Whether the *treatment* forest may split on the exposure
#'   index. Setting it `FALSE` forces all exposure-time shape into `beta`.
#' @param random_intercept Whether to fit unit random intercepts.
#' @param gamma_prior_a,gamma_prior_b Inverse-gamma prior on the unit-intercept
#'   variance.
#' @param sigma_prior_a,sigma_prior_b Inverse-gamma prior on the error variance;
#'   `(0, 0)` is the improper reference prior.
#' @param sigma_b Prior standard deviation of the adaptive coding weights.
#' @param sigma_alpha Prior standard deviation of the prognostic scale.
#' @param outcome `"continuous"` or `"binary"` (probit).
#' @param a_scaling Whether to sample the prognostic scale `alpha`.
#' @param b_scaling Whether to sample the adaptive coding weights `b0`, `b1`.
#'   On by default: this is an exactly conjugate move along the global scale of
#'   the treatment term.
#' @param ridge_move Whether to run the Metropolis move along the beta-nu ridge.
#' @param ridge_proposal_sigma Proposal standard deviation for `log c`.
#' @param standardize Whether to centre and scale a continuous outcome internally.
#' @param random_seed Base PRNG seed.
#' @param device `"auto"`, `"cpu"` or `"gpu"`.
#' @param sur Whether to enable full-precision triangular SUR coupling (inert for scalar `longbet`).
#' @param sur_prior_var Prior variance for loading regression parameters.
#' @param num_shared_trees Shared treatment trees; must be 0 for scalar fits.
#'   Use [longbet_multi()] to share treatment partitions across outcomes.
#' @param shared_variance_fraction Shared treatment-forest prior variance
#'   fraction, strictly between 0 and 1; inert for scalar fits.
#' @param verbose Whether to print progress messages.
#' @param ... Reserved. Passing an unsupported argument is an error rather than
#'   being silently ignored.
#' @return An object of class `longbet`.
#' @seealso [predict.longbet()], [get_att()], [att_stability()]
#' @export
longbet <- function(y, x, z, t = NULL,
                    x_trt = NULL, x_tv = NULL, x_trt_tv = NULL, ps = NULL,
                    num_sweeps = 250, num_burnin = 2000, n_skip = 2,
                    num_chains = 4, inner_loop_length = NULL,
                    num_trees_pr = 20, num_trees_trt = 20,
                    min_points_per_leaf_pr = 10, min_points_per_leaf_trt = 10,
                    max_depth_pr = 10, max_depth_trt = 10,
                    alpha_split_pr = 0.95, beta_split_pr = 2.0,
                    alpha_split_trt = 0.25, beta_split_trt = 3.0,
                    num_cutpoints = 100,
                    sig_knl = 1.0, lambda_knl = 1.0, kernel_type = "se",
                    gp_jitter = 1e-6, sigma_m = 1.0, gp_constant_mean = TRUE,
                    split_time_ps = TRUE, split_time_trt = TRUE,
                    random_intercept = TRUE,
                    gamma_prior_a = 1.0, gamma_prior_b = 0.1,
                    sigma_prior_a = 0.0, sigma_prior_b = 0.0,
                    sigma_b = 0.7071067811865476, sigma_alpha = 1.0,
                    outcome = c("continuous", "binary"),
                    a_scaling = FALSE, b_scaling = TRUE,
                    ridge_move = TRUE, ridge_proposal_sigma = 0.2,
                    standardize = TRUE, random_seed = 0,
                    device = c("auto", "cpu", "gpu"),
                    sur = TRUE, sur_prior_var = 1.0,
                    num_shared_trees = 0, shared_variance_fraction = 0.5,
                    verbose = FALSE, ...) {

  .reject_unsupported(...)
  .check_shared_treatment_options(num_shared_trees, shared_variance_fraction)
  if (num_shared_trees != 0) {
    stop("num_shared_trees requires longbet_multi(); a scalar fit cannot share trees across outcomes.", call. = FALSE)
  }
  outcome <- match.arg(outcome)
  device <- match.arg(device)

  lb <- longbet_py()

  y <- as.matrix(y)
  x <- as.matrix(x)
  z <- as.matrix(z)
  if (is.null(t)) t <- seq_len(ncol(y))
  t <- as.numeric(t)

  if (!identical(dim(y), dim(z))) {
    stop(sprintf("y is %d x %d but z is %d x %d; they must match.",
                 nrow(y), ncol(y), nrow(z), ncol(z)), call. = FALSE)
  }
  if (nrow(x) != nrow(y)) {
    stop(sprintf("x has %d rows but y has %d.", nrow(x), nrow(y)), call. = FALSE)
  }

  config <- lb$LongBetConfig(
    num_sweeps = as.integer(num_sweeps),
    num_burnin = as.integer(num_burnin),
    n_skip = as.integer(n_skip),
    num_chains = as.integer(num_chains),
    inner_loop_length = if (is.null(inner_loop_length)) NULL else as.integer(inner_loop_length),
    num_trees_pr = as.integer(num_trees_pr),
    num_trees_trt = as.integer(num_trees_trt),
    min_points_per_leaf_pr = as.integer(min_points_per_leaf_pr),
    min_points_per_leaf_trt = as.integer(min_points_per_leaf_trt),
    max_depth_pr = as.integer(max_depth_pr),
    max_depth_trt = as.integer(max_depth_trt),
    alpha_split_pr = as.numeric(alpha_split_pr),
    beta_split_pr = as.numeric(beta_split_pr),
    alpha_split_trt = as.numeric(alpha_split_trt),
    beta_split_trt = as.numeric(beta_split_trt),
    num_cutpoints = as.integer(num_cutpoints),
    sig_knl = as.numeric(sig_knl),
    lambda_knl = as.numeric(lambda_knl),
    kernel_type = kernel_type,
    gp_jitter = as.numeric(gp_jitter),
    sigma_m = as.numeric(sigma_m),
    gp_constant_mean = as.logical(gp_constant_mean),
    split_time_ps = as.logical(split_time_ps),
    split_time_trt = as.logical(split_time_trt),
    random_intercept = as.logical(random_intercept),
    gamma_prior_a = as.numeric(gamma_prior_a),
    gamma_prior_b = as.numeric(gamma_prior_b),
    sigma_prior_a = as.numeric(sigma_prior_a),
    sigma_prior_b = as.numeric(sigma_prior_b),
    sigma_b = as.numeric(sigma_b),
    sigma_alpha = as.numeric(sigma_alpha),
    outcome = outcome,
    sample_alpha = as.logical(a_scaling),
    adaptive_coding = as.logical(b_scaling),
    ridge_move = as.logical(ridge_move),
    ridge_proposal_sigma = as.numeric(ridge_proposal_sigma),
    standardize = as.logical(standardize),
    random_seed = as.integer(random_seed),
    device = device,
    sur = as.logical(sur),
    sur_prior_var = as.numeric(sur_prior_var),
    num_shared_trees = as.integer(num_shared_trees),
    shared_variance_fraction = as.numeric(shared_variance_fraction)
  )

  if (isTRUE(verbose)) {
    message(sprintf("longbet: fitting %d x %d panel on device '%s' (%d chain%s).",
                    nrow(y), ncol(y), device, num_chains,
                    if (num_chains == 1) "" else "s"))
  }

  py_model <- lb$LongBet(config)
  py_model$fit(
    y = .as_np_matrix(y, "y"),
    x = .as_np_matrix(x, "x"),
    z = .as_np_matrix(z, "z"),
    t = reticulate::r_to_py(t),
    x_trt = .as_np_matrix(x_trt, "x_trt"),
    x_tv = .as_np_array3(x_tv, "x_tv"),
    x_trt_tv = .as_np_array3(x_trt_tv, "x_trt_tv"),
    ps = if (is.null(ps)) NULL else reticulate::r_to_py(if (is.matrix(ps)) as.matrix(ps) else as.numeric(ps))
  )

  # Serialize the engine state to a raw vector: the R object must hold plain R
  # data, never a live Python handle, or saveRDS() produces something that
  # cannot be reopened.
  tmp <- tempfile(fileext = ".npz")
  on.exit(unlink(tmp), add = TRUE)
  py_model$save(tmp)
  raw_model <- readBin(tmp, what = "raw", n = file.info(tmp)$size)

  trace <- py_model$trace
  as_arr <- function(v) as.array(reticulate::py_to_r(reticulate::import("numpy")$asarray(v)))

  # Draws go last in R and first in JAX; permute once, here.
  flatten_draws <- function(a) {
    if (length(dim(a)) == 3L) {           # (chains, draws, k) -> (k, chains*draws)
      matrix(aperm(a, c(3, 2, 1)), nrow = dim(a)[3])
    } else {                              # (draws, k) -> (k, draws)
      t(a)
    }
  }

  # (chains, draws) -> (chains*draws,) chain-major. In R, as.numeric() traverses
  # column-major, so transpose first to avoid interleaving chains.
  flatten_scalars <- function(a) {
    if (length(dim(a)) == 2L) {
      as.numeric(t(a))
    } else {
      as.numeric(a)
    }
  }

  beta_draws <- as_arr(trace$beta)
  gamma_draws <- as_arr(trace$gamma)
  sigma2_draws <- flatten_scalars(as_arr(trace$sigma2))

  # The live handle is cached in an environment, but the model object also
  # carries `raw_model`, which is what predict() falls back to after a
  # saveRDS/readRDS round trip leaves the cached pointer dangling.
  handle <- new.env(parent = emptyenv())
  handle$py_model <- py_model
  handle$pid <- Sys.getpid()

  fit <- list(
    beta_values = flatten_draws(beta_draws),
    gamma_draws = flatten_draws(gamma_draws),
    sigma_gamma_draws = flatten_scalars(as_arr(trace$sigma_gamma2)),
    # Standardized units, as the reference package reports it: multiply by sdy
    # to recover the residual standard deviation on the data scale.
    sigma0_draws = matrix(sqrt(sigma2_draws), nrow = 1),
    b0_draws = flatten_scalars(as_arr(trace$b0)),
    b1_draws = flatten_scalars(as_arr(trace$b1)),
    alpha_draws = flatten_scalars(as_arr(trace$alpha)),
    sdy = as.numeric(py_model$sdy),
    meany = as.numeric(py_model$meany),
    random_intercept = as.logical(random_intercept),
    outcome = outcome,
    model_params = list(
      burnin = as.integer(num_burnin),
      num_sweeps = as.integer(num_sweeps),
      num_chains = as.integer(num_chains),
      num_trees_pr = as.integer(num_trees_pr),
      num_trees_trt = as.integer(num_trees_trt),
      device = device
    ),
    raw_model = raw_model,
    handle = handle,
    t = t
  )
  class(fit) <- "longbet"
  fit
}

#' @export
print.longbet <- function(x, ...) {
  p <- x$model_params
  cat("LongBet fit (JAX engine)\n")
  cat(sprintf("  Draws:   %d saved, %d burn-in, %d chain%s\n",
              p$num_sweeps, p$burnin, p$num_chains,
              if (p$num_chains == 1) "" else "s"))
  cat(sprintf("  Trees:   %d prognostic / %d treatment\n",
              p$num_trees_pr, p$num_trees_trt))
  cat(sprintf("  Outcome: %s\n", x$outcome))
  cat(sprintf("  Unit random intercept: %s\n",
              if (isTRUE(x$random_intercept)) "yes" else "no"))
  cat(sprintf("  Residual SD (posterior mean): %.4f\n",
              mean(x$sigma0_draws) * x$sdy))
  if (p$num_chains < 2) {
    cat("  Note: R-hat needs at least two chains; set num_chains >= 2.\n")
  }
  invisible(x)
}

#' @export
summary.longbet <- function(object, ...) {
  print(object, ...)
  invisible(object)
}

#' Inspect available JAX compute devices
#'
#' @return Character vector such as `c("cpu:0", "gpu:0")`.
#' @export
longbet_devices <- function() {
  as.character(reticulate::py_to_r(longbet_py()$available_devices()))
}
