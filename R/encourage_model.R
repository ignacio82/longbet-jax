# Plain R conversion is essential here: neither model nor prediction objects may
# retain reticulate handles across saveRDS or a new R process.
.encouragement_plain <- function(value) {
  if (inherits(value, "python.builtin.object") && reticulate::py_has_attr(value, "_asdict")) {
    value <- value$`_asdict`()
  }
  value <- .encouragement_as_r(value)
  if (is.data.frame(value)) {
    attr(value, "pandas.index") <- NULL
    rownames(value) <- NULL
    for (name in names(value)) value[[name]] <- .encouragement_plain(value[[name]])
  } else if (is.list(value)) {
    value <- lapply(value, .encouragement_plain)
  } else if (inherits(value, "python.builtin.object")) {
    stop("The engine returned an unsupported live Python value.", call. = FALSE)
  }
  value
}

.encouragement_load <- function(object) {
  if (!inherits(object, "longbet_encourage") || !is.raw(object$raw_archive)) {
    stop("object must be a fitted longbet_encourage with a raw archive.", call. = FALSE)
  }
  path <- tempfile(fileext = ".npz")
  on.exit(unlink(path), add = TRUE)
  writeBin(object$raw_archive, path)
  longbet_py()
  reticulate::import("longbet", convert = FALSE)$LongBetEncourage$load(path)
}

.encouragement_config_args <- function(config, lb) {
  if (!is.list(config) || (length(config) &&
      (is.null(names(config)) || anyNA(names(config)) ||
       any(!nzchar(names(config))) || anyDuplicated(names(config))))) {
    stop("config must be a list of uniquely named sampler options.", call. = FALSE)
  }
  defaults <- .encouragement_plain(reticulate::import("dataclasses")$asdict(lb$LongBetConfig()))
  if (any(!names(config) %in% names(defaults)) || "outcome" %in% names(config)) {
    stop("config names must be LongBetConfig options; set outcome separately.", call. = FALSE)
  }
  # R's ordinary numeric literals are doubles. Respect Python integer fields
  # without silently rounding noninteger values or maintaining a second config.
  for (name in names(config)) {
    if (is.integer(defaults[[name]]) || identical(name, "inner_loop_length")) {
      value <- config[[name]]
      if (is.null(value) && identical(name, "inner_loop_length")) next
      if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
          value != trunc(value) || abs(value) > .Machine$integer.max) {
        stop(sprintf("config$%s must be a finite integer.", name), call. = FALSE)
      }
      config[[name]] <- as.integer(value)
    }
  }
  config
}

#' Joint reduced forms for a randomized encouragement experiment
#'
#' Fits the outcome and actual adoption jointly on assigned encouragement using
#' the Python engine. Choose the LPM or probability-scale probit first stage
#' explicitly; calibration has not established a preferred default.
#'
#' @param y Complete outcome matrix, `[N x T]`.
#' @param d,z Actual adoption and assigned encouragement, complete absorbing
#'   binary `[N x T]` matrices. Supports a single wave and a permanent holdout.
#' @param x Baseline covariates, `[N x P]`.
#' @param t Increasing calendar vector with whole-unit gaps; defaults to `1:T`.
#' @param x_trt Optional baseline treatment-forest covariates.
#' @param first_stage Required choice, `"lpm"` or `"probit"`.
#' @param outcome `"continuous"` or `"binary"` for `y`.
#' @param config Named list of Python `LongBetConfig` sampler options. With no
#'   variance options supplied, the wrapper uses proper IG(2,1) innovation priors.
#'   Explicit improper innovation priors are rejected. Other sampler defaults
#'   follow `longbet()`. Set `num_chains`, `num_burnin`, `num_sweeps`, and
#'   `random_seed` here; convergence still requires checking.
#' @param verbose Whether to print a fitting message.
#' @return A `longbet_encourage` object containing a versioned raw archive,
#'   metadata and configuration, with no live Python handles. The archive
#'   includes original outcomes, adoption, assignment and covariates so prediction
#'   replays the same units and target. It can be saved using `saveRDS()`.
#' @details Both ITTs average over all original study units with equal weights.
#'   Binary contrasts are probability differences. The Wald ratio is not
#'   automatically CACE; exclusion, monotonicity, relevance and appropriate
#'   adoption-history assumptions are additional requirements. SUR couples
#'   innovations when a continuous equation is present; unit-intercept priors
#'   are independent across equations. With binary Y and a probit first stage,
#'   both innovation-loading rows are fixed at zero, so `sur = TRUE` does not
#'   couple the innovations. Such a fit is independent across equations by
#'   default. Optional shared treatment trees can couple forest uncertainty,
#'   but do not create correlated binary innovations or unit intercepts.
#' @export
longbet_encourage <- function(y, d, z, x, t = NULL, x_trt = NULL,
                             first_stage, outcome = "continuous",
                             config = list(), verbose = FALSE) {
  if (missing(first_stage)) stop("Choose first_stage = 'lpm' or 'probit' explicitly.", call. = FALSE)
  first_stage <- match.arg(first_stage, c("lpm", "probit"))
  outcome <- match.arg(outcome, c("continuous", "binary"))
  lb <- longbet_py()
  options <- .encouragement_config_args(config, lb)
  engine <- do.call(lb$LongBetEncourage,
                   c(list(first_stage = first_stage, outcome = outcome), options))
  if (isTRUE(verbose)) message("Fitting joint encouragement reduced forms.")
  engine$fit(y = as.matrix(y), d = as.matrix(d), z = as.matrix(z), x = as.matrix(x),
             t = if (is.null(t)) NULL else as.numeric(t),
             x_trt = if (is.null(x_trt)) NULL else as.matrix(x_trt))
  path <- tempfile(fileext = ".npz")
  on.exit(unlink(path), add = TRUE)
  engine$save(path)
  structure(list(
    raw_archive = readBin(path, what = "raw", n = file.info(path)$size),
    metadata = .encouragement_plain(engine$metadata),
    config = .encouragement_plain(reticulate::import("dataclasses")$asdict(engine$config)),
    call = match.call()
  ), class = "longbet_encourage")
}

#' Predict common-population encouragement effects
#'
#' @param object A fitted `longbet_encourage` object, including one read by
#'   `readRDS()` in another session.
#' @param summary_only Avoid retaining full unit/period/draw arrays. Aggregate
#'   ITT and ratio draws are always retained.
#' @param groups Optional prespecified baseline group labels, length `N`.
#' @param block_size Optional positive integer cells evaluated per block.
#' @param standardization `"conditional"` retains each fitted unit intercept;
#'   `"population"` integrates a fresh normal intercept and has a different
#'   target from the finite-study-unit reference.
#' @param alpha Tail probability for pointwise intervals.
#' @param min_ess,max_rhat Diagnostic thresholds; passing is not a coverage claim.
#' @param practical_threshold Optional positive adoption risk difference for a
#'   separate posterior practical-relevance probability, not an IV strength cutoff.
#' @param ... Reserved; unsupported arguments raise an error.
#' @return A plain-R `longbet_encourage.pred` list with `effects`, `reference`,
#'   `wald`, `wald_unchecked`, `first_stage`, `comparison`, `diagnostics`, `draws`,
#'   `cell_draws`, `cell_summaries`, `weights`, group information and metadata.
#'   Draw arrays retain axes `(group, horizon, chain, retained_draw)`.
#'   Cell arrays have axes `(unit, period, chain-major draw)` when requested.
#'   Wald summaries use medians, never a default ratio mean. `wald` withholds
#'   unconverged intervals; `wald_unchecked` allows inspection of those summaries.
#'   The `reference` contains separate normal Fieller/Anderson--Rubin-style
#'   confidence sets with infinite endpoints and unavailable cases preserved.
#' @export
predict.longbet_encourage <- function(object, summary_only = TRUE, groups = NULL,
                                     block_size = NULL, standardization = "conditional",
                                     alpha = 0.05, min_ess = 400, max_rhat = 1.01,
                                     practical_threshold = NULL, ...) {
  .reject_unsupported(...)
  if (!is.null(block_size)) {
    if (!is.numeric(block_size) || length(block_size) != 1L ||
        !is.finite(block_size) || block_size <= 0 || block_size != trunc(block_size) ||
        block_size > .Machine$integer.max) {
      stop("block_size must be a positive integer.", call. = FALSE)
    }
    block_size <- as.integer(block_size)
  }
  if (is.factor(groups)) groups <- as.character(groups)
  engine <- .encouragement_load(object)
  pred <- engine$predict(summary_only = summary_only, groups = groups,
                         block_size = block_size, standardization = standardization,
                         alpha = alpha)
  standardized <- pred$standardized
  covariance <- pred$itt_covariance()
  structure(list(
    effects = .encouragement_plain(pred$effects(min_ess = min_ess, max_rhat = max_rhat)),
    reference = .encouragement_plain(pred$reference),
    wald = .encouragement_plain(pred$wald(min_ess = min_ess, max_rhat = max_rhat)),
    wald_unchecked = .encouragement_plain(pred$wald(
      require_convergence = FALSE, min_ess = min_ess, max_rhat = max_rhat)),
    first_stage = .encouragement_plain(pred$first_stage(
      practical_threshold = practical_threshold, min_ess = min_ess, max_rhat = max_rhat)),
    comparison = .encouragement_plain(pred$comparison()),
    diagnostics = .encouragement_plain(pred$stability(min_ess = min_ess, max_rhat = max_rhat)),
    draws = .encouragement_plain(pred$draws),
    itt_covariance = .encouragement_plain(covariance$to_numpy()),
    itt_covariance_index = .encouragement_plain(covariance$index$to_frame(index = FALSE)),
    cell_draws = .encouragement_plain(standardized$cell_draws),
    cell_summaries = .encouragement_plain(standardized$cell_summaries),
    weights = .encouragement_plain(standardized$weights),
    group_labels = unlist(.encouragement_plain(standardized$group_labels), use.names = FALSE),
    group_counts = .encouragement_plain(standardized$group_counts),
    periods = .encouragement_plain(standardized$periods),
    period_indices = .encouragement_plain(standardized$period_indices),
    horizons = .encouragement_plain(standardized$horizons),
    metadata = .encouragement_plain(pred$metadata),
    alpha = alpha
  ), class = "longbet_encourage.pred")
}

#' Extract encouragement posterior and comparison tables
#'
#' These return tables computed by the Python engine during prediction; they
#' require no live Python process and survive `saveRDS()`.
#' @param pred A `longbet_encourage.pred` object.
#' @param require_convergence Withhold posterior ratio intervals failing the
#'   diagnostic thresholds selected at prediction. Raw draws remain available.
#' @return A data frame. `encouragement_wald()` returns posterior ratio summaries;
#'   frequentist confidence sets are instead `pred$reference`.
#'   `encouragement_first_stage()` separates precision and practical relevance.
#'   `encouragement_comparison()` gives descriptive differences and does not
#'   fabricate a disagreement flag without paired sampling covariance.
#' @export
encouragement_wald <- function(pred, require_convergence = TRUE) {
  if (!inherits(pred, "longbet_encourage.pred")) stop("pred must be an encouragement prediction.", call. = FALSE)
  if (!is.logical(require_convergence) || length(require_convergence) != 1L || is.na(require_convergence)) {
    stop("require_convergence must be TRUE or FALSE.", call. = FALSE)
  }
  if (require_convergence) pred$wald else pred$wald_unchecked
}

#' @rdname encouragement_wald
#' @export
encouragement_first_stage <- function(pred) {
  if (!inherits(pred, "longbet_encourage.pred")) stop("pred must be an encouragement prediction.", call. = FALSE)
  pred$first_stage
}

#' @rdname encouragement_wald
#' @export
encouragement_comparison <- function(pred) {
  if (!inherits(pred, "longbet_encourage.pred")) stop("pred must be an encouragement prediction.", call. = FALSE)
  pred$comparison
}

#' @export
print.longbet_encourage <- function(x, ...) {
  cat(sprintf("LongBet encouragement model: %s outcome, %s first stage\n",
              x$metadata$outcome, x$metadata$first_stage))
  cat(sprintf("  %d study units, %d periods; posterior calibration not established\n",
              x$metadata$n_units, x$metadata$n_periods))
  invisible(x)
}

#' @export
print.longbet_encourage.pred <- function(x, ...) {
  cat(sprintf("Encouragement prediction: %d groups, %d horizons (%s)\n",
              length(x$group_labels), length(x$horizons), x$metadata$standardization))
  cat("  Posterior summaries and design-based reference confidence sets are separate.\n")
  invisible(x)
}

#' Paired bootstrap comparison of model and reference encouragement effects
#'
#' Refits both estimators on the same whole-unit bootstrap samples, retaining
#' their paired covariance. This is model/reference comparison, not a new IV
#' identification assumption or an exact randomization test.
#' @param object A fitted `longbet_encourage` object.
#' @param replicates Number of refitted bootstrap samples; at least 100 and no
#'   failed fits/diagnostics are needed before any disagreement flag is issued.
#' @param groups Optional fixed baseline group labels.
#' @param random_seed Seed for bootstrap sampling and refit seed generation.
#' @param alpha,min_ess,max_rhat Interval and diagnostic settings.
#' @return Plain-R list containing `table`, `replicates`, `failures`, and
#'   `metadata`. Replicate axes are `(replicate, group, horizon, quantity,
#'   estimator)`, with quantities Y,D and estimators model,reference. Failures
#'   remain missing and are reported rather than removed.
#' @export
encouragement_bootstrap_comparison <- function(object, replicates = 200L, groups = NULL,
                                              random_seed = 0L, alpha = 0.05,
                                              min_ess = 400, max_rhat = 1.01) {
  for (name in c("replicates", "random_seed")) {
    value <- get(name)
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
        value != trunc(value) || value < 0 || value > .Machine$integer.max) {
      stop(sprintf("%s must be a nonnegative integer.", name), call. = FALSE)
    }
  }
  if (is.factor(groups)) groups <- as.character(groups)
  result <- .encouragement_load(object)$bootstrap_comparison(
    replicates = as.integer(replicates), groups = groups,
    random_seed = as.integer(random_seed), alpha = alpha,
    min_ess = min_ess, max_rhat = max_rhat)
  list(table = .encouragement_plain(result$table),
       replicates = .encouragement_plain(result$replicates),
       failures = .encouragement_plain(result$failures),
       metadata = .encouragement_plain(result$metadata))
}

.encouragement_mapping <- function(value, name) {
  if (is.null(value)) return(NULL)
  if (is.null(names(value)) || anyNA(names(value)) || any(!nzchar(names(value))) ||
      anyDuplicated(names(value))) {
    stop(sprintf("%s must have unique nonempty names.", name), call. = FALSE)
  }
  as.list(value)
}

#' Declare the randomization scheme for encouragement inference
#'
#' @param assignment `"complete"`, `"bernoulli"`, or `"staggered"`. The last
#'   means independent categorical cohort assignment, not fixed cohort quotas.
#' @param blocks,clusters Baseline labels, length N. Each cluster must lie within
#'   one block and share an encouragement path.
#' @param target `"unit"` gives equal participant weight; `"cluster"` gives
#'   equal weight to included clusters and equal within-cluster subgroup weights.
#' @param probabilities Known scalar or vector Bernoulli assignment probabilities.
#'   For staggered assignment supply a named list mapping zero-based start column
#'   indices and optionally `"never"` to probabilities; they sum to one per
#'   independent randomization unit. Vectors may have length N or cluster count.
#' @param block_weights Optional named fixed nonnegative block weights summing
#'   to one. Every positive-weight block must contain the subgroup.
#' @param cohort_weights Required for staggered designs: named positive weights
#'   summing to one over the finite cohorts of interest, using zero-based indices.
#' @param control `"not_yet"` uses later cohorts and permanent holdouts;
#'   `"never"` uses holdouts. Both require no anticipation.
#' @return A plain R `encouragement_design` specification. Array and support
#'   checks occur in the Python engine when effects are computed.
#' @export
encouragement_design <- function(assignment = "complete", blocks = NULL, clusters = NULL,
                                 target = "unit", probabilities = NULL, block_weights = NULL,
                                 cohort_weights = NULL, control = "not_yet") {
  assignment <- match.arg(assignment, c("complete", "bernoulli", "staggered"))
  target <- match.arg(target, c("unit", "cluster"))
  control <- match.arg(control, c("not_yet", "never"))
  if (is.factor(blocks)) blocks <- as.character(blocks)
  if (is.factor(clusters)) clusters <- as.character(clusters)
  if (assignment == "staggered") probabilities <- .encouragement_mapping(probabilities, "probabilities")
  structure(list(assignment = assignment, blocks = blocks, clusters = clusters,
                 target = target, probabilities = probabilities,
                 block_weights = .encouragement_mapping(block_weights, "block_weights"),
                 cohort_weights = .encouragement_mapping(cohort_weights, "cohort_weights"),
                 control = control), class = "encouragement_design")
}

#' Encouragement effects under an explicit randomization design
#'
#' @param y,d,z Complete outcome, actual-adoption and encouragement panels `[N x T]`.
#' @param t Increasing calendar vector with whole-unit gaps; defaults to `1:T`.
#' @param design An [encouragement_design()] specification; NULL selects
#'   individual-unit complete randomization with equal unit weights.
#' @param groups Optional prespecified baseline subgroup labels, length N.
#' @param alpha Tail probability for pointwise normal intervals and AR-style sets.
#' @return A plain-R list with `table`, numeric `covariance`, `covariance_index`,
#'   and `metadata`. Rows and columns of `covariance` both follow the rows of
#'   `covariance_index`, whose columns are `contrast_id` and `quantity` (Y or D ITT).
#' @details Uses fixed-population Horvitz--Thompson contrasts and joint
#'   randomization-unit covariance, preserving shared controls, cross-horizon
#'   dependence and small or unsupported groups. Blocked complete designs use
#'   Neyman covariance. Bernoulli/categorical designs use a conservative uncentered
#'   covariance bound that can be loose and depends on outcome location.
#'   Subgroup HT estimates do not divide by random subgroup arm sizes.
#'   Missing cohort support makes a specified aggregate unavailable without
#'   renormalizing weights. Intervals are pointwise normal approximations, not
#'   simultaneous bands, exact permutation tests or posterior intervals.
#' @export
design_encouragement_effects <- function(y, d, z, t = NULL, design = NULL,
                                        groups = NULL, alpha = 0.05) {
  if (is.null(design)) design <- encouragement_design()
  if (!inherits(design, "encouragement_design")) {
    stop("design must be an encouragement_design() specification.", call. = FALSE)
  }
  if (is.factor(groups)) groups <- as.character(groups)
  longbet_py()
  lb <- reticulate::import("longbet", convert = FALSE)
  declared <- do.call(lb$EncouragementDesign, unclass(design))
  result <- lb$design_encouragement_effects(y = as.matrix(y), d = as.matrix(d), z = as.matrix(z),
    t = if (is.null(t)) NULL else as.numeric(t), design = declared, groups = groups, alpha = alpha)
  list(table = .encouragement_plain(result$table),
       covariance = .encouragement_plain(result$covariance$to_numpy()),
       covariance_index = .encouragement_plain(result$covariance$index$to_frame(index = FALSE)),
       metadata = .encouragement_plain(result$metadata))
}
