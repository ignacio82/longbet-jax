#' Prediction-assisted longitudinal instrumental variables
#'
#' Experimental two-fold conditional cross-fitting with a LongBet trajectory
#' predictor. Uncertainty comes from the joint residual Neyman covariance and
#' analytic Fieller inversion, not posterior quantiles.
#'
#' @param y Complete finite outcome panel, [N x T].
#' @param d Complete binary absorbing adoption panel, [N x T].
#' @param z Binary single-wave encouragement panel, [N x T], or a vector of
#'   randomized assignments. A vector treats every supplied period as post-wave.
#' @param x Optional baseline covariates, [N x P]. All pre-wave y/d columns are
#'   appended automatically; held-out post-wave data are never used as features.
#' @param t Optional increasing observation times.
#' @param learner_config Named list of LongBetIVNuisanceConfig options. Integer
#'   sampling/tree controls must be whole numbers. The predictor selects temporal
#'   pooling on an inner account holdout within each training fold.
#' @param folds Number of folds; this prototype supports exactly two.
#' @param seed Nonnegative integer controlling the conditional split and, unless
#'   overridden in learner_config, the prediction RNG.
#' @param alpha Significance level; default 0.05.
#' @param design Must be "complete_randomization" for individual panel units.
#' @return A serializable list containing table, estimates, covariance,
#'   predictions, unit_scores, fold_ids and inference metadata. Covariance order
#'   is Y at horizon 1, D at horizon 1, Y at horizon 2, D at horizon 2, etc.
#' @details ITTs are unbiased under the stated assignment and splitting design.
#'   Covariance and intervals are asymptotic and pointwise; they require stable
#'   predictions and suitable moment conditions. Full unbounded/disconnected
#'   confidence sets are retained in lower/upper and lower2/upper2. A CACE
#'   interpretation additionally requires exclusion, monotonicity, relevance,
#'   and appropriate adoption-history restrictions. This does not implement an
#'   exact sharp-null permutation test or prove superiority over other learners.
#' @export
longbet_iv <- function(y, d, z, x = NULL, t = NULL,
                       learner_config = NULL, folds = 2L, seed = 0L,
                       alpha = 0.05, design = "complete_randomization") {
  integer_option <- function(value, name) {
    if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
        !is.finite(value) || value != floor(value) ||
        value < 0 || value > .Machine$integer.max) {
      stop(name, " must be a nonnegative integer.", call. = FALSE)
    }
    as.integer(value)
  }
  folds <- integer_option(folds, "folds")
  seed <- integer_option(seed, "seed")
  if (!is.null(learner_config)) {
    if (!is.list(learner_config) ||
        (length(learner_config) && (is.null(names(learner_config)) ||
          any(!nzchar(names(learner_config))) || anyDuplicated(names(learner_config))))) {
      stop("learner_config must be a named list or NULL.", call. = FALSE)
    }
    integer_names <- c("baseline_trees", "effect_trees", "cutpoints", "burnin",
                       "draws", "chains", "max_interaction_rules", "seed")
    for (name in intersect(names(learner_config), integer_names)) {
      learner_config[[name]] <- integer_option(learner_config[[name]], name)
    }
  }
  lb <- longbet_py()
  res <- lb$longbet_iv(
    y = reticulate::r_to_py(as.matrix(y)),
    d = reticulate::r_to_py(as.matrix(d)),
    z = reticulate::r_to_py(if (is.null(dim(z))) as.numeric(z) else as.matrix(z)),
    x = if (is.null(x)) NULL else reticulate::r_to_py(as.matrix(x)),
    t = if (is.null(t)) NULL else reticulate::r_to_py(as.numeric(t)),
    learner_config = if (is.null(learner_config) || !length(learner_config)) NULL
      else reticulate::r_to_py(learner_config),
    folds = folds, seed = seed, alpha = alpha, design = design
  )
  rows <- .encouragement_as_r(res$table)
  table <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
  rownames(table) <- NULL
  list(table = table,
       estimates = .encouragement_as_r(res$estimates),
       covariance = .encouragement_as_r(res$covariance),
       predictions = .encouragement_as_r(res$predictions),
       unit_scores = .encouragement_as_r(res$unit_scores),
       fold_ids = as.integer(.encouragement_as_r(res$fold_ids)),
       metadata = .encouragement_as_r(res$metadata))
}
