# Internal helpers shared by the R front door.

.as_np_vector <- function(x) {
  # reticulate converts length-one R vectors to Python scalars by default.
  reticulate::import("numpy", convert = FALSE)$atleast_1d(reticulate::r_to_py(x))
}

#' @keywords internal
.as_np_matrix <- function(x, name) {
  if (is.null(x)) return(NULL)
  x <- as.matrix(x)
  if (!is.numeric(x)) {
    stop(sprintf("`%s` must be numeric; got %s.", name, class(x)[1]), call. = FALSE)
  }
  reticulate::r_to_py(x)
}

#' @keywords internal
.as_np_array3 <- function(x, name) {
  if (is.null(x)) return(NULL)
  if (length(dim(x)) != 3L) {
    stop(sprintf("`%s` must be a 3-D array with dimensions [N, T, P].", name), call. = FALSE)
  }
  reticulate::r_to_py(x)
}

#' @keywords internal
.reject_unsupported <- function(...) {
  extras <- list(...)
  if (length(extras) == 0L) return(invisible(NULL))
  nms <- names(extras)
  nms <- nms[nzchar(nms)]

  # pcat is the one argument of the reference C++ interface with no equivalent
  # here. Silently ignoring it would fit unordered categoricals as ordered
  # numerics, which is a different model, so refuse instead.
  if ("pcat" %in% nms && !identical(extras$pcat, 0)) {
    stop(
      "`pcat` is not supported. This engine has no notion of an unordered ",
      "categorical: every column is treated as ordered numeric.\n",
      "One-hot encode categorical variables with more than two levels before ",
      "calling longbet(), and note that this is a real modelling difference ",
      "from the reference C++ implementation.",
      call. = FALSE
    )
  }
  unknown <- setdiff(nms, "pcat")
  if (length(unknown)) {
    stop("Unsupported argument(s): ", paste(sQuote(unknown), collapse = ", "),
         ".\nSee ?longbet for the supported arguments.", call. = FALSE)
  }
  invisible(NULL)
}
.check_shared_treatment_options <- function(num_shared_trees, shared_variance_fraction) {
  if (!is.numeric(num_shared_trees) || length(num_shared_trees) != 1L ||
      !is.finite(num_shared_trees) || num_shared_trees < 0 ||
      num_shared_trees != floor(num_shared_trees)) {
    stop("num_shared_trees must be a nonnegative integer scalar.", call. = FALSE)
  }
  if (!is.numeric(shared_variance_fraction) || length(shared_variance_fraction) != 1L ||
      !is.finite(shared_variance_fraction) || shared_variance_fraction <= 0 ||
      shared_variance_fraction >= 1) {
    stop("shared_variance_fraction must be a finite scalar strictly between 0 and 1.", call. = FALSE)
  }
}
