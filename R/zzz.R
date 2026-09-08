.onLoad <- function(libname, pkgname) {
  if (requireNamespace("reticulate", quietly = TRUE)) {
    # Declares the requirement; it does not initialise Python. The ephemeral
    # uv-backed environment is created only when reticulate first starts a
    # Python session, so loading the package downloads nothing.
    tryCatch(
      reticulate::py_require("longbet-jax>=0.1,<0.2"),
      error = function(e) NULL
    )
  }
}

# Cached module handle; reticulate objects must never be stored in a fitted
# model object, because they do not survive saveRDS.
.longbet_env <- new.env(parent = emptyenv())

#' Import the Python engine
#'
#' @return The `longbet` Python module.
#' @keywords internal
longbet_py <- function() {
  if (!is.null(.longbet_env$module)) {
    return(.longbet_env$module)
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required to use longbet.\n",
         "Install it with install.packages('reticulate').", call. = FALSE)
  }
  module <- tryCatch(
    reticulate::import("longbet", delay_load = FALSE),
    error = function(e) {
      stop("The Python engine 'longbet-jax' is not available.\n",
           "Install it with:  pip install longbet-jax\n",
           "or point reticulate at an environment that has it, e.g.\n",
           "  reticulate::use_virtualenv('/path/to/.venv', required = TRUE)\n",
           "Original error: ", conditionMessage(e), call. = FALSE)
    }
  )
  .longbet_env$module <- module
  module
}
