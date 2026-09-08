# Every test needs the Python engine; skip cleanly when it is absent so that
# `R CMD check` passes on a machine with no Python at all.
skip_without_engine <- function() {
  testthat::skip_if_not_installed("reticulate")
  ok <- tryCatch({
    reticulate::import("longbet")
    TRUE
  }, error = function(e) FALSE)
  testthat::skip_if_not(ok, "the longbet-jax Python engine is not available")
}

make_panel <- function(seed = 42, n = 40, Tn = 8, p = 3, t0 = 4, effect = 1.2) {
  set.seed(seed)
  x <- matrix(rnorm(n * p), n, p)
  z <- matrix(0, n, Tn)
  treated <- seq_len(floor(n / 2))
  z[treated, t0:Tn] <- 1
  t_vec <- seq_len(Tn)
  s <- ifelse(z == 1, pmax(col(z) - (t0 - 1), 0), 0)
  gamma <- rnorm(n, 0, 0.5)
  y <- 0.5 * x[, 1] + gamma + 0.1 * col(z) + effect * sqrt(s) +
    matrix(rnorm(n * Tn, 0, 0.25), n, Tn)
  list(x = x, y = y, z = z, t = t_vec, s = s, gamma = gamma, effect = effect)
}
