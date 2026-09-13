test_that("longbet_orthogonal_iv fits and predicts conditional CACE without denominator division", {
  skip_without_engine()
  set.seed(42)
  n <- 12L; periods <- 4L
  x <- matrix(rnorm(n * 2), n, 2)
  z <- matrix(0, n, periods); z[1:6, 2:4] <- 1
  d <- matrix(0, n, periods)
  for (i in seq_len(n)) {
    adopted <- FALSE
    for (s in 2:periods) {
      if (adopted) {
        d[i, s] <- 1
      } else if (z[i, s] == 1 && runif(1) > 0.3) {
        d[i, s] <- 1
        adopted <- TRUE
      }
    }
  }
  y <- 10 + x[, 1] + 3.5 * d + matrix(rnorm(n * periods), n, periods)
  cfg <- list(num_chains = 1, num_sweeps = 6, num_burnin = 4,
              num_trees_pr = 2, num_trees_trt = 2, random_seed = 101)

  fit <- longbet_orthogonal_iv(y, d, z, x, config = cfg)
  expect_s3_class(fit, "longbet_orthogonal_iv")

  pred <- predict_conditional(fit, cost = 2.0, hurdle = 0.5)
  expect_s3_class(pred, "longbet_conditional_pred")
  expect_equal(dim(pred$cace$mean), c(12L, 3L))
  expect_equal(dim(pred$citt_y$mean), c(12L, 3L))
  expect_equal(dim(pred$citt_d$mean), c(12L, 3L))
  expect_true(all(is.finite(pred$cace$median)))

  # Out of sample
  x_new <- matrix(rnorm(5 * 2), 5, 2)
  pred_new <- predict(fit, new_x = x_new, cost = 2.0, capacity = 2)
  expect_equal(dim(pred_new$cace$mean), c(5L, 3L))
  expect_lte(sum(pred_new$decision), 2)
})
