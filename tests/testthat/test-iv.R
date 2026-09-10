test_that("IV front door preserves Python estimates and serializes full results", {
  skip_without_engine()
  dat <- make_panel(n = 48, Tn = 4, t0 = 3)
  config <- list(baseline_trees = 1, effect_trees = 1, burnin = 2, draws = 3,
                 chains = 1, length_scales = list(0), interaction_partitions = FALSE)
  result <- longbet_iv(dat$y, dat$z, dat$z, dat$x, learner_config = config, seed = 7)
  expect_equal(dim(result$estimates), c(2L, 2L))
  expect_equal(dim(result$covariance), c(4L, 4L))
  expect_equal(dim(result$predictions), c(48L, 2L, 2L, 2L))
  expect_equal(nrow(result$table), 6L)
  expect_equal(apply(result$unit_scores, c(2, 3), mean), result$estimates)
  expect_identical(result$metadata$inference, "asymptotic_average_itt_moment")
  expect_false(result$metadata$exact_randomization)
  expect_false(result$metadata$posterior_causal_intervals)
  expect_identical(unserialize(serialize(result, NULL)), result)
  replay <- longbet_iv(dat$y, dat$z, dat$z, dat$x, learner_config = config, seed = 7)
  expect_equal(replay$estimates, result$estimates)
  expect_equal(replay$covariance, result$covariance)
  expect_error(longbet_iv(dat$y, dat$z, dat$z, seed = 1.5), "integer")
  expect_error(longbet_iv(dat$y, dat$z, dat$z, learner_config = list(draws = 1.5)), "integer")
})

test_that("IV front door defaults match the shared contract", {
  skip_if_not_installed("yaml")
  path <- system.file("contract", "longbet-api.yaml", package = "longbet")
  spec <- yaml::read_yaml(path)$crossfit_iv_api
  actual <- formals(longbet_iv)
  expect_setequal(names(actual), c(spec$required, names(spec$defaults)))
  for (name in names(spec$defaults)) {
    expect_equal(eval(actual[[name]]), spec$defaults[[name]], info = name)
  }
})
