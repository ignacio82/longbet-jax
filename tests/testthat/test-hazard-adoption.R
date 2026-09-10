test_that("hazard metadata states the fixed-basis posterior and calibration limits", {
  skip_without_engine()
  n <- 12L
  z <- matrix(0, n, 3); z[1:6, 2:3] <- 1
  d <- z; d[1:2, ] <- 0; d[9:10, 3] <- 1
  result <- hazard_adoption_effects(d, z, trees = 1, chains = 2,
                                    burnin = 5, draws = 12, seed = 81)
  expect_true(result$metadata$experimental)
  expect_identical(result$metadata$calibration_status, "not_established")
  expect_identical(result$metadata$topology, "fixed_random_stumps_shared_across_chains")
  expect_false(result$metadata$weak_instrument_robust)
  expect_false(result$metadata$causal_outcome_inference_supported)
  expect_identical(result$metadata$inference_version, "hazard_fixed_basis_v2")
  expect_equal(dim(result$draws$stock_itt), c(2L, 12L, 2L))
  expect_equal(dim(result$draws$xi), c(2L, 12L))
  expect_identical(unserialize(serialize(result, NULL)), result)
  expect_error(hazard_adoption_effects(d, z, draws = 1.5), "integer")
  expect_error(hazard_adoption_effects(d, z, seed = -1), "integer")
})
