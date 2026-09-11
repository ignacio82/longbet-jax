# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

test_that("shared treatment options validate without silently truncating", {
  expect_error(longbet_multi(NULL, NULL, NULL, num_shared_trees = 1.5), "integer scalar")
  expect_error(longbet_multi(NULL, NULL, NULL, num_shared_trees = TRUE), "integer scalar")
  expect_error(longbet_multi(NULL, NULL, NULL, shared_variance_fraction = NA_real_), "finite scalar")
  expect_error(longbet(NULL, NULL, NULL, num_shared_trees = 1), "requires longbet_multi")
})

test_that("shared/private three-outcome fits survive fresh-process R roundtrips", {
  skip_without_engine()
  d <- make_panel(n = 24)
  y <- list(gmv = d$y, hours = -d$y + .2*d$x[, 1],
            complaint = matrix(as.numeric(d$y > mean(d$y)), nrow(d$y)))
  fit <- longbet_multi(y, d$x, d$z, t = d$t,
    outcome = c("continuous", "continuous", "binary"),
    num_burnin = 5, num_sweeps = 6, n_skip = 2, num_chains = 2,
    num_trees_pr = 2, num_trees_trt = 4, num_shared_trees = 2,
    sigma_prior_a = 2, sigma_prior_b = 1,
    shared_variance_fraction = .4, max_depth_pr = 3, max_depth_trt = 3,
    min_points_per_leaf_pr = 2, min_points_per_leaf_trt = 2,
    random_seed = 52)
  expect_identical(fit$sampler_semantics, "shared_private_sur_v1")
  expect_identical(fit$num_shared_trees, 2L)
  expect_identical(fit$shared_variance_fraction, .4)
  expect_equal(fit["complaint"]$multi_origin$provenance, fit$provenance)
  p <- predict(fit, d$x, d$z, t = d$t)
  expect_identical(p$sampler_semantics, "shared_private_sur_v1")
  conditions <- list(gmv = function(e) e > 0, hours = function(e) e < 0,
                     complaint = function(e) e < 0)
  joint <- joint_prob(p, conditions)
  manual <- apply((effect_draws(p, "gmv") > 0) & (effect_draws(p, "hours") < 0) &
                  (effect_draws(p, "complaint") < 0), c(1,2), mean)
  expect_equal(joint, manual)
  expect_equal(dim(effect_draws(p, "complaint")), c(nrow(d$y), ncol(d$y), 12L))
  expect_output(print(fit), "2 shared.*2 private")
  payload <- tempfile(fileext = ".rds"); output <- tempfile(fileext = ".rds")
  on.exit(unlink(c(payload, output)), add = TRUE)
  saveRDS(list(fit = fit, child = fit["complaint"], pred = p, conditions = conditions, d = d), payload)
  loader <- if (requireNamespace("pkgload", quietly = TRUE) && pkgload::is_dev_package("longbet")) {
    sprintf("pkgload::load_all(%s, quiet=TRUE)", deparse(getNamespaceInfo(asNamespace("longbet"), "path")))
  } else "library(longbet)"
  script <- sprintf(paste0("%s; a <- readRDS(%s); p <- predict(a$fit,a$d$x,a$d$z); ",
    "c <- predict(a$child,a$d$x,a$d$z); saveRDS(list(e=effect_draws(p,'complaint'), ",
    "child=effect_draws(c), j=joint_prob(a$pred,a$conditions), ",
    "origin=reticulate::py_to_r(rehydrate_jax_model(a$child)$multi_origin)),%s)"),
    loader,deparse(payload),deparse(output))
  logs <- system2(file.path(R.home("bin"), "Rscript"), c("-e", shQuote(script)), stdout=TRUE, stderr=TRUE)
  expect_null(attr(logs,"status"), info=paste(logs,collapse="\n"))
  fresh <- readRDS(output)
  expect_equal(fresh$e, effect_draws(p, "complaint"))
  expect_equal(fresh$child, effect_draws(p, "complaint"))
  expect_equal(fresh$j, joint)
  expect_equal(fresh$origin, fit["complaint"]$multi_origin)
})
