% Generated from R/multi.R -- regenerate with roxygen2::roxygenise().
\name{longbet_multi}
\alias{longbet_multi}
\title{LongBetMulti: Coupled Bayesian ensemble trees for multiple outcomes}
\usage{
longbet_multi(y, x, z, t = NULL, x_trt = NULL, x_tv = NULL, x_trt_tv = NULL,
  ps = NULL, num_sweeps = 250, num_burnin = 2000, n_skip = 2, num_chains = 4,
  inner_loop_length = NULL, num_trees_pr = 20, num_trees_trt = 20,
  min_points_per_leaf_pr = 10, min_points_per_leaf_trt = 10,
  max_depth_pr = 10, max_depth_trt = 10, alpha_split_pr = 0.95,
  beta_split_pr = 2, alpha_split_trt = 0.25, beta_split_trt = 3,
  num_cutpoints = 100, sig_knl = 1, lambda_knl = 1, kernel_type = "se",
  gp_jitter = 1e-06, sigma_m = 1, gp_constant_mean = TRUE,
  split_time_ps = TRUE, split_time_trt = TRUE, random_intercept = TRUE,
  gamma_prior_a = 1, gamma_prior_b = 0.1, sigma_prior_a = 0,
  sigma_prior_b = 0, sigma_b = 0.7071067811865476, sigma_alpha = 1,
  outcome = "continuous", outcome_names = NULL, a_scaling = FALSE,
  b_scaling = TRUE, ridge_move = TRUE, ridge_proposal_sigma = 0.2,
  standardize = TRUE, random_seed = 0, device = c("auto", "cpu", "gpu"),
  sur = TRUE, sur_prior_var = 1, num_shared_trees = 0,
  shared_variance_fraction = 0.5, verbose = FALSE, ...)
}
\arguments{
\item{y}{Outcome panels: a list of \code{[N x T]} matrices, or a 3-D numeric array
\code{[N x T x M]}.}
\item{x}{Covariates for the prognostic forest, \code{[N x P]}.}
\item{z}{Binary treatment indicator, \code{[N x T]}. Must be absorbing.}
\item{t}{Calendar time vector of length \code{T}. Defaults to \code{1:ncol(y[[1]])}.}
\item{x_trt}{Covariates for the treatment forest, \code{[N x P_trt]}. Defaults
to \code{x}.}
\item{x_tv}{Time-varying prognostic covariates, \code{[N x T x P]}.}
\item{x_trt_tv}{Time-varying treatment covariates, \code{[N x T x P]}.}
\item{ps}{Propensity score, length \code{N} or \code{[N x T]}.}
\item{num_sweeps}{Posterior draws saved per chain, after burn-in.}
\item{num_burnin}{Burn-in iterations discarded.}
\item{n_skip}{Thinning interval.}
\item{num_chains}{Parallel MCMC chains.}
\item{inner_loop_length}{Sweeps per dispatch.}
\item{num_trees_pr, num_trees_trt}{Trees in each forest.}
\item{min_points_per_leaf_pr, min_points_per_leaf_trt}{Minimum leaf sizes.}
\item{max_depth_pr, max_depth_trt}{Maximum tree depths.}
\item{alpha_split_pr, beta_split_pr}{Tree-depth prior, prognostic forest.}
\item{alpha_split_trt, beta_split_trt}{Tree-depth prior, treatment forest.}
\item{num_cutpoints}{Maximum quantile bins per continuous covariate.}
\item{sig_knl}{Kernel standard deviation for the exposure trajectory.}
\item{lambda_knl}{Kernel lengthscale over the exposure index.}
\item{kernel_type}{One of \code{"se"}, \code{"matern32"}, \code{"matern52"},
\code{"ar1"}.}
\item{gp_jitter}{Relative diagonal jitter on the kernel.}
\item{sigma_m}{Prior standard deviation of the marginalized constant GP mean.}
\item{gp_constant_mean}{Whether the projection reverts to an estimated common level.}
\item{split_time_ps}{Whether prognostic forest may split on calendar time.}
\item{split_time_trt}{Whether treatment forest may split on exposure index.}
\item{random_intercept}{Whether to fit unit random intercepts.}
\item{gamma_prior_a, gamma_prior_b}{Inverse-gamma prior on unit-intercept variance.}
\item{sigma_prior_a, sigma_prior_b}{Inverse-gamma prior on innovation variance.
Both must be explicitly positive for coupled continuous outcomes. For
standardized responses, \code{2} and \code{1} specify a proper IG(2,1) prior
with mean variance one. With \code{standardize=FALSE}, choose these in outcome units.}
\item{sigma_b}{Prior standard deviation of adaptive coding weights.}
\item{sigma_alpha}{Prior standard deviation of prognostic scale.}
\item{outcome}{Outcome type(s): \code{"continuous"}, \code{"binary"}, or a vector of length \code{M}.}
\item{outcome_names}{Optional character vector of length \code{M} naming the outcomes.}
\item{a_scaling}{Whether to sample the prognostic scale \code{alpha}.}
\item{b_scaling}{Whether to sample adaptive coding weights \code{b0}, \code{b1}.}
\item{ridge_move}{Whether to run Metropolis move along beta-nu ridge.}
\item{ridge_proposal_sigma}{Proposal standard deviation for \code{log c}.}
\item{standardize}{Whether to centre and scale continuous outcomes internally.}
\item{random_seed}{Base PRNG seed.}
\item{device}{\code{"auto"}, \code{"cpu"}, or \code{"gpu"}.}
\item{sur}{Whether to enable full-precision triangular SUR coupling.}
\item{sur_prior_var}{Prior variance for loading regression parameters.}
\item{num_shared_trees}{Number of shared treatment trees, replacing this many
of \code{num_trees_trt}. Default 0 disables sharing; at least one private tree
must remain per outcome. Shared leaves have outcome-specific values and common
split settings. Minimum leaf size counts cells observed for any outcome.}
\item{shared_variance_fraction}{Fixed fraction of the unit treatment-forest
prior variance allocated to shared trees, strictly between 0 and 1. The
remainder regularizes private trees. Not a learned correlation or mixing weight;
inert when sharing is disabled.}
\item{verbose}{Whether to print progress messages.}
\item{...}{Reserved; unsupported arguments raise an error.}
}
\value{An object of class \code{longbet_multi}.}
\description{
Fits outcome-specific prognostic forests on a shared panel,
using the full precision of a triangular SUR (seemingly unrelated regressions)
likelihood (\code{full_precision_sur_v1}). Downstream observations inform earlier
means and binary latent responses. Binary equations come first, with zero
incoming loadings and unit marginal latent variance; complete or appropriately
nested observation masks are required when coupling is active.
Coupled fits remain experimental: aligned draws do not establish calibrated
joint inference. Treatment trees are outcome-specific by default. With
\code{num_shared_trees > 0}, that many partitions have vector-valued leaves,
with remaining private trees (\code{shared_private_sur_v1}). Shared leaf values
may have different magnitudes and signs across outcomes. \code{sur = FALSE}
disables residual correlation, not structural sharing.
Check convergence of the effects and decision events before interpreting
their probabilities. Both scalar and multi-outcome fits made before the
2026-09-07 dynamic leaf-precision cache repair must be refitted, not reloaded.
Coupled fits containing continuous outcomes require explicitly positive
\code{sigma_prior_a} and \code{sigma_prior_b}. The default improper reference
prior can produce an improper joint posterior and is rejected, including in
saved archives. Refit from the original data with a justified proper prior;
this does not guarantee convergence.
}
\seealso{\code{\link{predict.longbet_multi}}, \code{\link{effect_draws}},
\code{\link{joint_prob}}, \code{\link{outcome_correlation}}}
\examples{
\dontrun{
n <- 100; Tn <- 8
x <- matrix(rnorm(n * 3), n, 3)
z <- matrix(0, n, Tn); z[1:50, 5:Tn] <- 1
y1 <- 0.5 * x[, 1] + 1.2 * z + matrix(rnorm(n * Tn, 0, 0.3), n, Tn)
y2 <- matrix(as.numeric(0.3 * x[, 2] - 0.8 * z + rnorm(n * Tn) > 0), n, Tn)
fit <- longbet_multi(
  y = list(rev = y1, churn = y2),
  x = x, z = z, t = 1:Tn,
  outcome = c(rev = "continuous", churn = "binary"),
  sigma_prior_a = 2, sigma_prior_b = 1,
  num_chains = 2
)
pred <- predict(fit, x = x, z = z, t = 1:Tn, summary_only = TRUE)
}
}
