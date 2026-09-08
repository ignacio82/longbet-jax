% Generated from R/longbet.R -- regenerate with roxygen2::roxygenise().
\name{longbet}
\alias{longbet}
\title{LongBet: Bayesian ensemble trees for causal inference on longitudinal data}
\usage{
longbet(y, x, z, t = NULL, x_trt = NULL, x_tv = NULL, x_trt_tv = NULL,
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
  outcome = c("continuous", "binary"), a_scaling = FALSE, b_scaling = TRUE,
  ridge_move = TRUE, ridge_proposal_sigma = 0.2, standardize = TRUE,
  random_seed = 0, device = c("auto", "cpu", "gpu"),
  sur = TRUE, sur_prior_var = 1, num_shared_trees = 0,
  shared_variance_fraction = 0.5, verbose = FALSE, ...)
}
\arguments{
\item{y}{Outcome panel, \code{[N x T]}. \code{NA} marks an unobserved cell;
unbalanced panels are marginalized, not imputed.}
\item{x}{Covariates for the prognostic forest, \code{[N x P]}.}
\item{z}{Binary treatment indicator, \code{[N x T]}. Must be absorbing.}
\item{t}{Calendar time vector of length \code{T}. Defaults to \code{1:ncol(y)}.}
\item{x_trt}{Covariates for the treatment forest, \code{[N x P_trt]}. Defaults
to \code{x}, in which case both forests share one block of columns.}
\item{x_tv}{Time-varying prognostic covariates, \code{[N x T x P]}.}
\item{x_trt_tv}{Time-varying treatment covariates, \code{[N x T x P]}.}
\item{ps}{Propensity score, length \code{N} or \code{[N x T]}. Shown to the
prognostic forest only.}
\item{num_sweeps}{Posterior draws saved per chain, after burn-in.}
\item{num_burnin}{Initial burn-in iterations discarded. This does not
guarantee convergence or sufficient effective sample size.}
\item{n_skip}{Thinning interval; 1 saves every post-burn-in iteration.
At fixed retained draw count, larger values add sampling work while keeping
prediction cost and trace storage approximately fixed.}
\item{num_chains}{Parallel MCMC chains. At least 2 are needed for R-hat.}
\item{inner_loop_length}{Sweeps per dispatch; bounds compile time.}
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
\item{gp_constant_mean}{Whether the projection reverts to an estimated common
level rather than to zero.}
\item{split_time_ps}{Whether the \emph{prognostic} forest may split on calendar
time.}
\item{split_time_trt}{Whether the \emph{treatment} forest may split on the
exposure index.}
\item{random_intercept}{Whether to fit unit random intercepts.}
\item{gamma_prior_a, gamma_prior_b}{Inverse-gamma prior on the unit-intercept
variance.}
\item{sigma_prior_a, sigma_prior_b}{Inverse-gamma prior on the error variance;
\code{(0, 0)} is the improper reference prior.}
\item{sigma_b}{Prior standard deviation of the adaptive coding weights.}
\item{sigma_alpha}{Prior standard deviation of the prognostic scale.}
\item{outcome}{\code{"continuous"} or \code{"binary"} (probit).}
\item{a_scaling}{Whether to sample the prognostic scale \code{alpha}.}
\item{b_scaling}{Whether to sample the adaptive coding weights \code{b0},
\code{b1}. On by default.}
\item{ridge_move}{Whether to run the Metropolis move along the beta-nu ridge.}
\item{ridge_proposal_sigma}{Proposal standard deviation for \code{log c}.}
\item{standardize}{Whether to centre and scale a continuous outcome internally.}
\item{random_seed}{Base PRNG seed.}
\item{device}{\code{"auto"}, \code{"cpu"} or \code{"gpu"}.}
\item{sur}{Whether to enable full-precision triangular SUR coupling (inert for scalar \code{longbet}).}
\item{sur_prior_var}{Prior variance for loading regression parameters.}
\item{num_shared_trees}{Must be zero for scalar fits. Use
\code{\link{longbet_multi}} to share treatment partitions across outcomes.}
\item{shared_variance_fraction}{Shared treatment-forest prior variance fraction,
strictly between 0 and 1; inert for scalar fits.}
\item{verbose}{Whether to print progress messages.}
\item{...}{Reserved. Passing an unsupported argument is an error rather than
being silently ignored.}
}
\value{An object of class \code{longbet}.}
\description{
Fits time-varying heterogeneous treatment effects on panel data with separate
prognostic and treatment forests, a shared Gaussian-process trajectory over the
exposure index, and unit-level random intercepts. The sampler is the JAX engine
in the \code{longbet-jax} Python package.
Fits made before the 2026-09-07 dynamic leaf-precision cache repair must be
refitted from the original data. Loading old posterior draws cannot repair
their uncertainty; archive loaders reject missing sampler provenance.
}
\section{Estimand}{
The reported effect is the contrast between being \code{S} periods into
treatment and not being treated at all,
\deqn{\tau_t(X_i, S) = b_1 \beta_S \nu(X_i, S, t) - b_0 \beta_0 \nu(X_i, 0, t),}
so the treatment forest is evaluated on both the factual exposure and a copy of
the design with \eqn{S = 0}.
}
\section{Sampling defaults}{
The defaults use 2,000 burn-in iterations and 250 retained draws, thinned by
2, across four chains: 2,500 iterations per chain and 1,000 retained draws
overall. These are starting settings, not a convergence guarantee. Check
\code{\link{att_stability}} and the subgroup or probability-scale effects used
in decisions. More burn-in can remove an initial transient, but cannot by itself
repair persistent poor mixing; the retained chains also need adequate
effective sample size.
}

\section{Categorical covariates}{
Every column is treated as ordered numeric; there is no \code{pcat} argument.
One-hot encode unordered categorical variables with more than two levels before
calling this function. This is a real modelling difference from the reference
C++ implementation.
}
\examples{
\dontrun{
n <- 100; Tn <- 8
x <- matrix(rnorm(n * 3), n, 3)
z <- matrix(0, n, Tn); z[1:50, 5:Tn] <- 1
y <- 0.5 * x[, 1] + 1.2 * z + matrix(rnorm(n * Tn, 0, 0.3), n, Tn)
fit <- longbet(y = y, x = x, z = z, t = 1:Tn, num_chains = 2)
pred <- predict(fit, x = x, z = z, t = 1:Tn, summary_only = TRUE)
get_att(pred)
}
}
\seealso{\code{\link{predict.longbet}}, \code{\link{get_att}},
\code{\link{att_stability}}}
