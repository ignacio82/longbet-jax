\name{longbet_direct_smooth}
\alias{longbet_direct_smooth}
\title{Direct joint Gaussian forest for encouragement panels}
\usage{
longbet_direct_smooth(y, d, z, x, t = NULL,
  num_trees = 6L, num_sweeps = 200L, num_burnin = 50L,
  correlated_intercepts = TRUE, seed = 42L, num_chains = 4L, n_skip = 1L)
}
\arguments{
\item{y}{Complete continuous outcome panel, N by T.}
\item{d, z}{Complete absorbing binary adoption and encouragement panels, N by T.}
\item{x}{Baseline covariate matrix, N by P.}
\item{t}{Optional increasing calendar periods.}
\item{num_trees}{Number of trees in each baseline and effect ensemble.}
\item{num_sweeps}{Number of retained draws per chain.}
\item{num_burnin}{Number of discarded burn-in sweeps.}
\item{correlated_intercepts}{Learn a cross-equation covariance for unit intercepts.}
\item{seed}{Nonnegative random seed.}
\item{num_chains}{Number of independent MCMC chains.}
\item{n_skip}{Sweeps per retained draw after burn-in.}
}
\description{Fits Gaussian encouragement reduced forms with smooth stump leaves
and exact conditional Gibbs updates. The first stage is a working LPM.}
\details{Sampling controls are forwarded with integer validation. The default
short run guarantees neither convergence nor interval coverage. Inspect
\code{predict(fit)$diagnostics}. For further forest and prior options, use
\code{longbet_encourage(engine = "direct_smooth", direct_config = list(...))}.}
\value{A serializable \code{longbet_encourage} model. Use \code{predict()} for
ITTs, posterior ratios, separate reference confidence sets and diagnostics.}
\seealso{\code{\link{longbet_encourage}}, \code{\link{predict.longbet_encourage}}}
