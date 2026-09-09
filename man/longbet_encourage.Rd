\name{longbet_encourage}
\alias{longbet_encourage}
\title{Joint reduced forms for randomized encouragement}
\usage{
longbet_encourage(y, d, z, x, t = NULL, x_trt = NULL,
  first_stage, outcome = "continuous", config = list(), verbose = FALSE)
}
\arguments{
\item{y}{Complete numeric outcome matrix, N by T.}
\item{d, z}{Complete absorbing binary adoption and encouragement matrices.}
\item{x}{Baseline covariate matrix, N by P.}
\item{t}{Increasing calendar vector with whole-unit gaps; defaults to 1:T.}
\item{x_trt}{Optional baseline treatment-forest covariates.}
\item{first_stage}{Required choice: \code{"lpm"} or \code{"probit"}.}
\item{outcome}{\code{"continuous"} or \code{"binary"} for y.}
\item{config}{Named list of Python LongBetConfig sampler options, such as
num_chains, num_sweeps, num_burnin, random_seed and variance priors. Unspecified
innovation priors use proper IG(2,1); explicit improper priors are rejected.}
\item{verbose}{Whether to print a fitting message.}
}
\description{Fits Y and actual adoption jointly on assigned encouragement using
the Python engine, for one encouragement wave and a permanent control arm.}
\details{Prediction averages both ITTs over all original study units using equal
weights. Probit uses probability differences. Neither first-stage model is a
calibrated default. SUR couples innovations when a continuous equation is
present; unit-intercept priors are independent across equations. With binary Y
and a probit first stage, both innovation-loading rows are fixed at zero, so
\code{sur = TRUE} does not couple innovations. These equations are independent
by default. Optional shared treatment trees couple forest uncertainty only;
they do not introduce correlated binary innovations or unit intercepts.
The Wald ratio needs exclusion, monotonicity, relevance and
appropriate adoption-history assumptions for a CACE interpretation.}
\value{A longbet_encourage object containing a versioned raw archive, configuration
and metadata. The archive includes original outcomes, adoption, assignment and
covariates, retaining the original target. No live Python handles are stored;
the object survives saveRDS and readRDS in a fresh R session.}
\seealso{\code{\link{predict.longbet_encourage}}, \code{\link{encouragement_effects}}}
