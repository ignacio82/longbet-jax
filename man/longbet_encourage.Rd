\name{longbet_encourage}
\alias{longbet_encourage}
\title{The adoption-clock model for a randomized encouragement experiment}
\usage{
longbet_encourage(y, d, z, x, t = NULL, x_trt = NULL,
  outcome = "continuous", config = list(), verbose = FALSE)
}
\arguments{
\item{y}{Complete numeric outcome matrix, N by T.}
\item{d, z}{Complete absorbing binary adoption and encouragement matrices.}
\item{x}{Baseline covariate matrix, N by P.}
\item{t}{Increasing calendar vector with whole-unit gaps; defaults to 1:T.}
\item{x_trt}{Optional baseline treatment-forest covariates.}
\item{outcome}{\code{"continuous"} or \code{"binary"} for y; adoption is always a
binary hazard.}
\item{config}{Named list of Python LongBetConfig sampler options applied to both
equations, such as num_chains, num_sweeps, num_burnin and random_seed.}
\item{verbose}{Whether to print a fitting message.}
}
\description{Fits two LongBet equations and composes them: the outcome on the
observed adoption clock (periods since adoption, both arms contributing adoption
events) and adoption as a discrete-time hazard on the offer clock, with the unit
intercept of the hazard acting as a frailty. The offer's effects on the outcome
and on adoption follow by composition, integrating the frailty outside the
survival product.}
\details{This uses the exclusion restriction and assumes adoption timing is not
confounded with the outcome innovations given covariates and the unit intercept.
The design-based reference returned by \code{predict()} needs neither assumption
and is the check on the population offer effects. Both ITTs average over all
original study units with equal weights and each unit's own fitted intercepts.
The Wald ratio needs exclusion, monotonicity, relevance and appropriate
adoption-history assumptions for a CACE interpretation.}
\value{A longbet_encourage object containing a versioned raw archive, configuration
and metadata. The archive includes both fitted equations and the original
outcomes, adoption, assignment and covariates. No live Python handles are stored;
the object survives saveRDS and readRDS in a fresh R session.}
\seealso{\code{\link{predict.longbet_encourage}}, \code{\link{predict_conditional}},
\code{\link{encouragement_effects}}}
