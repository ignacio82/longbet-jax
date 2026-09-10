\name{hazard_adoption_effects}
\alias{hazard_adoption_effects}
\title{Experimental discrete-time hazard adoption forest}
\usage{
hazard_adoption_effects(d, z, x = NULL, t = NULL, trees = 4L,
  chains = 2L, burnin = 200L, draws = 300L, seed = 42L)
}
\arguments{
\item{d, z}{Binary absorbing adoption and encouragement panels, N by T.}
\item{x}{Optional baseline covariate matrix, N by P.}
\item{t}{Optional calendar periods.}
\item{trees}{Number of trees in each ensemble.}
\item{chains}{Number of MCMC chains conditional on the same random basis.}
\item{burnin}{Number of discarded initial sweeps.}
\item{draws}{Number of retained sweeps.}
\item{seed}{Random seed.}
}
\description{Computes working-model adoption hazard, stock probability and expected
exposure contrasts using a relevance indicator.}
\details{One seeded random stump basis remains fixed and is shared across chains.
Probit utilities and a collapsed relevance/effect block target the posterior
conditional on this basis. Posterior interval calibration is unestablished;
inspect convergence and sensitivity to the random basis and priors. Allowing
zero relevance does not identify an outcome effect or eliminate weak-IV bias.}
\value{A list with \code{table}, \code{metadata}, and \code{draws}. Metadata records
experimental status, calibration limitations, and the conditional fixed-basis
posterior target. Raw contrasts have (chain, retained_draw, horizon) axes;
relevance draws have the first two axes, permitting convergence assessment.}
