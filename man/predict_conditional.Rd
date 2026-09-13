\name{predict_conditional}
\alias{predict_conditional}
\alias{predict_conditional.longbet_encourage}
\title{Predict unit-specific conditional encouragement effects and risk-aware policy decisions}
\usage{
predict_conditional(object, ...)

\method{predict_conditional}{longbet_encourage}(object, new_x, new_z = NULL,
  cost = NULL, hurdle = 0.5, alpha = 0.05, cace_stabilization = 1e-4, ...)
}
\arguments{
\item{object}{A fitted \code{longbet_encourage} object.}
\item{new_x}{Baseline covariates for new or in-sample units (matrix or 2D array).}
\item{new_z}{Optional counterfactual encouragement matrix [N x T]. Defaults to 1 for post-treatment periods.}
\item{cost}{Optional per-unit intervention cost for policy decision analysis.}
\item{hurdle}{Posterior breakeven probability hurdle (e.g. 0.8 for 80\% confidence).}
\item{alpha}{Tail probability for credible intervals (default 0.05 for 95\% intervals).}
\item{cace_stabilization}{Small ridge regularizer to prevent division-by-zero instability in CACE posterior draws.}
\item{...}{Reserved for method compatibility.}
}
\value{
An object of class \code{longbet_conditional_pred} containing:
\describe{
  \item{\code{citt_y}}{Posterior summaries (mean, median, lower, upper, sd, draws) for conditional ITT on outcome Y.}
  \item{\code{citt_d}}{Posterior summaries for conditional ITT on adoption/treatment D (first stage).}
  \item{\code{cace}}{Posterior summaries for stabilized conditional average complier effect (CACE).}
  \item{\code{cumulative_lift}}{Posterior summaries for cumulative outcome gain across post-treatment horizons.}
  \item{\code{breakeven_prob}}{Vector of unit-specific posterior breakeven probabilities \eqn{\Pr(\text{Cumulative Lift} > \text{cost})}.}
  \item{\code{decision}}{Logical vector indicating optimal targeted policy decisions at the specified hurdle.}
}
}
\description{
Computes out-of-sample and in-sample unit-specific posterior draws for conditional intention-to-treat
effects on outcomes (\eqn{\text{CITT}_Y}), adoption (\eqn{\text{CITT}_D}), and complier average
causal effects (\eqn{\text{CACE}}). In addition, computes posterior breakeven probabilities and
risk-aware commercial policy allocations under arbitrary cost structures and hurdle rates.
}
