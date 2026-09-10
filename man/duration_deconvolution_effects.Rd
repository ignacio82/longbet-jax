% Generated from R/encourage.R; keep the public duration-IV contract synchronized.
\name{duration_deconvolution_effects}
\alias{duration_deconvolution_effects}
\title{Panel instrumental variables for a common treatment-duration response}
\usage{
duration_deconvolution_effects(y, d, z, t = NULL,
  model_type = "linear", max_duration = NULL, alpha = 0.05)
}
\arguments{
\item{y}{Finite balanced outcome panel, an N by T matrix. Units must be independent clusters.}
\item{d}{Binary absorbing adoption panel, an N by T matrix.}
\item{z}{Binary absorbing encouragement panel, an N by T matrix. Encouraged units must share a common onset, and never-encouraged units must be present.}
\item{t}{Optional finite, strictly increasing, equally spaced calendar periods of length T. Duration counts complete observed periods, independently of the numerical units of t.}
\item{model_type}{One of \code{"linear"}, \code{"stepwise"}, or \code{"quadratic"}. The legacy name \code{"spline"} aliases a quadratic polynomial without knots. The default is \code{"linear"}.}
\item{max_duration}{Optional positive integer giving the maximum duration to evaluate, capped at the largest observed duration. For \code{"stepwise"}, also limits the basis and imposes a constant response beyond that duration.}
\item{alpha}{Significance level, strictly between zero and one. Default 0.05.}
}
\value{
A list containing \code{summary_table}, \code{durations}, \code{effects},
\code{se}, \code{ci_lower}, and \code{ci_upper}, together with the fitted
\code{coefficients}, \code{coefficient_covariance}, \code{instrument_rank},
\code{regressor_rank}, \code{residual_df}, \code{n_clusters}, and
\code{inference_method}.

The historical \code{cumulative_effect} column in \code{summary_table} is the
effect on the current outcome after cumulative exposure, not an outcome summed
across periods.

\code{first_stage_f} is a unit-cluster Wald statistic divided by instrument rank
for a single endogenous regressor. It is \code{NaN} for multiple regressors or
singular cluster covariance. \code{first_stage_status} distinguishes
\code{available}, \code{perfect_fit}, \code{singular_cluster_covariance}, and
\code{unavailable_for_multiple_endogenous_regressors}. A perfectly fitted first
stage has infinite F. \code{first_stage_diagnostics} reports each regressor's
cluster Wald statistic, status, instrument rank, cluster and residual degrees of
freedom, and partial R-squared. Individual diagnostics do not establish joint
strength for multiple endogenous regressors.
}
\description{
Conventional unregularized panel two-stage least squares for a common response
to cumulative treatment exposure. Assignment-by-post-period interactions
instrument duration regressors after removing unit and period fixed effects.
Rank-deficient models are rejected rather than regularized into finite estimates.
}
\details{
Identification requires a correctly specified common duration response, additive
unit and period effects, exclusion through treatment history, and full rank of
the instrumented duration regressors. Random assignment alone does not identify
an unrestricted heterogeneous duration response. The linear specification uses
duration, the quadratic specification uses duration and its square, and the
stepwise specification uses indicators that duration is at least each modeled
level. All impose a zero effect at zero exposure.

Confidence intervals are pointwise and use unit-cluster CR1 covariance with
t(N-1) critical values. The covariance correction is
N/(N-1) times (NT-1)/(NT-K), where K=N+T-1+p includes all fixed effects and p
endogenous regressors. They require many independent units and strong
instruments; they are not exact or weak-instrument-robust confidence sets.
The cluster Wald diagnostic must not be compared to Stock--Yogo critical values
for a homoskedastic first-stage F statistic.

Missing observations, irregular periods, nonabsorbing panels, and staggered
encouragement onsets are not supported by this estimator. Exposure is counted
from the beginning of the supplied panel; prior exposure is not reconstructed.
}
