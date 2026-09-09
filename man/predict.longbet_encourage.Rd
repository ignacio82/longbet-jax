\name{predict.longbet_encourage}
\alias{predict.longbet_encourage}
\title{Predict common-population encouragement effects}
\usage{
\method{predict}{longbet_encourage}(object, summary_only = TRUE, groups = NULL,
  block_size = NULL, standardization = "conditional", alpha = 0.05,
  min_ess = 400, max_rhat = 1.01, practical_threshold = NULL, ...)
}
\arguments{
\item{object}{A fitted longbet_encourage object, including one loaded with readRDS.}
\item{summary_only}{Suppress full unit/period/draw arrays; aggregate draws remain.}
\item{groups}{Optional prespecified baseline subgroup labels, length N.}
\item{block_size}{Optional positive integer cells evaluated per block.}
\item{standardization}{Conditional retains each fitted unit intercept. Population
integrates a fresh normal unit intercept and targets a different population
from the finite-study-unit reference.}
\item{alpha}{Tail probability for pointwise intervals.}
\item{min_ess, max_rhat}{Diagnostic thresholds. Passing is not a coverage claim.}
\item{practical_threshold}{Optional positive adoption risk difference for a
separate practical-relevance probability, not an instrument-strength cutoff.}
\item{...}{Reserved; unsupported arguments raise an error.}
}
\description{Replays the original population and averages natural-scale joint
posterior contrasts over the same units for both outcomes.}
\value{A plain-R longbet_encourage.pred list containing effects, reference,
wald, wald_unchecked, first_stage, comparison, diagnostics, draws,
cell_draws, cell_summaries, weights, group information and metadata. Joint ITT
covariance is a numeric matrix in itt_covariance; both axes follow rows of
itt_covariance_index, with group, horizon and quantity labels.
Aggregate draw axes are group, horizon, chain, retained draw; cell draw axes
are unit, period, chain-major draw. Period indices are zero based.}
\details{Wald summaries use medians and quantiles, with no default ratio mean.
The wald table withholds intervals failing diagnostics; wald_unchecked permits
inspection. No denominator draws are clipped or removed. Frequentist confidence
sets remain separate in reference. Infinite endpoints, unavailable references
and unsupported groups remain represented. All returned objects survive saveRDS.}
