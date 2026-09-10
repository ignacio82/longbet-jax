\name{longbet_iv}
\alias{longbet_iv}
\title{Prediction-assisted longitudinal instrumental variables}
\usage{
longbet_iv(y, d, z, x = NULL, t = NULL, learner_config = NULL,
  folds = 2L, seed = 0L, alpha = 0.05, design = "complete_randomization")
}
\arguments{
\item{y}{Complete finite outcome panel, units by periods.}
\item{d}{Complete binary absorbing adoption panel matching y.}
\item{z}{Single-wave encouragement panel or assignment vector. A vector treats
every supplied period as post-assignment.}
\item{x}{Optional baseline covariates. All pre-assignment y/d columns are appended.}
\item{t}{Optional increasing observation times.}
\item{learner_config}{Named list of LongBetIVNuisanceConfig predictor options.}
\item{folds}{Exactly two folds are supported.}
\item{seed}{Nonnegative integer RNG seed.}
\item{alpha}{Significance level, default 0.05.}
\item{design}{Individual-unit complete randomization. Other designs are unsupported.}
}
\value{A serializable list with table, estimates, covariance, predictions,
unit_scores, fold_ids and metadata. Covariance orders each horizon's Y then D.
The table retains both components of disconnected Fieller confidence sets.}
\description{Experimental LongBet trajectory prediction inside conditional
cross-fitted augmented ITTs and joint residual Neyman/Fieller inference.}
\details{All observations from a unit stay together. Inner tuning sees only the
outer training accounts. New-account predictions integrate out their random
intercepts and use only baseline covariates. The ITTs are unbiased under the
stated complete-randomization and splitting design. Confidence statements are
asymptotic and pointwise and require stable predictions and moment conditions.
Posterior quantiles are not used as causal intervals. Weak first stages may yield
unbounded, disconnected or all-real sets. CACE interpretation requires exclusion,
monotonicity, relevance and appropriate restrictions on treatment history.
This is not an exact sharp-null randomization test.}
\seealso{\code{\link{encouragement_effects}}, \code{\link{longbet_direct_smooth}}}
