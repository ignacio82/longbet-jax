"""Conditional cross-fitting for a completely randomized encouragement panel.

The prediction model assists estimation of assignment effects; it does not supply
the uncertainty calculation or identify effects of adoption. The split-by-arm
construction and residual Neyman variance follow Lu, Shi, Liu and Ding (2025),
https://arxiv.org/abs/2508.15664, Sections 3.1 and 5.3. Two folds are supported.

Conditional on the partition, its two assignment vectors are independent complete
randomizations with known arm counts. Consequently the augmented ITTs are exactly
unbiased over assignment and the randomized split, even for misspecified learners.
The reported covariance is only asymptotically conservative: replacing fitted
learners by fixed oracle prediction functions requires finite-population L2
stability. Bounded residual fourth moments, nondegenerate contrast variances and
arm fractions bounded away from zero are also required. The unidentifiable
residual treatment-effect covariance is omitted in the Neyman construction.

Analytic Fieller inversion concerns the average moment ITT_Y - beta * ITT_D = 0.
It permits weak or zero first stages and retains the entire confidence set. It is
neither a finite-sample sharp-null randomization test nor proof that the ratio is
a causal adoption effect. No posterior-interval calibration is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.stats import norm

from longbet._encourage import _critical, _validate, _wald_set


@dataclass
class CrossfitEncouragementResult:
    """Assignment-effect estimates and design-based joint uncertainty.

    ``estimates`` and ``unit_scores`` have trailing axes (horizon, Y/D).
    ``covariance`` flattens those two axes in C order, so the ordering is
    (Y at first horizon, D at first horizon, Y at second horizon, ...).
    ``predictions`` has axes (unit, horizon, assignment arm 0/1, response Y/D).
    ``table`` retains both components of disjoint Fieller sets. ``unit_scores``
    average to the estimates; their ordinary sample covariance is NOT the
    reported finite-population covariance.
    """

    estimates: np.ndarray
    covariance: np.ndarray
    table: list[dict[str, Any]]
    unit_scores: np.ndarray
    predictions: np.ndarray
    fold_ids: np.ndarray
    metadata: dict[str, Any]

    def confidence_set(self, horizon: int = 1) -> dict[str, Any]:
        """Return the full pointwise Fieller set for a one-based horizon index."""
        index = self._horizon_index(horizon)
        return dict(self.table[3 * index + 2])

    def ar_test(self, beta: float, horizon: int = 1) -> dict[str, Any]:
        """Test the average ITT moment using asymptotic normal inference.

        ``beta`` is held fixed when taking a linear contrast of the joint ITTs.
        No division by the estimated first stage occurs. This is not an exact
        permutation test under a heterogeneous-effect causal null.
        """
        index = self._horizon_index(horizon)
        if isinstance(beta, (bool, np.bool_)) or not np.isscalar(beta):
            raise ValueError("beta must be a finite number.")
        try:
            beta = float(beta)
        except (TypeError, ValueError) as exc:
            raise ValueError("beta must be a finite number.") from exc
        if not np.isfinite(beta):
            raise ValueError("beta must be a finite number.")
        # Normalization avoids overflow in beta**2 for very large finite beta.
        scale = max(1.0, abs(beta))
        contrast = np.array([1.0 / scale, -beta / scale])
        effect = self.estimates[index]
        covariance = self.covariance[2 * index:2 * index + 2, 2 * index:2 * index + 2]
        moment = float(contrast @ effect)
        variance = max(float(contrast @ covariance @ contrast), 0.0)
        statistic = (moment / np.sqrt(variance) if variance > 0
                     else (0.0 if moment == 0 else np.copysign(np.inf, moment)))
        p_value = float(2 * norm.sf(abs(statistic)))
        return dict(horizon=int(horizon), beta=beta, statistic=float(statistic),
                    p_value=p_value, reject=bool(abs(statistic) > self.metadata["critical_value"]),
                    inference="asymptotic_average_itt_moment", exact_randomization=False)

    def _horizon_index(self, horizon: int) -> int:
        if (isinstance(horizon, (bool, np.bool_))
                or not isinstance(horizon, (int, np.integer))
                or not 1 <= horizon <= len(self.estimates)):
            raise ValueError(f"horizon must be an integer from 1 to {len(self.estimates)}.")
        return int(horizon) - 1


def _split_by_assignment(assignment: np.ndarray, seed: int) -> np.ndarray:
    """Uniformly partition each arm with prespecified floor/ceiling counts.

    The selection probability for a particular partition is constant over all
    compatible assignments. Given the partition, the original complete
    randomization therefore factorizes into independent within-fold complete
    randomizations. This differs from an unconditioned random unit partition.
    """
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(assignment), dtype=np.int64)
    for arm in (0, 1):
        indices = rng.permutation(np.flatnonzero(assignment == arm))
        cut = len(indices) // 2
        fold_ids[indices[:cut]] = 0
        fold_ids[indices[cut:]] = 1
    return fold_ids


def _infer_from_predictions(
    responses: np.ndarray,
    assignment: np.ndarray,
    predictions: np.ndarray,
    fold_ids: np.ndarray,
    times: np.ndarray,
    *,
    alpha: float = .05,
    metadata: dict[str, Any] | None = None,
) -> CrossfitEncouragementResult:
    """Algebra shared with tests; inputs must satisfy the public fit contract.

    In particular, externally supplied predictions or partitions are NOT thereby
    certified as honest conditional cross-fitting. The public entry point owns
    the randomized partition and training separation.
    """
    critical = _critical(alpha)
    n, horizons, _ = responses.shape
    scores = np.empty_like(responses, dtype=float)
    covariance = np.zeros((2 * horizons, 2 * horizons))
    fold_details = []
    for fold in (0, 1):
        held_out = fold_ids == fold
        fold_n = int(held_out.sum())
        fold_z = assignment[held_out]
        count1 = int(fold_z.sum())
        count0 = fold_n - count1
        if min(count0, count1) < 2:
            raise ValueError("Each fold must have at least two units in each assignment arm.")
        probability = count1 / fold_n
        observed = responses[held_out]
        pred = predictions[held_out]
        z = fold_z[:, None, None]
        scores[held_out] = (pred[:, :, 1] - pred[:, :, 0]
                           + z * (observed - pred[:, :, 1]) / probability
                           - (1 - z) * (observed - pred[:, :, 0]) / (1 - probability))
        residuals = observed - pred[np.arange(fold_n), :, fold_z.astype(int), :]
        flat_residuals = residuals.reshape(fold_n, 2 * horizons)
        for arm, count in ((0, count0), (1, count1)):
            residual = flat_residuals[fold_z == arm]
            centered = residual - residual.mean(axis=0)
            covariance += (fold_n / n) ** 2 * (centered.T @ centered) / (count * (count - 1))
        fold_details.append(dict(fold=fold, n_units=fold_n, n_control=count0,
                                 n_encouraged=count1, assignment_probability=probability))
    estimates = scores.mean(axis=0)
    if not np.all(np.isfinite(estimates)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Nonfinite estimated moments; rescale outcomes or predictions before inference.")
    rows = []
    for index, effect in enumerate(estimates):
        variance = covariance[2 * index:2 * index + 2, 2 * index:2 * index + 2]
        for response, quantity in enumerate(("itt_y", "itt_d")):
            standard_error = float(np.sqrt(max(variance[response, response], 0)))
            rows.append(dict(horizon=index + 1, period=float(times[index]), quantity=quantity,
                             estimate=float(effect[response]), se=standard_error,
                             lower=float(effect[response] - critical * standard_error),
                             upper=float(effect[response] + critical * standard_error),
                             lower2=np.nan, upper2=np.nan, set_type="bounded"))
        interval = _wald_set(effect[0], effect[1], variance[0, 0], variance[1, 1],
                             variance[0, 1], critical)
        rows.append(dict(horizon=index + 1, period=float(times[index]), quantity="wald",
                         estimate=float(effect[0] / effect[1]) if effect[1] != 0 else np.nan,
                         se=np.nan, lower=interval["wald_lower_1"], upper=interval["wald_upper_1"],
                         lower2=interval["wald_lower_2"], upper2=interval["wald_upper_2"],
                         set_type=interval["wald_set_type"]))
    details = dict(metadata or {})
    details.update(
        design="complete_randomization", randomization_unit="individual_panel_unit",
        fold_scheme="conditional_split_by_assignment", folds=fold_details,
        conditional_assignment_probabilities=[v["assignment_probability"] for v in fold_details],
        covariance_method="fold_weighted_finite_population_neyman_residual",
        covariance_order=[(float(period), response) for period in times for response in ("itt_y", "itt_d")],
        interval_scope="pointwise", alpha=float(alpha), critical_value=critical,
        inference="asymptotic_average_itt_moment", exact_randomization=False,
        prediction_misspecification_robust_itt_unbiasedness=True,
        covariance_requires=["finite_population_L2_prediction_stability", "bounded_residual_fourth_moments",
                             "nondegenerate_finite_population_contrast_CLT", "positive_limiting_fold_arm_fractions"],
        omitted_covariance="unidentifiable_residual_assignment_effect_covariance",
        causal_ratio_requires=["exclusion", "monotonicity", "relevance", "appropriate_adoption_history_restrictions"],
        references=["https://arxiv.org/abs/2508.15664"],
    )
    return CrossfitEncouragementResult(estimates, covariance, rows, scores,
                                      predictions.copy(), fold_ids.copy(), details)


def crossfit_encouragement(
    y: Any,
    d: Any,
    z: Any,
    x: Any,
    t: Any = None,
    *,
    learner_factory: Callable[[], Any],
    folds: int = 2,
    seed: int = 0,
    alpha: float = .05,
    design: str = "complete_randomization",
) -> CrossfitEncouragementResult:
    """Cross-fit prediction-assisted assignment ITTs and Fieller confidence sets.

    ``y`` and ``d`` are complete (units, periods) outcome and binary absorbing
    adoption arrays. A binary absorbing panel ``z`` must contain one common
    encouragement wave and a permanent control arm. Only post-encouragement
    columns are analyzed. Alternatively ``z`` can be a unit assignment vector;
    then ALL supplied outcome/adoption columns are regarded as post-assignment.
    ``x`` is a complete unit-by-covariate matrix containing only information
    measured before assignment, including pre-assignment outcomes if desired.
    Baseline preprocessing may use all units' pre-assignment information but must
    never use held-out post-assignment outcomes, adoption or assignment labels.

    ``learner_factory()`` must return a fresh learner with methods
    ``fit(x_train, assignment_train, responses_train, times_post)`` and
    ``predict(x_test)``. Responses have shape (units, horizons, 2), ordered Y/D;
    predictions have shape (units, horizons, 2 arms, 2 responses). Each account's
    entire trajectory stays together. Hyperparameter selection must be confined
    to the training fold. Prediction randomness must be independent of assignment
    and held-out responses. ``seed`` controls the split only; configure learner
    randomness in the factory. No clipping or monotonicity projection is applied
    to predictions by this estimator.

    The user asserts individual-unit complete randomization via ``design``.
    Bernoulli, stratified, clustered, matched-pair, unequal-probability and
    staggered encouragement designs are unsupported. At least four units per
    arm are necessary for the two within-fold residual covariances; asymptotic
    validity requires much larger samples. Two-fold fitting can lose efficiency
    from smaller training sets. If arm counts are odd, conditional fold assignment
    fractions can differ slightly; they are used exactly and reported.

    The covariance retains dependence across responses and periods within a unit.
    Wald ratios have a causal interpretation only under additional IV and
    adoption-history restrictions. Inference is asymptotic, pointwise and
    conservative under the stability/moment conditions in this module's docs.
    """
    if design != "complete_randomization":
        raise ValueError("Only individual-unit complete_randomization is supported.")
    if isinstance(folds, (bool, np.bool_)) or not isinstance(folds, (int, np.integer)) or folds != 2:
        raise ValueError("Only folds=2 is supported by this prototype.")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer.")
    _critical(alpha)
    if not callable(learner_factory):
        raise ValueError("learner_factory must be a callable returning a fresh predictor.")
    outcomes = np.asarray(y)
    if outcomes.ndim != 2 or outcomes.dtype.kind not in "biuf" or min(outcomes.shape) < 1:
        raise ValueError("y must be a finite numeric unit-by-period matrix.")
    outcomes = outcomes.astype(float)
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("y must be complete and finite; no units or periods are dropped.")
    assignment_input = np.asarray(z)
    if assignment_input.ndim == 1:
        if len(assignment_input) != len(outcomes):
            raise ValueError("A vector z must contain one assignment per unit.")
        assignment_input = np.broadcast_to(assignment_input[:, None], outcomes.shape)
    panel = _validate(assignment_input, d, t)
    if outcomes.shape != panel.z.shape:
        raise ValueError(f"y must have shape {panel.z.shape} matching d and z.")
    features = np.asarray(x)
    if features.ndim != 2 or features.shape[0] != len(outcomes) or features.dtype.kind not in "biuf":
        raise ValueError("x must be a finite numeric unit-by-covariate matrix.")
    features = features.astype(float)
    if not np.all(np.isfinite(features)):
        raise ValueError("x must be complete and finite; imputation must use pre-assignment information.")
    assignment = panel.assigned.astype(np.int64)
    if min(int(assignment.sum()), int((1 - assignment).sum())) < 4:
        raise ValueError("Two-fold inference requires at least four units in each assignment arm.")
    fold_ids = _split_by_assignment(assignment, int(seed))
    response = np.stack([outcomes, panel.d], axis=-1)[:, panel.start:]
    times = panel.t[panel.start:]
    predictions = np.empty((len(outcomes), len(times), 2, 2))
    learner_details = []
    learners = []
    for fold in (0, 1):
        test = fold_ids == fold
        train = ~test
        learner = learner_factory()
        if any(learner is other for other in learners):
            raise ValueError("learner_factory must return a fresh learner for each fold.")
        learners.append(learner)
        learner.fit(features[train].copy(), assignment[train].copy(), response[train].copy(), times.copy())
        values = np.asarray(learner.predict(features[test].copy()))
        expected = (int(test.sum()), len(times), 2, 2)
        if values.shape != expected or values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)):
            raise ValueError(f"Fold {fold}: learner predictions must be finite numeric arrays with shape {expected}.")
        predictions[test] = values
        learner_details.append(dict(fold=fold, learner=type(learner).__name__,
                                    metadata=getattr(learner, "metadata_", getattr(learner, "metadata", {}))))
    return _infer_from_predictions(response, assignment, predictions, fold_ids, times, alpha=alpha,
                                  metadata=dict(panel=panel.metadata(), split_seed=int(seed),
                                                learner_seed_control="learner_factory", learners=learner_details))
