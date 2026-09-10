"""Independent design, algebra, honesty and confidence-set checks."""

from itertools import combinations, product

import numpy as np
import pytest
from scipy.stats import norm

from longbet._orthogonal_iv import (
    _infer_from_predictions,
    crossfit_encouragement,
)


class ZeroLearner:
    def fit(self, x, z, responses, times):
        self.horizons = len(times)
        return self

    def predict(self, x):
        return np.zeros((len(x), self.horizons, 2, 2))


def panel(n=24):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(n, 3))
    assignment = np.arange(n) % 2
    z = assignment[:, None] * np.array([0, 1, 1])[None, :]
    d = (x[:, :1] + assignment[:, None] > .3) * np.array([0, 1, 1])[None, :]
    y = 2 * x[:, :1] + np.array([0, 1, 2])[None, :] + 3 * d + rng.normal(size=(n, 3))
    return dict(y=y, d=d, z=z, x=x, t=np.arange(3))


def contains(row, beta):
    return any(row[lo] <= beta <= row[hi] for lo, hi in (("lower", "upper"), ("lower2", "upper2")))


def conditional_assignments():
    """Every assignment in two independent four-unit/two-treated CREs."""
    for left, right in product(combinations(range(4), 2), combinations(range(4, 8), 2)):
        assignment = np.zeros(8, dtype=int)
        assignment[list(left + right)] = 1
        yield assignment


def test_zero_predictions_give_arm_means_with_even_counts_and_post_only():
    data = panel()
    result = crossfit_encouragement(**data, learner_factory=ZeroLearner)
    assigned = data["z"][:, -1] == 1
    expected = np.stack([data[name][assigned, 1:].mean(axis=0)
                         - data[name][~assigned, 1:].mean(axis=0) for name in ("y", "d")], axis=-1)
    np.testing.assert_allclose(result.estimates, expected, atol=1e-14)
    np.testing.assert_allclose(result.unit_scores.mean(axis=0), expected, atol=1e-14)
    assert result.predictions.shape == (24, 2, 2, 2)
    assert result.metadata["panel"]["encouragement_index"] == 1
    assert result.metadata["conditional_assignment_probabilities"] == [.5, .5]
    assert len(result.table) == 6


def test_joint_covariance_matches_independent_saturated_regression_hc2():
    """A saturated fold-by-arm regression gives an independent HC2 sandwich."""
    data = panel(n=26)  # Odd arm counts also exercise unequal fold sizes.

    class Predictor(ZeroLearner):
        def predict(self, x):
            base = np.stack([2 * x[:, 0], .1 * x[:, 1]], axis=-1)
            return (base[:, None, None, :]
                    + np.array([1., 2.])[None, :, None, None]
                    + np.array([0., .2])[None, None, :, None])

    result = crossfit_encouragement(**data, learner_factory=Predictor, seed=43)
    assignment = data["z"][:, -1].astype(int)
    response = np.stack([data["y"], data["d"]], axis=-1)[:, 1:]
    residual = response - result.predictions[np.arange(26), :, assignment, :]
    matrix = np.column_stack([(result.fold_ids == fold) & (assignment == arm)
                              for fold in (0, 1) for arm in (0, 1)]).astype(float)
    inverse = np.linalg.inv(matrix.T @ matrix)
    fitted = matrix @ inverse @ matrix.T @ residual.reshape(26, 4)
    regression_residual = residual.reshape(26, 4) - fitted
    leverage = np.diag(matrix @ inverse @ matrix.T)
    contrast = np.array([(-1 if arm == 0 else 1) * np.mean(result.fold_ids == fold)
                         for fold in (0, 1) for arm in (0, 1)])
    weights = contrast @ inverse @ matrix.T
    adjusted_residual = regression_residual * (weights / np.sqrt(1 - leverage))[:, None]
    expected_covariance = adjusted_residual.T @ adjusted_residual
    np.testing.assert_allclose(result.covariance, expected_covariance, rtol=1e-13, atol=1e-14)
    expected_estimates = (result.predictions[:, :, 1] - result.predictions[:, :, 0]).mean(axis=0)
    expected_estimates += (contrast @ inverse @ matrix.T @ residual.reshape(26, 4)).reshape(2, 2)
    np.testing.assert_allclose(result.estimates, expected_estimates, rtol=1e-13, atol=1e-14)
    assert abs(result.covariance[0, 2]) > .01  # Retains serial dependence.
    assert np.linalg.eigvalsh(result.covariance).min() > -1e-12
    assert not np.allclose(result.covariance, np.cov(result.unit_scores.reshape(26, 4).T) / 26)


def test_conditional_randomization_exact_unbiasedness_with_bad_adaptive_predictor():
    """Enumerate every conditional assignment; allow severely wrong learned fits."""
    unit = np.arange(8.)
    control = np.stack([np.sin(unit), unit ** 2 / 10], axis=-1)[:, None, :]
    encouraged = control + np.stack([.1 * unit, .3 - unit / 20], axis=-1)[:, None, :]
    folds = np.repeat([0, 1], 4)
    estimates = []
    for assignment in conditional_assignments():
        responses = np.where(assignment[:, None, None], encouraged, control)
        predictions = np.empty((8, 1, 2, 2))
        for fold in (0, 1):
            # An intentionally unstable/misspecified predictor learned entirely
            # on the other fold; unbiasedness needs neither consistency nor fit.
            other = responses[folds != fold]
            coefficient = np.sin(other.sum()) + other.mean() ** 2
            predictions[folds == fold, 0, 0] = coefficient * (1 + unit[folds == fold, None])
            predictions[folds == fold, 0, 1] = -coefficient * unit[folds == fold, None] ** 2
        estimates.append(_infer_from_predictions(responses, assignment, predictions, folds, np.array([1])).estimates)
    np.testing.assert_allclose(np.mean(estimates, axis=0), (encouraged - control).mean(axis=0), atol=2e-13)


def test_neyman_covariance_omits_exactly_residual_effect_covariance_for_fixed_predictions():
    """Enumerate assignment to compare the true covariance to mean reported V."""
    rng = np.random.default_rng(71)
    potential = rng.normal(size=(8, 2, 2, 2))
    predictions = rng.normal(size=(8, 2, 2, 2))
    folds = np.repeat([0, 1], 4)
    results = []
    for assignment in conditional_assignments():
        responses = potential[np.arange(8), :, assignment, :]
        results.append(_infer_from_predictions(responses, assignment, predictions, folds, np.array([1, 2])))
    estimates = np.array([r.estimates.reshape(-1) for r in results])
    true_covariance = np.cov(estimates.T, ddof=0)
    expected_omitted = np.zeros((4, 4))
    for fold in (0, 1):
        residual = potential[folds == fold] - predictions[folds == fold]
        residual_effect = (residual[:, :, 1] - residual[:, :, 0]).reshape(4, 4)
        expected_omitted += .25 * np.cov(residual_effect.T, ddof=1) / 4
    observed_omitted = np.mean([r.covariance for r in results], axis=0) - true_covariance
    np.testing.assert_allclose(observed_omitted, expected_omitted, atol=1e-14)
    assert np.linalg.eigvalsh(observed_omitted).min() >= -1e-14


def test_split_uses_conditional_probability_when_arm_counts_are_not_proportional():
    data = panel(n=25)
    result = crossfit_encouragement(**data, learner_factory=ZeroLearner)
    assignment = data["z"][:, -1]
    expected = np.zeros((2, 2))
    response = np.stack([data["y"], data["d"]], axis=-1)[:, 1:]
    for fold in (0, 1):
        idx = result.fold_ids == fold
        expected += idx.mean() * (response[idx & (assignment == 1)].mean(axis=0)
                                  - response[idx & (assignment == 0)].mean(axis=0))
    np.testing.assert_allclose(result.estimates, expected, atol=1e-14)
    assert result.metadata["conditional_assignment_probabilities"] == [.5, 6 / 13]


def test_no_held_out_response_reaches_learner_and_entire_trajectory_stays_together():
    data = panel()
    data["x"] = np.arange(24.)[:, None]
    seen = []

    class AuditedLearner(ZeroLearner):
        def fit(self, x, z, responses, times):
            self.train_ids = x[:, 0].astype(int)
            np.testing.assert_array_equal(responses[:, :, 0], data["y"][self.train_ids, 1:])
            np.testing.assert_array_equal(responses[:, :, 1], data["d"][self.train_ids, 1:])
            np.testing.assert_array_equal(z, data["z"][self.train_ids, -1])
            np.testing.assert_array_equal(times, [1, 2])
            self.metadata_ = {"n_train": len(x)}
            return super().fit(x, z, responses, times)

        def predict(self, x):
            test_ids = x[:, 0].astype(int)
            assert not set(test_ids) & set(self.train_ids)
            assert set(test_ids) | set(self.train_ids) == set(range(24))
            seen.append(test_ids)
            return super().predict(x)

    result = crossfit_encouragement(**data, learner_factory=AuditedLearner)
    np.testing.assert_array_equal(np.sort(np.concatenate(seen)), np.arange(24))
    assert result.metadata["learners"][0]["metadata"] == {"n_train": 12}


def test_vector_assignment_means_all_supplied_periods_are_post():
    data = panel()
    full = crossfit_encouragement(**data, learner_factory=ZeroLearner, seed=9)
    vector = crossfit_encouragement(data["y"][:, 1:], data["d"][:, 1:], data["z"][:, -1],
                                    data["x"], data["t"][1:], learner_factory=ZeroLearner, seed=9)
    np.testing.assert_array_equal(full.estimates, vector.estimates)
    np.testing.assert_array_equal(full.covariance, vector.covariance)


def test_fieller_full_set_membership_equals_studentized_moment_at_every_grid_point():
    data = panel()
    result = crossfit_encouragement(**data, learner_factory=ZeroLearner)
    for horizon in (1, 2):
        row = result.confidence_set(horizon)
        for beta in np.r_[np.linspace(-50, 50, 201), -1e100, 1e100]:
            assert contains(row, beta) == (not result.ar_test(beta, horizon)["reject"])
    assert result.metadata["exact_randomization"] is False
    assert result.metadata["interval_scope"] == "pointwise"


@pytest.mark.parametrize("denominator", [0.0, .03, 2.0])
def test_weak_stage_fieller_has_nominal_gaussian_moment_coverage_on_deterministic_quantiles(denominator):
    """No random simulation: 400 equal-probability normal quantiles give 95%.

    This checks inversion under its limiting Gaussian model, not finite-sample
    exactness. A tiny or zero denominator never enters the test statistic.
    """
    beta = 1.75
    errors = norm.ppf((np.arange(400) + .5) / 400)
    # Residual patterns give joint ITT covariance [[1, .25], [.25, 1]].
    covariance = np.array([[1., .25], [.25, 1.]])
    cholesky = np.linalg.cholesky(covariance)
    variance_moment = np.array([1., -beta]) @ covariance @ np.array([1., -beta])
    # Two fold/arm replicates form four residual vectors with sample covariance
    # 2V; sum of four contributions (1/4)*(2V/2) is V.
    residual = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    residual = residual @ cholesky.T * np.sqrt(3)
    assignment = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 2)
    folds = np.repeat([0, 1], 8)
    # For n_arm=4 each of the four arm covariance terms contributes V/4.
    # residual sample covariance is 2V, so use sqrt(2) additional scaling.
    residual *= np.sqrt(2)
    covered = []
    for error in errors:
        effect = np.array([beta * denominator + np.sqrt(variance_moment) * error, denominator])
        response = np.tile(residual, (4, 1)) + assignment[:, None] * effect
        result = _infer_from_predictions(response[:, None, :], assignment,
                                         np.zeros((16, 1, 2, 2)), folds, np.array([1]))
        np.testing.assert_allclose(result.covariance, covariance, atol=1e-14)
        covered.append(contains(result.confidence_set(), beta))
    assert np.mean(covered) == .95


def test_zero_stage_all_real_and_empty_sets_are_preserved():
    data = panel()
    data["d"][:] = 0
    data["y"][:] = 0
    all_real = crossfit_encouragement(**data, learner_factory=ZeroLearner)
    assert all_real.confidence_set()["set_type"] == "all_real"
    assert np.isnan(all_real.confidence_set()["estimate"])
    assert all_real.ar_test(1)["p_value"] == 1
    data["y"] = data["z"].astype(float)
    empty = crossfit_encouragement(**data, learner_factory=ZeroLearner)
    assert empty.confidence_set()["set_type"] == "empty"
    assert empty.ar_test(1)["p_value"] == 0


def test_split_is_reproducible_and_independent_of_responses():
    data = panel()
    first = crossfit_encouragement(**data, learner_factory=ZeroLearner, seed=817)
    data["y"] *= -100
    data["d"][:] = 0
    second = crossfit_encouragement(**data, learner_factory=ZeroLearner, seed=817)
    np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
    third = crossfit_encouragement(**data, learner_factory=ZeroLearner, seed=818)
    assert not np.array_equal(first.fold_ids, third.fold_ids)


@pytest.mark.parametrize("kwargs, message", [
    ({"design": "bernoulli"}, "complete_randomization"),
    ({"design": "cluster"}, "complete_randomization"),
    ({"folds": 3}, "folds=2"),
    ({"folds": True}, "folds=2"),
    ({"seed": -1}, "nonnegative integer"),
    ({"seed": 1.2}, "nonnegative integer"),
    ({"alpha": 0}, "alpha"),
    ({"alpha": np.nan}, "alpha"),
    ({"learner_factory": None}, "callable"),
])
def test_invalid_configuration(kwargs, message):
    options = dict(learner_factory=ZeroLearner)
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        crossfit_encouragement(**panel(), **options)


@pytest.mark.parametrize("field", ["x", "y", "d", "z"])
def test_nonfinite_data_are_rejected_without_dropping_units(field):
    data = panel()
    data[field] = data[field].astype(float)
    data[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        crossfit_encouragement(**data, learner_factory=ZeroLearner)


def test_missing_arms_small_arms_staggered_assignment_and_nonabsorbing_adoption_rejected():
    for change in ("all_control", "small_arm", "staggered", "nonabsorbing"):
        data = panel()
        if change == "all_control":
            data["z"][:] = 0
        elif change == "small_arm":
            data["z"][6:] = 0
        elif change == "staggered":
            data["z"][1, 1] = 0
        else:
            data["d"][0] = [0, 1, 0]
        with pytest.raises(ValueError):
            crossfit_encouragement(**data, learner_factory=ZeroLearner)


@pytest.mark.parametrize("failure", ["nonfinite", "wrong_shape", "reused"])
def test_learner_contract_rejected(failure):
    class Invalid(ZeroLearner):
        def predict(self, x):
            result = super().predict(x)
            return result[:, :, 0] if failure == "wrong_shape" else result + (np.nan if failure == "nonfinite" else 0)
    learner = Invalid()
    factory = (lambda: learner) if failure == "reused" else Invalid
    with pytest.raises(ValueError, match="fresh learner|predictions"):
        crossfit_encouragement(**panel(), learner_factory=factory)


@pytest.mark.parametrize("horizon", [0, 3, True, 1.5])
def test_invalid_horizon_rejected(horizon):
    result = crossfit_encouragement(**panel(), learner_factory=ZeroLearner)
    with pytest.raises(ValueError, match="horizon"):
        result.confidence_set(horizon)
    with pytest.raises(ValueError, match="horizon"):
        result.ar_test(1, horizon)


@pytest.mark.parametrize("beta", [np.inf, np.nan, True, [1], "invalid", 1 + 2j])
def test_invalid_null_coefficient_rejected(beta):
    result = crossfit_encouragement(**panel(), learner_factory=ZeroLearner)
    with pytest.raises(ValueError, match="beta"):
        result.ar_test(beta)
