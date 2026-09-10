"""Account-held-out prediction and exact Gaussian checks for IV nuisance trees."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import block_diag

from longbet._direct_smooth import ForestDesign, time_covariance
from longbet._iv_nuisance import (
    LongBetIVNuisance,
    LongBetIVNuisanceConfig,
    _evaluate_forest,
    _intercept_whitening,
    _prediction_tree_space,
    _rule_masks,
    inner_validation_split,
    nuisance_validation_score,
)


def data(seed=321, n=32, h=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    z = np.tile([0, 1], n // 2)
    d = np.maximum.accumulate((rng.uniform(size=(n, h)) < (0.1 + 0.4 * z[:, None])).astype(float), axis=1)
    y = x[:, 0, None] + x[:, 1, None] * x[:, 2, None] + 2 * d + rng.normal(size=(n, h))
    return x, z, np.stack((y, d), axis=-1), np.arange(h, dtype=float)


def small_config(**kwargs):
    return LongBetIVNuisanceConfig(baseline_trees=2, effect_trees=2, cutpoints=2,
                                   burnin=5, draws=8, chains=2, **kwargs)


def test_split_is_stratified_disjoint_reproducible_and_unit_level():
    z = np.repeat([0, 1], [39, 41])
    train, valid = inner_validation_split(z, 10)
    assert len(valid) == 7 + 8
    assert not set(train) & set(valid)
    np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.arange(len(z)))
    np.testing.assert_array_equal(valid, inner_validation_split(z, 10)[1])
    assert not np.array_equal(valid, inner_validation_split(z, 11)[1])


def test_score_uses_inner_training_scales_and_observed_arm_only():
    training = np.array([[[0.0, 0.0]], [[2.0, 0.0]]])
    responses = np.array([[[3.0, 1.0]], [[5.0, 1.0]]])
    pred = np.array([[[[2.0, 0.75], [10000.0, 10000.0]]],
                     [[[10000.0, 10000.0], [4.0, 0.75]]]])
    score = nuisance_validation_score(pred, np.array([0, 1]), responses, training)
    assert score == pytest.approx((1.0 + (0.25 / 0.05)**2) / 2)


def test_finite_depth_two_support_partitions_each_account_and_replays():
    x, *_ = data(n=80)
    config = small_config(max_interaction_rules=2)
    space = _prediction_tree_space(x, config)
    np.testing.assert_allclose(space.mask.sum(axis=1), 1)
    np.testing.assert_allclose(np.exp(space.log_prior).sum(), 1)
    assert np.sum(space.feature[:, 1] >= 0) == 2
    np.testing.assert_array_equal(space.mask, _rule_masks(x, space.feature, space.threshold))
    np.testing.assert_array_equal(space.feature, _prediction_tree_space(x, config).feature)
    no_interaction = _prediction_tree_space(x, replace(config, interaction_partitions=False))
    assert np.all(no_interaction.feature[:, 1] == -1)


def test_constant_covariates_have_one_valid_root_tree():
    space = _prediction_tree_space(np.ones((12, 2)), small_config())
    assert space.mask.shape == (1, 4, 12)
    np.testing.assert_allclose(space.mask[0, 0], 1)
    np.testing.assert_allclose(space.log_prior, 0)


def test_predictive_rule_average_matches_individual_tree_draw_average():
    x, *_ = data()
    space = _prediction_tree_space(x, small_config())
    rng = np.random.default_rng(887)
    rule_draws = rng.integers(len(space.feature), size=(7, 3))
    leaf_draws = rng.normal(size=(7, 3, 4, 4, 2))
    accumulated = np.zeros((len(space.feature), 4, 4, 2))
    explicit = np.zeros((len(x), 4, 2))
    for rules, leaves in zip(rule_draws, leaf_draws):
        np.add.at(accumulated, rules, leaves)
        for rule, leaf in zip(rules, leaves):
            for unit in range(len(x)):
                explicit[unit] += leaf[np.flatnonzero(space.mask[rule, :, unit])[0]]
    np.testing.assert_allclose(_evaluate_forest(space.mask, accumulated / len(rule_draws)), explicit / len(rule_draws))


@pytest.mark.parametrize("periods", [1, 3, 11])
def test_intercept_whitening_matches_dense_covariance(periods):
    sigma2, gamma2 = 0.7, 2.4
    whitening, root = _intercept_whitening(sigma2, gamma2, periods)
    covariance = sigma2 * np.eye(periods) + gamma2 * np.ones((periods, periods))
    np.testing.assert_allclose(root @ root, covariance, atol=1e-13)
    np.testing.assert_allclose(whitening @ covariance @ whitening, np.eye(periods), atol=1e-13)
    np.testing.assert_allclose(root @ whitening, np.eye(periods), atol=1e-13)


@pytest.mark.parametrize("effect", [False, True])
def test_whitened_tree_posterior_matches_independent_dense_gaussian_algebra(effect):
    rng = np.random.default_rng(539)
    n, h = 12, 3
    x = rng.normal(size=(n, 2))
    space = _prediction_tree_space(x, small_config())
    weight = np.resize([-0.6, 0.4], n) if effect else np.ones(n)
    prior = time_covariance(np.arange(h), sd=0.8, length_scale=2, nugget=0.1)
    sigma2, gamma2 = 0.9, 1.7
    whitening, root = _intercept_whitening(sigma2, gamma2, h)
    design = ForestDesign.build(space, weight, whitening @ prior @ whitening)
    residual = rng.normal(size=(n, h))
    score, variance, projected = design.collapsed(residual @ whitening, 1.0)
    noise_covariance = sigma2 * np.eye(h) + gamma2 * np.ones((h, h))
    noise = block_diag(*[noise_covariance] * n)
    prior_dense = block_diag(*[prior] * 4)
    scores_dense = []
    for rule in range(len(space.feature)):
        matrix = np.zeros((n * h, 4 * h))
        for unit in range(n):
            for leaf in range(4):
                matrix[unit * h:(unit + 1) * h, leaf * h:(leaf + 1) * h] = np.eye(h) * space.mask[rule, leaf, unit] * weight[unit]
        precision = np.linalg.inv(prior_dense) + matrix.T @ np.linalg.solve(noise, matrix)
        covariance_dense = np.linalg.inv(precision)
        mean_dense = covariance_dense @ matrix.T @ np.linalg.solve(noise, residual.ravel())
        mean_white = (variance[rule] * projected[rule]) @ design.vectors.T
        np.testing.assert_allclose((mean_white @ root).ravel(), mean_dense, atol=3e-13)
        for leaf in range(4):
            covariance_white = (design.vectors * variance[rule, leaf]) @ design.vectors.T
            np.testing.assert_allclose(root @ covariance_white @ root,
                                       covariance_dense[leaf*h:(leaf+1)*h, leaf*h:(leaf+1)*h], atol=3e-13)
        marginal = noise + matrix @ prior_dense @ matrix.T
        scores_dense.append(space.log_prior[rule] - 0.5 * np.linalg.slogdet(marginal)[1]
                            - 0.5 * residual.ravel() @ np.linalg.solve(marginal, residual.ravel()))
    np.testing.assert_allclose(score - score[0], np.asarray(scores_dense) - scores_dense[0], atol=2e-13)


def test_fit_predict_replays_and_does_not_depend_on_other_test_accounts():
    x, z, responses, times = data()
    config = small_config(length_scales=(0.0, 2.0, 8.0), seed=29)
    model = LongBetIVNuisance(config).fit(x, z, responses, times)
    repeated = LongBetIVNuisance(config).fit(x, z, responses, times)
    prediction = model.predict(x[:5])
    assert prediction.shape == (5, len(times), 2, 2)
    assert np.all((prediction[..., 1] >= 0) & (prediction[..., 1] <= 1))
    np.testing.assert_array_equal(prediction, repeated.predict(x[:5]))
    np.testing.assert_array_equal(prediction, model.predict(np.r_[x[:5], x * 1000])[:5])
    assert model.metadata_["posterior_interval_inference"] is False
    assert len(model.metadata_["prediction_chain_rms_discrepancy"]) == 2
    before = prediction.copy()
    responses[:] = -999
    x[5:] = 500
    np.testing.assert_array_equal(before, model.predict(x[:5]))


def test_inner_training_has_no_validation_accounts(monkeypatch):
    x, z, responses, times = data()
    x[:, 0] = np.arange(len(x))
    model = LongBetIVNuisance(small_config(seed=17))
    original = model._fit_fixed
    seen = []
    def spy(xx, zz, yy, tt, length_scale, seed):
        seen.append(xx[:, 0].astype(int))
        return original(xx, zz, yy, tt, length_scale, seed)
    monkeypatch.setattr(model, "_fit_fixed", spy)
    model.fit(x, z, responses, times)
    train, valid = inner_validation_split(z, 17)
    assert len(seen) == 4
    for indices in seen[:-1]:
        np.testing.assert_array_equal(indices, train)
        assert not set(indices) & set(valid)
    np.testing.assert_array_equal(seen[-1], np.arange(len(x)))
    assert model.metadata_["inner_validation_indices"] == valid.tolist()


def test_training_only_affine_outcome_scaling_and_time_unit_invariance():
    x, z, responses, times = data()
    config = small_config(length_scales=(2.0,))
    first = LongBetIVNuisance(config).fit(x, z, responses, times)
    shifted = responses.copy()
    shifted[..., 0] = 10 + 3 * shifted[..., 0]
    second = LongBetIVNuisance(config).fit(x, z, shifted, 7 * times + 18)
    prediction = first.predict(x)
    prediction[..., 0] = 10 + 3 * prediction[..., 0]
    np.testing.assert_allclose(second.predict(x), prediction, atol=1e-10)


@pytest.mark.parametrize("kwargs", [dict(draws=0), dict(chains=0), dict(burnin=-1),
    dict(seed=True), dict(length_scales=()), dict(length_scales=(-1,)),
    dict(length_scales=(0, 0)), dict(validation_fraction=0.5), dict(nugget=0),
    dict(interaction_partitions=1), dict(max_interaction_rules=-1)])
def test_config_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        LongBetIVNuisanceConfig(**kwargs)


def test_fit_validation_and_unfitted_prediction():
    x, z, responses, times = data()
    model = LongBetIVNuisance(small_config(length_scales=(2.0,)))
    with pytest.raises(RuntimeError, match="Fit"):
        model.predict(x)
    with pytest.raises(ValueError, match="binary vector"):
        model.fit(x, z + 1, responses, times)
    with pytest.raises(ValueError, match="binary adoption"):
        model.fit(x, z, responses + 0.5, times)
    with pytest.raises(ValueError, match="strictly increasing"):
        model.fit(x, z, responses, times[::-1])
    with pytest.raises(ValueError, match="at least six"):
        LongBetIVNuisance(small_config()).fit(x[:8], z[:8], responses[:8], times)
    model.fit(x, z, responses, times)
    with pytest.raises(ValueError, match="feature count"):
        model.predict(x[:, :2])
    assert model.predict(np.empty((0, 3))).shape == (0, len(times), 2, 2)
