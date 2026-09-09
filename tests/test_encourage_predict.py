"""Natural-scale encouragement standardization keeps units and joint draws aligned."""

import copy

import numpy as np
import pytest
from scipy.special import ndtr

from longbet import LongBetConfig, LongBetMulti
from longbet._encourage_predict import _target_weights, standardize_encouragement


@pytest.fixture(scope="module")
def mixed_fit():
    """One small fit exercises both scales and nontrivial chain ordering."""
    rng = np.random.default_rng(719)
    n, periods = 8, 4
    x = rng.normal(size=(n, 2)).astype(np.float32)
    z = np.zeros((n, periods))
    z[:4, 1:] = 1
    t = np.array([0.1, 1.1, 3.1, 4.1])
    y = rng.normal(size=(n, periods)) + z * (1 + x[:, :1])
    d = (rng.uniform(size=(n, periods)) < (0.2 + 0.55 * z)).astype(float)
    cfg = LongBetConfig(
        sigma_prior_a=2, sigma_prior_b=1,
        num_sweeps=6, num_burnin=4, num_chains=2,
        num_trees_pr=2, num_trees_trt=2, random_seed=86,
    )
    model = LongBetMulti(cfg).fit(
        y={"outcome": y, "takeup": d}, x=x, z=z, t=t,
        outcome={"outcome": "continuous", "takeup": "binary"},
    )
    return model, x, z, t


@pytest.mark.parametrize("standardization", ["conditional", "population"])
def test_blocked_full_equivalence_and_common_population(mixed_fit, monkeypatch, standardization):
    from longbet import _encourage_predict as ep

    model, x, z, t = mixed_fit
    groups = np.array(["a", "b", "a", "b", "b", "b", "a", "a"])
    weights = np.arange(1, len(x) + 1, dtype=float)
    kwargs = dict(groups=groups, weights=weights, alpha=0.2, standardization=standardization)
    full = standardize_encouragement(model, x, z, t, summary_only=False, **kwargs)
    calls = []
    original = ep._eval_block

    def bounded_eval(design, lo, hi, width, trace):
        calls.append((lo, hi, width))
        assert hi - lo <= width <= 5
        return original(design, lo, hi, width, trace)

    monkeypatch.setattr(ep, "_eval_block", bounded_eval)
    blocked = standardize_encouragement(model, x, z, t, block_size=5, **kwargs)
    assert len(calls) > 2
    assert blocked.cell_draws is None
    assert blocked.group_labels == ("all", "a", "b")
    np.testing.assert_array_equal(blocked.group_counts, [8, 4, 4])
    np.testing.assert_array_equal(blocked.periods, [1.1, 3.1, 4.1])
    np.testing.assert_array_equal(blocked.period_indices, [1, 2, 3])
    np.testing.assert_array_equal(blocked.horizons, [1, 3, 4])
    np.testing.assert_allclose(blocked.weights.sum(axis=1), 1)
    assert blocked.effects.shape == (2, 6, 3, 3, 2)
    assert blocked.provenance == model.provenance
    for m, name in enumerate(model.outcome_names):
        np.testing.assert_allclose(blocked.draws[name], full.draws[name], rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(blocked.cell_summaries[name], full.cell_summaries[name], atol=2e-7)
        cells = full.cell_draws[name]
        np.testing.assert_array_equal(cells[:, 0], 0)
        expected = np.einsum("gn,nhd->ghd", full.weights, cells[:, 1:]).reshape(3, 3, 2, 6)
        np.testing.assert_allclose(blocked.draws[name], expected, rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(blocked.effects[..., m], expected.transpose(2, 3, 0, 1), atol=2e-7)
        np.testing.assert_allclose(full.cell_summaries[name].mean, cells.mean(axis=2), atol=2e-7)
        np.testing.assert_allclose(full.cell_summaries[name].lower, np.quantile(cells, .1, axis=2), atol=2e-7)
        # Summary-only prediction must not create the existing optional forest cache.
        assert not hasattr(model[name], "_eval_cache")


def test_probability_scale_and_population_marginalization_algebra(mixed_fit):
    model, x, z, t = mixed_fit
    conditional = standardize_encouragement(model, x, z, t, summary_only=False)
    population = standardize_encouragement(model, x, z, t, summary_only=False,
                                          standardization="population")
    all_encouraged = np.zeros_like(z)
    all_encouraged[:, 1:] = 1
    ordinary = model.predict(x=x, z=all_encouraged, t=t, summary_only=False)
    for name in model.outcome_names:
        child = model[name]
        pred = ordinary[name]
        if child.config.outcome == "continuous":
            expected = pred.tauhats.copy()
            np.testing.assert_allclose(population.draws[name], conditional.draws[name], atol=2e-7)
        else:
            expected = ndtr(pred.muhats0 + pred.tauhats) - ndtr(pred.muhats0)
            gamma = np.asarray(child.trace.gamma).reshape(-1, len(x)).T[:, None, :]
            variance = np.asarray(child.trace.sigma_gamma2).reshape(-1)
            eta0 = pred.muhats0 - gamma * child.sdy
            denominator = np.sqrt(1 + variance * child.sdy ** 2)
            marginal = ndtr((eta0 + pred.tauhats) / denominator) - ndtr(eta0 / denominator)
            marginal[:, 0] = 0
            np.testing.assert_allclose(population.cell_draws[name], marginal, rtol=2e-5, atol=3e-7)
            assert np.max(np.abs(marginal - expected)) > 1e-4
            assert np.all(np.abs(conditional.cell_draws[name]) <= 1)
        expected[:, 0] = 0
        np.testing.assert_allclose(conditional.cell_draws[name], expected, rtol=2e-5, atol=3e-7)


def test_alias_and_fitted_calendar_default(mixed_fit):
    model, x, z, t = mixed_fit
    actual = standardize_encouragement(model, x, z, standardization="observed")
    expected = standardize_encouragement(model, x, z, t, standardization="conditional")
    assert actual.standardization == "conditional"
    for name in model.outcome_names:
        np.testing.assert_array_equal(actual.draws[name], expected.draws[name])


def test_fractional_origin_keeps_original_labels_and_recovers_fitted_offsets(mixed_fit):
    model, x, z, t = mixed_fit
    explicit = standardize_encouragement(model, x, z, t)
    implicit = standardize_encouragement(model, x, z)
    np.testing.assert_array_equal(explicit.periods, t[1:])
    np.testing.assert_array_equal(implicit.periods,
                                  float(model.fitted_t_[0]) + np.array([1, 3, 4]))
    np.testing.assert_array_equal(explicit.horizons, implicit.horizons)
    for name in model.outcome_names:
        np.testing.assert_array_equal(explicit.draws[name], implicit.draws[name])


def test_fitted_nonwhole_gaps_and_calendar_rounding_clock_drift_are_rejected(mixed_fit):
    model, x, z, t = mixed_fit
    nonwhole = copy.copy(model)
    nonwhole.fitted_t_ = np.array([.1, 1.2, 3.2, 4.2], dtype=np.float32)
    with pytest.raises(ValueError, match="whole-unit gaps"):
        standardize_encouragement(nonwhole, x, z)
    # Equal float32 representations alone are insufficient: integer original
    # offsets can change by an entire period when the calendar is very large.
    original = np.array([16777216., 16777219., 16777222., 16777225.])
    drifted = copy.copy(model)
    drifted.fitted_t_ = original.astype(np.float32)
    with pytest.raises(ValueError, match="fitted exposure clock"):
        standardize_encouragement(drifted, x, z, original)


@pytest.mark.parametrize("kwargs, message", [
    ({"block_size": 0}, "block_size"),
    ({"block_size": True}, "block_size"),
    ({"block_size": 1.5}, "block_size"),
    ({"alpha": 0}, "alpha"),
    ({"standardization": "latent"}, "standardization"),
    ({"groups": ["all"] * 8}, "cannot be 'all'"),
    ({"groups": [None] * 8}, "nonmissing"),
    ({"weights": [0] * 8}, "positive total"),
    ({"weights": [-1] * 8}, "nonnegative"),
    ({"weights": [float("inf")] * 8}, "finite"),
])
def test_invalid_options(mixed_fit, kwargs, message):
    model, x, z, t = mixed_fit
    with pytest.raises(ValueError, match=message):
        standardize_encouragement(model, x, z, t, **kwargs)


def test_rejects_missing_fit_rows_calendar_and_legacy_semantics(mixed_fit):
    model, x, z, t = mixed_fit
    with pytest.raises(RuntimeError, match="fitted"):
        standardize_encouragement(LongBetMulti(), x, z, t)
    with pytest.raises(ValueError, match="all fitted study units"):
        standardize_encouragement(model, x[:-1], z[:-1], t)
    with pytest.raises(ValueError, match="fitted calendar"):
        standardize_encouragement(model, x, z, t + 1)
    bad_x = x.copy()
    bad_x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        standardize_encouragement(model, bad_x, z, t)
    legacy = copy.copy(model)
    legacy.sampler_semantics = "legacy"
    with pytest.raises(ValueError, match="current joint sampler semantics"):
        standardize_encouragement(legacy, x, z, t)


def test_fixed_weights_robust_to_overflow_and_ambiguous_groups():
    labels, counts, weights = _target_weights(4, ["a", "b", "a", "b"], [1e308] * 4)
    assert labels == ("all", "a", "b")
    np.testing.assert_array_equal(counts, [4, 2, 2])
    np.testing.assert_allclose(weights, [[.25] * 4, [.5, 0, .5, 0], [0, .5, 0, .5]])
    with pytest.raises(ValueError, match="positive total weight"):
        _target_weights(4, ["a", "a", "b", "b"], [1, 1, 0, 0])
    with pytest.raises(ValueError, match="unique as strings"):
        _target_weights(2, np.array([1, "1"], dtype=object), None)


def test_no_unit_intercept_makes_conditional_and_population_equal(mixed_fit):
    """Use a copy with zero intercept draws to isolate the integration identity."""
    import dataclasses

    model, x, z, t = mixed_fit
    view = copy.copy(model)
    view.fits = [copy.copy(child) for child in model.fits]
    for child in view.fits:
        child.config = dataclasses.replace(child.config, random_intercept=False)
        child.trace = dataclasses.replace(child.trace, gamma=np.zeros_like(child.trace.gamma))
    conditional = standardize_encouragement(view, x, z, t)
    population = standardize_encouragement(view, x, z, t, standardization="population")
    for name in model.outcome_names:
        np.testing.assert_array_equal(conditional.draws[name], population.draws[name])
