"""Checks of the experimental modular distribution and its inference boundary."""

import numpy as np
import pytest

from longbet._coupled_hazard_iv import (
    CoupledHazardConfig,
    CoupledHazardIV,
    CoupledHazardResult,
    _outcome_posterior,
)


def panel(seed=42):
    rng = np.random.default_rng(seed)
    n, t_len = 60, 4
    z = np.zeros((n, t_len))
    z[: n // 2, 1:] = 1
    d = np.zeros_like(z)
    for s in range(1, t_len):
        d[:, s] = np.maximum(d[:, s - 1], rng.random(n) < (.15 + .5 * z[:, s]))
    y = 1.2 * d + rng.normal(0, .5, size=d.shape)
    return y, d, z


def test_working_coefficients_never_export_a_causal_interval():
    y, d, z = panel()
    cfg = CoupledHazardConfig(num_sweeps=50, num_burnin=20, seed=123)
    with pytest.warns(UserWarning, match="causal_ci is withheld"):
        res = CoupledHazardIV(cfg).fit(y, d, z)
    assert isinstance(res, CoupledHazardResult)
    assert res.beta_post.shape == res.rho_post.shape == (30,)
    assert np.isfinite(res.beta_post).all()
    assert res.causal_ci is None
    assert res.metadata["causal_inference_supported"] is False
    assert res.metadata["weak_instrument_robust"] is False
    assert res.metadata["distribution"] == "modular_cut"
    assert "working coefficient" in res.summary_table.parameter.iloc[0]
    np.testing.assert_allclose(res.beta_ci, np.quantile(res.beta_post, [.025, .975]))


def test_conditional_sampler_matches_full_regression_with_time_intercepts():
    """Independent augmented-regression calculation checks the marginalized NIG."""
    y, d, z = panel()
    d_hat = .1 + .4 * z
    mean, covariance, shape, scale, rank = _outcome_posterior(y, d, d_hat, 4., 1.)
    n, periods = y.shape
    intercepts = np.tile(np.eye(periods), (n, 1))
    design = np.column_stack([d.ravel(), (d - d_hat).ravel(), intercepts])
    prior = np.diag([.25, 1., *np.zeros(periods)])
    full_covariance = np.linalg.inv(design.T @ design + prior)
    full_mean = full_covariance @ design.T @ y.ravel()
    residual = y.ravel() - design @ full_mean
    expected_scale = 1 + .5 * (residual @ residual + full_mean @ prior @ full_mean)
    np.testing.assert_allclose(mean, full_mean[:2], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(covariance, full_covariance[:2, :2], rtol=1e-10, atol=1e-10)
    assert shape == 2 + .5 * (n - 1) * periods
    assert scale == pytest.approx(expected_scale)
    assert rank == 2


def test_zero_relevance_exposes_nonidentification_instead_of_prior_precision_as_information():
    y, d, z = panel()
    cfg = CoupledHazardConfig(num_sweeps=30, num_burnin=10, prior_pi0=0.)
    with pytest.warns(UserWarning, match="experimental"):
        res = CoupledHazardIV(cfg).fit(y, d, z)
    assert res.xi_inclusion_prob == 0
    assert res.metadata["null_relevance_fraction"] == 1
    assert res.metadata["rank_deficient_fraction"] == 1
    assert res.causal_ci is None
    # Even with many observations, the two equal columns only identify their sum.
    # The beta variance relative to sigma^2 cannot fall below the conditional
    # prior variance 4*1/(4+1) in this direction.
    _, covariance, _, _, rank = _outcome_posterior(
        np.tile(y, (10, 1)), np.tile(d, (10, 1)), np.full((len(y) * 10, y.shape[1]), .3), 4., 1.,
    )
    assert rank == 1
    assert covariance[0, 0] >= .8 - 1e-10


def test_modular_hazard_does_not_receive_outcome_feedback():
    y, d, z = panel()
    cfg = CoupledHazardConfig(num_sweeps=40, num_burnin=10)
    with pytest.warns(UserWarning):
        result = CoupledHazardIV(cfg).fit(y, d, z)
    with pytest.warns(UserWarning):
        shifted = CoupledHazardIV(cfg).fit(y + 100 * d, d, z)
    assert shifted.xi_inclusion_prob == result.xi_inclusion_prob
    assert shifted.metadata["null_relevance_fraction"] == result.metadata["null_relevance_fraction"]
    assert shifted.beta_mean != pytest.approx(result.beta_mean)
    assert result.metadata["outcome_feedback_to_hazard"] is False


def test_outcome_time_shifts_are_integrated_exactly():
    y, d, z = panel()
    cfg = CoupledHazardConfig(num_sweeps=30, num_burnin=10)
    with pytest.warns(UserWarning):
        result = CoupledHazardIV(cfg).fit(y, d, z)
    with pytest.warns(UserWarning):
        shifted = CoupledHazardIV(cfg).fit(y + np.array([100., -10., 300., 1.]), d, z)
    np.testing.assert_allclose(result.beta_post, shifted.beta_post, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.sigma_y_post, shifted.sigma_y_post, rtol=1e-10, atol=1e-10)


def test_exhausted_and_separated_risk_sets_have_proper_draws():
    y, d, z = panel()
    d[:, 1:] = 1
    with pytest.warns(UserWarning):
        result = CoupledHazardIV(CoupledHazardConfig(num_sweeps=30, num_burnin=10)).fit(y, d, z)
    assert np.isfinite(result.beta_post).all()
    assert np.isfinite(result.sigma_y_post).all()
    assert result.metadata["rank_deficient_fraction"] == 1


@pytest.mark.parametrize("kwargs", [
    {"num_sweeps": 0}, {"num_sweeps": 40, "num_burnin": 40}, {"num_burnin": -1},
    {"num_sweeps": True}, {"seed": -1}, {"prior_pi0": -0.1}, {"prior_pi0": np.nan},
    {"slab_var": 0}, {"beta_prior_var": np.inf}, {"rho_prior_var": -1},
    {"hazard_intercept_prior_var": 0},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CoupledHazardConfig(**kwargs)


def test_unused_covariates_and_irregular_times_are_rejected():
    y, d, z = panel()
    model = CoupledHazardIV()
    with pytest.raises(ValueError, match="x adjustment"):
        model.fit(y, d, z, x=np.zeros((len(y), 1)))
    with pytest.raises(ValueError, match="equally spaced"):
        model.fit(y, d, z, t=np.array([1, 2, 4, 5]))
    with pytest.raises(ValueError, match="aligned"):
        model.fit(y[:-1], d, z)
    d[:, -1] = 0
    with pytest.raises(ValueError, match="switches back"):
        model.fit(y, d, z)
