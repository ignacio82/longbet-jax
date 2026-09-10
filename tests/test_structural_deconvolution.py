"""Tests for structural treatment-clock duration deconvolution."""

import numpy as np
import pytest
from scipy import stats

from longbet._structural_deconvolution import (
    DurationDeconvolutionResult,
    duration_deconvolution_effects,
)


def test_structural_deconvolution_basic():
    np.random.seed(42)
    n = 50
    t_len = 6

    # Encouragement at t=2
    z = np.zeros((n, t_len))
    z[:25, 2:] = 1.0

    # Adoption: treated adopt earlier (t=2), control adopt later (t=4)
    d = np.zeros((n, t_len))
    d[:25, 2:] = 1.0
    d[25:, 4:] = 1.0

    duration = np.cumsum(d, axis=1)
    y = 0.5 * duration + np.random.normal(0, 0.1, size=(n, t_len))

    # Linear
    res_lin = duration_deconvolution_effects(y, d, z, model_type="linear")
    assert isinstance(res_lin, DurationDeconvolutionResult)
    assert len(res_lin.durations) == 4
    assert len(res_lin.effects) == 4
    assert res_lin.first_stage_f > 0
    assert not res_lin.summary_table.empty

    # Stepwise
    res_step = duration_deconvolution_effects(y, d, z, model_type="stepwise")
    assert len(res_step.effects) == 4

    # Spline
    res_spline = duration_deconvolution_effects(y, d, z, model_type="spline")
    assert len(res_spline.effects) == 4


def test_acceleration_scenario_recovers_duration_effect():
    """Test where static stock first stage vanishes at t=3, but duration effect is identified."""
    np.random.seed(123)
    n = 100
    t_len = 5

    # Half treated, half control
    z = np.zeros((n, t_len))
    z[:50, 1:] = 1.0

    # Encouraged adopt at period 1, controls adopt at period 3
    d = np.zeros((n, t_len))
    d[:50, 1:] = 1.0
    d[50:, 3:] = 1.0

    # Note that at t=3 and t=4, d is 1 for both groups!
    # Static stock difference at t=3 is 0:
    assert np.all(d[:50, 3] == 1.0)
    assert np.all(d[50:, 3] == 1.0)

    # True duration effect: beta = 1.5 per period of exposure
    duration = np.cumsum(d, axis=1)
    true_beta = 1.5
    unit_fe = np.random.normal(0, 0.5, size=(n, 1))
    time_fe = np.sin(np.arange(t_len)).reshape(1, -1)
    noise = np.random.normal(0, 0.1, size=(n, t_len))
    y = unit_fe + time_fe + true_beta * duration + noise

    res = duration_deconvolution_effects(y, d, z, model_type="linear")

    # Marginal effect per period is effects[0]
    estimated_beta_per_period = res.effects[0]
    assert np.isclose(estimated_beta_per_period, true_beta, atol=0.2)
    # 95% CI covers true_beta
    assert res.ci_lower[0] <= true_beta <= res.ci_upper[0]
    assert res.first_stage_f > 20.0


def test_error_handling():
    y = np.ones((10, 4))
    d = np.ones((10, 4))
    z = np.ones((10, 4))

    with pytest.raises(ValueError, match="2D arrays"):
        duration_deconvolution_effects(y[0], d, z)

    with pytest.raises(ValueError, match="No adoption"):
        duration_deconvolution_effects(y, np.zeros((10, 4)), z)

    with pytest.raises(ValueError, match="At least 3 units"):
        duration_deconvolution_effects(y[:2], d[:2], z[:2])


def _varied_panel(seed=942, n=80, periods=7, onset=1):
    rng = np.random.default_rng(seed)
    assigned = np.arange(n) < n // 2
    z = assigned[:, None] * (np.arange(periods)[None, :] >= onset)
    adoption = rng.integers(1, periods + 1, size=n)
    adoption[assigned] = np.maximum(1, adoption[assigned] - 1)
    d = (np.arange(periods)[None, :] >= adoption[:, None]).astype(float)
    duration = d.cumsum(axis=1)
    y = (
        rng.normal(size=(n, 1))
        + np.arange(periods)[None, :] ** 2 / 10
        + 1.7 * duration - 0.08 * duration**2
        + rng.normal(size=(n, periods))
    )
    return y, d, z


def _dummy_2sls_reference(y, d, z, model_type, max_duration):
    """Full-dummy reference, independent of within transforms/SVD in the API."""
    n, periods = y.shape
    duration = d.cumsum(axis=1).ravel()
    if model_type == "linear":
        endogenous = duration[:, None]
    elif model_type == "stepwise":
        endogenous = (duration[:, None] >= np.arange(1, max_duration + 1)).astype(float)
    else:
        endogenous = np.column_stack((duration, duration**2))
    p = endogenous.shape[1]
    unit = np.repeat(np.arange(n), periods)
    wave = np.tile(np.arange(periods), n)
    fixed = np.column_stack((np.eye(n)[unit], np.eye(periods)[wave, 1:]))
    first_wave = max(1, int(np.flatnonzero(z.any(axis=0))[0]))
    excluded = np.column_stack([
        z[:, -1][unit] * (wave == j) for j in range(first_wave, periods)
    ])
    design = np.column_stack((fixed, endogenous))
    instruments = np.column_stack((fixed, excluded))
    instrument_inverse = np.linalg.inv(instruments.T @ instruments)
    first_beta = np.linalg.lstsq(instruments, endogenous, rcond=None)[0]
    fitted_design = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
    beta = np.linalg.lstsq(fitted_design, y.ravel(), rcond=None)[0]
    residual = y.ravel() - design @ beta
    bread = np.linalg.inv(fitted_design.T @ fitted_design)
    scores = np.array([fitted_design[unit == i].T @ residual[unit == i] for i in range(n)])
    correction = n / (n - 1) * (n * periods - 1) / (n * periods - design.shape[1])
    cov = correction * bread @ (scores.T @ scores) @ bread
    first_f = np.nan
    if p == 1:
        first_residual = endogenous[:, 0] - instruments @ first_beta[:, 0]
        first_scores = np.array([
            instruments[unit == i].T @ first_residual[unit == i] for i in range(n)
        ])
        first_correction = n / (n - 1) * (n * periods - 1) / (n * periods - instruments.shape[1])
        first_cov = first_correction * instrument_inverse @ (first_scores.T @ first_scores) @ instrument_inverse
        q = excluded.shape[1]
        first_f = first_beta[-q:, 0] @ np.linalg.solve(first_cov[-q:, -q:], first_beta[-q:, 0]) / q
    return beta[-p:], cov[-p:, -p:], residual.reshape(n, periods), first_f


@pytest.mark.parametrize("model_type", ["linear", "stepwise", "quadratic"])
@pytest.mark.parametrize("onset", [0, 1, 3])
def test_matches_full_dummy_2sls_and_unit_cluster_covariance(model_type, onset):
    y, d, z = _varied_panel(onset=onset)
    result = duration_deconvolution_effects(y, d, z, model_type=model_type, max_duration=3)
    beta, cov, residual, first_f = _dummy_2sls_reference(y, d, z, model_type, 3)
    np.testing.assert_allclose(result.coefficients, beta, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result.coefficient_covariance, cov, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result.residuals, residual, rtol=1e-8, atol=1e-10)
    n, periods = y.shape
    assert result.instrument_rank == periods - max(1, onset)
    assert result.regressor_rank == len(beta)
    assert result.residual_df == n * periods - (n + periods - 1) - len(beta)
    assert result.n_clusters == n
    np.testing.assert_allclose(
        result.ci_upper - result.effects,
        stats.t.ppf(0.975, n - 1) * result.se,
    )
    if model_type == "linear":
        assert result.first_stage_status == "available"
        assert result.first_stage_f == pytest.approx(first_f, rel=1e-9)
    else:
        assert np.isnan(result.first_stage_f)
        assert result.first_stage_status == "unavailable_for_multiple_endogenous_regressors"


def test_zero_first_stage_with_variable_adoption_is_rejected():
    adoption = np.tile([1, 2, 3, 5], 2)
    d = (np.arange(5)[None, :] >= adoption[:, None]).astype(float)
    z = np.zeros_like(d)
    z[:4, 1:] = 1
    y = d.cumsum(axis=1) + np.arange(8)[:, None]
    assert np.std(d, axis=0).max() > 0
    with pytest.raises(ValueError, match="instrumented regressor rank is 0"):
        duration_deconvolution_effects(y, d, z)


def test_fixed_effects_remove_uninformative_duration_regressor():
    y, d, z = _varied_panel()
    d[:] = d[0]
    with pytest.raises(ValueError, match="no within-panel variation"):
        duration_deconvolution_effects(y, d, z)


def test_insufficient_instrument_rank_is_rejected():
    y, d, z = _varied_panel(onset=6)
    with pytest.raises(ValueError, match="2 endogenous regressors but only 1"):
        duration_deconvolution_effects(y, d, z, model_type="quadratic")


def test_collinear_endogenous_regressors_are_rejected():
    y, d, z = _varied_panel()
    d[:] = 0
    d[:25, -1] = 1  # s and s**2 coincide, despite six independent instruments.
    with pytest.raises(ValueError, match="instrumented regressor rank is 1"):
        duration_deconvolution_effects(y, d, z, model_type="quadratic")


def test_ridge_cannot_manufacture_identification():
    y, d, z = _varied_panel()
    with pytest.raises(ValueError, match="regularization cannot supply IV identification"):
        duration_deconvolution_effects(y, d, z, ridge_lambda=1e-5)


def test_weak_first_stage_is_reported_without_shrinking_coefficient():
    # Independent arm assignments and adoption dates give a weak realized stage.
    rng = np.random.default_rng(771)
    n, periods = 400, 5
    assigned = np.arange(n) < n // 2
    z = assigned[:, None] * (np.arange(periods)[None, :] >= 1)
    adoption = rng.integers(1, periods + 1, n)
    d = (np.arange(periods)[None, :] >= adoption[:, None]).astype(float)
    y = 2 * d.cumsum(axis=1) + rng.normal(size=(n, periods))
    result = duration_deconvolution_effects(y, d, z)
    assert result.first_stage_status == "available"
    assert 0 < result.first_stage_f < 5
    assert result.first_stage_diagnostics.partial_r_squared.iloc[0] < 0.05
    # With no ridge penalty, changing outcome units cannot change the fit's
    # inferential content even when the first stage is weak.
    scaled = duration_deconvolution_effects(1e-8 * y, d, z)
    np.testing.assert_allclose(scaled.effects / 1e-8, result.effects, rtol=1e-10)
    np.testing.assert_allclose(scaled.se / 1e-8, result.se, rtol=1e-10)
    assert scaled.first_stage_f == pytest.approx(result.first_stage_f)


def test_perfect_first_stage_has_explicit_status():
    z = np.zeros((20, 5))
    z[:10, 1:] = 1
    y = 2 * z.cumsum(axis=1) + np.random.default_rng(5).normal(size=z.shape)
    result = duration_deconvolution_effects(y, z, z)
    assert np.isinf(result.first_stage_f)
    assert result.first_stage_status == "perfect_fit"


def test_singular_first_stage_cluster_covariance_is_unavailable():
    # Only two distinct uptake histories within each arm cannot support the
    # covariance of four instrument coefficients, even with many repeated units.
    d = np.zeros((40, 5))
    d[:15, 1:] = 1
    d[20:25, 1:] = 1
    z = np.zeros_like(d)
    z[:20, 1:] = 1
    y = d.cumsum(axis=1) + np.random.default_rng(91).normal(size=d.shape)
    result = duration_deconvolution_effects(y, d, z)
    assert result.instrument_rank == 4
    assert np.isnan(result.first_stage_f)
    assert result.first_stage_status == "singular_cluster_covariance"


@pytest.mark.parametrize("name", ["y", "d", "z"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_missing_or_nonfinite_panels_are_rejected(name, bad):
    y, d, z = _varied_panel()
    panels = dict(y=y, d=d, z=z.astype(float))
    panels[name][0, 0] = bad
    with pytest.raises(ValueError, match="finite balanced panels"):
        duration_deconvolution_effects(**panels)


@pytest.mark.parametrize("name", ["d", "z"])
def test_nonbinary_and_reversible_panels_are_rejected(name):
    y, d, z = _varied_panel()
    panels = dict(y=y, d=d, z=z.astype(float))
    panels[name][0, 0] = 0.5
    with pytest.raises(ValueError, match=f"{name} must be binary"):
        duration_deconvolution_effects(**panels)
    panels[name][0, 0] = 1
    panels[name][0, 1] = 0
    with pytest.raises(ValueError, match=f"{name} must be absorbing"):
        duration_deconvolution_effects(**panels)


@pytest.mark.parametrize("assignment", [0, 1])
def test_no_assignment_variation_is_rejected(assignment):
    y, d, z = _varied_panel()
    with pytest.raises(ValueError, match="both assigned and never-encouraged"):
        duration_deconvolution_effects(y, d, np.full_like(z, assignment))


def test_staggered_encouragement_is_rejected():
    y, d, z = _varied_panel()
    z[0, 1] = 0
    with pytest.raises(ValueError, match="common encouragement onset"):
        duration_deconvolution_effects(y, d, z)


@pytest.mark.parametrize("times", [
    [0, 1, 2, 3, 4, 5, 7],
    [0, 1, 2, 3, 4, 4, 5],
    [6, 5, 4, 3, 2, 1, 0],
    [0, 1, 2, 3, 4, 5, np.nan],
])
def test_irregular_or_invalid_time_is_rejected(times):
    y, d, z = _varied_panel()
    with pytest.raises(ValueError, match="t must be"):
        duration_deconvolution_effects(y, d, z, t=times)


def test_time_units_do_not_change_number_of_observed_exposure_periods():
    y, d, z = _varied_panel()
    regular = duration_deconvolution_effects(y, d, z)
    calendar = duration_deconvolution_effects(y, d, z, t=2000 + 7 * np.arange(y.shape[1]))
    np.testing.assert_array_equal(calendar.effects, regular.effects)
    np.testing.assert_array_equal(calendar.se, regular.se)


@pytest.mark.parametrize("bad", [0, -1, 2.1, np.nan, np.inf, True])
def test_invalid_maximum_duration_is_rejected(bad):
    y, d, z = _varied_panel()
    with pytest.raises(ValueError, match="positive integer"):
        duration_deconvolution_effects(y, d, z, max_duration=bad)


@pytest.mark.parametrize("bad", [0, 1, -0.1, np.nan])
def test_invalid_alpha_is_rejected(bad):
    y, d, z = _varied_panel()
    with pytest.raises(ValueError, match="alpha must be"):
        duration_deconvolution_effects(y, d, z, alpha=bad)


def test_legacy_spline_name_is_quadratic_alias():
    y, d, z = _varied_panel()
    legacy = duration_deconvolution_effects(y, d, z, model_type="spline")
    quadratic = duration_deconvolution_effects(y, d, z, model_type="quadratic")
    np.testing.assert_array_equal(legacy.effects, quadratic.effects)
    np.testing.assert_array_equal(legacy.coefficient_covariance, quadratic.coefficient_covariance)


def test_linear_maximum_duration_changes_only_evaluation_grid():
    y, d, z = _varied_panel()
    full = duration_deconvolution_effects(y, d, z)
    short = duration_deconvolution_effects(y, d, z, max_duration=2)
    np.testing.assert_array_equal(short.effects, full.effects[:2])
    np.testing.assert_array_equal(short.coefficients, full.coefficients)
