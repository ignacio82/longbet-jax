"""Tests for structural treatment-clock duration deconvolution."""

import numpy as np
import pytest

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
    true_beta = 2.0
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
