"""Unit tests for heterogeneous exposure-duration instrumental variables."""
import numpy as np
import pytest

from longbet import heterogeneous_duration_iv, HeterogeneousDurationIVResult


def simulate_duration_panel(
    n_units: int = 300,
    n_periods: int = 12,
    beta_0: float = 8.0,
    beta_1: float = 3.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_units, 1))
    z = rng.binomial(1, 0.5, size=(n_units, 1))

    # Unit fixed effects and time fixed effects
    alpha_i = rng.standard_normal((n_units, 1)) * 2.0
    lambda_t = np.sin(np.linspace(0, 3, n_periods))[None, :]

    # Adoption hazard: encouraged units adopt earlier
    base_hazard = 0.08
    enc_hazard = 0.28
    d = np.zeros((n_units, n_periods))
    for i in range(n_units):
        p_adopt = enc_hazard if z[i, 0] == 1 else base_hazard
        adopted = False
        for t in range(n_periods):
            if adopted or rng.random() < p_adopt:
                adopted = True
                d[i, t] = 1.0

    # Accumulated exposure duration S_{it}
    s = np.cumsum(d, axis=1)

    # Heterogeneous slope beta_i = beta_0 + beta_1 * X_i
    beta_i = beta_0 + beta_1 * x  # (n_units, 1)
    y_true = alpha_i + lambda_t + s * beta_i
    eps = rng.standard_normal((n_units, n_periods)) * 1.0
    y = y_true + eps

    z_panel = np.broadcast_to(z, (n_units, n_periods))
    return y, d, z_panel, x, beta_0, beta_1


def test_heterogeneous_duration_iv_recovers_truth():
    y, d, z, x, true_b0, true_b1 = simulate_duration_panel(
        n_units=400, n_periods=14, beta_0=10.0, beta_1=4.0, seed=123
    )

    res = heterogeneous_duration_iv(y, d, z, x=x, model_type="linear", segments=3)

    assert isinstance(res, HeterogeneousDurationIVResult)
    assert res.first_stage_status == "strong"
    assert res.first_stage_f > 20.0

    # Estimated coefficients: [duration, duration_x_covariate_1]
    # Note: covariate is standardized, so population_effect is in-sample mean slope
    in_sample_mean_slope = float(np.mean(true_b0 + true_b1 * x))
    assert abs(res.population_effect - in_sample_mean_slope) < 0.1
    assert res.population_ci[0] <= in_sample_mean_slope <= res.population_ci[1]

    # Check interaction coefficient
    assert abs(res.coefficients[1] - true_b1) < 1.0


def test_heterogeneous_duration_iv_without_covariates():
    y, d, z, x, true_b0, _ = simulate_duration_panel(
        n_units=250, n_periods=10, beta_0=6.0, beta_1=0.0, seed=456
    )

    # Pass None for x
    res = heterogeneous_duration_iv(y, d, z, x=None, model_type="linear")
    assert isinstance(res, HeterogeneousDurationIVResult)
    assert len(res.coefficients) == 1
    assert res.regressor_names == ["duration"]
    assert abs(res.population_effect - true_b0) < 1.0

    # Pass empty array for x
    res_empty = heterogeneous_duration_iv(y, d, z, x=np.zeros((250, 0)), model_type="linear")
    assert len(res_empty.coefficients) == 1
    assert abs(res_empty.population_effect - res.population_effect) < 1e-10


def test_heterogeneous_duration_iv_predict_curve():
    y, d, z, x, true_b0, true_b1 = simulate_duration_panel(
        n_units=200, n_periods=8, beta_0=5.0, beta_1=2.0, seed=789
    )
    res = heterogeneous_duration_iv(y, d, z, x=x)

    x_test = np.array([[-1.0], [0.0], [1.0]])
    durations = np.array([1, 2, 4])
    curves = res.predict_curve(x_test, durations=durations)

    assert curves.shape == (3, 3)
    # Monotonically increasing across durations since slopes > 0
    for i in range(3):
        assert curves[i, 0] < curves[i, 1] < curves[i, 2]
    # Curve at x=1 should have steeper slope than at x=-1
    assert curves[2, 2] > curves[0, 2]


def test_heterogeneous_duration_iv_segment_table():
    y, d, z, x, _, _ = simulate_duration_panel(
        n_units=300, n_periods=10, beta_0=5.0, beta_1=3.0, seed=101
    )
    res = heterogeneous_duration_iv(y, d, z, x=x, segments=4)

    tbl = res.segment_table
    assert len(tbl) == 4
    assert list(tbl.columns) == ["segment", "n_accounts", "mean_duration_slope", "se", "ci_lower", "ci_upper"]
    # Slopes should increase monotonically across segments since beta_1 > 0
    slopes = tbl["mean_duration_slope"].values
    assert slopes[0] < slopes[-1]


def test_heterogeneous_duration_iv_quadratic():
    rng = np.random.default_rng(202)
    n_units, n_periods = 300, 10
    x = rng.standard_normal((n_units, 1))
    z = rng.binomial(1, 0.5, size=(n_units, 1))
    z_panel = np.broadcast_to(z, (n_units, n_periods))

    d = np.zeros((n_units, n_periods))
    for i in range(n_units):
        p = 0.3 if z[i, 0] == 1 else 0.08
        ad = False
        for t in range(n_periods):
            if ad or rng.random() < p:
                ad = True
                d[i, t] = 1.0
    s = np.cumsum(d, axis=1)
    y = 3.0 * s - 0.2 * (s ** 2) + rng.standard_normal((n_units, n_periods))

    res = heterogeneous_duration_iv(y, d, z_panel, x=x, model_type="quadratic")
    assert "duration_sq" in res.regressor_names
    assert res.coefficients[0] > 0
    assert res.coefficients[-1] < 0  # negative quadratic term


def test_heterogeneous_duration_iv_input_validation():
    y = np.ones((50, 10))
    d = np.ones((50, 10))
    z = np.zeros((50, 10))
    z[25:, :] = 1.0
    x_bad = np.ones((40, 2))  # mismatched units

    with pytest.raises(ValueError, match="match the number of accounts"):
        heterogeneous_duration_iv(y, d, z, x=x_bad)
