"""Unit tests for Phase 1 nuisance upgrades: Matérn kernel, linear ANCOVA backbone, and hazard adoption."""
import numpy as np
import pytest

from longbet._direct_smooth import time_covariance
from longbet._iv_nuisance import (
    LongBetIVNuisance,
    LongBetIVNuisanceConfig,
)


def make_test_data(seed=42, n=40, h=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    z = np.tile([0, 1], n // 2)
    # Absorbing adoption
    adopt_time = rng.integers(1, h + 3, size=n)
    d = (np.arange(1, h + 1)[None, :] >= adopt_time[:, None]).astype(float)
    # Linear outcome plus noise
    y = 2.0 + 1.5 * x[:, :1] + 3.0 * d + rng.normal(scale=0.2, size=(n, h))
    times = np.arange(1, h + 1, dtype=float)
    return x, z, np.stack([y, d], axis=-1), times


def test_matern32_time_covariance_is_positive_definite():
    times = np.linspace(1, 10, 10)
    cov = time_covariance(times, sd=1.0, length_scale=2.0, nugget=0.05, kernel="matern32")
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0)
    np.testing.assert_allclose(np.diag(cov), 1.0)


def test_matern12_time_covariance_is_positive_definite():
    times = np.linspace(1, 10, 10)
    cov = time_covariance(times, sd=1.0, length_scale=2.0, nugget=0.05, kernel="matern12")
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0)


def test_invalid_kernel_raises():
    times = np.arange(5)
    with pytest.raises(ValueError, match="Unknown kernel"):
        time_covariance(times, sd=1.0, length_scale=1.0, nugget=0.1, kernel="invalid_kernel")


def test_hazard_adoption_is_strictly_monotonic_and_bounded():
    x, z, responses, times = make_test_data()
    cfg = LongBetIVNuisanceConfig(
        baseline_trees=2, effect_trees=2, burnin=10, draws=20, chains=1,
        adoption_model="hazard", linear_ancova_backbone=True, kernel="matern32"
    )
    model = LongBetIVNuisance(cfg).fit(x, z, responses, times)
    pred = model.predict(x)
    adopt_pred = pred[..., 1]
    assert np.all(adopt_pred >= 0.0)
    assert np.all(adopt_pred <= 1.0)
    diffs = np.diff(adopt_pred, axis=1)
    assert np.all(diffs >= -1e-12)


def test_linear_ancova_backbone_fits_pure_linear_dgp():
    rng = np.random.default_rng(99)
    n, h = 60, 3
    x = rng.normal(size=(n, 2))
    z = np.tile([0, 1], n // 2)
    y = np.repeat(3.0 + 2.0 * x[:, :1] + 1.0 * z[:, None], h, axis=1)
    d = np.zeros((n, h))
    responses = np.stack([y, d], axis=-1)
    times = np.arange(1, h + 1, dtype=float)

    cfg = LongBetIVNuisanceConfig(
        baseline_trees=2, effect_trees=2, burnin=10, draws=20, chains=1,
        linear_ancova_backbone=True, ridge_penalty=1e-6
    )
    model = LongBetIVNuisance(cfg).fit(x, z, responses, times)
    pred = model.predict(x)
    truth_arm1 = 4.0 + 2.0 * x[:, :1]
    np.testing.assert_allclose(pred[:, :, 1, 0], np.tile(truth_arm1, (1, h)), atol=0.05)


@pytest.mark.parametrize("bad_cfg", [
    {"kernel": "unknown"},
    {"adoption_model": "invalid"},
    {"linear_ancova_backbone": "yes"},
    {"ridge_penalty": -1.0},
    {"ridge_penalty": 0.0},
])
def test_config_validates_new_options(bad_cfg):
    with pytest.raises(ValueError):
        LongBetIVNuisanceConfig(**bad_cfg)
