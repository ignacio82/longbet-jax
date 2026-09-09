"""Unit and integration tests for Direct Smooth encouragement forests and correlated intercepts."""
import numpy as np
import pytest

import longbet
from longbet import DirectSmoothConfig, LongBetDirectSmooth, LongBetEncourage


def make_test_data(n=60, t_len=5, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    z = np.zeros((n, t_len))
    # Encouragement starts at period index 2
    z[:n // 2, 2:] = 1
    # Compliance: uptake depends on encouragement and baseline x
    latent_u = rng.normal(size=n)  # common unobserved confounder
    d = np.zeros((n, t_len))
    for t in range(t_len):
        prob = 1 / (1 + np.exp(-(0.5 * latent_u + 1.5 * z[:, t] + 0.3 * x[:, 0] - 1.0)))
        d[:, t] = rng.binomial(1, prob)
    # Enforce absorbing
    for t in range(1, t_len):
        d[:, t] = np.maximum(d[:, t], d[:, t - 1])
    # Outcome: depends on uptake, latent u, and x
    y = np.zeros((n, t_len))
    for t in range(t_len):
        y[:, t] = 2.0 * d[:, t] + 1.2 * latent_u + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=n)
    t = np.arange(1.0, t_len + 1.0)
    return y, d, z, x, t


def test_direct_smooth_exports():
    assert "DirectSmoothConfig" in longbet.__all__
    assert "LongBetDirectSmooth" in longbet.__all__


def test_direct_smooth_fit_and_predict():
    y, d, z, x, t = make_test_data()
    config = DirectSmoothConfig(
        baseline_trees=3, effect_trees=3,
        joint_baseline_intercept=True, joint_effect_leaves=True,
    )
    model = LongBetDirectSmooth(config)
    model.fit(y, d, z, x, t, seed=123, chains=2, burnin=40, draws=60)
    assert model.fitted_
    assert "itt_y" in model.draws
    assert "itt_d" in model.draws
    assert model.draws["itt_y"].shape == (2, 60, 3)  # chains, draws, horizons (5 - 2 = 3)

    pred = model.predict()
    assert set(pred.draws.keys()) == {"itt_y", "itt_d", "wald"}
    assert len(pred.horizons) == 3

    effects = pred.effects(min_ess=10, max_rhat=1.5)
    assert len(effects) == 3 * 3  # 3 quantities * 3 horizons
    assert set(effects.quantity) == {"itt_y", "itt_d", "wald"}

    first_stage = pred.first_stage(min_ess=10, max_rhat=1.5)
    assert len(first_stage) == 3
    assert (first_stage.posterior_probability_positive > 0.5).all()

    wald = pred.wald(require_convergence=False, min_ess=10, max_rhat=1.5)
    assert len(wald) == 3


def test_direct_smooth_correlated_intercepts():
    y, d, z, x, t = make_test_data(seed=99)
    config = DirectSmoothConfig(
        baseline_trees=3, effect_trees=3,
        correlated_intercepts=True, intercept_df=6.0, intercept_scale=2.0,
    )
    model = LongBetDirectSmooth(config)
    model.fit(y, d, z, x, t, seed=456, chains=2, burnin=40, draws=60)
    assert "intercept_cov" in model.draws
    cov = model.draws["intercept_cov"]
    assert cov.shape == (2, 60, 2, 2)
    # Check positive definiteness of posterior covariance draws
    for c in range(2):
        for d_idx in range(60):
            mat = cov[c, d_idx]
            assert np.linalg.det(mat) > 0
            assert mat[0, 0] > 0 and mat[1, 1] > 0


def test_direct_smooth_save_and_load(tmp_path):
    y, d, z, x, t = make_test_data()
    model = LongBetDirectSmooth(DirectSmoothConfig(baseline_trees=2, effect_trees=2))
    model.fit(y, d, z, x, t, seed=789, chains=2, burnin=20, draws=30)

    save_path = tmp_path / "direct_model.npz"
    model.save(save_path)
    loaded = LongBetDirectSmooth.load(save_path)
    assert loaded.fitted_
    np.testing.assert_allclose(loaded.draws["itt_y"], model.draws["itt_y"])
    np.testing.assert_allclose(loaded.draws["itt_d"], model.draws["itt_d"])


def test_longbet_encourage_direct_smooth_engine():
    y, d, z, x, t = make_test_data()
    config = DirectSmoothConfig(baseline_trees=2, effect_trees=2)
    wrapper = LongBetEncourage(first_stage="lpm", engine="direct_smooth", direct_config=config)
    wrapper.fit(y, d, z, x, t)
    pred = wrapper.predict()
    assert pred is not None
    assert "itt_y" in pred.draws
