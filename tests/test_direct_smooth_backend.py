"""Unit and integration tests for Direct Smooth encouragement forests and correlated intercepts."""
import numpy as np
import pytest

import longbet
from longbet import DirectSmoothConfig, LongBetDirectSmooth, LongBetEncourage
from longbet import _direct_smooth as backend


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
    model = LongBetDirectSmooth(DirectSmoothConfig(
        baseline_trees=2, effect_trees=2, correlated_intercepts=True))
    model.fit(y, d, z, x, t, seed=789, chains=2, burnin=20, draws=30)

    save_path = tmp_path / "subdir" / "direct_model.archive"
    model.save(save_path)
    loaded = LongBetDirectSmooth.load(save_path)
    assert loaded.fitted_
    np.testing.assert_allclose(loaded.draws["itt_y"], model.draws["itt_y"])
    np.testing.assert_allclose(loaded.draws["itt_d"], model.draws["itt_d"])
    np.testing.assert_array_equal(loaded.draws["intercept_cov"], model.draws["intercept_cov"])
    np.testing.assert_array_equal(loaded.draws["gamma2"],
                                  np.diagonal(loaded.draws["intercept_cov"], axis1=-2, axis2=-1))
    assert loaded.metadata == model.metadata


def test_longbet_encourage_direct_smooth_engine(tmp_path):
    y, d, z, x, t = make_test_data()
    config = DirectSmoothConfig(baseline_trees=2, effect_trees=2)
    wrapper = LongBetEncourage(first_stage="lpm", engine="direct_smooth", direct_config=config,
                              random_seed=91, num_chains=2, num_burnin=3, num_sweeps=8, n_skip=2)
    wrapper.fit(y, d, z, x, t)
    pred = wrapper.predict()
    assert pred is not None
    assert "itt_y" in pred.draws
    assert pred.draws["itt_y"].shape == (1, 3, 2, 8)
    direct = LongBetDirectSmooth(config).fit(y, d, z, x, t, seed=91,
                                            chains=2, burnin=3, draws=16)
    np.testing.assert_array_equal(wrapper._direct_model.draws["itt_y"], direct.draws["itt_y"][:, 1::2])
    path = tmp_path / "wrapped.npz"
    wrapper.save(path)
    loaded = LongBetEncourage.load(path)
    assert loaded.engine == "direct_smooth"
    assert loaded.config == wrapper.config
    assert loaded.metadata == wrapper.metadata
    np.testing.assert_array_equal(loaded.predict().draws["itt_y"], pred.draws["itt_y"])
    # Full short-trace diagnostics survive archive replay, including failed checks.
    import pandas as pd
    pd.testing.assert_frame_equal(loaded.predict().stability(), pred.stability())


def test_average_contrast_is_not_average_observed_treatment_contribution():
    config = DirectSmoothConfig(baseline_trees=1, effect_trees=1)
    space = backend.make_tree_space(np.ones((6, 1)), config)
    assignment = np.array([0, 0, 0, 1, 1, 1])
    design = backend.ForestDesign.build(space, assignment - .5, np.eye(2))
    base = backend.ForestDesign.build(space, np.ones(6), np.eye(3))
    state = backend.init_equation_state(np.random.default_rng(7), base, design, config,
                                        np.zeros((6, 3)), 1)
    state.effect_leaves[0] = [[2., 3.], [9., 11.]]
    state.effect_fits[0] = (space.mask[0].T @ state.effect_leaves[0]) * design.weight[:, None]
    np.testing.assert_allclose(state.effect_fits.mean(axis=(0, 1)), [0., 0.])
    np.testing.assert_allclose(backend._effect_contrast(state, design), [2., 3.])


@pytest.mark.parametrize("change", [dict(seed=-1), dict(seed=True), dict(chains=0),
                                    dict(draws=0), dict(burnin=-1), dict(n_skip=1.5)])
def test_invalid_sampler_controls_fail_before_sampling(change):
    with pytest.raises(ValueError):
        LongBetDirectSmooth().fit(*make_test_data(), **change)


@pytest.mark.parametrize("field", ["y", "x"])
def test_nonfinite_and_mismatched_inputs_rejected(field):
    inputs = dict(zip(("y", "d", "z", "x", "t"), make_test_data()))
    inputs[field][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        LongBetDirectSmooth().fit(**inputs, chains=1, burnin=0, draws=1)
    inputs[field] = inputs[field][:-1]
    with pytest.raises(ValueError, match="matrix"):
        LongBetDirectSmooth().fit(**inputs, chains=1, burnin=0, draws=1)


@pytest.mark.parametrize("options, match", [
    (dict(engine="typo"), "engine"),
    (dict(engine="longbet", direct_config={}), "direct_config"),
    (dict(engine="direct_smooth", first_stage="probit"), "requires"),
    (dict(engine="direct_smooth", outcome="binary"), "requires"),
    (dict(engine="direct_smooth", num_trees_pr=2), "Unsupported"),
    (dict(engine="direct_smooth", config=longbet.LongBetConfig(sur=False)), "Unsupported"),
])
def test_wrapper_rejects_unknown_or_unsupported_options(options, match):
    with pytest.raises(ValueError, match=match):
        LongBetEncourage(**{**dict(first_stage="lpm"), **options})


def test_wrapper_rejects_unsupported_fit_and_predict_options():
    wrapper = LongBetEncourage(first_stage="lpm", engine="direct_smooth",
                              direct_config=dict(baseline_trees=1, effect_trees=1),
                              num_chains=1, num_burnin=0, num_sweeps=4, n_skip=1)
    inputs = dict(zip(("y", "d", "z", "x", "t"), make_test_data()))
    for option in (dict(x_trt=inputs["x"]), dict(key=np.array([1, 2]))):
        with pytest.raises(ValueError):
            wrapper.fit(**inputs, **option)
    wrapper.fit(**inputs)
    for option in (dict(groups=np.repeat("a", 60)), dict(summary_only=False),
                   dict(block_size=100), dict(standardization="population")):
        with pytest.raises(ValueError):
            wrapper.predict(**option)


@pytest.mark.parametrize("joint_effects", [False, True])
def test_joint_block_matches_dense_gaussian_with_conditional_intercept_prior(joint_effects):
    """Independent dense posterior check, including the correlated-prior mean."""
    rng = np.random.default_rng(8702)
    config = DirectSmoothConfig(baseline_trees=1, effect_trees=1)
    space = backend.make_tree_space(np.ones((4, 1)), config)
    baseline = backend.ForestDesign.build(space, np.ones(4), np.array([[1., .3], [.3, 1.]]))
    effect = backend.ForestDesign.build(space, np.array([-.5, -.5, .5, .5]), np.eye(1))
    y = rng.normal(size=(4, 2))
    state = backend.init_equation_state(rng, baseline, effect, config, y, 1)
    state.sigma2, state.gamma2 = .8, .7
    prior_mean = np.array([1.1, -.9, .6, -1.7])
    factor = baseline.vectors * np.sqrt(baseline.eigenvalues)
    matrix = np.einsum("jln,tq->ntjlq", space.mask[state.baseline_rules], factor).reshape(8, -1)
    if joint_effects:
        effect_matrix = np.zeros((4, 2, 2))
        effect_matrix[:, 1:, :] = (space.mask[state.effect_rules].transpose(2, 0, 1)
                                   * effect.weight[:, None, None])
        matrix = np.column_stack([matrix, effect_matrix.reshape(8, -1)])
    q = matrix.shape[1]
    design = np.column_stack([matrix, np.repeat(np.eye(4), 2, axis=0)])
    prior_precision = np.diag(np.r_[np.ones(q), np.full(4, 1 / state.gamma2)])
    target_variance = np.linalg.inv(prior_precision + design.T @ design / state.sigma2)
    response = y.copy()
    if not joint_effects:
        response[:, 1:] -= state.effect_fits.sum(axis=0)
    target_mean = target_variance @ (design.T @ response.ravel() / state.sigma2
                                     + np.r_[np.zeros(q), prior_mean / state.gamma2])
    samples = []
    for _ in range(5000):
        backend.joint_baseline_intercept_draw(rng, state, baseline, y, 1,
            effect=effect if joint_effects else None, gamma_prior_mean=prior_mean)
        whitened = np.linalg.solve(factor, state.baseline_leaves.reshape(-1, 2).T).T.ravel()
        if joint_effects:
            whitened = np.r_[whitened, state.effect_leaves.ravel()]
        samples.append(np.r_[whitened, state.gamma])
    samples = np.asarray(samples)
    mean_z = (samples.mean(axis=0) - target_mean) / np.sqrt(np.diag(target_variance) / len(samples))
    covariance_mcse = np.sqrt((np.outer(np.diag(target_variance), np.diag(target_variance))
                               + target_variance**2) / (len(samples) - 1))
    covariance_z = (np.cov(samples, rowvar=False) - target_variance) / covariance_mcse
    assert np.max(np.abs(mean_z)) < 5.5
    assert np.max(np.abs(covariance_z)) < 5.5
    np.testing.assert_allclose(state.residual, y - backend._fitted_equation(state, 1), atol=1e-13)


def test_correlated_equation_updates_preserve_dense_joint_posterior():
    """Start independent exact posterior draws, then apply both Gibbs blocks.

    This checks the cross-equation kernel against a separately constructed
    Gaussian posterior. Replacing the conditional prior with independent
    intercept updates changes these moments, even if a joint intercept draw
    follows later in a sweep.
    """
    from scipy.linalg import block_diag

    rng = np.random.default_rng(14920)
    n, periods = 3, 2
    config = DirectSmoothConfig(baseline_trees=1, effect_trees=1, correlated_intercepts=True)
    space = backend.make_tree_space(np.ones((n, 1)), config)
    baseline = backend.ForestDesign.build(space, np.ones(n), np.array([[1., .3], [.3, 1.]]))
    effect = backend.ForestDesign.build(space, np.array([-1 / 3, -1 / 3, 2 / 3]), np.eye(1))
    y = rng.normal(size=(n, periods, 2))
    states = [backend.init_equation_state(rng, baseline, effect, config, y[..., m], 1)
              for m in range(2)]
    covariance = np.array([[1.3, .9], [.9, 1.1]])
    factor = baseline.vectors * np.sqrt(baseline.eigenvalues)
    design_base = np.einsum("jln,tq->ntjlq", space.mask[states[0].baseline_rules], factor).reshape(6, -1)
    design_effect = np.zeros((n, periods, 2))
    design_effect[:, 1:, :] = (space.mask[states[0].effect_rules].transpose(2, 0, 1)
                               * effect.weight[:, None, None])
    design = np.column_stack([design_base, design_effect.reshape(6, -1),
                              np.repeat(np.eye(n), periods, axis=0)])
    p = design.shape[1]
    prior_covariance = np.eye(2 * p)
    for m in range(2):
        for k in range(2):
            prior_covariance[m * p + 6:(m + 1) * p, k * p + 6:(k + 1) * p] = covariance[m, k] * np.eye(n)
    full_design = block_diag(design, design)
    sigmas = [.7, .9]
    observation_precision = np.diag(np.repeat(1 / np.array(sigmas), n * periods))
    target_variance = np.linalg.inv(np.linalg.inv(prior_covariance)
                                    + full_design.T @ observation_precision @ full_design)
    target_mean = target_variance @ full_design.T @ observation_precision @ y.transpose(2, 0, 1).ravel()
    initial = rng.multivariate_normal(target_mean, target_variance, size=5000)
    samples = []
    for draw in initial:
        for m, state in enumerate(states):
            params = draw[m * p:(m + 1) * p]
            state.baseline_leaves = params[:4].reshape(1, 2, 2) @ factor.T
            state.baseline_fits = np.einsum("jln,jlt->jnt", space.mask[state.baseline_rules], state.baseline_leaves)
            state.effect_leaves = params[4:6].reshape(1, 2, 1)
            state.effect_fits = (np.einsum("jln,jlt->jnt", space.mask[state.effect_rules], state.effect_leaves)
                                  * effect.weight[None, :, None])
            state.gamma = params[6:].copy()
            state.sigma2 = sigmas[m]
            state.residual = y[..., m] - backend._fitted_equation(state, 1)
        for m, state in enumerate(states):
            mean, state.gamma2 = backend._conditional_intercept_prior(states, covariance, m)
            backend.joint_baseline_intercept_draw(rng, state, baseline, y[..., m], 1,
                                                   effect=effect, gamma_prior_mean=mean)
        result = []
        for state in states:
            whitened = np.linalg.solve(factor, state.baseline_leaves.reshape(-1, 2).T).T.ravel()
            result.extend(np.r_[whitened, state.effect_leaves.ravel(), state.gamma])
        samples.append(result)
    samples = np.asarray(samples)
    mean_z = (samples.mean(axis=0) - target_mean) / np.sqrt(np.diag(target_variance) / len(samples))
    covariance_mcse = np.sqrt((np.outer(np.diag(target_variance), np.diag(target_variance))
                               + target_variance**2) / (len(samples) - 1))
    covariance_z = (np.cov(samples, rowvar=False) - target_variance) / covariance_mcse
    assert np.max(np.abs(mean_z)) < 5.5
    assert np.max(np.abs(covariance_z)) < 5.5
