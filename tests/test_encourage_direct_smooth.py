"""Independent posterior-kernel checks for the benchmark direct smooth forest."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


path = Path(__file__).parents[1] / "benchmarks/encouragement/direct_smooth_candidate.py"
spec = importlib.util.spec_from_file_location("direct_smooth_candidate", path)
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)


def setup_design():
    config = candidate.ForestConfig(baseline_trees=2, effect_trees=2, cutpoints=2)
    x = np.arange(6)[:, None]
    baseline, effect = candidate.make_designs(x, np.arange(6) % 2, np.arange(3), 1, config)
    return config, baseline, effect


def test_leaf_conditional_matches_dense_gaussian_and_sample_moments():
    rng = np.random.default_rng(912)
    covariance = np.array([[1., .3], [.3, .8]])
    total, count, sigma2 = np.array([2., -1.]), 3.2, .7
    mean, variance = candidate.leaf_posterior(total, count, sigma2, covariance)
    target_var = np.linalg.inv(np.linalg.inv(covariance) + count / sigma2 * np.eye(2))
    target_mean = target_var @ total / sigma2
    np.testing.assert_allclose(mean, target_mean)
    np.testing.assert_allclose(variance, target_var)
    # One root tree gives the exact same conditional; sample_tree must draw with
    # covariance V, not V**2 or the Cholesky factor in the wrong orientation.
    space = candidate.make_tree_space(np.ones((4, 1)), candidate.ForestConfig())
    design = candidate.ForestDesign.build(space, np.ones(4), covariance)
    residual = rng.normal(size=(4, 2))
    exact_mean, exact_var = candidate.leaf_posterior(residual.sum(axis=0), 4., sigma2, covariance)
    values = np.array([candidate.sample_tree(rng, design, residual, sigma2)[1][0]
                       for _ in range(16000)])
    np.testing.assert_allclose(values.mean(axis=0), exact_mean,
                               atol=5 * np.sqrt(np.max(np.diag(exact_var)) / len(values)))
    np.testing.assert_allclose(np.cov(values, rowvar=False), exact_var, rtol=.04, atol=.006)


def test_enumerated_topology_matches_direct_dense_marginal_likelihood():
    config, _, effect = setup_design()
    rng = np.random.default_rng(803)
    residual = rng.normal(size=(6, 2))
    sigma2 = .6
    score, _, _ = effect.collapsed(residual, sigma2)
    probabilities = np.exp(score - logsumexp(score))
    covariance = (effect.vectors * effect.eigenvalues) @ effect.vectors.T
    exact = []
    for rule in range(len(score)):
        design = np.concatenate([np.kron(effect.weighted_masks[rule, leaf, :, None], np.eye(2))
                                 for leaf in range(2)], axis=1)
        marginal = sigma2 * np.eye(residual.size) + design @ np.kron(np.eye(2), covariance) @ design.T
        exact.append(effect.space.log_prior[rule]
                     + multivariate_normal.logpdf(residual.ravel(), cov=marginal))
    exact = np.asarray(exact)
    np.testing.assert_allclose(probabilities, np.exp(exact - logsumexp(exact)), atol=1e-12)
    sampled = np.bincount([candidate.sample_tree(rng, effect, residual, sigma2)[0]
                          for _ in range(12000)], minlength=len(score)) / 12000
    assert np.all(np.abs(sampled - probabilities)
                  < 5 * np.sqrt(probabilities * (1 - probabilities) / 12000))


@pytest.mark.parametrize("joint, effects", [(False, False), (True, False), (True, True)])
def test_full_residual_preserved_and_stumps_actually_change(joint, effects):
    config, baseline, effect = setup_design()
    config = candidate.ForestConfig(baseline_trees=2, effect_trees=2, cutpoints=2,
                                    joint_baseline_intercept=joint, joint_effect_leaves=effects)
    rng = np.random.default_rng(731)
    y = rng.normal(size=(6, 3))
    state = candidate.prior_state(rng, baseline, effect, config, y, 1)
    seen = set()
    for _ in range(40):
        candidate.sweep(rng, state, baseline, effect, config, y, 1)
        np.testing.assert_allclose(state.residual, y - candidate.fitted(state, 1), atol=2e-14)
        seen.update(state.effect_rules)
    assert len(seen) > 1


def small_data():
    rng = np.random.default_rng(903)
    assignment = np.arange(10) % 2
    z = assignment[:, None] * np.array([0, 1, 1])[None]
    return dict(y=rng.normal(size=(10, 3)) + z, d=z, z=z,
                assignment=assignment, x=rng.normal(size=(10, 2)))


@pytest.mark.parametrize("joint, effects", [(False, False), (True, False), (True, True)])
def test_all_unit_effect_archive_and_exact_resume(tmp_path, joint, effects):
    data = small_data()
    config = candidate.ForestConfig(baseline_trees=2, effect_trees=2, cutpoints=1,
                                    joint_baseline_intercept=joint, joint_effect_leaves=effects)
    values, full = candidate.fit_candidate(data, 1, seed=432, chains=2, burnin=3, draws=8, config=config)
    _, part = candidate.fit_candidate(data, 1, seed=432, chains=2, burnin=3, draws=4, config=config)
    path = tmp_path / "candidate.npz"
    np.savez_compressed(path, **part)
    with np.load(path, allow_pickle=False) as loaded:
        _, resumed = candidate.fit_candidate(data, 1, seed=999, chains=2, burnin=0,
                                              draws=4, config=config, resume=loaded)
    for name in ("baseline_rules", "baseline_leaves", "effect_rules", "effect_leaves",
                 "gamma", "gamma2", "sigma2", "itt"):
        np.testing.assert_allclose(resumed[name], full[name][:, 4:], atol=2e-14)
    if not joint:
        legacy = dict(part)
        settings = json.loads(str(legacy["config_json"]))
        del settings["joint_baseline_intercept"]
        del settings["joint_effect_leaves"]
        legacy["config_json"] = np.asarray(json.dumps(settings, sort_keys=True))
        _, replay = candidate.fit_candidate(data, 1, seed=999, chains=2, burnin=0,
                                             draws=4, config=config, resume=legacy)
        np.testing.assert_allclose(replay["itt"], resumed["itt"], atol=2e-14)
    assert values["outcome"].shape == (1, 2, 2, 8)
    # Independently replay the all-unit target from persisted leaves/topologies.
    _, effect = candidate.make_designs(data["x"], data["assignment"], np.arange(3), 1, config)
    rules, leaves = full["effect_rules"][0, 0, 0], full["effect_leaves"][0, 0, 0]
    tau = sum(effect.space.mask[rule].T @ leaf for rule, leaf in zip(rules, leaves))
    np.testing.assert_allclose(tau.mean(axis=0) * full["scale"][0], values["outcome"][0, :, 0, 0])
    with pytest.raises(ValueError, match="identical data"):
        candidate.fit_candidate(data, 1, seed=432, burnin=1, config=config, resume=part)


@pytest.mark.slow
@pytest.mark.parametrize("joint, effects", [(False, False), (True, False), (True, True)])
def test_geweke_preserves_joint_prior_moments(joint, effects):
    result = candidate.geweke(replications=3000, joint_baseline_intercept=joint,
                               joint_effect_leaves=effects)
    assert result["passed"], dict(zip(result["functions"], result["z"]))


@pytest.mark.slow
def test_geweke_detects_broken_leaf_conditional(monkeypatch):
    original = candidate.sample_tree

    def broken(*args):
        rule, leaves, fit = original(*args)
        return rule, np.zeros_like(leaves), np.zeros_like(fit)

    monkeypatch.setattr(candidate, "sample_tree", broken)
    result = candidate.geweke(replications=300)
    assert not result["passed"]
    assert np.isneginf(result["z"][1])
    assert np.isneginf(result["z"][3])


@pytest.mark.parametrize("joint_effects", [False, True])
def test_joint_baseline_gamma_block_matches_dense_conditional(joint_effects):
    config, baseline, effect = setup_design()
    rng = np.random.default_rng(7803)
    y = rng.normal(size=(6, 3))
    state = candidate.prior_state(rng, baseline, effect, config, y, 1)
    state.sigma2, state.gamma2 = .8, 1.3
    state.baseline_rules[:] = [0, 1]  # Includes an inactive augmented root leaf.
    masks = baseline.space.mask[state.baseline_rules]
    factor = baseline.vectors * np.sqrt(baseline.eigenvalues)
    x = np.einsum("jln,tq->ntjlq", masks, factor).reshape(18, -1)
    if joint_effects:
        state.effect_rules[:] = [0, 1]
        effect_masks = effect.space.mask[state.effect_rules]
        effect_factor = effect.vectors * np.sqrt(effect.eigenvalues)
        effect_matrix = np.zeros((6, 3, 8))
        effect_matrix[:, 1:] = (np.einsum("jln,tq->ntjlq", effect_masks, effect_factor).reshape(6, 2, -1)
                                * effect.weight[:, None, None])
        x = np.column_stack([x, effect_matrix.reshape(18, -1)])
    q = x.shape[-1]
    design = np.column_stack([x, np.repeat(np.eye(6), 3, axis=0)])
    prior_precision = np.diag(np.r_[np.ones(q), np.full(6, 1 / state.gamma2)])
    target_variance = np.linalg.inv(prior_precision + design.T @ design / state.sigma2)
    response = y.copy()
    if not joint_effects:
        response[:, 1:] -= state.effect_fits.sum(axis=0)
    target_mean = target_variance @ design.T @ response.ravel() / state.sigma2
    samples = []
    for _ in range(10000):
        candidate.joint_baseline_intercept_draw(rng, state, baseline, y, 1,
                                                 effect=effect if joint_effects else None)
        whitened = np.linalg.solve(factor, state.baseline_leaves.reshape(-1, 3).T).T.ravel()
        if joint_effects:
            effect_whitened = np.linalg.solve(effect_factor, state.effect_leaves.reshape(-1, 2).T).T.ravel()
            whitened = np.r_[whitened, effect_whitened]
        samples.append(np.r_[whitened, state.gamma])
    samples = np.asarray(samples)
    mean_z = (samples.mean(axis=0) - target_mean) / np.sqrt(np.diag(target_variance) / len(samples))
    covariance_mcse = np.sqrt((np.outer(np.diag(target_variance), np.diag(target_variance))
                               + target_variance**2) / (len(samples) - 1))
    covariance_z = (np.cov(samples, rowvar=False) - target_variance) / covariance_mcse
    assert np.max(np.abs(mean_z)) < 5.5
    assert np.max(np.abs(covariance_z)) < 5.5
    np.testing.assert_allclose(state.residual, y - candidate.fitted(state, 1), atol=2e-14)


@pytest.mark.slow
@pytest.mark.parametrize("joint_effects", [False, True])
def test_geweke_detects_broken_joint_block(monkeypatch, joint_effects):
    def broken(rng, state, baseline, y, start, **kwargs):
        state.baseline_leaves[:] = 0
        state.baseline_fits[:] = 0
        state.gamma[:] = 0
        state.residual = y - candidate.fitted(state, start)

    monkeypatch.setattr(candidate, "joint_baseline_intercept_draw", broken)
    result = candidate.geweke(replications=300, joint_baseline_intercept=True,
                               joint_effect_leaves=joint_effects)
    assert not result["passed"]
    assert np.isneginf(result["z"][1])
    assert np.isneginf(result["z"][4])


@pytest.mark.parametrize("change", [dict(assignment=np.ones(10)), dict(x=np.ones((10, 3, 2))),
                                   dict(d=np.tile([0, 1, 0], (10, 1))), dict(t=[0, 1, 1])])
def test_rejects_unsupported_data(change):
    data = {**small_data(), **change}
    with pytest.raises(ValueError):
        candidate.fit_candidate(data, 1, seed=1, burnin=0, draws=1)
