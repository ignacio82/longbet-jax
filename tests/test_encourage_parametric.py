"""Independent algebra and stationary-prior checks for the research control."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "encouragement"))
import parametric_candidate as candidate
from dgp import make_population


def dense_conditional(design, y, sigma2, gamma2):
    n, t, q = design.matrix.shape
    matrix = np.column_stack([design.matrix.reshape(n * t, q), np.repeat(np.eye(n), t, axis=0)])
    precision = matrix.T @ matrix / sigma2 + np.diag(np.r_[np.ones(q), np.full(n, 1 / gamma2)])
    covariance = np.linalg.solve(precision, np.eye(q + n))
    return covariance @ (matrix.T @ y.ravel() / sigma2), covariance


def test_marginal_coefficients_match_dense_joint_gaussian():
    rng = np.random.default_rng(871)
    design = candidate.GaussianDesign.build(rng.normal(size=(4, 3, 5)))
    y = rng.normal(size=(4, 3))
    mean, chol = candidate.coefficient_posterior(design, design.response_statistics(y), .3, 4.)
    expected_mean, expected_covariance = dense_conditional(design, y, .3, 4.)
    np.testing.assert_allclose(mean, expected_mean[:5], atol=1e-13)
    np.testing.assert_allclose(cho_solve((chol, True), np.eye(5)), expected_covariance[:5, :5], atol=1e-13)


def test_joint_conditional_samples_include_coefficient_intercept_covariance():
    rng = np.random.default_rng(91)
    design = candidate.GaussianDesign.build(rng.normal(size=(3, 2, 3)))
    y = rng.normal(size=(3, 2))
    statistics = design.response_statistics(y)
    mean, covariance = dense_conditional(design, y, .7, 1.4)
    samples = np.array([np.r_[*candidate.joint_gaussian_draw(rng, design, y, statistics, .7, 1.4)[:2]]
                        for _ in range(8000)])
    mean_se = np.sqrt(np.diag(covariance) / len(samples))
    assert np.max(np.abs(samples.mean(axis=0) - mean) / mean_se) < 4.5
    covariance_se = np.sqrt((covariance**2 + np.outer(np.diag(covariance), np.diag(covariance)))
                            / (len(samples) - 1))
    assert np.max(np.abs(np.cov(samples, rowvar=False) - covariance) / covariance_se) < 4.5
    # The sampled block must preserve nonzero coefficient/intercept dependence.
    assert np.max(np.abs(covariance[:3, 3:])) > .05


def test_marginalization_stable_at_large_unit_variance():
    rng = np.random.default_rng(431)
    matrix = np.repeat(rng.normal(size=(4, 1, 3)), 3, axis=1)
    design = candidate.GaussianDesign.build(matrix)
    mean, chol = candidate.coefficient_posterior(design, design.response_statistics(np.ones((4, 3))), 1e-8, 1e12)
    np.testing.assert_allclose(mean, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(chol, np.eye(3), atol=1e-10)


def test_sweep_residual_and_proper_variances():
    rng = np.random.default_rng(754)
    design = candidate.GaussianDesign.build(rng.normal(size=(5, 4, 3)))
    y = rng.normal(size=(5, 4))
    state = candidate.prior_state(rng, design, y)
    for _ in range(15):
        np.testing.assert_allclose(state.residual,
                                   y - np.einsum("itq,q->it", design.matrix, state.coefficients)
                                   - state.gamma[:, None], atol=1e-13)
        assert state.sigma2 > 0 and state.gamma2 > 0
        state = candidate.sweep(rng, state, design, y)


def test_geweke_prior_preservation():
    result = candidate.geweke(replications=3000, seed=11347)
    assert result["pass"], result


def test_geweke_detects_broken_conditional():
    def broken(rng, state, design, y, statistics, config):
        return candidate.GaussianState(np.zeros_like(state.coefficients), np.zeros_like(state.gamma), 1., 1., y.copy())

    result = candidate.geweke(replications=40, seed=2, sweep_function=broken)
    assert not result["pass"]
    assert np.isinf(result["z"][[1, 3, 6, 7]]).all()


def test_prior_average_function_variance_matches_forest_scale():
    rng = np.random.default_rng(13)
    x = np.column_stack([rng.normal(size=10), np.ones(10)])
    design, details = candidate.make_design(x, np.arange(10) < 5, np.arange(4), 1)
    np.testing.assert_allclose(np.mean(np.sum(details["features"]**2, axis=1)), 1., atol=1e-14)
    p, t = details["features"].shape[1], 4
    # Baseline coefficient priors are identity in whitened coordinates.
    np.testing.assert_allclose(np.mean(np.sum(design.matrix[:, :, :p * t]**2, axis=2), axis=0), np.ones(t), atol=1e-14)
    np.testing.assert_allclose(np.diag(details["effect_cholesky"] @ details["effect_cholesky"].T), np.ones(3))


def test_fit_targets_all_units_subgroups_and_archive_replay():
    population = make_population(20, "strong", seed=710)
    data = population.assign(711)
    groups = {"positive": data["x"][:, 0] > 0}
    values, archive = candidate.fit_candidate(data, population.start, seed=712, chains=4, burnin=15, draws=20, groups=groups)
    assert values["outcome"].shape == (2, 4, 4, 20)
    assert np.isfinite(values["takeup"]).all()
    assert archive["baseline_probe"].shape == (4, 20, 3, 6, 2)
    assert archive["effect_probe"].shape == (4, 20, 3, 4, 2)
    assert len(np.unique(archive["initial_variances"][:, 0, 0])) == 4
    p, t, h = archive["features"].shape[1], len(data["t"]), len(data["t"]) - population.start
    tau = archive["coefficient_draws"][..., p * t:, :].reshape(4, 20, p, h, 2)
    tau = np.einsum("ckpqm,hq->ckphm", tau, archive["effect_cholesky"])
    for g, mask in enumerate(archive["group_membership"]):
        replay = np.einsum("p,ckphm->hckm", archive["features"][mask].mean(axis=0), tau) * archive["response_scale"]
        np.testing.assert_allclose(values["outcome"][g], replay[..., 0])
        np.testing.assert_allclose(values["takeup"][g], replay[..., 1])


def test_resume_is_identical_to_uninterrupted_sampling_and_pickle_free(tmp_path):
    population = make_population(12, "strong", seed=11)
    data = population.assign(12)
    full_values, full = candidate.fit_candidate(data, population.start, seed=13, chains=2, burnin=10, draws=30)
    first_values, first = candidate.fit_candidate(data, population.start, seed=13, chains=2, burnin=10, draws=12)
    path = tmp_path / "candidate.npz"
    np.savez_compressed(path, **first)
    with np.load(path, allow_pickle=False) as saved:
        next_values, next_archive = candidate.fit_candidate(data, population.start, seed=999, chains=2,
                                                            burnin=0, draws=18, resume=saved)
    for name in full_values:
        np.testing.assert_array_equal(full_values[name], np.concatenate([first_values[name], next_values[name]], axis=-1))
    for name in ("coefficient_draws", "unit_intercept_draws", "innovation_variance_draws", "unit_variance_draws"):
        np.testing.assert_array_equal(full[name], np.concatenate([first[name], next_archive[name]], axis=1))
    np.testing.assert_array_equal(full["rng_state_json"], next_archive["rng_state_json"])
    with pytest.raises(ValueError, match="burnin=0"):
        candidate.fit_candidate(data, population.start, seed=1, chains=2, burnin=1, resume=first)
    changed = dict(data, y=data["y"] + .1)
    with pytest.raises(ValueError, match="does not match"):
        candidate.fit_candidate(changed, population.start, seed=1, chains=2, burnin=0, resume=first)


@pytest.mark.parametrize("option", ["length_scale", "nugget", "baseline_sd", "effect_sd", "sigma_shape", "sigma_scale", "gamma_shape", "gamma_scale"])
def test_invalid_prior_rejected(option):
    with pytest.raises(ValueError, match="finite and positive"):
        candidate.ParametricConfig(**{option: 0.})


def test_malformed_groups_and_assignments_rejected():
    population = make_population(12, "strong", seed=11)
    data = population.assign(12)
    with pytest.raises(ValueError, match="boolean"):
        candidate.fit_candidate(data, population.start, seed=13, groups={"bad": np.ones(12)})
    with pytest.raises(ValueError, match="both binary"):
        candidate.make_design(data["x"], np.zeros(12), data["t"], population.start)
    with pytest.raises(ValueError, match="absorbing binary"):
        candidate.fit_candidate(dict(data, d=np.full_like(data["d"], .5, dtype=float)), population.start, seed=13)
    with pytest.raises(ValueError, match="single randomized"):
        candidate.fit_candidate(dict(data, z=np.zeros_like(data["z"])), population.start, seed=13)
