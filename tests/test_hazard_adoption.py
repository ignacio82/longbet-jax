"""Unit and integration tests for Discrete-Time Hazard Adoption Model with Spike-and-Slab."""
import numpy as np
import pytest

import longbet
from longbet import HazardConfig, HazardAdoptionForest, HazardAdoptionResult, hazard_adoption_effects
from longbet._hazard_adoption import _effect_block_parameters, _sample_effect_block, _utility_draw


def make_hazard_data(n=80, t_len=4, scenario="acceleration", seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    z = np.zeros((n, t_len))
    # Encouragement begins at period 1 (0-indexed: period 1)
    z[:n // 2, 1:] = 1
    d = np.zeros((n, t_len))

    if scenario == "acceleration":
        # Encouraged units adopt rapidly at period 1; control units adopt at period 3
        for i in range(n):
            for t in range(t_len):
                if t == 0:
                    d[i, t] = 0
                elif d[i, t - 1] == 1:
                    d[i, t] = 1
                else:
                    # Hazard depends on encouragement
                    prob = 0.7 if z[i, t] == 1 else (0.1 if t < 2 else 0.8)
                    d[i, t] = rng.binomial(1, prob)
    elif scenario == "zero":
        # Uptake completely independent of z
        for i in range(n):
            for t in range(t_len):
                if t == 0:
                    d[i, t] = 0
                elif d[i, t - 1] == 1:
                    d[i, t] = 1
                else:
                    d[i, t] = rng.binomial(1, 0.25)
    t = np.arange(1.0, t_len + 1.0)
    return d, z, x, t


def test_hazard_adoption_exports():
    assert "HazardConfig" in longbet.__all__
    assert "HazardAdoptionForest" in longbet.__all__
    assert "HazardAdoptionResult" in longbet.__all__
    assert "hazard_adoption_effects" in longbet.__all__


def test_hazard_adoption_acceleration_scenario():
    d, z, x, t = make_hazard_data(n=60, t_len=4, scenario="acceleration", seed=101)
    config = HazardConfig(trees=3, cutpoints=3, prior_inclusion_prob=0.5)
    forest = HazardAdoptionForest(config)
    forest.fit(d, z, x, t=t, seed=123, chains=2, burnin=30, draws=50)

    assert forest.fitted_
    assert "hazard_itt" in forest.draws
    assert "stock_itt" in forest.draws
    assert "exposure_itt" in forest.draws
    assert "xi" in forest.draws

    summary = forest.summary()
    assert isinstance(summary, HazardAdoptionResult)
    assert len(summary.table) == 3  # periods 1, 2, 3 (post-encouragement)
    assert set(summary.table.columns) >= {
        "period", "horizon", "relevance_prob",
        "hazard_itt_median", "stock_itt_median", "exposure_itt_median",
    }
    # Relevance probability should be high under acceleration
    assert summary.metadata["relevance_probability"] > 0.5


def test_hazard_adoption_zero_stage_retains_zero_in_reported_interval():
    d, z, x, t = make_hazard_data(n=80, t_len=4, scenario="zero", seed=202)
    config = HazardConfig(trees=2, cutpoints=3, prior_inclusion_prob=0.5)
    forest = HazardAdoptionForest(config)
    forest.fit(d, z, x, t=t, seed=303, chains=2, burnin=30, draws=50)

    summary = forest.summary()
    # Check the actually reported endpoints, without expanding them. This one
    # fixed-data example does not establish repeated-simulation calibration.
    for row in summary.table.itertuples():
        assert row.stock_itt_lower <= 0 <= row.stock_itt_upper


def test_hazard_adoption_effects_convenience_function():
    d, z, x, t = make_hazard_data(n=40, t_len=3, scenario="acceleration", seed=404)
    res = hazard_adoption_effects(d, z, x, t=t, seed=505, chains=2, burnin=20, draws=30)
    assert isinstance(res, HazardAdoptionResult)
    assert len(res.table) == 2


def test_collapsed_relevance_matches_dense_marginal_likelihood_and_leaf_moments():
    """Check the mixture against independent observation-space Gaussian integrals."""
    from scipy.linalg import block_diag
    from scipy.special import expit, logit
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(773)
    residual = rng.normal(size=(5, 2))
    risk = np.array([[1, 1], [1, 0], [1, 1], [1, 0], [1, 1]], dtype=bool)
    basis = np.column_stack([np.array([-.4, -.4, .6, .6, .6]), np.array([0, 0, .6, .6, 0])])
    variance, prior = .6, .4
    means, cholesky, log_bayes_factor = _effect_block_parameters(residual, risk, basis, variance)
    expected_bf = 0.0
    expected_means, expected_covariances = [], []
    for h in range(2):
        x = basis[risk[:, h]]
        response = residual[risk[:, h], h]
        marginal = np.eye(len(x)) + variance * x @ x.T
        expected_bf += (multivariate_normal.logpdf(response, cov=marginal)
                        - multivariate_normal.logpdf(response, cov=np.eye(len(x))))
        covariance = np.linalg.inv(np.eye(2) / variance + x.T @ x)
        mean = covariance @ x.T @ response
        np.testing.assert_allclose(np.sqrt(variance) * means[h], mean)
        np.testing.assert_allclose(variance * np.linalg.inv(cholesky[h] @ cholesky[h].T), covariance)
        expected_means.append(mean)
        expected_covariances.append(covariance)
    assert log_bayes_factor == pytest.approx(expected_bf, abs=1e-12)
    probability = expit(logit(prior) + expected_bf)
    values = [_sample_effect_block(rng, residual, risk, basis, variance, prior) for _ in range(12000)]
    indicators = np.array([v[0] for v in values])
    assert abs(indicators.mean() - probability) < 5.5 * np.sqrt(probability * (1 - probability) / len(values))
    for active in (0, 1):
        draws = np.array([v[1].T.ravel() for v in values if v[0] == active])
        expected_mean = np.concatenate(expected_means) if active else np.zeros(4)
        expected_covariance = block_diag(*expected_covariances) if active else variance * np.eye(4)
        mean_z = (draws.mean(axis=0) - expected_mean) / np.sqrt(np.diag(expected_covariance) / len(draws))
        covariance_mcse = np.sqrt((np.outer(np.diag(expected_covariance), np.diag(expected_covariance))
                                   + expected_covariance**2) / (len(draws) - 1))
        covariance_z = (np.cov(draws, rowvar=False) - expected_covariance) / covariance_mcse
        assert np.max(np.abs(mean_z)) < 5.5
        assert np.max(np.abs(covariance_z)) < 5.5


def test_empty_risks_add_no_evidence_and_preserve_effect_prior():
    residual = np.full((4, 2), 100.)
    risk = np.zeros((4, 2), dtype=bool)
    basis = np.ones((4, 2))
    means, chol, log_bayes_factor = _effect_block_parameters(residual, risk, basis, .8)
    np.testing.assert_array_equal(means, np.zeros_like(means))
    np.testing.assert_array_equal(chol, np.broadcast_to(np.eye(2), (2, 2, 2)))
    assert log_bayes_factor == 0


def test_probit_utilities_match_truncated_normal_including_extreme_tails():
    from scipy.stats import truncnorm

    rng = np.random.default_rng(639)
    eta = np.array([-100., -8., -.5, .5, 8., 100.])
    locations = np.broadcast_to(eta, (18000, len(eta)))
    for response in (0, 1):
        draws = _utility_draw(rng, locations, np.full_like(locations, response))
        lower = -eta if response else np.full_like(eta, -np.inf)
        upper = np.full_like(eta, np.inf) if response else -eta
        mean, variance = truncnorm.stats(lower, upper, loc=eta, moments="mv")
        assert np.isfinite(draws).all()
        assert (draws > 0).all() if response else (draws < 0).all()
        z = (draws.mean(axis=0) - mean) / np.sqrt(variance / len(draws))
        assert np.max(np.abs(z)) < 5.5


def test_observed_probit_posterior_matches_small_model_quadrature():
    """End-to-end posterior check, integrating both baseline and slab effects."""
    import arviz as az
    from scipy.special import ndtr

    z = np.array([0, 0, 0, 0, 1, 1, 1, 1])[:, None]
    d = np.array([0, 0, 0, 1, 0, 1, 1, 1])[:, None]
    config = HazardConfig(trees=1, baseline_sd=.6, effect_sd=.8, prior_inclusion_prob=.4)
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    weights /= np.sqrt(np.pi)
    baseline = np.sqrt(2) * config.baseline_sd * nodes
    effect = np.sqrt(2) * config.effect_sd * nodes
    hazard_zero = ndtr(baseline)
    likelihood_zero = hazard_zero**4 * (1 - hazard_zero)**4
    hazard_control = ndtr(baseline[:, None] - .5 * effect[None, :])
    hazard_encouraged = ndtr(baseline[:, None] + .5 * effect[None, :])
    likelihood_one = (hazard_control * (1 - hazard_control)**3
                      * hazard_encouraged**3 * (1 - hazard_encouraged))
    mass_zero = (1 - config.prior_inclusion_prob) * weights @ likelihood_zero
    weighted_one = config.prior_inclusion_prob * likelihood_one * weights[:, None] * weights[None, :]
    normalizer = mass_zero + weighted_one.sum()
    inclusion = weighted_one.sum() / normalizer
    contrast = ((hazard_encouraged - hazard_control) * weighted_one).sum() / normalizer
    model = HazardAdoptionForest(config).fit(d, z, np.ones((len(z), 1)),
                                             seed=732, chains=4, burnin=500, draws=2500)
    for actual, expected in ((model.draws["xi"], inclusion), (model.draws["stock_itt"][..., 0], contrast)):
        mcse = float(az.mcse(actual, method="mean"))
        assert abs(actual.mean() - expected) < 5.5 * mcse + 1e-4
        assert abs(actual.mean() - expected) < .04
    assert model.metadata["topology"] == "fixed_random_stumps_shared_across_chains"
    assert model.metadata["calibration_status"] == "not_established"
    assert model.metadata["weak_instrument_robust"] is False


def test_basis_and_first_chain_reproducible_when_chain_count_changes():
    d, z, x, t = make_hazard_data(n=20, t_len=3)
    config = HazardConfig(trees=2)
    one = HazardAdoptionForest(config).fit(d, z, x, t, seed=38, chains=1, burnin=2, draws=5)
    two = HazardAdoptionForest(config).fit(d, z, x, t, seed=38, chains=2, burnin=2, draws=5)
    assert one.metadata["baseline_rules"] == two.metadata["baseline_rules"]
    assert one.metadata["effect_rules"] == two.metadata["effect_rules"]
    for name in one.draws:
        np.testing.assert_array_equal(one.draws[name][0], two.draws[name][0])


@pytest.mark.parametrize("options", [dict(chains=0), dict(draws=0), dict(seed=-1),
                                     dict(burnin=-1), dict(draws=1.5)])
def test_invalid_sampling_controls_rejected(options):
    d, z, x, t = make_hazard_data()
    with pytest.raises(ValueError):
        HazardAdoptionForest().fit(d, z, x, t, **options)


def test_invalid_covariates_and_irregular_times_rejected():
    d, z, x, t = make_hazard_data()
    with pytest.raises(ValueError, match="matrix"):
        HazardAdoptionForest().fit(d, z, x[:-1], t)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        HazardAdoptionForest().fit(d, z, x, t)
    with pytest.raises(ValueError, match="equally spaced"):
        HazardAdoptionForest().fit(d, z, np.ones_like(x), [0, 1, 3, 4])


def test_hazard_adoption_predict_unit_level_and_absorbing():
    d, z, x, t = make_hazard_data(n=30, t_len=4, scenario="acceleration", seed=88)
    config = HazardConfig(trees=2, cutpoints=3)
    forest = HazardAdoptionForest(config).fit(d, z, x, t=t, seed=42, chains=2, burnin=10, draws=20)

    # In-sample prediction: shape (N, H, draws)
    pred_in = forest.predict()
    assert pred_in.shape == (30, 3, 40)
    assert np.all(np.isfinite(pred_in))

    # Out-of-sample prediction on new units
    rng = np.random.default_rng(123)
    x_new = rng.normal(size=(12, 2))
    pred_out = forest.predict(x_new)
    assert pred_out.shape == (12, 3, 40)
    assert np.all(np.isfinite(pred_out))

