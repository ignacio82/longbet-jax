"""Independent checks of binary likelihoods, sampler kernels and sharp bounds."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import linprog
from scipy.special import ndtr
from scipy.stats import beta, truncnorm

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/encouragement"))
import binary_cells as cells
import binary_forest as forest
import binary_dgp as dgp
import binary_benchmark as benchmark


def data():
    rng = np.random.default_rng(830)
    x = np.tile(np.arange(3), 40)[:, None]
    z = rng.permutation(np.arange(len(x)) % 2)
    d = rng.binomial(1, .15 + .55 * z)
    y = rng.binomial(1, .1 + .65 * d)
    return y, d, z, x


def test_joint_cell_posterior_matches_analytic_covariance():
    fit = cells.fit_cells(*data(), seed=81, chains=4, draws=30000)
    exact = cells.beta_joint_moments(fit["uptake_counts"], fit["outcome_counts"])
    p, r = fit["p"].reshape(-1, 3, 2), fit["r"].reshape(-1, 3, 2)
    for values, name in ((p, "p"), (r, "r")):
        np.testing.assert_allclose(values.mean(axis=0), exact[name + "_mean"], atol=.0012)
        np.testing.assert_allclose(values.var(axis=0), exact[name + "_variance"], rtol=.025)
    covariance = np.mean((p - p.mean(axis=0)) * (r - r.mean(axis=0)), axis=0)
    np.testing.assert_allclose(covariance, exact["covariance"], rtol=.035)
    draw = cells.effects(fit)
    target_cov = np.sum(fit["weights"]**2 * exact["covariance"].sum(axis=-1))
    assert abs(np.cov(draw["itt_y"].ravel(), draw["itt_d"].ravel())[0, 1] - target_cov) < .00004
    comparison = cells.effects(cells.fit_cells(*data(), seed=81, draws=30000, independent_reduced_forms=True))
    assert abs(np.cov(comparison["itt_y"].ravel(), comparison["itt_d"].ravel())[0, 1]) < .00004


def test_ordered_beta_matches_numerical_integral_and_exhaustion_is_visible():
    a, b = np.array([4., 2.]), np.array([2., 4.])
    sample, attempts = cells.ordered_beta(np.random.default_rng(91), a, b, 50000)
    denominator = quad(lambda v: beta.pdf(v, a[0], b[0]) * beta.sf(v, a[1], b[1]), 0, 1)[0]
    numerator = quad(lambda v: v * beta.pdf(v, a[0], b[0]) * beta.sf(v, a[1], b[1]), 0, 1)[0]
    assert abs(sample[:, 0].mean() - numerator / denominator) < .003
    assert np.all(sample[:, 1] >= sample[:, 0]) and attempts > len(sample)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        cells.ordered_beta(np.random.default_rng(1), [100, 1], [1, 100], 10, max_candidates=100)


def test_raw_ratio_keeps_negative_and_tiny_stages_and_exact_null_undefined():
    fit = cells.fit_cells(*data(), seed=21, draws=5, first_stage="null")
    assert np.isnan(cells.effects(fit)["wald"]).all()
    synthetic = dict(p=np.array([[[[.5, .5 + 1e-12]], [[.6, .5]]]]),
                     r=np.array([[[[.1, .3]], [[.1, .3]]]]), weights=np.ones(1))
    ratio = cells.effects(synthetic)["wald"]
    assert ratio[0, 0] > 1e11 and ratio[0, 1] == pytest.approx(-2)
    bound = cells.identification_bounds(fit["p"], fit["q"], fit["weights"])
    assert not bound["defined"].any() and np.isnan(bound["treatment_z0"]).all()


def lp_bounds(p, q, w, delta):
    """Independent primal LP in eight stratum outcome means per baseline cell."""
    groups = len(p)
    equality, target, upper, limits = [], [], [], []
    objective = {k: np.zeros(groups * 8) for k in ("complier_encouragement", "treatment_z0", "treatment_z1")}
    mass = 0.
    for g in range(groups):
        a, c, n = p[g, 0], p[g, 1] - p[g, 0], 1 - p[g, 1]
        mass += w[g] * c
        def row(**entries):
            value = np.zeros(groups * 8)
            for index, coefficient in entries.items():
                value[g * 8 + int(index)] = coefficient
            return value
        # a10,a11,n00,n01,c00,c01,c10,c11
        equality.extend([row(**{"0": a}), row(**{"3": n}),
                         row(**{"1": a, "7": c}), row(**{"2": n, "4": c})])
        target.extend([a * q[g, 1, 0], n * q[g, 0, 1], p[g, 1] * q[g, 1, 1], (1 - p[g, 0]) * q[g, 0, 0]])
        for (one, two), tolerance in zip(((0, 1), (2, 3), (4, 5), (6, 7)), (delta[0], delta[1], delta[2], delta[2])):
            direction = row(**{str(one): 1, str(two): -1})
            upper.extend([direction, -direction])
            limits.extend([tolerance, tolerance])
        for name, positive, negative in (("complier_encouragement", 7, 4), ("treatment_z0", 6, 4), ("treatment_z1", 7, 5)):
            objective[name] += row(**{str(positive): w[g] * c, str(negative): -w[g] * c})
    result = {}
    for name, direction in objective.items():
        endpoints = []
        for sign in (1, -1):
            solved = linprog(sign * direction, A_ub=upper, b_ub=limits, A_eq=equality, b_eq=target, bounds=(0, 1), method="highs")
            if not solved.success:
                assert solved.status == 2  # infeasible, not an optimizer failure
                return None
            endpoints.append(sign * solved.fun / mass)
        result[name] = endpoints
    return result


@pytest.mark.parametrize("delta", [(1, 1, 1), (.1, .2, .05), (0, 0, 0)])
def test_bounds_match_independent_joint_linear_program(delta):
    rng = np.random.default_rng(947)
    for _ in range(10):
        p = np.sort(rng.uniform(.05, .95, (2, 2)), axis=-1)
        q = rng.uniform(.05, .95, (2, 2, 2))
        actual = cells.identification_bounds(p, q, [.4, .6], delta=delta)
        expected = lp_bounds(p, q, [.4, .6], delta)
        assert bool(actual["compatible"]) == (expected is not None)
        if expected is not None:
            for name, endpoints in expected.items():
                np.testing.assert_allclose(actual[name], endpoints, atol=1e-12)


def test_exclusion_collapse_and_direct_effect_distinction_and_zero_cell():
    p = np.array([[.2, .6], [.1, .1]])
    # a=.2, c=.4, n=.4. Exclusion means a=.5,n=.2,c0=.3,c1=.6.
    q = np.array([[[.25, .2], [.5, 17 / 30]], [[.2, .2], [.5, .5]]])
    bounds = cells.identification_bounds(p, q, [.7, .3], delta=0)
    assert bounds["available"]
    for name in ("complier_encouragement", "treatment_z0", "treatment_z1"):
        np.testing.assert_allclose(bounds[name], [.3, .3], atol=1e-14)
    free = cells.identification_bounds(p[:1], q[:1], [1.], delta=1)
    assert np.diff(free["treatment_z0"])[0] > np.diff(free["complier_encouragement"])[0]
    q[0, 1, 1] = 0
    empty = cells.identification_bounds(p, q, [.7, .3], delta=0)
    assert not empty["compatible"] and np.isnan(empty["treatment_z0"]).all()
    with pytest.raises(ValueError, match="no-defiers"):
        cells.identification_bounds(p[:, ::-1], q, [.7, .3])


@pytest.mark.parametrize("eta,response", [(-8., 1), (1.5, 1), (0., 0), (6., 0)])
def test_utility_draw_matches_truncated_normal_moments(eta, response):
    values = forest.utility_draw(np.random.default_rng(301), np.full(70000, eta), response)
    a, b = (-eta, np.inf) if response else (-np.inf, -eta)
    mean, variance = truncnorm.stats(a, b, loc=eta, moments="mv")
    assert abs(values.mean() - mean) < 5 * np.sqrt(variance / len(values))
    assert abs(values.var() / variance - 1) < .03
    assert ((values > 0) == response).all()


def test_rare_monotone_augmentation_and_extreme_utilities():
    rng = np.random.default_rng(563)
    f, g = -.4, .8
    rf, rg = forest.monotone_labels(rng, np.full(80000, f), g, 0)
    expected = np.array([(1 - ndtr(f)) * (1 - ndtr(g)), ndtr(f) * (1 - ndtr(g)), (1 - ndtr(f)) * ndtr(g)])
    expected /= expected.sum()
    actual = np.array([np.mean((rf == 0) & (rg == 0)), rf.mean(), rg.mean()])
    np.testing.assert_allclose(actual, expected, atol=.006)
    assert not np.any(rf * rg)
    for value in (-40., 40.):
        rf, rg = forest.monotone_labels(rng, np.full(50, value), value, 0)
        assert not np.any(rf * rg)
        for label in (0, 1):
            utility = forest.utility_draw(rng, np.full(50, value), label)
            assert np.isfinite(utility).all() and ((utility > 0) == label).all()


def test_joint_leaf_conditional_moments():
    rng = np.random.default_rng(339)
    matrix = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1.]])
    y = np.array([1., 2., -.5, -.1])
    covariance = np.linalg.inv(matrix.T @ matrix + np.eye(4) / .7)
    mean = covariance @ matrix.T @ y
    values = np.array([forest.joint_leaf_draw(rng, matrix, y, .7) for _ in range(18000)])
    np.testing.assert_allclose(values.mean(axis=0), mean, atol=.017)
    np.testing.assert_allclose(np.cov(values.T), covariance, atol=.018)


@pytest.mark.parametrize("stage", ["unrestricted", "monotone"])
@pytest.mark.parametrize("marginal", [False, True])
def test_binary_forest_archive_replay_and_empty_outcome_cells(stage, marginal, tmp_path):
    y, d, z, x = data()
    d = z.copy()  # two observational cells empty, retain their proper priors
    config = forest.BinaryConfig(trees=2, cutpoints=2, marginal_uptake_refresh=marginal)
    full = forest.fit_binary(y, d, z, x, seed=72, burnin=3, draws=10, chains=2, config=config, first_stage=stage)
    part = forest.fit_binary(y, d, z, x, seed=72, burnin=3, draws=4, chains=2, config=config, first_stage=stage)
    np.savez_compressed(tmp_path / "binary.npz", **part)
    with np.load(tmp_path / "binary.npz", allow_pickle=False) as saved:
        rest = forest.fit_binary(y, d, z, x, seed=99, burnin=0, draws=6, chains=2, config=config, first_stage=stage, resume=saved)
    for name in ("p", "q", "r", "eta", "rules", "leaves"):
        np.testing.assert_allclose(rest[name], full[name][:, 4:], atol=1e-14)
    assert np.ptp(full["q"][..., 0, 1]) > .1
    if stage == "monotone":
        assert np.all(full["p"][..., 1] >= full["p"][..., 0])
    with pytest.raises(ValueError, match="unchanged"):
        forest.fit_binary(1-y, d, z, x, seed=99, burnin=0, draws=6, chains=2, config=config, first_stage=stage, resume=part)


def geweke_statistic(stage, *, repetitions=900, marginal=False):
    """Independent prior-predictive / posterior-transition paired replicates."""
    rng = np.random.default_rng(471)
    x = np.arange(8)[:, None] % 3
    z = np.arange(8) % 2
    config = forest.BinaryConfig(trees=2, cutpoints=2, marginal_uptake_refresh=marginal)
    grid = np.unique(x, axis=0)
    space, grid_mask, _ = forest.setup(x, grid, z, z, config)
    differences = []
    for _ in range(repetitions):
        prior = [forest.ProbitForest(rng, space, grid_mask, np.arange(8), config) for _ in range(6)]
        eta = np.array([f.eta() for f in prior])
        p, q, _ = forest.probabilities(eta, stage)
        d = rng.binomial(1, p[np.arange(8), z])
        y = rng.binomial(1, q[np.arange(8), d, z])
        _, _, outcome_rows = forest.setup(x, grid, z, d, config)
        rows = ([np.arange(8), np.flatnonzero(z == 0)] if stage == "monotone"
                else [np.flatnonzero(z == zz) for zz in (0, 1)]) + outcome_rows
        posterior = [forest.ProbitForest(rng, space, grid_mask, row, config, rules=f.rules, leaves=f.leaves)
                     for row, f in zip(rows, prior)]
        def statistics(fs):
            eta = np.array([f.predict_grid() for f in fs])
            # Joint moments exercise component dependence; leaf/topology checks
            # include inactive leaves, which must still follow their priors.
            return np.r_[eta.mean(axis=1), (eta**2).mean(axis=1),
                         np.mean(eta[0] * eta[1]), np.mean(eta[0] * eta[2]),
                         [np.mean(f.rules == 0) for f in fs],
                         [np.mean(f.leaves**2) for f in fs]]
        before = statistics(posterior)
        forest.model_sweep(rng, posterior, d, z, y, stage)
        differences.append(statistics(posterior) - before)
    differences = np.array(differences)
    return differences.mean(axis=0) / (differences.std(axis=0, ddof=1) / np.sqrt(repetitions))


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["unrestricted", "monotone"])
@pytest.mark.parametrize("marginal", [False, True])
def test_binary_geweke_prior_preservation(stage, marginal):
    assert np.max(np.abs(geweke_statistic(stage, marginal=marginal))) < 4.5


@pytest.mark.slow
def test_binary_geweke_detects_broken_conditional(monkeypatch):
    correct = forest.joint_leaf_draw
    monkeypatch.setattr(forest, "joint_leaf_draw", lambda rng, matrix, utility, variance:
                        correct(rng, matrix, utility, variance / 4))
    assert np.max(np.abs(geweke_statistic("monotone", repetitions=350))) > 7


@pytest.mark.slow
def test_geweke_detects_broken_marginal_slice_prior(monkeypatch):
    correct = forest.marginal_uptake_refresh
    def broken(rng, forests, d, z, first_stage):
        for f in forests[:2]:
            f.variance /= 9  # wrong Gaussian ellipse prior
        correct(rng, forests, d, z, first_stage)
        for f in forests[:2]:
            f.variance *= 9
    monkeypatch.setattr(forest, "marginal_uptake_refresh", broken)
    assert np.max(np.abs(geweke_statistic("monotone", repetitions=500, marginal=True))) > 7


@pytest.mark.parametrize("stage", ["unrestricted", "monotone"])
def test_marginal_uptake_likelihood_equals_direct_binary_probability(stage):
    rng = np.random.default_rng(23)
    eta = rng.normal(size=(2, 12))
    d, z = rng.binomial(1, .5, (2, 12))
    if stage == "monotone":
        p = np.where(z, ndtr(eta[0]), ndtr(eta[0]) * ndtr(eta[1]))
    else:
        p = ndtr(eta[z, np.arange(12)])
    np.testing.assert_allclose(forest.uptake_loglik(eta, d, z, stage), np.log(np.where(d, p, 1-p)).sum())
    assert np.isfinite(forest.uptake_loglik(np.full((2, 12), 40.), d, z, stage))


def test_binary_residual_and_topology_changes():
    rng = np.random.default_rng(777)
    y, d, z, x = data()
    config = forest.BinaryConfig(trees=2, cutpoints=2)
    space, mask, _ = forest.setup(x, np.unique(x, axis=0), z, d, config)
    f = forest.ProbitForest(rng, space, mask, np.arange(len(y)), config)
    seen = set()
    for _ in range(30):
        utility = f.sweep(rng, y)
        np.testing.assert_allclose(f.residual, utility - f.eta(), atol=1e-13)
        seen.update(f.rules)
    assert len(seen) > 1


@pytest.mark.parametrize("scenario", dgp.SCENARIOS)
def test_binary_dgp_potential_consistency_truth_and_bounds(scenario):
    value = dgp.make_binary_data(scenario, n=240, population_seed=233, assignment_seed=45)
    changed = dgp.make_binary_data(scenario, n=240, population_seed=233, assignment_seed=46)
    for name in ("potential_y", "potential_d", "strata", "conditional_itt", "finite_itt"):
        np.testing.assert_array_equal(value[name], changed[name])
    assert not np.array_equal(value["z"], changed["z"])
    assert value["z"].sum() == 120
    np.testing.assert_array_equal(value["d"], value["potential_d"][np.arange(240), value["z"]])
    np.testing.assert_array_equal(value["y"], value["potential_y"][np.arange(240), value["d"], value["z"]])
    np.testing.assert_allclose(value["r_truth"], value["p_truth"] * value["q_truth"][:, 1] + (1-value["p_truth"]) * value["q_truth"][:, 0])
    truth = dgp.targets(value)
    if scenario == "zero":
        assert truth["finite"]["itt_d"] == 0 and np.isnan(truth["finite"]["wald"])
        assert np.isnan(truth["conditional"]["complier_encouragement"])
    elif scenario == "defiers":
        assert truth["conditional"]["itt_d"] < 0
        complier, defier = value["strata"] == 2, value["strata"] == 3
        treatment_contrast = value["potential_y"][:, 1, 0] - value["potential_y"][:, 0, 0]
        # Exclusion now gives a signed mixture of complier and defier effects;
        # numerical equality with a CACE can occur without identification.
        assert truth["finite"]["itt_y"] == pytest.approx(
            np.mean(complier * treatment_contrast - defier * treatment_contrast))
    else:
        bound = cells.identification_bounds(value["p_truth"], value["q_truth"], value["weights"], delta=.12 if scenario == "direct_effect" else 0)
        assert bound["available"]
        for name in benchmark.BOUND_NAMES:
            lo, hi = bound[name]
            assert lo - 1e-12 <= truth["conditional"][name] <= hi + 1e-12
        if scenario != "direct_effect":
            assert truth["finite"]["wald"] == pytest.approx(truth["finite"]["complier_encouragement"])
        else:
            assert abs(truth["conditional"]["wald"] - truth["conditional"]["complier_encouragement"]) > .05


def test_dgp_randomization_covariance_identity_by_enumeration():
    from itertools import combinations
    value = dgp.make_binary_data("strong", n=12, population_seed=214)
    dy = value["potential_d"]
    yy = np.stack([value["potential_y"][np.arange(12), dy[:, z], z] for z in (0, 1)], axis=-1)
    observed, estimates = [], []
    for chosen in combinations(range(12), 6):
        assigned = np.zeros(12, bool)
        assigned[list(chosen)] = True
        v1 = np.stack([yy[assigned, 1], dy[assigned, 1]], axis=-1)
        v0 = np.stack([yy[~assigned, 0], dy[~assigned, 0]], axis=-1)
        observed.append(v1.mean(axis=0) - v0.mean(axis=0))
        estimates.append((np.cov(v1.T, ddof=1) + np.cov(v0.T, ddof=1)) / 6)
    np.testing.assert_allclose(np.cov(np.array(observed).T, ddof=0), value["assignment_covariance"], atol=1e-13)
    np.testing.assert_allclose(np.mean(estimates, axis=0), value["expected_neyman_covariance"], atol=1e-13)


def test_summary_does_not_condition_away_incompatible_draws():
    data = dgp.make_binary_data("strong", n=24)
    archive = cells.fit_cells(*[data[k] for k in ("y", "d", "z", "x")], seed=23, draws=120, first_stage="monotone")
    summary = benchmark.summarize(archive, dgp.targets(data))
    assert summary["bounds"]["0.0"]["available_fraction"] < 1
    assert "withheld" in summary["bounds"]["0.0"]["treatment_z0"]["status"]
    assert "mean" not in summary["diagnostics"]["wald"]
    assert "mcse_mean" not in summary["diagnostics"]["wald"]
    assert summary["diagnostics"]["bounds_delta_0.0_treatment_z0[0]"]["status"] == "undefined_draws"
    assert summary["bounds"]["1.0"]["available_fraction"] == 1


def test_aggregation_retains_failures_and_undefined_null_ratios(tmp_path):
    data = dgp.make_binary_data("zero", n=24)
    archive = cells.fit_cells(*[data[k] for k in ("y", "d", "z", "x")], seed=23, draws=120, first_stage="null")
    truth = dgp.targets(data)
    summary = benchmark.summarize(archive, truth, full_diagnostics=False)
    for replicate in range(2):
        directory = tmp_path / f"zero_{replicate:04d}"
        directory.mkdir()
        record = dict(scenario="zero", model="cells_null", replicate=replicate, targets=truth,
                      status="failed" if replicate else "complete")
        record.update(traceback="deliberate test failure" if replicate else None, summary=summary)
        benchmark.write_json(directory / "cells_null.json", record)
    result = benchmark.aggregate(tmp_path)[0]
    assert result["failures"] == 1 and result["attempts"] == 2 and result["complete"] == 1
    assert result["targets"]["finite"]["coverage"]["wald"]["n"] == 0
    assert result["mean_interval_width"]["wald"] is None


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["unrestricted", "monotone"])
def test_constant_covariate_forest_matches_independent_exact_posterior(stage):
    """With pointwise N(0,1) priors, probit probabilities are Uniform(0,1).

    For product-monotone uptake the (p0,p1) density is 1/p1 on the triangle.
    Multiplying by the binomial likelihood gives ordered Betas with one fewer
    p1 success pseudo-count. This provides an independent end-to-end oracle.
    """
    import arviz as az
    z = np.repeat([0, 1], 16)
    d = np.r_[np.arange(16) < 3, np.arange(16) < 10].astype(int)
    y = np.random.default_rng(814).binomial(1, .2 + .5*d)
    x = np.zeros((32, 1))
    fit = forest.fit_binary(y, d, z, x, seed=914, burnin=500, draws=1500, chains=4,
                            first_stage=stage, config=forest.BinaryConfig(trees=2, marginal_uptake_refresh=stage == "monotone"))
    exact = cells.fit_cells(y, d, z, x, seed=281, draws=50000, chains=4)
    if stage == "monotone":
        counts = exact["uptake_counts"][0]
        ordered, _ = cells.ordered_beta(np.random.default_rng(792), counts[:, 1]+[1, 0],
                                        counts[:, 0]+1, 200000)
        exact["p"] = ordered.reshape(4, 50000, 1, 2)
        exact["r"] = exact["p"] * exact["q"][..., 1, :] + (1-exact["p"]) * exact["q"][..., 0, :]
    def quantities(archive):
        shape = archive["p"].shape[:2]
        return np.concatenate([archive[k].reshape(*shape, -1) for k in ("p", "q", "r")]
                              + [(archive["p"] * archive["r"]).reshape(*shape, -1)], axis=-1)
    observed, expected = quantities(fit), quantities(exact)
    for j in range(observed.shape[-1]):
        error = observed[..., j].mean() - expected[..., j].mean()
        se = np.hypot(float(az.mcse(observed[..., j], method="mean")), expected[..., j].std() / np.sqrt(200000))
        assert abs(error) < 4.5 * se, (stage, j, error, se)
