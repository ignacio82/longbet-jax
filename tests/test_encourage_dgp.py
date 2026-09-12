"""Potential-outcome and covariance identities used by calibration runners."""

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "encouragement"))
from dgp import SCENARIOS, Scenario, make_population, population_standardized_truth


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_potential_paths_are_fixed_before_assignment(scenario):
    p = make_population(80, scenario, seed=10)
    for value in (p.d0, p.d1):
        assert np.all(np.isin(value, [0, 1]))
        assert np.all(np.diff(value, axis=1) >= 0)
    for seed in (11, 12):
        data = p.assign(seed)
        a = data["assignment"]
        assert a.sum() == 40
        np.testing.assert_array_equal(data["y"], np.where(a[:, None], p.y1, p.y0))
        np.testing.assert_array_equal(data["d"], np.where(a[:, None], p.d1, p.d0))
    np.testing.assert_array_equal(p.y0[:, :p.start], p.y1[:, :p.start])
    if p.scenario.binary:
        assert np.isin(p.y0, [0, 1]).all() and np.isin(p.y1, [0, 1]).all()
        assert ((p.mean_y1 >= 0) & (p.mean_y1 <= 1)).all()
    if not p.scenario.defier_share:
        assert np.all(p.d1 >= p.d0)


def test_finite_population_covariance_matches_complete_assignment_enumeration():
    p = make_population(8, Scenario(complier_share=.375, always_share=.25), seed=19)
    a = np.c_[p.y1[:, p.start:], p.d1[:, p.start:]]
    b = np.c_[p.y0[:, p.start:], p.d0[:, p.start:]]
    estimates, covariances = [], []
    for assigned in combinations(range(8), 4):
        z = np.zeros(8, dtype=bool)
        z[list(assigned)] = True
        estimates.append(a[z].mean(axis=0) - b[~z].mean(axis=0))
        covariances.append(np.cov(a[z], rowvar=False, ddof=1) / 4
                           + np.cov(b[~z], rowvar=False, ddof=1) / 4)
    exact = p.covariance_truth()
    # Enumeration is the entire assignment distribution, hence ddof=0 here.
    np.testing.assert_allclose(np.cov(estimates, rowvar=False, ddof=0), exact["randomization"], atol=1e-14)
    np.testing.assert_allclose(np.mean(covariances, axis=0), exact["expected_neyman"], atol=1e-14)
    assert np.linalg.eigvalsh(exact["unidentified_neyman_gap"]).min() > -1e-14


def test_acceleration_changes_history_after_stock_first_stage_disappears():
    p = make_population(100, Scenario(complier_share=0, always_share=0,
                                     accelerated_share=1, effect_heterogeneity=0), seed=1)
    np.testing.assert_array_equal(p.truth["itt_d"], [1, 1, 0, 0])
    assert (p.truth["itt_y"] > 0).all()
    assert np.isnan(p.truth["wald"][-2:]).all()


def test_zero_stage_and_assumption_failures_are_labeled_without_cace_claim():
    p = make_population(400, "zero")
    np.testing.assert_array_equal(p.truth["itt_d"], np.zeros(4))
    assert np.isnan(p.truth["wald"]).all()
    assert SCENARIOS["defiers"].assumption_failures == ["monotonicity"]
    assert SCENARIOS["exclusion_violation"].assumption_failures == ["exclusion"]


def test_population_marginal_target_is_reproducible_and_has_separate_mcse():
    p = make_population(40, "binary", seed=11)
    result = population_standardized_truth(p, draws=8, seed=32)
    again = population_standardized_truth(p, draws=8, seed=32)
    np.testing.assert_array_equal(result["itt_y"], again["itt_y"])
    assert (result["mcse_itt_y"] > 0).any()
    assert result["target"] == "new_latent_units_at_empirical_baseline_covariates"


@pytest.mark.parametrize("kwargs", [dict(complier_share=-.1), dict(always_share=.9),
                                    dict(unit_correlation=1), dict(serial_correlation=-1),
                                    dict(unit_correlation=np.nan),
                                    dict(adoption_delay=-1), dict(effect=np.inf)])
def test_invalid_scenario_rejected(kwargs):
    with pytest.raises(ValueError):
        Scenario(**kwargs)


def test_calibration_summary_preserves_undefined_ratio_and_strict_json():
    from model_calibration import summarize_draws
    from reference_matrix import portable
    import json

    rng = np.random.default_rng(8)
    values = {"outcome": rng.normal(size=(1, 2, 2, 20)), "takeup": np.zeros((1, 2, 2, 20))}
    stats, raw = summarize_draws(values, {"test": {"itt_y": np.ones(2), "itt_d": np.zeros(2),
                                                 "wald": np.full(2, np.nan)}})
    assert np.isnan(raw["wald"]).all()
    assert np.isnan(stats["wald"]["median"]).all()
    assert all(d["undefined_fraction"] == 1 for d in stats["wald"]["diagnostics"])
    assert all("mcse_mean" not in d for d in stats["wald"]["diagnostics"])
    assert all("mcse_median" in d for d in stats["wald"]["diagnostics"])
    json.dumps(portable(stats), allow_nan=False)


def test_candidate_geweke_detects_a_broken_conditional(monkeypatch):
    import intercept_candidate as candidate

    def broken(rng, state, *args, **kwargs):
        return candidate.GaussianState(np.zeros_like(state.beta), np.zeros_like(state.gamma),
                                       np.zeros_like(state.covariance))

    monkeypatch.setattr(candidate, "sweep", broken)
    result = candidate.geweke(replications=20, seed=11)
    for value in result["results"].values():
        assert not value["pass"]
        assert np.isinf(value["z"][[1, 2, 3]]).all()


def test_model_calibration_never_pools_different_sample_sizes(tmp_path):
    from model_calibration import aggregate, summarize_draws
    from reference_matrix import portable
    import json

    rng = np.random.default_rng(4)
    truth = {"itt_y": [1., 1.], "itt_d": [.5, .5], "wald": [2., 2.]}
    targets = {"conditional_mean": truth, "population_marginal": truth}
    stats, _ = summarize_draws({"outcome": rng.normal(size=(1, 2, 2, 30)),
                               "takeup": rng.normal(size=(1, 2, 2, 30))}, targets)
    for n in (8, 16):
        folder = tmp_path / f"n{n}"
        folder.mkdir()
        record = {"status": "complete", "n": n, "scenario": {"name": "strong", "effect": 1.},
                  "variant": "joint_lpm", "config_id": "same_sampler", "targets": targets,
                  "standardizations": {"conditional": stats, "population": stats}}
        (folder / "result.json").write_text(json.dumps(portable(record)))
    aggregate(tmp_path)
    rows = json.loads((tmp_path / "summary.json").read_text())
    assert len(rows) == 2
    assert sorted(row["n"] for row in rows) == [8, 16]
    assert all(row["independent_datasets"] == 1 for row in rows)


def test_reference_calibration_late_failure_does_not_count_as_success(tmp_path, monkeypatch):
    import reference_matrix as matrix

    original = matrix.encouragement_effects
    calls = 0

    def delayed_failure(*args, **kwargs):
        nonlocal calls
        table = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            def broken(*args, **kwargs):
                raise RuntimeError("deliberate failure after estimates were stored")
            table.to_dict = broken
        return table

    monkeypatch.setattr(matrix, "encouragement_effects", delayed_failure)
    result = matrix.run_setting("strong", 20, 3, 9, tmp_path)
    assert result["successful_replications"] == 2
    assert len(result["failures"]) == 1
    with np.load(tmp_path / result["raw_archive"]) as archive:
        assert np.isnan(archive["estimates"][0]).all()
        assert np.isnan(archive["coverage"][0]).all()
