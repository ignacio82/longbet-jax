"""Deterministic metric checks; a few simulated successes cannot prove calibration."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.encouragement import sbc_calibration as coverage


def test_potential_panel_and_prespecified_binary_target():
    panel = coverage.generate_coverage_panel(n=100, instrument_effect=1.2, seed=123, binary_threshold=.5)
    assert panel.y.shape == panel.d.shape == panel.z.shape == (100, 5)
    assert np.all(np.diff(panel.d, axis=1) >= 0)
    assert np.all(panel.d_z1 >= panel.d_z0)
    assert panel.z[:, -1].sum() == 50
    # Assignment must be actually randomized, rather than the first half of rows.
    assert panel.z[50:, -1].sum() > 0
    np.testing.assert_allclose(panel.y - panel.true_beta * panel.d, panel.untreated_y)
    compliers = panel.d_z1[:, 1] > panel.d_z0[:, 1]
    binary_y1 = (panel.untreated_y[:, 1] + panel.true_beta > .5).astype(float)
    binary_y0 = (panel.untreated_y[:, 1] > .5).astype(float)
    assert panel.binary_complier_target == np.mean((binary_y1 - binary_y0)[compliers])
    null = coverage.generate_coverage_panel(n=40, instrument_effect=0)
    assert null.binary_complier_target is None
    np.testing.assert_array_equal(null.d_z0, null.d_z1)


def test_disconnected_ar_grid_retains_gaps_and_unknown_tails():
    table = pd.DataFrame({"beta": [-3., -2., -1., 0., 1., 2., 3.],
                          "accepted": [True, True, False, False, False, True, True]})
    summary = coverage.summarize_ar_grid(table, 1.)
    assert summary["covers_target"] is False
    assert summary["grid_runs"] == [(-3., -2.), (2., 3.)]
    assert summary["accepted_grid_points"] == 4
    assert summary["left_tail"] == summary["right_tail"] == "unknown"
    assert summary["between_grid_points"] == "unknown"
    assert "width" not in summary
    with pytest.raises(ValueError, match="exact target"):
        coverage.summarize_ar_grid(table, 1.1)
    table["accepted"] = False
    empty = coverage.summarize_ar_grid(table, 1.)
    assert empty["grid_runs"] == []
    assert empty["right_tail"] == "unknown"


@pytest.mark.parametrize("lower,upper,target,expected", [
    (0., .8, 1., False), (.2, .2, .2, True), (np.nan, 2., 1., False),
    (-np.inf, np.inf, 1., True), (2., 1., 1.5, False),
])
def test_interval_coverage_never_pads_or_only_checks_nonemptiness(lower, upper, target, expected):
    assert coverage.interval_contains(lower, upper, target) is expected


def mock_methods(monkeypatch):
    panel = replace(coverage.generate_coverage_panel(n=20, instrument_effect=1.2), binary_complier_target=.7)
    monkeypatch.setattr(coverage, "generate_coverage_panel", lambda **kwargs: panel)
    ar_table = pd.DataFrame({"horizon": [1, 1, 1], "beta": [0., 1., 2.], "accepted": [True, False, True]})
    monkeypatch.setattr(coverage, "randomization_ar", lambda *args, **kwargs: SimpleNamespace(table=ar_table))
    ref = pd.DataFrame([dict(horizon=1, itt_d=.1, itt_d_se=.1, wald_set_type="disjoint",
                             wald_lower_1=-np.inf, wald_upper_1=0., wald_lower_2=.8, wald_upper_2=np.inf)])
    monkeypatch.setattr(coverage, "encouragement_effects", lambda *args, **kwargs: ref)
    bounds = pd.DataFrame([dict(estimand="complier_encouragement", lower_median=0., upper_median=.2,
                                lower_ci_lower=-.1, upper_ci_upper=.9, compatible_fraction=.6)])
    monkeypatch.setattr(coverage, "encouragement_bounds", lambda *args, **kwargs: SimpleNamespace(table=bounds))
    working = SimpleNamespace(beta_ci=(0., .8), metadata={"null_relevance_fraction": 1.})
    monkeypatch.setattr(coverage, "CoupledHazardIV", lambda config: SimpleNamespace(fit=lambda *args: working))
    return ar_table


def test_report_compares_distinct_true_targets_with_actual_endpoints(monkeypatch):
    mock_methods(monkeypatch)
    summaries = coverage.evaluate_coverage(replications=2, include_experimental=True)
    for summary in summaries:
        assert summary.ar_coverage == 0
        assert summary.ar_mean_grid_components == 2
        assert summary.fieller_coverage == 1  # Truth is in the second component.
        assert summary.fieller_unbounded_rate == 1
        assert summary.binary_median_bounds_containment == 0  # Nonempty but excludes truth.
        assert summary.binary_posterior_outer_interval_coverage == 1
        assert summary.coupled_working_interval_coverage == 0  # Old +/- .25 padding would cover.
        assert summary.coupled_causal_inference_supported is False
        assert summary.binary_target_defined_replications == 2


def test_failed_replications_stay_in_coverage_denominator(monkeypatch):
    table = mock_methods(monkeypatch)
    table["accepted"] = True
    calls = 0

    def sometimes_fails(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 2:
            raise RuntimeError("simulated numerical failure")
        return SimpleNamespace(table=table)

    monkeypatch.setattr(coverage, "randomization_ar", sometimes_fails)
    with pytest.warns(RuntimeWarning, match="AR failed"):
        summaries = coverage.evaluate_coverage(replications=2)
    assert all(summary.ar_coverage == .5 for summary in summaries)
    assert all(summary.failures["ar"] == 1 for summary in summaries)


def test_small_real_experiment_smoke():
    summaries = coverage.evaluate_coverage(replications=1, n=40, t_len=3,
                                            permutations=19, bounds_draws=20, seed=42)
    assert len(summaries) == 4
    assert summaries[0].scenario == "null"
    assert summaries[0].binary_target_defined_replications == 0
    assert summaries[0].binary_median_bounds_containment is None
    for summary in summaries:
        assert summary.replications == 1
        assert sum(summary.failures.values()) == 0
        assert summary.coupled_working_interval_coverage is None
        assert 0 <= summary.ar_coverage <= 1


def test_legacy_sbc_names_deprecate_mislabel():
    with pytest.warns(DeprecationWarning, match="not SBC"):
        y, d, z = coverage.generate_sbc_panel(n=20)
    assert y.shape == d.shape == z.shape
