"""Independent covariance and reporting checks for the matched IV comparison."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks" / "encouragement"
sys.path.insert(0, str(BENCHMARKS))
import iv_comparison as comparison
import summarize_iv_comparison as summary


def test_arm_adjustment_agrees_with_interacted_regression_hc2_sandwich():
    """Compare to a full interacted design's standard sandwich calculation."""
    data = comparison.generate_panel("smooth", 101, n=160)
    estimates, covariance = comparison.adjusted_contrasts(data, spline=True)
    a = comparison.feature_matrix(data, spline=True)
    z = data["assignment"].astype(float)
    design = np.column_stack([a * (1-z[:, None]), a * z[:, None]])
    contrast = np.r_[-a.mean(axis=0), a.mean(axis=0)]
    bread = np.linalg.inv(design.T @ design)
    hat_diag = np.diag(design @ bread @ design.T)
    for horizon in range(4):
        responses = np.column_stack([data["y"][:, horizon+2], data["d"][:, horizon+2]])
        coefficients = np.linalg.lstsq(design, responses, rcond=None)[0]
        residual = responses - design @ coefficients
        expected = contrast @ coefficients
        np.testing.assert_allclose(estimates[horizon], expected, rtol=1e-6, atol=1e-8)
        for j in range(2):
            for k in range(2):
                meat = design.T @ ((residual[:, j] * residual[:, k] / (1-hat_diag))[:, None] * design)
                expected_covariance = contrast @ bread @ meat @ bread @ contrast
                np.testing.assert_allclose(covariance[horizon, j, k], expected_covariance, rtol=1e-5, atol=1e-8)


def test_intercept_only_hc2_matches_unadjusted_arm_covariance(monkeypatch):
    data = comparison.generate_panel("linear", 123, n=160)
    monkeypatch.setattr(comparison, "feature_matrix", lambda data, spline=False: np.ones((len(data["y"]), 1)))
    estimate, covariance = comparison.adjusted_contrasts(data)
    ref = comparison.encouragement_effects(data["y"], data["d"], data["z"], data["t"])
    np.testing.assert_allclose(estimate, ref[["itt_y", "itt_d"]], atol=1e-12)
    np.testing.assert_allclose(covariance[:, 0, 0], ref.itt_y_se**2, atol=1e-12)
    np.testing.assert_allclose(covariance[:, 1, 1], ref.itt_d_se**2, atol=1e-12)
    np.testing.assert_allclose(covariance[:, 0, 1], ref.itt_y_d_cov, atol=1e-12)


@pytest.mark.parametrize("row,target,expected", [
    ({"lower": "-Infinity", "upper": 0., "lower2": 2., "upper2": "Infinity"}, 1., False),
    ({"lower": "-Infinity", "upper": 0., "lower2": 2., "upper2": "Infinity"}, 3., True),
    ({"lower": "-Infinity", "upper": "Infinity"}, 1., True),
    ({"lower": None, "upper": None}, 1., False),
    ({"lower": 0., "upper": .8}, 1., False),
    ({"lower": 0., "upper": 2.}, np.nan, False),
])
def test_json_confidence_set_membership(row, target, expected):
    assert summary.contains(row, target) is expected


def test_paired_rmse_bootstrap_keeps_horizons_together_and_rejects_incomplete_pairs():
    comparator = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.], [2., 1., 3., 4.]])
    candidate = .8 * comparator
    result = summary.paired_rmse_ratio(candidate, comparator, draws=100)
    assert result["rmse_ratio"] == pytest.approx(.8)
    assert result["lower"] == pytest.approx(.8)
    assert result["upper"] == pytest.approx(.8)
    candidate[0, 0] = np.nan
    incomplete = summary.paired_rmse_ratio(candidate, comparator, draws=100)
    assert incomplete["available"] is False
    assert incomplete["complete_pairs"] == 2
    assert incomplete["total_pairs"] == 3


def test_null_stage_has_undefined_ratio_target():
    data = comparison.generate_panel("zero", 123, n=160)
    np.testing.assert_array_equal(data["truth"]["itt_d"], np.zeros(4))
    np.testing.assert_array_equal(data["truth"]["itt_y"], np.zeros(4))
    assert np.isnan(data["truth"]["wald"]).all()


def test_summary_keeps_failed_runs_in_coverage_and_marks_rmse_incomplete(tmp_path):
    manifest = {"arguments": {"replications": 2, "scenarios": ["weak"]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "runs").mkdir()
    truth = {name: [1.] * 4 for name in ("itt_y", "itt_d", "wald")}
    rows = [dict(quantity=name, horizon=h, estimate=1., lower=0., upper=2.,
                 set_type="posterior_equal_tail", diagnostics_passed=True)
            for name in truth for h in range(1, 5)]
    for rep in range(2):
        result = (dict(status="completed", rows=rows, seconds=1.) if rep == 0
                  else dict(status="failed", error="numerical failure", seconds=1.))
        record = {"truth": truth, "methods": {"longbet_direct": result}}
        (tmp_path / "runs" / f"weak_{rep:04d}.json").write_text(json.dumps(record))
    report = summary.summarize(tmp_path)
    for aggregate in report["aggregate"]:
        assert aggregate["min_coverage"] == .5
        assert aggregate["max_coverage"] == .5
        assert aggregate["completed"] == 1
        assert aggregate["datasets"] == 2
        assert np.isnan(aggregate["rmse"])
