"""Design and comparator checks for the held-out IV redesign benchmark."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1]/"benchmarks"/"encouragement"
sys.path.insert(0, str(BENCHMARKS))
import orthogonal_comparison as comparison
import iv_comparison as original


@pytest.mark.parametrize("scenario", original.SCENARIOS)
def test_original_designs_are_exactly_preserved(scenario):
    old, new = original.generate_panel(scenario, 381), comparison.generate_panel(scenario, 381)
    for key in ("y", "d", "z", "x", "t", "assignment"):
        np.testing.assert_array_equal(new[key], old[key])
    for quantity in old["truth"]:
        np.testing.assert_array_equal(new["truth"][quantity], old["truth"][quantity])


@pytest.mark.parametrize("scenario", ("long_smooth", "long_abrupt"))
def test_long_panels_have_valid_current_adoption_cace_and_varied_timing(scenario):
    data = comparison.generate_panel(scenario, 947)
    assert data["y"].shape == (240, 22)
    assert data["start"] == 6
    assert data["assignment"].sum() == 120
    d0, d1 = data["potential_d0"], data["potential_d1"]
    y0, y1 = data["potential_y0"], data["potential_y1"]
    assert np.all(d1 >= d0)
    assert np.all(np.diff(d0, axis=1) >= 0)
    assert np.all(np.diff(d1, axis=1) >= 0)
    assert len(np.unique(data["adoption_time1"])) > 6
    assert np.all(d0[:, :6] == 0)
    assert np.all(d1[:, :6] == 0)
    np.testing.assert_array_equal(y0[:, :6], y1[:, :6])
    np.testing.assert_array_equal(y0[d0 == d1], y1[d0 == d1])
    np.testing.assert_allclose(data["truth"]["itt_y"], (y1-y0).mean(axis=0)[6:])
    np.testing.assert_allclose(data["truth"]["itt_d"], (d1-d0).mean(axis=0)[6:])
    for h in range(16):
        compliers = d1[:, 6+h] > d0[:, 6+h]
        assert compliers.mean() >= .25
        np.testing.assert_allclose(data["truth"]["wald"][h], (y1-y0)[compliers, 6+h].mean())
    np.testing.assert_array_equal(data["y"], np.where(data["assignment"][:, None], y1, y0))
    np.testing.assert_array_equal(data["d"], np.where(data["assignment"][:, None], d1, d0))


def test_potential_outcomes_do_not_depend_on_assignment_draw(monkeypatch):
    """Perturb only the final permutation, preserving the entire DGP RNG stream."""
    real_default_rng = np.random.default_rng

    class ComplementFinalPermutation:
        def __init__(self, seed):
            self.rng = real_default_rng(seed)

        def __getattr__(self, name):
            return getattr(self.rng, name)

        def permutation(self, values):
            return 1-self.rng.permutation(values)

    original_data = comparison.generate_panel("long_smooth", 291)
    monkeypatch.setattr(np.random, "default_rng", ComplementFinalPermutation)
    changed = comparison.generate_panel("long_smooth", 291)
    np.testing.assert_array_equal(changed["assignment"], ~original_data["assignment"])
    for key in ("potential_y0", "potential_y1", "potential_d0", "potential_d1"):
        np.testing.assert_array_equal(changed[key], original_data[key])
    for quantity in changed["truth"]:
        np.testing.assert_array_equal(changed["truth"][quantity], original_data["truth"][quantity])


def test_baseline_features_include_every_pre_observation_and_no_post_outcomes():
    pytest.importorskip("longbet._iv")
    data = comparison.generate_panel("long_abrupt", 94)
    features = comparison.baseline_features(data)
    expected = np.column_stack((data["x"], data["y"][:, :6], data["d"][:, :6]))
    np.testing.assert_array_equal(features, expected)
    data["y"][:, 6:] += 100
    np.testing.assert_array_equal(comparison.baseline_features(data), expected)


@pytest.mark.parametrize("kind", ("spline", "extratrees"))
def test_comparators_return_counterfactual_trajectories_and_do_not_fit_test_accounts(kind):
    pytest.importorskip("sklearn")
    data = comparison.generate_panel("smooth", 107)
    x = np.column_stack((data["x"], data["y"][:, :2]))
    response = np.stack((data["y"][:, 2:], data["d"][:, 2:]), axis=-1)
    fit = comparison.TunedComparator(kind, seed=319, trees=8).fit(
        x[:120], data["assignment"][:120], response[:120], data["t"][2:])
    pred = fit.predict(x[120:])
    assert pred.shape == (40, 4, 2, 2)
    assert np.isfinite(pred).all()
    assert np.all((pred[..., 1] >= 0) & (pred[..., 1] <= 1))
    assert len(fit.metadata_["validation_scores"]) == 3
    assert fit.metadata_["training_count"] == 120
    np.testing.assert_allclose(fit.predict(x[120:130]), pred[:10])
    np.testing.assert_allclose(fit.predict(x[120:][::-1]), pred[::-1])


def test_comparators_and_longbet_share_inner_validation_accounts():
    pytest.importorskip("sklearn")
    from longbet._iv_nuisance import inner_validation_split
    import hashlib
    data = comparison.generate_panel("linear", 204)
    x = np.column_stack((data["x"], data["y"][:, :2]))
    response = np.stack((data["y"][:, 2:], data["d"][:, 2:]), axis=-1)
    _, valid = inner_validation_split(data["assignment"], seed=96)
    expected = hashlib.sha256(np.asarray(valid, dtype="<i8").tobytes()).hexdigest()
    for kind in ("spline", "extratrees"):
        fit = comparison.TunedComparator(kind, seed=96, trees=4).fit(
            x, data["assignment"], response, data["t"][2:])
        assert fit.metadata_["inner_holdout_sha256"] == expected


def test_spline_fits_a_known_arm_specific_smooth_trajectory():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(759)
    x = rng.uniform(-1, 1, (160, 2))
    z = np.arange(160) % 2
    times = np.arange(5)
    y = (1 + x[:, :1] + times[None, :]*x[:, 1:2]/4 + z[:, None]*(2 + times[None, :]/3))
    d = np.broadcast_to(.3 + .2*z[:, None], y.shape)
    fit = comparison._SplineFit(1e-9).fit(x, z, np.stack((y, d), axis=-1), times)
    pred = fit.predict(x)
    for arm in (0, 1):
        truth = 1 + x[:, :1] + times[None, :]*x[:, 1:2]/4 + arm*(2 + times[None, :]/3)
        np.testing.assert_allclose(pred[:, :, arm, 0], truth, atol=2e-5)
        np.testing.assert_allclose(pred[:, :, arm, 1], .3+.2*arm, atol=2e-7)


def test_manifest_refuses_source_or_configuration_changes(tmp_path):
    path = tmp_path/"manifest.json"
    manifest = {"arguments": {"replications": 500}, "source_sha256": {"implementation": "abc"}}
    comparison.write_manifest(path, manifest)
    comparison.write_manifest(path, manifest)
    assert json.loads(path.read_text()) == manifest
    with pytest.raises(ValueError, match="manifest differs"):
        comparison.write_manifest(path, {**manifest, "source_sha256": {"implementation": "def"}})
    with pytest.raises(ValueError, match="manifest differs"):
        comparison.write_manifest(path, {**manifest, "arguments": {"replications": 501}})


def test_driver_keeps_identical_inference_and_records_a_learner_failure(tmp_path, monkeypatch):
    import hashlib
    class ArmMeanLearner:
        def fit(self, x, z, response, times):
            self.means = np.stack([response[z == arm].mean(axis=0) for arm in (0, 1)], axis=1)
            self.metadata_ = {"training_count": len(x)}
            return self

        def predict(self, x):
            return np.broadcast_to(self.means, (len(x), *self.means.shape)).copy()

    def factory(method, *, seed, config):
        if method == "crossfit_longbet":
            raise RuntimeError("Deliberate learner failure")
        return ArmMeanLearner

    monkeypatch.setattr(comparison, "learner_factory", factory)
    manifest_path = tmp_path/"manifest.json"
    manifest_path.write_text(json.dumps({"source_sha256": comparison.source_hashes()}))
    task = dict(scenario="long_abrupt", replicate=0, seed=137, n=120,
        output=str(tmp_path), folds=2, learner_config={},
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    comparison.run_one(task)
    path = tmp_path/"runs"/"long_abrupt_0000.json"
    record = json.loads(path.read_text())
    assert record["horizons"] == 16
    assert record["feature_count"] == 15
    spline, trees = [record["methods"][method] for method in comparison.CORRECTED_METHODS[:2]]
    assert spline["status"] == trees["status"] == "completed"
    assert spline["rows"] == trees["rows"]
    assert spline["fold_ids"] == trees["fold_ids"]
    assert np.asarray(spline["covariance"]).shape == (32, 32)
    assert len(spline["rows"]) == 48
    assert record["methods"]["crossfit_longbet"]["status"] == "failed"
    assert "Deliberate learner failure" in record["methods"]["crossfit_longbet"]["error"]
    assert not list(path.parent.glob("*.part"))


def test_zero_moment_nulls_preserve_undefined_cace_and_use_joint_covariance():
    from scipy.stats import norm
    effect = np.array([[.4, .1]])
    covariance = np.array([[[.09, .015], [.015, .01]]])
    tests = comparison.reference_null_tests(effect, covariance)
    assert [r["beta"] for r in tests] == list(comparison.NULL_MOMENT_BETAS)
    for test in tests:
        beta = test["beta"]
        statistic = (.4-.1*beta)/np.sqrt(.09+beta**2*.01-2*beta*.015)
        assert test["statistic"] == pytest.approx(statistic)
        assert test["p_value"] == pytest.approx(2*norm.sf(abs(statistic)))
    zero_data = comparison.generate_panel("zero", 49)
    assert np.isnan(zero_data["truth"]["wald"]).all()


@pytest.mark.parametrize("config,message", [
    ({"validation_fraction": .25}, "same 20%"),
    ({"length_scales": [0., 2.]}, "exactly three"),
])
def test_cli_rejects_unequal_inner_tuning_budgets(tmp_path, monkeypatch, capsys, config, message):
    monkeypatch.setattr(sys, "argv", ["orthogonal_comparison", "--output", str(tmp_path),
        "--replications", "100", "--longbet-config", json.dumps(config)])
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path/"manifest.json").exists()
