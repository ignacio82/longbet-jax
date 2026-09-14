"""Encouragement wrapper: composition, matched targets, honest ratios, archive replay."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from longbet import LongBetConfig, LongBetEncourage
from longbet._encourage_model import (
    EncouragementDraws, EncouragementPrediction, adoption_distribution, offer_effect_on_outcome,
)


def encouragement_panel(seed=918, n=12, t=4):
    """A small panel with organic adoption, offer-driven adoption and an adoption jump in y."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    z = np.zeros((n, t)); z[: n // 2, 1:] = 1
    hazard = np.where(z == 1, 0.6, 0.15)
    hit = rng.uniform(size=(n, t)) < hazard
    d = np.zeros((n, t))
    for i in range(n):
        first = np.flatnonzero(hit[i])
        if first.size:
            d[i, first[0]:] = 1
    d[0] = 1  # adopted from the first period
    d[1] = 0  # never adopts
    exposure = np.cumsum(d, axis=1)
    y = rng.normal(size=(n, t)) + 1.5 * np.log1p(exposure) + x[:, :1] + rng.normal(size=(n, 1))
    return dict(y=y, d=d, z=z, x=x, t=np.array([1, 2, 4, 5], dtype=float))


@pytest.fixture(scope="module")
def fitted():
    p = encouragement_panel()
    fit = LongBetEncourage(num_chains=2, num_burnin=4, num_sweeps=8, n_skip=1, num_trees_pr=2,
                           num_trees_trt=2, max_depth_pr=3, max_depth_trt=3, min_points_per_leaf_pr=2,
                           min_points_per_leaf_trt=2, random_seed=5)
    return fit.fit(p["y"], p["d"], p["z"], p["x"], t=p["t"])


def test_constructor_validation():
    assert LongBetEncourage().config == LongBetConfig()
    assert LongBetEncourage(num_chains=2).config.num_chains == 2
    with pytest.raises(ValueError, match="outcome must be"):
        LongBetEncourage(outcome="ordinal")
    with pytest.raises(ValueError, match="outcome=..."):
        LongBetEncourage(LongBetConfig(outcome="binary"))
    with pytest.raises(TypeError, match="config"):
        LongBetEncourage({"num_chains": 2})


def test_both_equations_are_fitted_as_designed(fitted):
    assert fitted.outcome_model.config.outcome == "continuous"
    assert fitted.adoption_model.config.outcome == "binary"
    events = LongBetEncourage._first_adoption_events(fitted._data["d"])
    # Cells after adoption leave the risk set; the adoption period itself is an event.
    assert np.isnan(events[0, 1:]).all() and events[0, 0] == 1
    assert (events[1] == 0).all()
    assert fitted.metadata["engine"] == "adoption_clock"
    assert fitted.metadata["inference_version"] == "adoption_clock_v1"


def test_wrapper_preserves_target_diagnostics_and_joint_draws(fitted):
    pred = fitted.predict(groups=np.tile(["a", "b"], 6), block_size=7)
    assert pred.group_labels == ("all", "a", "b")
    assert pred.draws["itt_y"].shape == (3, 3, 2, 8)
    np.testing.assert_allclose(pred.draws["wald"], pred.draws["itt_y"] / pred.draws["itt_d"])
    assert pred.horizons.tolist() == [1, 3, 4]
    assert set(pred.reference.group) == {"all", "a", "b"}
    assert len(pred.stability()) == 27
    assert pred.metadata["target_aligned"]
    assert pred.metadata["calibration_status"] == "not_established"
    assert pred.table.query("quantity == 'wald'").posterior_mean.isna().all()
    assert pred.wald().posterior_median.isna().all()  # short traces cannot pass ESS
    assert pred.wald(require_convergence=False).posterior_median.notna().all()
    stage = pred.first_stage(practical_threshold=.03, min_ess=1, max_rhat=100)
    np.testing.assert_allclose(stage.posterior_probability_above_threshold,
                               (pred.draws["itt_d"] > .03).mean(axis=(-2, -1)).ravel())
    assert pred.comparison().disagreement.isna().all()
    # Blocks are an implementation detail: the standardized draws do not depend on them.
    whole = fitted.predict(groups=np.tile(["a", "b"], 6))
    for name in whole.draws:
        np.testing.assert_allclose(whole.draws[name], pred.draws[name], rtol=1e-4, atol=1e-6)


def test_missing_group_arm_is_retained(fitted):
    groups = np.where(fitted._data["z"][:, -1], "encouraged_only", "controls_only")
    pred = fitted.predict(groups=groups)
    ref = pred.reference.query("group != 'all'")
    assert ref.itt_y.isna().all()
    assert set(ref.wald_reason) == {"missing_group_arm"}
    assert np.isfinite(pred.draws["itt_y"]).all()


def test_roundtrip_original_inputs_and_metadata(fitted, tmp_path):
    path = tmp_path / "encourage.npz"
    fitted.save(path)
    loaded = LongBetEncourage.load(path)
    assert loaded.metadata == fitted.metadata
    assert loaded.config == fitted.config
    for name, values in fitted._data.items():
        np.testing.assert_array_equal(loaded._data[name], values)
        assert not loaded._data[name].flags.writeable
    before, after = fitted.predict(block_size=5), loaded.predict(block_size=5)
    for name in before.draws:
        np.testing.assert_array_equal(before.draws[name], after.draws[name])
    pd.testing.assert_frame_equal(before.reference, after.reference)
    with np.load(path) as archive:
        original = {name: archive[name] for name in archive.files}
    assert {"outcome_model", "adoption_model"} <= set(original)
    changed = dict(original)
    changed["z"] = changed["z"].copy()
    changed["z"][0, 0] = 1
    np.savez(path, **changed)
    with pytest.raises(ValueError, match="digest mismatch"):
        LongBetEncourage.load(path)
    for field, value in (("archive_version", 999), ("engine", "reduced_form"),
                         ("n_units", 99), ("target", "treated_only")):
        meta = json.loads(str(original["metadata"]))
        meta[field] = value
        np.savez(path, **{**original, "metadata": json.dumps(meta)})
        with pytest.raises(ValueError, match="Invalid encouragement archive"):
            LongBetEncourage.load(path)


def synthetic_prediction(dy, dd):
    standardized = EncouragementDraws(
        draws={"outcome": np.asarray(dy, float)[None, None],
               "takeup": np.asarray(dd, float)[None, None]},
        group_labels=("all",), group_counts=np.array([8]), weights=np.full((1, 8), 1/8),
        periods=np.array([2]), period_indices=np.array([1]), horizons=np.array([1]),
        standardization="conditional", provenance="synthetic")
    return EncouragementPrediction(standardized, pd.DataFrame({"group": ["all"],
        "horizon": [1], "itt_y": [1.], "itt_d": [.1], "n_units": [8],
        "n_encouraged": [4], "n_control": [4]}), {})


def test_zero_and_negative_denominators_are_never_discarded():
    dd = np.array([[.1, -.1, .001, -.001, .2, -.2, .3, -.3]] * 2)
    pred = synthetic_prediction(np.ones_like(dd), dd)
    np.testing.assert_allclose(pred.draws["wald"][0, 0], 1/dd)
    frame = pred.wald(require_convergence=False)
    assert frame.posterior_lower.iloc[0] < 0 < frame.posterior_upper.iloc[0]
    dd[0, 0] = 0
    pred = synthetic_prediction(np.ones_like(dd), dd)
    assert np.isnan(pred.draws["wald"][0, 0, 0, 0])
    frame = pred.wald(require_convergence=False)
    assert frame.undefined_fraction.iloc[0] == 1/16
    assert np.isnan(frame.posterior_median.iloc[0])
    assert frame.posterior_status.iloc[0] == "undefined_draws"


def test_ratio_tail_diagnostics_and_no_mean_mcse():
    rng = np.random.default_rng(119)
    pred = synthetic_prediction(rng.normal(1, .1, (4, 600)), rng.normal(.2, .1, (4, 600)))
    frame = pred.stability()
    wald = frame.query("quantity == 'wald'").iloc[0]
    assert np.isfinite(wald.ess_tail)
    assert np.isfinite(wald.mcse_lower)
    assert np.isnan(wald.mcse_mean)
    assert np.isfinite(frame.query("quantity != 'wald'").mcse_mean).all()
    for kwargs in ({"min_ess": 0}, {"max_rhat": .9}):
        with pytest.raises(ValueError):
            pred.stability(**kwargs)


def test_bootstrap_uses_paired_sampling_covariance(fitted, monkeypatch):
    """A shared bootstrap perturbation cancels in the paired difference SE."""
    original = fitted.predict()
    index = {"i": 0}
    def predict_stub(self, **kwargs):
        if self is fitted:
            return original
        index["i"] += 1
        shift = float(index["i"])
        std = dataclasses.replace(original.standardized, draws={
            "outcome": original.draws["itt_y"] + shift,
            "takeup": original.draws["itt_d"] + shift})
        ref = original.reference.copy()
        ref[["itt_y", "itt_d"]] += shift
        return EncouragementPrediction(std, ref, original.metadata)
    monkeypatch.setattr(LongBetEncourage, "predict", predict_stub)
    monkeypatch.setattr(LongBetEncourage, "fit", lambda self, *args, **kwargs: self)
    result = fitted.bootstrap_comparison(replicates=4)
    np.testing.assert_allclose(result.table.difference_se, 0, atol=1e-14)
    assert (result.table.model_reference_covariance > 0).all()
    assert result.table.disagreement.isna().all()  # both short fit and replication count fail
    assert result.replicates.shape == (4, 1, 3, 2, 2)
    assert len(result.failures) == 4


def test_fit_validation_precedes_sampling():
    n = 4
    z = np.array([[0, 1], [0, 1], [0, 0], [0, 0]])
    fit = LongBetEncourage(outcome="binary")
    with pytest.raises(ValueError, match="binary outcome"):
        fit.fit(np.full((n, 2), .5), z, z, np.ones((n, 1)))
    with pytest.raises(ValueError, match="x must be"):
        fit.fit(z, z, z, np.ones((n, 2, 1)))
    with pytest.raises(RuntimeError, match="fitted"):
        fit.predict()
    with pytest.raises(RuntimeError, match="fitted"):
        fit.predict_conditional()


def test_fractional_calendar_and_covariate_roundtrip(fitted, tmp_path):
    data = fitted._data
    t = np.array([.1, 1.1, 3.1, 4.1])
    model = LongBetEncourage(fitted.config)
    model.fit(data["y"], data["d"], data["z"], data["x"], t, x_trt=data["x"][:, :1])
    before = model.predict()
    np.testing.assert_array_equal(before.periods, t[1:])
    cond = model.predict_conditional(x=data["x"][:3], x_trt=data["x"][:3, :1])
    assert np.all(np.isfinite(cond.citt_y.mean))
    with pytest.raises(ValueError, match="x_trt"):
        model.predict_conditional(x=data["x"][:3])
    path = tmp_path / "fractional.npz"
    model.save(path)
    loaded = LongBetEncourage.load(path)
    after = loaded.predict()
    np.testing.assert_array_equal(after.periods, t[1:])
    np.testing.assert_array_equal(before.draws["itt_y"], after.draws["itt_y"])
    np.testing.assert_array_equal(loaded._data["x_trt"], data["x"][:, :1])


def test_bootstrap_refuses_unavailable_reference_uncertainty(fitted, monkeypatch):
    pred = fitted.predict()
    pred.reference.loc[0, "itt_y_se"] = np.nan
    monkeypatch.setattr(LongBetEncourage, "predict", lambda self, **kwargs: pred)
    monkeypatch.setattr(LongBetEncourage, "fit", lambda self, *args, **kwargs: self)
    checks = pred.stability()
    checks["diagnostics_passed"] = True
    monkeypatch.setattr(EncouragementPrediction, "stability", lambda self, **kwargs: checks)
    comparison = fitted.bootstrap_comparison(replicates=100)
    assert not comparison.metadata["original_reference_inference_available"]
    assert comparison.table.disagreement.isna().all()
    assert set(comparison.failures.reason) == {"unavailable_reference_inference"}


def test_binary_outcome_contrasts_are_probability_differences(fitted, tmp_path):
    data = fitted._data
    y = (data["y"] > np.median(data["y"])).astype(float)
    fit = LongBetEncourage(fitted.config, outcome="binary").fit(y, data["d"], data["z"], data["x"], data["t"])
    assert fit.outcome_model.config.outcome == "binary"
    pred = fit.predict()
    assert np.all(np.abs(pred.draws["itt_y"]) <= 1)
    assert np.all(np.abs(pred.draws["itt_d"]) <= 1)
    cond = fit.predict_conditional(x=data["x"][:3])
    assert np.all(np.abs(cond.citt_y.draws) <= 1)
    path = tmp_path / "binary.npz"
    fit.save(path)
    loaded = LongBetEncourage.load(path)
    assert loaded.metadata == fit.metadata
    np.testing.assert_array_equal(loaded.predict().draws["itt_y"], pred.draws["itt_y"])


def test_predict_conditional_in_and_out_of_sample(fitted):
    cond_in = fitted.predict_conditional()
    n_in, t_post = fitted._data["x"].shape[0], fitted.predict().horizons.shape[0]
    assert cond_in.citt_y.mean.shape == (n_in, t_post)
    assert cond_in.citt_d.mean.shape == (n_in, t_post)
    assert cond_in.cace.median.shape == (n_in, t_post)
    assert np.all(np.isfinite(cond_in.citt_y.mean))
    assert np.all(np.isfinite(cond_in.cace.median))
    n_new = 15
    rng = np.random.default_rng(42)
    x_new = rng.normal(size=(n_new, fitted._data["x"].shape[1]))
    cond_out = fitted.predict_conditional(x=x_new)
    assert cond_out.citt_y.mean.shape == (n_new, t_post)
    assert cond_out.citt_d.mean.shape == (n_new, t_post)
    assert cond_out.cace.median.shape == (n_new, t_post)
    cum = cond_out.cumulative_lift()
    assert cum.shape == (n_new, cond_out.citt_y.draws.shape[-1])
    prob = cond_out.breakeven_probability(cost=1.0)
    assert prob.shape == (n_new,) and np.all((prob >= 0.0) & (prob <= 1.0))
    policy = cond_out.optimal_policy(cost=1.0, hurdle=0.5)
    assert policy.shape == (n_new,) and policy.dtype == bool
    assert np.isfinite(cond_out.policy_value(cost=1.0, hurdle=0.5))
    with pytest.raises(ValueError, match="columns"):
        fitted.predict_conditional(x=x_new[:, :1])
    with pytest.raises(ValueError, match="shape"):
        fitted.predict_conditional(x=x_new, z=np.zeros((n_new, 2)))


def test_monotonic_first_stage_and_cace_floor(fitted):
    cond_mono = fitted.predict_conditional(monotonic_first_stage=True)
    assert np.all(cond_mono.citt_d.draws >= 0.0)
    cond_raw = fitted.predict_conditional(monotonic_first_stage=False)
    assert cond_raw.citt_d.draws.shape == cond_mono.citt_d.draws.shape
    np.testing.assert_allclose(cond_mono.cace.draws,
                               cond_mono.citt_y.draws / (cond_mono.citt_d.draws + 0.02))
    with pytest.raises(ValueError, match="cace_stabilization"):
        fitted.predict_conditional(cace_stabilization=0)


def test_knapsack_policy_and_budget_constraints(fitted):
    cond = fitted.predict_conditional()
    n_units = fitted._data["x"].shape[0]
    policy_cap = cond.knapsack_policy(cost=0.5, capacity=3)
    assert policy_cap.shape == (n_units,) and policy_cap.dtype == bool and np.sum(policy_cap) <= 3
    assert np.sum(cond.knapsack_policy(cost=1.0, budget=2.5)) <= 2
    assert np.sum(cond.knapsack_policy(cost=2.0, budget=1.0)) == 0
    assert np.sum(cond.knapsack_policy(cost=1.0, capacity=2, budget=5.0)) <= 2
    for metric in ["expected_net_value", "certainty_adjusted", "breakeven_probability"]:
        assert np.sum(cond.knapsack_policy(cost=0.5, capacity=2, ranking_metric=metric)) <= 2
    with pytest.raises(ValueError, match="Unknown ranking_metric"):
        cond.knapsack_policy(cost=0.5, ranking_metric="unknown")
    assert np.isfinite(cond.policy_value(cost=0.5, budget=2.0, capacity=2))


def test_principal_strata_estimation(fitted):
    cond = fitted.predict_conditional()
    df = cond.principal_strata()
    assert set(df["stratum"].unique()) == {"complier", "always_taker", "never_taker"}
    for h in df["horizon"].unique():
        sub = df[df["horizon"] == h]
        np.testing.assert_allclose(sub["prob_mean"].sum(), 1.0, atol=0.05)
        np.testing.assert_allclose(sub["count_mean"].sum(), sub["n_total"].iloc[0], atol=0.6)
    summary_txt = cond.strata_summary(horizon=0)
    for word in ("Principal Strata Estimates", "Complier", "Always-Taker", "Never-Taker"):
        assert word in summary_txt


def _manual_paths(model, x, z, t, intercepts):
    """Adoption paths from the binary equation's draws, integrating a frailty by hand."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eta = np.asarray(model.predict(x, z, t=t, summary_only=False).yhats, dtype=np.float64)
    if len(x) == model.N_:
        gamma = np.asarray(model.trace.gamma).reshape(eta.shape[-1], -1)
        eta = eta - gamma.T[:, None, :]
    if intercepts is None:
        nodes, weights = np.polynomial.hermite_e.hermegauss(12)
        weights = weights / weights.sum()
        sd = np.sqrt(np.asarray(model.trace.sigma_gamma2).reshape(-1))
    else:
        nodes, weights, sd = np.zeros(1), np.ones(1), np.zeros(eta.shape[-1])
        eta = eta + np.asarray(intercepts).T[:, None, :]
    p_adopt = np.zeros_like(eta); stock = np.zeros_like(eta)
    for node, w in zip(nodes, weights):
        lam = ndtr(eta + node * sd)
        surv = np.cumprod(1 - lam, axis=1)
        first = lam * np.concatenate([np.ones((len(x), 1, eta.shape[-1])), surv[:, :-1]], axis=1)
        p_adopt += w * first; stock += w * (1 - surv)
    return p_adopt, stock


def test_composition_matches_direct_evaluation(fitted):
    """The offer effects are exactly the hazard-weighted exposure responses."""
    data = fitted._data
    x_new = np.asarray(data["x"][:5], dtype=np.float32) + 0.1
    t, n_periods = data["t"], data["z"].shape[1]
    z1 = np.zeros((5, n_periods), dtype=np.float32); z1[:, 1:] = 1
    z0 = np.zeros_like(z1)
    p1, s1 = adoption_distribution(fitted.adoption_model, x_new, z1, t)
    p0, s0 = adoption_distribution(fitted.adoption_model, x_new, z0, t)
    for (p, s), z in (((p1, s1), z1), ((p0, s0), z0)):
        p_ref, s_ref = _manual_paths(fitted.adoption_model, x_new, z, t, None)
        np.testing.assert_allclose(p, p_ref, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(s, s_ref, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(s, np.cumsum(p, axis=1), rtol=1e-6, atol=1e-8)
    effect = offer_effect_on_outcome(fitted.outcome_model, x_new, t, p1, p0)
    expected = np.zeros_like(effect)
    for a in range(n_periods):
        schedule = np.zeros((5, n_periods), dtype=np.float32); schedule[:, a:] = 1
        tau = np.asarray(fitted.outcome_model.predict(x_new, schedule, t=t, summary_only=False).tauhats)
        expected += (p1 - p0)[:, a, None, :] * tau
    np.testing.assert_allclose(effect, expected, rtol=1e-6, atol=1e-8)
    cond = fitted.predict_conditional(x=x_new)
    np.testing.assert_allclose(cond.citt_y.draws, effect[:, 1:, :], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(cond.always_takers.draws, s0[:, 1:, :], rtol=1e-6, atol=1e-8)


def test_population_prediction_uses_the_study_units_own_frailties(fitted):
    """predict() conditions on each unit's fitted intercept draws, in blocks or not."""
    data = fitted._data
    n, n_periods = data["z"].shape
    z1 = np.zeros((n, n_periods), dtype=np.float32); z1[:, 1:] = 1
    z0 = np.zeros_like(z1)
    gamma = np.asarray(fitted.adoption_model.trace.gamma).reshape(-1, n)
    p1, s1 = _manual_paths(fitted.adoption_model, data["x"], z1, data["t"], gamma)
    p0, s0 = _manual_paths(fitted.adoption_model, data["x"], z0, data["t"], gamma)
    citt_d = (s1 - s0)[:, 1:, :].mean(axis=0)
    pred = fitted.predict(block_size=4)
    np.testing.assert_allclose(pred.draws["itt_d"][0].reshape(3, -1).T, citt_d.T, rtol=1e-5, atol=1e-7)
