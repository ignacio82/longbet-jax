"""Encouragement wrapper: matched targets, honest ratios, and archive replay."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from longbet import LongBetConfig, LongBetEncourage
from longbet._encourage_model import EncouragementPrediction
from longbet._encourage_predict import EncouragementDraws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(918)
    n, t = 12, 4
    x = rng.normal(size=(n, 2))
    z = np.zeros((n, t))
    z[:6, 1:] = 1
    d = z.copy()
    d[0] = 1  # always-taker
    d[1] = 0  # never-taker
    y = rng.normal(size=(n, t)) + 1.5 * d + x[:, :1]
    fit = LongBetEncourage(first_stage="lpm", num_chains=2, num_burnin=4,
        num_sweeps=8, n_skip=1, num_trees_pr=2, num_trees_trt=2,
        max_depth_pr=3, max_depth_trt=3, min_points_per_leaf_pr=2,
        min_points_per_leaf_trt=2, random_seed=5).fit(y, d, z, x, t=[1, 2, 4, 5])
    return fit


def test_proper_prior_and_explicit_first_stage():
    with pytest.raises(TypeError, match="first_stage"):
        LongBetEncourage()
    assert LongBetEncourage(first_stage="lpm").config.sigma_prior_a == 2
    assert LongBetEncourage(first_stage="probit").config.sigma_prior_b == 1
    with pytest.raises(ValueError, match="proper innovation"):
        LongBetEncourage(LongBetConfig(), first_stage="lpm")
    for kwargs in ({"sigma_prior_b": 0}, {"gamma_prior_a": 0}, {"gamma_prior_b": np.nan}):
        with pytest.raises(ValueError, match="proper"):
            LongBetEncourage(first_stage="lpm", **kwargs)
    with pytest.raises(ValueError, match="first_stage"):
        LongBetEncourage(first_stage="auto")


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
    assert fitted.predict(standardization="population").comparison().difference.isna().all()


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
    for name, values in fitted._data.items():
        np.testing.assert_array_equal(loaded._data[name], values)
        assert not loaded._data[name].flags.writeable
    before, after = fitted.predict(block_size=5), loaded.predict(block_size=5)
    for name in before.draws:
        np.testing.assert_array_equal(before.draws[name], after.draws[name])
    pd.testing.assert_frame_equal(before.reference, after.reference)
    with np.load(path) as archive:
        original = {name: archive[name] for name in archive.files}
    changed = dict(original)
    changed["z"] = changed["z"].copy()
    changed["z"][0, 0] = 1
    np.savez(path, **changed)
    with pytest.raises(ValueError, match="digest mismatch"):
        LongBetEncourage.load(path)
    for field, value in (("archive_version", 999), ("provenance", "unrelated"),
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
        cell_draws=None, cell_summaries={}, standardization="conditional", provenance="synthetic")
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
    fit = LongBetEncourage(first_stage="probit", outcome="binary")
    with pytest.raises(ValueError, match="binary outcome"):
        fit.fit(np.full((n, 2), .5), z, z, np.ones((n, 1)))
    with pytest.raises(ValueError, match="x must be"):
        fit.fit(z, z, z, np.ones((n, 2, 1)))
    with pytest.raises(RuntimeError, match="fitted"):
        fit.predict()


def test_fractional_calendar_first_period_and_covariate_roundtrip(fitted, tmp_path):
    import warnings
    data = fitted._data
    z = data["z"].copy()
    z[:6, 0] = 1
    t = np.array([.1, 1.1, 3.1, 4.1])
    model = LongBetEncourage(fitted.config, first_stage="lpm")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data["y"], data["d"], z, data["x"], t, x_trt=data["x"][:, :1])
    messages = [str(w.message) for w in caught]
    assert any("Retain both randomized arms" in message for message in messages)
    assert not any("dropping" in message for message in messages)
    before = model.predict()
    np.testing.assert_array_equal(before.periods, t)
    path = tmp_path / "fractional.npz"
    model.save(path)
    loaded = LongBetEncourage.load(path)
    after = loaded.predict()
    np.testing.assert_array_equal(after.periods, t)
    np.testing.assert_array_equal(before.draws["itt_y"], after.draws["itt_y"])
    np.testing.assert_array_equal(loaded._data["x_trt"], data["x"][:, :1])


def test_bootstrap_refuses_unavailable_reference_uncertainty(fitted, monkeypatch):
    pred = fitted.predict()
    pred.reference.loc[0, "itt_y_se"] = np.nan
    monkeypatch.setattr(LongBetEncourage, "predict", lambda self, **kwargs: pred)
    monkeypatch.setattr(LongBetEncourage, "fit", lambda self, *args, **kwargs: self)
    # Make diagnostics pass: reference support must be checked independently.
    checks = pred.stability()
    checks["diagnostics_passed"] = True
    monkeypatch.setattr(EncouragementPrediction, "stability", lambda self, **kwargs: checks)
    comparison = fitted.bootstrap_comparison(replicates=100)
    assert not comparison.metadata["original_reference_inference_available"]
    assert comparison.table.disagreement.isna().all()
    assert set(comparison.failures.reason) == {"unavailable_reference_inference"}


def test_binary_pair_does_not_claim_sur_innovation_coupling(fitted, tmp_path):
    data = fitted._data
    y = (data["y"] > np.median(data["y"])).astype(float)
    fit = LongBetEncourage(fitted.config, first_stage="probit", outcome="binary").fit(
        y, data["d"], data["z"], data["x"], data["t"])
    assert fit.model.sur_active  # requested option, distinct from the actual loading structure
    assert fit.metadata["innovation_coupling"] == "independent_binary"
    assert not fit.metadata["shared_treatment_partitions"]
    np.testing.assert_array_equal(fit.model.Gamma_draws, 0)
    pred = fit.predict()
    assert np.all(np.abs(pred.draws["itt_y"]) <= 1)
    assert np.all(np.abs(pred.draws["itt_d"]) <= 1)
    path = tmp_path / "binary.npz"
    fit.save(path)
    loaded = LongBetEncourage.load(path)
    assert loaded.metadata == fit.metadata
    np.testing.assert_array_equal(loaded.predict().draws["itt_y"], pred.draws["itt_y"])


def test_predict_conditional_in_and_out_of_sample(fitted):
    # In-sample conditional prediction
    cond_in = fitted.predict_conditional()
    n_in, t_post = fitted._data["x"].shape[0], fitted.predict().horizons.shape[0]
    assert cond_in.citt_y.mean.shape == (n_in, t_post)
    assert cond_in.citt_d.mean.shape == (n_in, t_post)
    assert cond_in.cace.median.shape == (n_in, t_post)
    assert np.all(np.isfinite(cond_in.citt_y.mean))
    assert np.all(np.isfinite(cond_in.cace.median))

    # Out-of-sample prediction on new units
    n_new = 15
    rng = np.random.default_rng(42)
    x_new = rng.normal(size=(n_new, fitted._data["x"].shape[1]))
    cond_out = fitted.predict_conditional(x=x_new)
    assert cond_out.citt_y.mean.shape == (n_new, t_post)
    assert cond_out.citt_d.mean.shape == (n_new, t_post)
    assert cond_out.cace.median.shape == (n_new, t_post)

    # Decision functions
    cum = cond_out.cumulative_lift()
    assert cum.shape == (n_new, cond_out.citt_y.draws.shape[-1])
    prob = cond_out.breakeven_probability(cost=1.0)
    assert prob.shape == (n_new,)
    assert np.all((prob >= 0.0) & (prob <= 1.0))
    policy = cond_out.optimal_policy(cost=1.0, hurdle=0.5)
    assert policy.shape == (n_new,)
    assert policy.dtype == bool
    val = cond_out.policy_value(cost=1.0, hurdle=0.5)
    assert np.isfinite(val)


def test_monotonic_first_stage(fitted):
    # Monotonicity enforced: all citt_d draws must be >= 0
    cond_mono = fitted.predict_conditional(monotonic_first_stage=True)
    assert np.all(cond_mono.citt_d.draws >= 0.0)
    assert np.all(cond_mono.citt_d.mean >= 0.0)

    # Monotonicity disabled: citt_d draws are unconstrained
    cond_raw = fitted.predict_conditional(monotonic_first_stage=False)
    assert cond_raw.citt_d.draws.shape == cond_mono.citt_d.draws.shape


def test_cace_adaptive_shrinkage(fitted):
    # Adaptive shrinkage
    cond_adapt = fitted.predict_conditional(cace_shrinkage="adaptive", shrinkage_lambda=1.0)
    assert np.all(np.isfinite(cond_adapt.cace.median))
    assert np.all(np.isfinite(cond_adapt.cace.draws))

    # Ridge shrinkage
    cond_ridge = fitted.predict_conditional(cace_shrinkage="ridge", cace_stabilization=0.05)
    assert np.all(np.isfinite(cond_ridge.cace.median))

    # None
    cond_none = fitted.predict_conditional(cace_shrinkage="none")
    assert np.all(np.isfinite(cond_none.cace.median))

    # Invalid shrinkage mode raises ValueError
    with pytest.raises(ValueError, match="Unknown cace_shrinkage"):
        fitted.predict_conditional(cace_shrinkage="invalid_mode")


def test_knapsack_policy_and_budget_constraints(fitted):
    cond = fitted.predict_conditional()
    n_units = fitted._data["x"].shape[0]

    # Capacity constraint: treating at most 3 accounts
    policy_cap = cond.knapsack_policy(cost=0.5, capacity=3)
    assert policy_cap.shape == (n_units,)
    assert policy_cap.dtype == bool
    assert np.sum(policy_cap) <= 3

    # Budget constraint: budget = 2.5 with cost = 1.0 => max 2 units
    policy_bud = cond.knapsack_policy(cost=1.0, budget=2.5)
    assert np.sum(policy_bud) <= 2

    # Insufficient budget for even 1 unit => 0 units treated
    policy_zero = cond.knapsack_policy(cost=2.0, budget=1.0)
    assert np.sum(policy_zero) == 0

    # Both capacity and budget
    policy_both = cond.knapsack_policy(cost=1.0, capacity=2, budget=5.0)
    assert np.sum(policy_both) <= 2

    # Different ranking metrics
    for metric in ["expected_net_value", "certainty_adjusted", "breakeven_probability"]:
        pol = cond.knapsack_policy(cost=0.5, capacity=2, ranking_metric=metric)
        assert np.sum(pol) <= 2

    with pytest.raises(ValueError, match="Unknown ranking_metric"):
        cond.knapsack_policy(cost=0.5, ranking_metric="unknown")

    # Policy value with budget/capacity
    val_budget = cond.policy_value(cost=0.5, budget=2.0, capacity=2)
    assert np.isfinite(val_budget)


def test_principal_strata_estimation(fitted):
    cond = fitted.predict_conditional()
    df = cond.principal_strata()
    assert isinstance(df, pd.DataFrame)
    assert set(df["stratum"].unique()) == {"complier", "always_taker", "never_taker"}
    assert "prob_mean" in df.columns
    assert "count_mean" in df.columns
    assert "n_total" in df.columns

    # Probabilities per horizon should sum to ~1.0
    for h in df["horizon"].unique():
        sub = df[df["horizon"] == h]
        total_p = sub["prob_mean"].sum()
        np.testing.assert_allclose(total_p, 1.0, atol=0.05)
        total_cnt = sub["count_mean"].sum()
        np.testing.assert_allclose(total_cnt, sub["n_total"].iloc[0], atol=0.6)

    # Human-readable summary
    summary_txt = cond.strata_summary(horizon=0)
    assert "Principal Strata Estimates" in summary_txt
    assert "Complier" in summary_txt
    assert "Always-Taker" in summary_txt
    assert "Never-Taker" in summary_txt


def test_longbet_encourage_hazard_first_stage(fitted):
    data = fitted._data
    cfg = LongBetConfig(
        num_chains=1,
        num_burnin=4,
        num_sweeps=8,
        num_trees_pr=2,
        num_trees_trt=2,
        random_seed=77,
        sigma_prior_a=2.0,
        sigma_prior_b=1.0,
    )
    model = LongBetEncourage(cfg, first_stage="hazard", outcome="continuous")
    t_eq = np.arange(1, data["y"].shape[1] + 1, dtype=float)
    model.fit(data["y"], data["d"], data["z"], data["x"], t=t_eq)

    pred = model.predict_conditional()
    assert pred.citt_d.mean.shape == (len(data["x"]), len(pred.horizons))
    assert pred.citt_y.mean.shape == (len(data["x"]), len(pred.horizons))
    assert pred.cace.median.shape == (len(data["x"]), len(pred.horizons))
    assert np.all(pred.citt_d.mean >= 0.0)
    assert np.all(np.isfinite(pred.cace.median))

    # Out-of-sample prediction
    rng = np.random.default_rng(91)
    x_new = rng.normal(size=(5, data["x"].shape[1]))
    pred_out = model.predict_conditional(x=x_new)
    assert pred_out.citt_d.mean.shape == (5, len(pred.horizons))
    assert pred_out.cace.median.shape == (5, len(pred.horizons))



