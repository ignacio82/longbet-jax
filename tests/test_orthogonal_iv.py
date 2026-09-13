"""Unit tests for Direct Bayesian Orthogonal IV (Two-Stage BCF)."""
import numpy as np
import pytest

from longbet import LongBetConfig, LongBetEncourage, LongBetOrthogonalIV


@pytest.fixture(scope="module")
def red_panel():
    rng = np.random.default_rng(777)
    n, t = 16, 4
    x = rng.normal(size=(n, 2))
    z = np.zeros((n, t))
    z[:8, 1:] = 1.0  # Encouragement starts at period 1

    # First stage compliance: subgroup x[:, 0] > 0 has high takeup
    subgroup = x[:, 0] > 0
    p1 = np.where(subgroup, 0.85, 0.15)
    p0 = 0.05
    d = np.zeros((n, t))
    for i in range(n):
        adopted = False
        for s in range(1, t):
            if adopted:
                d[i, s] = 1.0
            else:
                prob = p1[i] if z[i, s] == 1 else p0
                if rng.uniform() < prob:
                    d[i, s] = 1.0
                    adopted = True

    # True CACE = 4.0 for subgroup, 1.0 otherwise
    cace_true = np.where(subgroup[:, None], 4.0, 1.0)
    y = 15.0 + x[:, :1] + cace_true * d + rng.normal(scale=0.5, size=(n, t))
    t_vec = np.arange(1, t + 1)
    return {"y": y, "d": d, "z": z, "x": x, "t": t_vec, "subgroup": subgroup}


def test_orthogonal_iv_fit_and_predict_in_sample(red_panel):
    cfg = LongBetConfig(
        num_chains=1,
        num_burnin=4,
        num_sweeps=8,
        num_trees_pr=2,
        num_trees_trt=2,
        random_seed=42,
    )
    model = LongBetOrthogonalIV(cfg, min_compliance=0.02, monotonic_first_stage=True)
    model.fit(
        red_panel["y"],
        red_panel["d"],
        red_panel["z"],
        red_panel["x"],
        t=red_panel["t"],
    )

    pred = model.predict_conditional()
    n_units = len(red_panel["x"])
    t_post = 3  # periods 2, 3, 4

    assert pred.cace.mean.shape == (n_units, t_post)
    assert pred.cace.median.shape == (n_units, t_post)
    assert pred.citt_d.mean.shape == (n_units, t_post)
    assert pred.citt_y.mean.shape == (n_units, t_post)
    assert np.all(np.isfinite(pred.cace.mean))
    assert np.all(np.isfinite(pred.cace.median))
    assert np.all(pred.citt_d.draws >= 0.0)  # Monotonicity enforced


def test_orthogonal_iv_out_of_sample_and_knapsack(red_panel):
    cfg = LongBetConfig(
        num_chains=1,
        num_burnin=4,
        num_sweeps=8,
        num_trees_pr=2,
        num_trees_trt=2,
        random_seed=123,
    )
    model = LongBetOrthogonalIV(cfg)
    model.fit(
        red_panel["y"],
        red_panel["d"],
        red_panel["z"],
        red_panel["x"],
        t=red_panel["t"],
    )

    n_new = 10
    rng = np.random.default_rng(999)
    x_new = rng.normal(size=(n_new, red_panel["x"].shape[1]))

    pred_out = model.predict_conditional(x=x_new)
    assert pred_out.cace.mean.shape == (n_new, 3)
    assert pred_out.citt_y.mean.shape == (n_new, 3)

    # Knapsack policy on out-of-sample prediction
    policy_cap = pred_out.knapsack_policy(cost=2.0, capacity=3)
    assert len(policy_cap) == n_new
    assert policy_cap.dtype == bool
    assert np.sum(policy_cap) <= 3

    policy_bud = pred_out.knapsack_policy(cost=2.0, budget=4.5)
    assert np.sum(policy_bud) <= 2

    val = pred_out.policy_value(cost=2.0, capacity=3)
    assert np.isfinite(val)


def test_longbet_encourage_engine_orthogonal_iv(red_panel):
    cfg = LongBetConfig(
        num_chains=1,
        num_burnin=4,
        num_sweeps=8,
        num_trees_pr=2,
        num_trees_trt=2,
        random_seed=55,
    )
    enc_model = LongBetEncourage(
        cfg,
        first_stage="lpm",
        outcome="continuous",
        engine="orthogonal_iv",
    )
    enc_model.fit(
        red_panel["y"],
        red_panel["d"],
        red_panel["z"],
        red_panel["x"],
        t=red_panel["t"],
    )

    pred = enc_model.predict_conditional()
    assert pred.cace.median.shape == (len(red_panel["x"]), 3)
    assert np.all(np.isfinite(pred.cace.median))


def test_orthogonal_iv_input_validation(red_panel):
    with pytest.raises(ValueError, match="min_compliance must be positive"):
        LongBetOrthogonalIV(min_compliance=0.0)

    model = LongBetOrthogonalIV()
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict_conditional()


def test_orthogonal_iv_multi_chain(red_panel):
    cfg = LongBetConfig(
        num_chains=2,
        num_burnin=4,
        num_sweeps=8,
        num_trees_pr=2,
        num_trees_trt=2,
        random_seed=88,
    )
    model = LongBetOrthogonalIV(cfg, min_compliance=0.02, monotonic_first_stage=True)
    model.fit(
        red_panel["y"],
        red_panel["d"],
        red_panel["z"],
        red_panel["x"],
        t=red_panel["t"],
    )

    pred = model.predict_conditional()
    # Total draws should equal num_chains * num_sweeps = 2 * 8 = 16
    assert pred.cace.draws.shape == (len(red_panel["x"]), 3, 16)
    assert np.all(np.isfinite(pred.cace.median))
