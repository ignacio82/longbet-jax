"""Unit and integration tests for Discrete-Time Hazard Adoption Model with Spike-and-Slab."""
import numpy as np
import pytest

import longbet
from longbet import HazardConfig, HazardAdoptionForest, HazardAdoptionResult, hazard_adoption_effects


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


def test_hazard_adoption_zero_stage_relevance_prob():
    d, z, x, t = make_hazard_data(n=80, t_len=4, scenario="zero", seed=202)
    config = HazardConfig(trees=2, cutpoints=3, prior_inclusion_prob=0.5)
    forest = HazardAdoptionForest(config)
    forest.fit(d, z, x, t=t, seed=303, chains=2, burnin=30, draws=50)

    summary = forest.summary()
    # On zero stage, credible interval for stock ITT should cover zero
    for row in summary.table.itertuples():
        assert row.stock_itt_lower <= 0.15
        assert row.stock_itt_upper >= -0.15


def test_hazard_adoption_effects_convenience_function():
    d, z, x, t = make_hazard_data(n=40, t_len=3, scenario="acceleration", seed=404)
    res = hazard_adoption_effects(d, z, x, t=t, seed=505, chains=2, burnin=20, draws=30)
    assert isinstance(res, HazardAdoptionResult)
    assert len(res.table) == 2
