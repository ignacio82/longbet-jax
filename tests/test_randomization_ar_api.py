"""Unit tests for public randomization AR API."""
import numpy as np
import pytest

import longbet
from longbet import randomization_ar, RandomizationARResult


def test_randomization_ar_exports():
    assert "randomization_ar" in longbet.__all__
    assert "RandomizationARResult" in longbet.__all__


def test_randomization_ar_exact():
    # 6 units, 3 encouraged
    z = np.array([[0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1]])
    d = np.array([[0, 0], [0, 0], [0, 1], [0, 0], [0, 1], [0, 1]])
    y = np.array([[1.0, 1.2], [2.0, 2.1], [1.5, 3.0], [0.8, 1.1], [1.7, 3.2], [2.2, 3.8]])

    result = randomization_ar(y, d, z, beta=[0.0, 1.0], method="exact")
    assert isinstance(result, RandomizationARResult)
    assert len(result.table) == 2  # 2 betas, 1 post-encouragement horizon
    assert set(result.table.columns) == {
        "period", "horizon", "beta", "statistic", "p_value",
        "accepted", "exceedances", "mc_p_lower", "mc_p_upper",
    }
    assert (result.table.p_value >= 0).all() and (result.table.p_value <= 1).all()
    assert result.metadata["randomization_method"] == "exact"
    assert result.metadata["evaluated_assignments"] == 20  # comb(6, 3) = 20


def test_randomization_ar_monte_carlo():
    n = 20
    z = np.zeros((n, 2))
    z[n // 2:, 1] = 1
    d = z.copy()
    y = d * 2.0 + np.random.default_rng(42).normal(size=(n, 2))

    result = randomization_ar(y, d, z, beta=[1.0, 2.0, 3.0], method="monte_carlo", permutations=499, seed=123)
    assert isinstance(result, RandomizationARResult)
    assert len(result.table) == 3
    assert result.metadata["randomization_method"] == "monte_carlo"
    assert result.metadata["evaluated_assignments"] == 499
