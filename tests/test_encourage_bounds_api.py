"""Unit tests for public partial identification bounds API."""
import numpy as np
import pytest

import longbet
from longbet import encouragement_bounds, identification_bounds, IdentificationBoundsResult


def test_encouragement_bounds_exports():
    assert "encouragement_bounds" in longbet.__all__
    assert "identification_bounds" in longbet.__all__
    assert "IdentificationBoundsResult" in longbet.__all__


def test_encouragement_bounds_execution():
    rng = np.random.default_rng(42)
    n = 100
    z = rng.binomial(1, 0.5, size=n)
    # Compliers: adopt if encouraged
    d = np.where(z == 1, rng.binomial(1, 0.7, size=n), rng.binomial(1, 0.2, size=n))
    y = np.where(d == 1, rng.binomial(1, 0.6, size=n), rng.binomial(1, 0.3, size=n))
    x = rng.binomial(1, 0.5, size=(n, 2))

    res = encouragement_bounds(y, d, z, x, delta=1.0, seed=1, chains=2, draws=200)
    assert isinstance(res, IdentificationBoundsResult)
    assert len(res.table) == 3
    assert set(res.table.estimand) == {"complier_encouragement", "treatment_z0", "treatment_z1"}
    assert (res.table.available_fraction > 0).all()
    # Upper bound should be greater than or equal to lower bound
    for row in res.table.itertuples():
        assert row.upper_median >= row.lower_median - 1e-6


def test_identification_bounds_array_input():
    p = np.array([[0.2, 0.7]])  # 1 group, p0=0.2, p1=0.7
    q = np.array([[[0.3, 0.5], [0.4, 0.8]]])  # 1 group, D=0/1, Z=0/1
    w = np.array([1.0])

    bounds = identification_bounds(p, q, w, delta=1.0)
    assert bounds["available"]
    assert float(bounds["complier_share"]) == pytest.approx(0.5)
    ce = bounds["complier_encouragement"]
    assert ce.shape == (2,)
    assert ce[0] <= ce[1]
