"""Exhaustive assignment checks for research randomization AR inference."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/encouragement"))
from randomization_ar import randomization_ar, _statistics


def test_exact_sharp_null_test_size_over_entire_assignment_support():
    n, effect = 8, 2.
    baseline = np.array([-2., -.7, 0., .4, .9, 1.6, 2., 4.])
    d0 = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    d1 = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    pvalues = []
    for indices in combinations(range(n), 4):
        a = np.zeros(n, dtype=bool)
        a[list(indices)] = True
        d = np.where(a, d1, d0)
        result = randomization_ar((baseline + effect * d)[:, None], d[:, None],
                                  a[:, None], beta=[effect], method="exact")
        pvalues.append(result.table.p_value.iloc[0])
    pvalues = np.array(pvalues)
    for alpha in (.01, .05, .1, .2, .5):
        assert np.mean(pvalues <= alpha) <= alpha + 1e-14
    assert (pvalues < .2).any()  # check has power to rank random assignments


def test_monte_carlo_plus_one_reproducible_and_grid_is_not_full_inversion():
    a = np.arange(10) % 2
    d = a[:, None]
    y = (np.arange(10) + a)[:, None]
    first = randomization_ar(y, d, d, beta=[0, 1, 2], method="monte_carlo", permutations=199, seed=4)
    second = randomization_ar(y, d, d, beta=[0, 1, 2], method="monte_carlo", permutations=199, seed=4, block_size=7)
    np.testing.assert_array_equal(first.table.p_value, second.table.p_value)
    np.testing.assert_allclose(first.table.p_value, (1 + first.table.exceedances) / 200)
    assert (first.table.p_value > 0).all()
    assert first.metadata["left_tail"] == first.metadata["right_tail"] == "unknown"
    assert not first.metadata["finite_sample_exact_for_heterogeneous_late"]


def test_zero_stage_never_divides_by_first_stage_and_constant_null_accepts():
    a = (np.arange(6) % 2)[:, None]
    d = np.zeros_like(a)
    result = randomization_ar(np.ones_like(a), d, a, beta=[-1e12, 0, 1e12])
    np.testing.assert_array_equal(result.table.p_value, 1.)
    assert result.table.accepted.all()
    assert result.metadata["inversion"] == "evaluated grid only"


def test_degenerate_statistics_have_defined_tie_behavior():
    a = np.array([[0, 0, 1, 1]], dtype=bool)
    assert _statistics(np.zeros((4, 1)), a)[0, 0] == 0
    assert np.isposinf(_statistics(np.array([[0], [0], [1], [1]]), a)[0, 0])


def test_public_infinite_ties_and_affine_invariance():
    z = np.array([[0], [0], [1], [1]])
    d = np.zeros_like(z)
    result = randomization_ar(z, d, z, beta=[0])
    assert np.isposinf(result.table.statistic.iloc[0])
    assert result.table.p_value.iloc[0] == 2 / 6
    other = randomization_ar(1e6 + 3 * z, d, z, beta=[0])
    np.testing.assert_allclose(result.table.p_value, other.table.p_value)


@pytest.mark.parametrize("kwargs", [dict(design="cluster"), dict(method="wrong"),
                                    dict(beta=[]), dict(seed=None),
                                    dict(max_enumerations=1, method="exact")])
def test_unsupported_requests_rejected(kwargs):
    a = (np.arange(6) % 2)[:, None]
    with pytest.raises(ValueError):
        randomization_ar(a, a, a, **{"beta": [1], **kwargs})
