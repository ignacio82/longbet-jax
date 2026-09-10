import numpy as np
import pytest
import inspect
from pathlib import Path
import yaml

from longbet import LongBetIVNuisance, crossfit_encouragement, longbet_iv
from longbet._iv import baseline_features


def panel():
    rng = np.random.default_rng(291)
    n = 48
    x = rng.normal(size=(n, 2))
    a = np.arange(n) % 2
    z = a[:, None] * np.array([0, 0, 1, 1])
    d = z.copy()
    y = x[:, :1] + rng.normal(size=(n, 4)) + d
    return y, d, z, x


def test_baseline_features_cannot_read_followup_outcomes_or_adoption():
    y, d, z, x = panel()
    expected = np.column_stack([x, y[:, :2], d[:, :2]])
    np.testing.assert_array_equal(baseline_features(y, d, z, x), expected)
    y[:, 2:] += 1000
    d[:, 2:] = 1
    np.testing.assert_array_equal(baseline_features(y, d, z, x), expected)


def test_vector_assignment_has_no_implicit_baseline():
    y, d, z, x = panel()
    np.testing.assert_array_equal(baseline_features(y[:, 2:], d[:, 2:], z[:, -1], x), x)
    np.testing.assert_array_equal(baseline_features(y[:, 2:], d[:, 2:], z[:, -1]),
                                  np.ones((len(y), 1)))


def test_frontdoor_matches_explicit_feature_and_inference_pipeline():
    y, d, z, x = panel()
    config = dict(baseline_trees=1, effect_trees=1, burnin=2, draws=3,
                  chains=1, length_scales=(0.,), interaction_partitions=False)
    actual = longbet_iv(y, d, z, x, learner_config=config, seed=7)
    expected = crossfit_encouragement(
        y, d, z, baseline_features(y, d, z, x), seed=7,
        learner_factory=lambda: LongBetIVNuisance(**config, seed=7 + 7919))
    np.testing.assert_array_equal(actual.estimates, expected.estimates)
    np.testing.assert_array_equal(actual.covariance, expected.covariance)
    assert actual.metadata["posterior_causal_intervals"] is False


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"seed": -1},
                                   {"learner_config": []}])
def test_invalid_frontdoor_options(kwargs):
    with pytest.raises(ValueError):
        longbet_iv(*panel()[:3], **kwargs)


def test_iv_frontdoor_contract():
    root = Path(__file__).resolve().parents[1]
    contract = yaml.safe_load((root / "contract/longbet-api.yaml").read_text())["crossfit_iv_api"]
    signature = inspect.signature(longbet_iv)
    assert set(signature.parameters) == set(contract["required"]) | set(contract["defaults"])
    for key, value in contract["defaults"].items():
        assert signature.parameters[key].default == value
    assert "export(longbet_iv)" in (root / "NAMESPACE").read_text()
