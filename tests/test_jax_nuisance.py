import numpy as np
import pytest

from longbet._iv_nuisance import LongBetIVNuisanceConfig
from longbet._jax_nuisance import JAXLongBetIVNuisance


def make_test_data(seed=42, n=40, h=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    z = np.tile([0, 1], n // 2)
    adopt_time = rng.integers(1, h + 3, size=n)
    d = (np.arange(1, h + 1)[None, :] >= adopt_time[:, None]).astype(float)
    y = 2.0 + 1.5 * x[:, :1] + 3.0 * d + rng.normal(scale=0.2, size=(n, h))
    times = np.arange(1, h + 1, dtype=float)
    return x, z, np.stack([y, d], axis=-1), times


def test_jax_nuisance_fit_predict():
    x, z, responses, times = make_test_data()
    cfg = LongBetIVNuisanceConfig(
        baseline_trees=2, effect_trees=2, burnin=10, draws=20, chains=1,
        length_scales=(2.0,), linear_ancova_backbone=True, adoption_model="hazard"
    )
    model = JAXLongBetIVNuisance(cfg).fit(x, z, responses, times)
    pred = model.predict(x)
    assert pred.shape == (len(x), len(times), 2, 2)
    assert np.all((pred[..., 1] >= 0.0) & (pred[..., 1] <= 1.0))
    # Monotonic adoption check
    diffs = np.diff(pred[..., 1], axis=1)
    assert np.all(diffs >= -1e-12)
