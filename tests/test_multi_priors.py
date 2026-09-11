# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reject undefined joint targets before sampling or reusing old archives."""

import numpy as np
import pytest

from longbet import LongBet, LongBetConfig, LongBetMulti


@pytest.mark.parametrize("name", ["sigma_prior_a", "sigma_prior_b"])
@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "1"])
def test_invalid_variance_hyperparameters(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a finite nonnegative scalar"):
        LongBetConfig(**{name: value})


@pytest.mark.parametrize("shape,scale", [(0, 0), (0, 1), (1, 0)])
@pytest.mark.parametrize("shared_trees", [0, 1])
def test_coupled_fit_requires_proper_prior_before_sampling(monkeypatch, shape, scale, shared_trees):
    from longbet import _multi_model

    def fail_if_sampling_starts(*args, **kwargs):
        pytest.fail("an improper joint target reached the sampler")

    monkeypatch.setattr(_multi_model, "run_multi_longbet_mcmc", fail_if_sampling_starts)
    rng = np.random.default_rng(422)
    x = rng.normal(size=(6, 2))
    z = np.zeros((6, 4)); z[:3, 2:] = 1
    y = {"continuous": rng.normal(size=z.shape),
         "binary": (rng.normal(size=z.shape) > .2).astype(float)}
    model = LongBetMulti(sigma_prior_a=shape, sigma_prior_b=scale,
                        num_shared_trees=shared_trees)
    with pytest.raises(ValueError, match="require a proper innovation-variance prior"):
        model.fit(y, x, z, outcome={name: name for name in y})
    assert model.trace is None


@pytest.mark.parametrize("settings,outcomes", [
    ({}, ("binary", "binary")),
    ({"sur": False}, ("binary", "continuous")),
    ({"sur_prior_var": 0}, ("continuous", "continuous")),
    ({}, ("continuous",)),
    ({"sigma_prior_a": 2, "sigma_prior_b": 1}, ("binary", "continuous")),
])
def test_valid_targets_and_uncoupled_reductions_are_preserved(settings, outcomes):
    LongBetConfig(**settings).validate_multi_variance_prior(outcomes)


def test_binary_only_archives_and_children_keep_fixed_unit_variances(tmp_path):
    rng = np.random.default_rng(13)
    y = {name: rng.integers(0, 2, size=(6, 4)).astype(float) for name in ("a", "b")}
    x = rng.normal(size=(6, 2))
    z = np.zeros((6, 4)); z[:3, 2:] = 1
    model = LongBetMulti(num_burnin=2, num_sweeps=2, num_chains=1,
                        num_trees_pr=2, num_trees_trt=2,
                        max_depth_pr=2, max_depth_trt=2).fit(y, x, z, outcome="binary")
    path = tmp_path / "binary.npz"
    model.save(path)
    loaded = LongBetMulti.load(path)
    assert loaded.config.sigma_prior_a == loaded.config.sigma_prior_b == 0
    for name in y:
        child = tmp_path / f"{name}.npz"
        loaded[name].save(child)
        scalar = LongBet.load(child)
        np.testing.assert_array_equal(scalar.trace.sigma2, 1)
        assert scalar.multi_origin["outcomes"] == ["binary", "binary"]
