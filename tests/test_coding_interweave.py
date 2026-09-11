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

"""Prior invariance and actual state checks for coding/GP interweaving."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from longbet._coding_interweave import interweave_coding_gp, coding_interweave_step
from longbet._shared_forest import enable_x64
from longbet import LongBetMulti
from test_shared_forest import _panel, _cfg


def test_transformation_preserves_independent_joint_prior_and_effective_weights():
    # Independent exact prior draws followed by one conditional transition must
    # retain the full joint prior. Wrong beta/log-magnitude Jacobians change it.
    d, samples, sd = 5, 24000, .7
    grid = np.arange(d)
    K = .4+.8**np.abs(grid[:, None]-grid[None, :])
    L = np.linalg.cholesky(K)
    rng = np.random.default_rng(6881)
    beta = rng.normal(size=(samples, d))@L.T
    coding = sd*rng.normal(size=(samples, 2))
    with enable_x64(True):
        new_beta, new_coding, evaluations = jax.jit(jax.vmap(
            lambda k, b, c: interweave_coding_gp(k, b, c, jnp.asarray(L), sd)))(
                jax.random.split(jax.random.key(1123), samples), jnp.asarray(beta), jnp.asarray(coding))
    b, c = np.asarray(new_beta), np.asarray(new_coding)
    groups = (np.arange(d) > 0).astype(int)
    np.testing.assert_allclose(b*c[:, groups], beta*coding[:, groups], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(b.mean(0), 0., atol=.035)
    np.testing.assert_allclose(c.mean(0), 0., atol=.02)
    np.testing.assert_allclose(np.cov(b.T), K, atol=.055)
    np.testing.assert_allclose(np.cov(c.T), sd**2*np.eye(2), atol=.02)
    np.testing.assert_allclose((b[:, :, None]*c[:, None, :]).mean(0), 0., atol=.025)
    np.testing.assert_allclose((b[:, :, None]**2*c[:, None, :]**2).mean(0),
                               sd**2*np.diag(K)[:, None]*np.ones((1, 2)), atol=.045)
    assert np.all(np.isfinite(evaluations))
    assert np.mean(np.abs(c-coding)) > .1


@pytest.mark.parametrize('sample_beta,adaptive_coding', [(True, True), (False, True), (True, False)])
def test_state_transition_preserves_fits_and_respects_fixed_options(sample_beta, adaptive_coding):
    x, z, y, types = _panel()
    fit = LongBetMulti(_cfg(num_chains=2, sample_beta=sample_beta,
                           adaptive_coding=adaptive_coding)).fit(y, x, z, outcome=types)
    with enable_x64(True):
        updated = coding_interweave_step(jax.random.key(666), fit.state)
    for old, new in zip(fit.state.states, updated.states):
        if not (sample_beta and adaptive_coding):
            for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
                np.testing.assert_array_equal(a, b)
        else:
            w0 = np.where(old.z_vec == 1, old.b1[:, None], old.b0[:, None])*np.asarray(old.beta)[:, old.exposure_idx]
            w1 = np.where(new.z_vec == 1, new.b1[:, None], new.b0[:, None])*np.asarray(new.beta)[:, new.exposure_idx]
            np.testing.assert_allclose(w0, w1, rtol=2e-7, atol=2e-7)
            np.testing.assert_allclose(old.resid, new.resid, atol=2e-6)
            np.testing.assert_array_equal(old.nu_fit, new.nu_fit)
            np.testing.assert_array_equal(old.mu_fit, new.mu_fit)
    np.testing.assert_array_equal(updated.gamma_loadings, fit.state.gamma_loadings)
