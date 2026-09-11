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

"""Observed SUR/probit posterior oracle for ALL-forest collapsed root moves."""
import itertools
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal

from longbet._collapsed_exposure import collapsed_exposure_step
from longbet._multi_input import normalize_multi_inputs
from longbet._multi_state import init_multi_longbet
from longbet._shared_forest import enable_x64
from longbet._forest_cache import current_forest_fit
from test_shared_forest import _cfg


def test_joint_root_posterior_matches_independent_observed_gaussian_probit_integral():
    X = np.array([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], np.uint8)
    bits = np.array([0, 0, 1, 0, 1, 1])
    yc = np.array([-1.2, .3, .1, .5, 1.3, .7])
    n = len(bits)
    cfg = _cfg(num_trees_pr=1, num_trees_trt=2, num_shared_trees=1,
        max_depth_pr=2, max_depth_trt=1, min_points_per_leaf_pr=1,
        min_points_per_leaf_trt=1, sample_beta=False, adaptive_coding=False,
        random_intercept=False, standardize=False)
    norm = normalize_multi_inputs([bits[None, :].astype(float), yc[None, :]],
        outcome=['binary', 'continuous'], outcome_names=None, config=cfg)
    state = init_multi_longbet(X_unified=jnp.asarray(X), unit_idx=jnp.zeros(n, jnp.int32),
        time_idx=jnp.arange(n, dtype=jnp.int32), exposure_idx=jnp.zeros(n, jnp.int32),
        z_vec=jnp.zeros(n, jnp.float32), max_split_mu=jnp.array([2, 1], jnp.uint8),
        max_split_nu=jnp.zeros(2, jnp.uint8), norm_input=norm, config=cfg)
    G = jnp.array([[0., 0.], [.6, 0.]], jnp.float32)
    variances = jnp.array([1., .7], jnp.float32)
    children = []
    for m, child in enumerate(state.states):
        forest = eqx.tree_at(lambda f: f.split_tree, child.forest,
                            child.forest.split_tree.at[0, 1].set(1))
        children.append(eqx.tree_at(lambda c: (c.forest, c.sigma2, c.error_cov_inv.value),
            child, (forest, variances[m], 1/variances[m])))
    state = eqx.tree_at(lambda s: (s.states, s.gamma_loadings), state, (tuple(children), G))
    yc = np.asarray(state.states[1].y)
    offsets = [float(s.forest.offset) for s in state.states]
    choices = [(0, 1), (0, 2), (1, 1)]
    log_weights = []
    numerical_checks = []
    B = np.eye(2)-np.asarray(G, float)
    noise = np.linalg.solve(B, np.diag(np.asarray(variances)))@np.linalg.inv(B.T)
    sign = 2*bits-1
    for a, b in itertools.product(choices, repeat=2):
        covariances = []
        log_prior = 0.
        for m, (variable, cut) in enumerate((a, b)):
            child = state.states[m]
            ids = X[variable] >= cut
            membership = (ids[:, None] == np.array([False, True]))
            mu_prior = float(child.alpha)**2/float(child.forest.leaf_prior_cov_inv)
            nu_prior = (1/float(child.forest_nu.leaf_prior_cov_inv)
                         + 1/float(state.shared_forest.prior_precision[m, m]))
            covariances.append(noise[m, m]*np.eye(n)+mu_prior*(membership@membership.T)+nu_prior*np.ones((n, n)))
            log_prior += -np.log(2)-np.log(2 if variable == 0 else 1)
        cross = noise[0, 1]*np.eye(n)
        conditional_mean = offsets[0]+cross@np.linalg.solve(covariances[1], yc-offsets[1])
        conditional_cov = covariances[0]-cross@np.linalg.solve(covariances[1], cross.T)
        probabilities = [multivariate_normal.cdf(np.zeros(n), mean=-sign*conditional_mean,
            cov=sign[:, None]*conditional_cov*sign[None, :], maxpts=3000000,
            abseps=1e-7, releps=1e-7, rng=np.random.default_rng(seed)) for seed in (741, 852)]
        numerical_checks.append(abs(probabilities[0]-probabilities[1]))
        log_weights.append(log_prior+multivariate_normal.logpdf(yc,
            mean=np.full(n, offsets[1]), cov=covariances[1])+np.log(np.mean(probabilities)))
    assert max(numerical_checks) < 2e-6
    expected = softmax(log_weights)
    omega = np.linalg.inv(noise)
    def step(s, key):
        kz, kt = jax.random.split(key)
        child = s.states[0]
        score = omega[0, 0]*child.resid+omega[0, 1]*s.states[1].resid
        mean = child.z-score/omega[0, 0]
        sd = 1/jnp.sqrt(omega[0, 0])
        bound = -mean/sd
        latent = mean+sd*jax.random.truncated_normal(kz,
            jnp.where(bits == 1, bound, -jnp.inf), jnp.where(bits == 1, jnp.inf, bound), dtype=jnp.float64)
        latent = latent.astype(jnp.float32)
        child = eqx.tree_at(lambda c: (c.z, c.resid), child,
                            (latent, child.resid+latent-child.z))
        s = eqx.tree_at(lambda s: s.states, s, (child, s.states[1]))
        s, _ = collapsed_exposure_step(kt, s, capacity=8, proposal='tree')
        code = [jnp.where(c.forest.var_tree[0, 1] == 1, 2,
                         c.forest.split_tree[0, 1].astype(jnp.int32)-1) for c in s.states]
        return s, 3*code[0]+code[1]
    with enable_x64(True):
        final, codes = jax.jit(lambda: jax.lax.scan(step, state,
            jax.random.split(jax.random.key(762), 44000)))()
    actual = np.bincount(np.asarray(codes)[4000:], minlength=9)/40000
    np.testing.assert_allclose(actual, expected, atol=.016)
    for m, child in enumerate(final.states):
        mu = current_forest_fit(child.forest)
        nu = current_forest_fit(child.forest_nu)+final.shared_forest.fit[m]
        response = child.z if m == 0 else child.y
        np.testing.assert_allclose(child.resid, response-mu-nu, atol=2e-4)
