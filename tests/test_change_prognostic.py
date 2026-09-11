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

"""Independent enumerated posterior and state invariants for root changes."""
import itertools
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import softmax

from longbet._change_prognostic import root_move, rule_geometry, change_prognostic_step
from longbet._shared_forest import enable_x64
from longbet._forest_cache import current_forest_fit
from longbet._multi_state import split_multi_chain_fields
from longbet._multi_step import multi_step
from longbet import LongBetMulti, compute_ess
from test_shared_forest import _panel, _cfg


@pytest.mark.parametrize('variable_weights', [[.5, .5], [.8, .2]])
def test_tree_frequencies_match_independent_enumerated_posterior(variable_weights):
    X = np.array(list(itertools.product(range(3), repeat=2)), np.uint8).T
    X = np.tile(X, (1, 2))
    y = .4*(X[0].astype(float)-1)-.3*(X[1].astype(float)-1)
    w = np.linspace(.8, 1.7, X.shape[1])
    alpha, prior = -.7, 2.3
    pnt = np.array([0., .8, .35, .35, .2, .2, .2, .2]+[0.]*8)
    probabilities = np.asarray(variable_weights)
    trees, logps, encoded, means, covariances = [], [], [], [], []
    # Fixed shape: root and left child are decision nodes. Independent geometry
    # below works with predictor intervals and does not call bartz split_range.
    for rv, rc, cv, cc in itertools.product(range(2), (1, 2), range(2), (1, 2)):
        var, split = np.zeros(8, np.uint8), np.zeros(8, np.uint8)
        var[1:3], split[1:3] = (rv, cv), (rc, cc)
        bounds = {1: (np.zeros(2, int), np.full(2, 3))}
        log_prior, valid = 0., True
        for node in (1, 2, 3, 4, 5):
            low, high = bounds[node]
            available = high-low-1
            eligible = available > 0
            if node in (1, 2):
                v, cut = int(var[node]), int(split[node])
                if not (eligible[v] and low[v] < cut < high[v]):
                    valid = False; break
                log_prior += np.log(pnt[node])+np.log(probabilities[v])-np.log(probabilities[eligible].sum())-np.log(available[v])
                left, right = high.copy(), low.copy()
                left[v], right[v] = cut, cut
                bounds[2*node], bounds[2*node+1] = (low.copy(), left), (right, high.copy())
            else:
                log_prior += np.log(1-pnt[node]*eligible.any())
        if not valid:
            continue
        ids = np.where(X[rv] >= rc, 3, np.where(X[cv] >= cc, 5, 4))
        if any(np.sum(ids == leaf) < 1 for leaf in (3, 4, 5)):
            continue
        A = alpha*(ids[:, None] == np.array([3, 4, 5]))
        V = np.diag(1/w)+A@A.T/prior
        likelihood = -.5*(np.linalg.slogdet(V)[1]+y@np.linalg.solve(V, y))
        covariance = np.linalg.inv(prior*np.eye(3)+A.T@(w[:, None]*A))
        mean = covariance@A.T@(w*y)
        means.append(A@mean)
        covariances.append(A@covariance@A.T)
        with enable_x64(True):
            actual, _, _ = rule_geometry(jnp.asarray(var), jnp.asarray(split),
                jnp.array([2, 2], jnp.uint8), None, jnp.log(probabilities), jnp.asarray(pnt), 16)
        np.testing.assert_allclose(actual, log_prior, atol=1e-12)
        trees.append((var, split)); logps.append(log_prior+likelihood)
        encoded.append(rv*1000+rc*100+cv*10+cc)
    expected = softmax(logps)
    means = np.asarray(means)
    expected_mean = expected@means
    expected_covariance = np.einsum('i,ijk->jk', expected,
        np.asarray(covariances)+means[:, :, None]*means[:, None, :])-np.outer(expected_mean, expected_mean)
    initial = (*trees[0], np.zeros(16, np.float64))
    def step(carry, key):
        var, split, values = carry
        result = root_move(key, var, split, values, jnp.asarray(X), jnp.asarray(w),
            jnp.asarray(w*y), alpha, prior, jnp.array([2, 2], jnp.uint8), None, jnp.log(probabilities),
            jnp.asarray(pnt), 1, 1, 16)
        var, split, values = result[:3]
        code = var[1].astype(jnp.int32)*1000+split[1].astype(jnp.int32)*100+var[2].astype(jnp.int32)*10+split[2]
        return (var, split, values), (code, alpha*values[result[3]])
    with enable_x64(True):
        _, (draws, fitted) = jax.jit(lambda: jax.lax.scan(step, initial,
            jax.random.split(jax.random.key(805), 44000)))()
    draws = np.asarray(draws)[4000:]
    assert set(np.unique(draws)) == set(encoded)
    actual = np.array([np.mean(draws == code) for code in encoded])
    ess = np.array([float(compute_ess((draws == code)[None, :], method='bulk')) for code in encoded])
    mcse = np.sqrt(actual*(1-actual)/ess)
    assert np.max(np.abs(actual-expected)/mcse) < 5
    # Direct one-step detailed-balance check avoids relying on a long chain to
    # cross the less frequent rule configurations under unequal variable weights.
    trials = 16000
    tree_vars, tree_splits = [np.stack([t[i] for t in trees]) for i in (0, 1)]
    with enable_x64(True):
        def row(i):
            carry = (jnp.asarray(tree_vars)[i], jnp.asarray(tree_splits)[i], jnp.zeros(16, jnp.float64))
            keys = jax.random.split(jax.random.fold_in(jax.random.key(643), i), trials)
            codes = jax.vmap(lambda k: step(carry, k)[1][0])(keys)
            return (codes[:, None] == jnp.asarray(encoded)[None, :]).mean(axis=0)
        transition = np.asarray(jax.jit(lambda: jax.lax.map(row, jnp.arange(len(trees))))())
    flow = expected[:, None]*transition
    variance = expected[:, None]**2*transition*(1-transition)/trials
    error = np.sqrt(variance+variance.T)+1/trials
    assert np.max(np.abs(flow-flow.T)/error) < 5
    fitted = np.asarray(fitted)[4000:]
    np.testing.assert_allclose(fitted.mean(0), expected_mean, atol=.018)
    np.testing.assert_allclose(np.cov(fitted.T), expected_covariance, atol=.012, rtol=.08)


def test_state_caches_and_following_sweep_remain_consistent():
    x, z, y, types = _panel()
    y['h'][0, 0] = np.nan
    fit = LongBetMulti(_cfg(num_chains=2)).fit(y, x, z, outcome=types)
    old = fit.state
    with enable_x64(True):
        changed, info = change_prognostic_step(jax.random.key(381), old)
        following = multi_step(jax.random.key(382), changed, jnp.int32(2))
    def errors(state):
        result = []
        for m, child in enumerate(state.states):
            mu = current_forest_fit(child.forest)
            nu = current_forest_fit(child.forest_nu)+state.shared_forest.fit[m]
            response = child.z if child.outcome_type_str == 'binary' else child.y
            mean = child.alpha*mu+jnp.where(child.z_vec == 1, child.b1, child.b0)*child.beta[child.exposure_idx]*nu+child.gamma[child.unit_idx]
            result.extend([jnp.max(jnp.abs(mu-child.mu_fit)),
                jnp.max(jnp.abs(child.resid-jnp.where(child.obs_mask, response-mean, 0.)))])
        return jnp.stack(result)
    for candidate in (changed, following):
        per, constants = split_multi_chain_fields(candidate)
        err = jax.vmap(lambda p: errors(eqx.combine(p, constants)))(per)
        assert np.max(np.asarray(err)) < 3e-4
        np.testing.assert_array_equal(candidate.states[0].sigma2, 1.)
    for before, after in zip(old.states, changed.states):
        np.testing.assert_array_equal(before.beta, after.beta)
        np.testing.assert_array_equal(before.nu_fit, after.nu_fit)
        np.testing.assert_array_equal(before.z, after.z)
    assert info.shape == (2, 3, 2, 5)


def test_observed_binary_root_posterior_matches_probit_quadrature():
    from scipy.special import roots_hermitenorm, log_ndtr, logsumexp
    X = np.array([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], np.uint8)
    bits = np.array([0, 0, 1, 0, 1, 1])
    alpha, prior, offset = -.8, 1.4, .2
    pnt = np.array([0., .8, .3, .3])
    nodes, weights = roots_hermitenorm(120)
    means = offset+alpha*nodes/np.sqrt(prior)
    targets, codes = [], []
    for var, cut in ((0, 1), (0, 2), (1, 1)):
        ids = X[var] >= cut
        lp = -np.log(2)-np.log(2 if var == 0 else 1)
        for side in (False, True):
            ll = log_ndtr((2*bits[ids == side, None]-1)*means).sum(axis=0)
            lp += logsumexp(ll+np.log(weights))-.5*np.log(2*np.pi)
        targets.append(lp); codes.append(var*10+cut)
    expected = softmax(targets)
    def step(carry, key):
        var, split, leaves = carry
        kt, km = jax.random.split(key)
        ids = 2+(jnp.asarray(X)[var[1]] >= split[1]).astype(jnp.int32)
        mu = offset+alpha*leaves[ids]
        lower = jnp.where(bits == 1, -mu, -jnp.inf)
        upper = jnp.where(bits == 1, jnp.inf, -mu)
        latent = mu+jax.random.truncated_normal(kt, lower, upper, dtype=jnp.float64)
        result = root_move(km, var, split, leaves, jnp.asarray(X), jnp.ones(6, jnp.float64),
            latent-offset, alpha, prior, jnp.array([2, 1], jnp.uint8), None, None,
            jnp.asarray(pnt), 1, 1, 8)
        var, split, leaves = result[:3]
        return (var, split, leaves), var[1].astype(jnp.int32)*10+split[1]
    initial = (np.array([0, 0], np.uint8), np.array([0, 1], np.uint8), np.zeros(4))
    with enable_x64(True):
        _, draws = jax.jit(lambda: jax.lax.scan(step, initial,
            jax.random.split(jax.random.key(894), 44000)))()
    values = np.asarray(draws)[4000:]
    actual = np.array([np.mean(values == c) for c in codes])
    np.testing.assert_allclose(actual, expected, atol=.016)
