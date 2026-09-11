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

"""Independent dense Gaussian and state checks for the collapsed exposure move."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from longbet._collapsed_exposure import (collapsed_statistics, collapsed_draw,
    collapsed_exposure_step, _pack, coding_sufficient_statistics,
    coding_statistics, elliptical_slice)
from longbet._shared_forest import enable_x64
from longbet._multi_state import split_multi_chain_fields
from longbet._forest_cache import current_forest_fit
from longbet import LongBetMulti
from test_shared_forest import _panel, _cfg


@pytest.mark.parametrize('unit_sd', [0., .7])
def test_integral_and_joint_draws_match_dense_observation_covariance(unit_sd):
    rng = np.random.default_rng(6187)
    n, k, units = 27, 6, 8
    A = rng.normal(size=(n, k))
    A[:, 1] = A[:, 0] + .02*A[:, 1]  # correlated forest columns
    A[:, -1] = 0.  # packed padding is an independent prior coordinate
    y = rng.normal(size=n)
    w = rng.uniform(.1, 4., n)
    w[[0, 6, 18]] = 0.  # genuinely absent likelihood cells
    unit = rng.integers(0, units-1, n)  # last unit has prior-only coefficients
    U = unit_sd*np.eye(units)[unit]
    design = np.column_stack((A, U))
    P = np.eye(k+units)+design.T@(w[:, None]*design)
    covariance = np.linalg.inv(P)
    mean = covariance@design.T@(w*y)
    keep = w > 0
    noise = np.diag(1/w[keep])
    V = noise+design[keep]@design[keep].T
    log_integral = -.5*(np.linalg.slogdet(V)[1]-np.linalg.slogdet(noise)[1]
                       + y[keep]@np.linalg.solve(V, y[keep]))
    with enable_x64(True):
        stats = collapsed_statistics(A, y, w, jnp.asarray(unit), unit_sd, units)
        np.testing.assert_allclose(stats[-1], log_integral, atol=2e-10)
        samples = 24000
        def one(key):
            x, g = collapsed_draw(key, stats)
            return jnp.concatenate((x, g))
        draws = np.asarray(jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(15), samples)))
    sd = np.sqrt(np.diag(covariance))
    assert np.max(np.abs(draws.mean(0)-mean)/(sd/np.sqrt(samples))) < 4.5
    assert np.max(np.abs(np.cov(draws, rowvar=False)-covariance)/np.outer(sd, sd)) < .05


@pytest.mark.parametrize('sharing', [False, True])
@pytest.mark.parametrize('proposal', ['pcn', 'coding', 'joint', 'tree'])
def test_actual_move_preserves_residuals_shared_fit_and_binary_identification(sharing, proposal):
    x, z, y, types = _panel()
    # Predecessor-closed supported missing cell (last continuous outcome).
    y['h'][0, 0] = np.nan
    cfg = _cfg(num_chains=2, num_shared_trees=2 if sharing else 0)
    model = LongBetMulti(cfg).fit(y, x, z, outcome=types)
    old = model.state
    with enable_x64(True):
        new, info = collapsed_exposure_step(jax.random.key(17), old, capacity=32, proposal=proposal)
        jax.block_until_ready(new)
    assert np.all(np.asarray(info[..., 0]) == 1)
    assert np.all(np.isfinite(info))
    np.testing.assert_array_equal(new.states[0].sigma2, 1.)
    np.testing.assert_array_equal(new.states[0].z, old.states[0].z)
    np.testing.assert_array_equal(new.gamma_loadings, old.gamma_loadings)

    def invariants(before, after):
        errors = []
        from bartz.grove._grove import is_actual_leaf
        for m, s in enumerate(after.states):
            mu = current_forest_fit(s.forest)
            nu = current_forest_fit(s.forest_nu)
            if sharing:
                shared = after.shared_forest
                actual = jnp.sum(jax.vmap(lambda v, idx: v[:, idx])(
                    shared.leaf_tree, shared.leaf_indices), axis=0)
                errors.append(jnp.max(jnp.abs(actual-shared.fit)))
                nu += actual[m]
            response = s.z if s.outcome_type_str == 'binary' else s.y
            fitted = s.alpha*mu+jnp.where(s.z_vec == 1, s.b1, s.b0)*s.beta[s.exposure_idx]*nu+s.gamma[s.unit_idx]
            errors += [jnp.max(jnp.abs(s.resid-jnp.where(s.obs_mask, response-fitted, 0))),
                       jnp.max(jnp.abs(s.mu_fit-mu)), jnp.max(jnp.abs(s.nu_fit-nu))]
            for name in ('forest', 'forest_nu'):
                a, b = getattr(before.states[m], name), getattr(s, name)
                active = jax.vmap(lambda sp: is_actual_leaf(sp, add_bottom_level=True))(a.split_tree)
                errors.append(jnp.max(jnp.abs(jnp.where(active, 0., b.leaf_tree-a.leaf_tree))))
                if proposal != 'tree' or name != 'forest':
                    errors.append(jnp.max(jnp.abs(b.split_tree.astype(jnp.int32)-a.split_tree)))
        return jnp.stack(errors)
    old_per, constants = split_multi_chain_fields(old)
    new_per, _ = split_multi_chain_fields(new)
    errors = jax.vmap(lambda a, b: invariants(eqx.combine(a, constants), eqx.combine(b, constants)))(old_per, new_per)
    assert np.max(np.asarray(errors)) < 3e-4


@pytest.mark.parametrize('proposal', ['pcn', 'coding', 'joint', 'tree'])
def test_overflow_is_identity_and_fixed_beta_stays_fixed(proposal):
    x, z, y, types = _panel()
    model = LongBetMulti(_cfg(sample_beta=False, random_intercept=False)).fit(y, x, z, outcome=types)
    with enable_x64(True):
        unchanged, info = collapsed_exposure_step(jax.random.key(21), model.state, capacity=1, proposal=proposal)
        updated, info2 = collapsed_exposure_step(jax.random.key(22), model.state, capacity=32, proposal=proposal)
    assert np.all(np.asarray(info[..., 0]) == 0)
    for before, after in zip(jax.tree.leaves(model.state), jax.tree.leaves(unchanged)):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(model.state.states, updated.states):
        np.testing.assert_array_equal(before.beta, after.beta)
        np.testing.assert_array_equal(before.gamma, after.gamma)
    assert np.all(np.asarray(info2[..., 0]) == 1)


def test_coding_cache_with_offsets_matches_direct_integral():
    rng = np.random.default_rng(103)
    n, p, units = 29, 7, 6
    base = rng.normal(size=(n, p))
    is_trt = np.arange(p) % 2 == 0
    beta = rng.normal(size=n)
    z = rng.integers(2, size=n)
    response = rng.normal(size=n)
    precision = rng.uniform(.3, 2., size=n)
    precision[[2, 7, 12]] = 0
    unit = jnp.asarray(rng.integers(units-1, size=n))
    with enable_x64(True):
        sufficient = coding_sufficient_statistics(base, is_trt, -.8, beta, z,
            response, .37, precision, unit, .6, units)
        for coding in rng.normal(size=(15, 2)):
            w = coding[z]*beta
            expected = collapsed_statistics(base*np.where(is_trt, w[:, None], -.8),
                response-.37*w, precision, unit, .6, units)
            actual = coding_statistics(jnp.asarray(coding), sufficient)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=2e-10)


def test_collapsed_coding_slice_and_redraw_match_dense_posterior_quadrature():
    """Independent observation-covariance oracle, including coding/leaf coupling."""
    from scipy.special import logsumexp
    rng = np.random.default_rng(492)
    n, p, units = 7, 4, 3
    base = rng.normal(size=(n, p))*.4
    is_trt = np.array([False, True, False, True])
    beta = np.array([.7, .7, .7, -.3, .8, 1.2, .9])
    z = np.array([0, 0, 0, 1, 1, 1, 1])
    unit = np.array([0, 1, 2, 0, 1, 2, 2])
    y = np.array([.2, -.1, .8, -.3, 1., .4, -.7])
    precision = np.array([1., 1.4, .7, 2., 0., 1.3, 1.7])
    offset, sd = .21, .3
    bsd = .7
    grid = np.linspace(-4., 4., 201)
    b0, b1 = np.meshgrid(grid, grid, indexing='ij')
    points = np.column_stack((b0.ravel(), b1.ravel()))
    w = points[:, z]*beta
    A = base[None, :, :]*np.where(is_trt[None, None, :], w[:, :, None], 1.)
    U = sd*np.eye(units)[unit]
    keep = precision > 0
    F = np.concatenate((A, np.broadcast_to(U, (len(points), n, units))), axis=2)[:, keep]
    response = (y[None, :]-offset*w)[:, keep]
    V = np.diag(1/precision[keep])+F@F.transpose(0, 2, 1)
    solve = np.linalg.solve(V, response[..., None])[..., 0]
    logp = -.5*(np.linalg.slogdet(V)[1]+np.sum(response*solve, axis=1)
                  + np.sum(points**2, axis=1)/bsd**2)
    mass = np.exp(logp-logsumexp(logp))
    conditional_x = (F.transpose(0, 2, 1)@solve[..., None])[..., 0]
    expected_mean = mass@points
    expected_second = np.einsum('n,ni,nj->ij', mass, points, points)
    expected_coupling = mass@(points[:, 0]*conditional_x[:, 1])
    with enable_x64(True):
        sufficient = coding_sufficient_statistics(base, is_trt, 1., beta, z,
            y, offset, precision, jnp.asarray(unit), sd, units)
        def step(b, key):
            kd, ks, kx = jax.random.split(key, 3)
            b, stats, _, _ = elliptical_slice(ks, b,
                bsd*jax.random.normal(kd, (2,), jnp.float64),
                lambda v: coding_statistics(v, sufficient))
            x, _ = collapsed_draw(kx, stats)
            return b, (b, b[0]*x[1])
        _, (draws, coupling) = jax.jit(lambda: jax.lax.scan(step,
            jnp.array([.2, -.1], jnp.float64), jax.random.split(jax.random.key(6172), 26000)))()
    draws, coupling = np.asarray(draws)[2000:], np.asarray(coupling)[2000:]
    np.testing.assert_allclose(draws.mean(0), expected_mean, atol=.025)
    np.testing.assert_allclose(draws.T@draws/len(draws), expected_second, rtol=.06, atol=.015)
    np.testing.assert_allclose(coupling.mean(), expected_coupling, atol=.02)


@pytest.mark.parametrize('proposal', ['pcn', 'coding', 'joint', 'interweave'])
def test_mixed_observed_posterior_with_sampled_gp_matches_quadrature(proposal):
    """Actual probit/Gaussian likelihood and beta*nu product, not latent oracle.

    Constant trees permit integrating beta's Gaussian prior by Gauss-Hermite
    quadrature and both outcome means by a separate dense grid. The sampler
    still draws beta, mu, shared/private nu, and truncated binary latent values.
    """
    from scipy.special import log_ndtr, logsumexp, roots_hermitenorm
    from scipy.stats import norm as normal
    from longbet._multi_input import normalize_multi_inputs
    from longbet._multi_state import init_multi_longbet
    from longbet._multi_step import multi_single_step

    yb, yc = np.array([[0., 1., 0., 0., 1.]]), np.array([[-1.2, .3, .1, .5, 1.3]])
    cfg = _cfg(num_trees_pr=1, num_trees_trt=2, num_shared_trees=1,
        max_depth_pr=1, max_depth_trt=1, min_points_per_leaf_pr=1,
        min_points_per_leaf_trt=1, sample_beta=True, adaptive_coding=proposal != 'pcn',
        random_intercept=False, standardize=False)
    norm = normalize_multi_inputs([yb, yc], outcome=['binary', 'continuous'],
                                  outcome_names=None, config=cfg)
    state = init_multi_longbet(X_unified=jnp.zeros((1, 5), jnp.uint8),
        unit_idx=jnp.zeros(5, jnp.int32), time_idx=jnp.arange(5, dtype=jnp.int32),
        exposure_idx=jnp.zeros(5, jnp.int32), z_vec=jnp.zeros(5, jnp.float32),
        max_split_mu=jnp.zeros(1, jnp.uint8), max_split_nu=jnp.zeros(1, jnp.uint8),
        norm_input=norm, config=cfg)
    G = jnp.array([[0., 0.], [.6, 0.]], jnp.float32)
    v = jnp.array([1., .7], jnp.float32)
    def fix(s):
        children = tuple(eqx.tree_at(lambda c: (c.sigma2, c.error_cov_inv.value),
            c, (v[m], 1/v[m])) for m, c in enumerate(s.states))
        return eqx.tree_at(lambda t: (t.states, t.gamma_loadings), s, (children, G))
    state = fix(state)
    nodes, weights = roots_hermitenorm(100)
    grid = np.linspace(-5., 5., 651)
    priors, beta2_priors, coding2_priors = [], [], []
    for m, s in enumerate(state.states):
        beta2 = nodes**2*float(s.K_chol[0, 0])**2
        mu_var = 1/float(s.forest.leaf_prior_cov_inv)
        trt_var = (1/float(s.forest_nu.leaf_prior_cov_inv)
                   + 1/float(state.shared_forest.prior_precision[m, m]))
        if s.adaptive_coding:
            coding2 = nodes**2*float(s.sigma_b)**2
            product2 = (beta2[:, None]*coding2[None, :]).ravel()
            node_weights = np.outer(weights, weights).ravel()/(2*np.pi)
            beta2 = np.broadcast_to(beta2[:, None], (len(nodes), len(nodes))).ravel()
            coding2 = np.broadcast_to(coding2[None, :], (len(nodes), len(nodes))).ravel()
        else:
            product2, node_weights = beta2, weights/np.sqrt(2*np.pi)
            coding2 = np.ones_like(beta2)
        density = normal.pdf(grid[:, None], loc=float(s.forest.offset),
                             scale=np.sqrt(mu_var+trt_var*product2)[None, :])
        priors.append(density@node_weights)
        beta2_priors.append((density@(node_weights*beta2))/(density@node_weights))
        coding2_priors.append((density@(node_weights*coding2))/(density@node_weights))
    a, b = np.meshgrid(grid, grid, indexing='ij')
    logp = np.log(priors[0])[:, None]+np.log(priors[1])[None, :]
    var_c, cond_sd = .7+.6**2, np.sqrt(.7/(.7+.6**2))
    for bit, value in zip(yb[0], yc[0]):
        logp += normal.logpdf(value, loc=b, scale=np.sqrt(var_c))
        logp += log_ndtr((2*bit-1)*(a+.6/var_c*(value-b))/cond_sd)
    mass = np.exp(logp-logsumexp(logp))
    mean = np.array([np.sum(mass*a), np.sum(mass*b)])
    cov = np.array([[np.sum(mass*(a-mean[0])**2), np.sum(mass*(a-mean[0])*(b-mean[1]))],
                    [np.sum(mass*(a-mean[0])*(b-mean[1])), np.sum(mass*(b-mean[1])**2)]])
    expected_beta2 = np.array([np.sum(mass*beta2_priors[0][:, None]),
                              np.sum(mass*beta2_priors[1][None, :])])
    expected_coding2 = np.array([np.sum(mass*coding2_priors[0][:, None]),
                                np.sum(mass*coding2_priors[1][None, :])])
    def step(s, key):
        ko, kl, kc = jax.random.split(key, 3)
        s = fix(multi_single_step(jax.random.split(ko, 2), jax.random.split(kl, 2), s, jnp.int32(1)))
        s, info = collapsed_exposure_step(kc, s, capacity=4, proposal_scale=.7,
                                          proposal='coding' if proposal == 'interweave' else proposal)
        if proposal == 'interweave':
            from longbet._coding_interweave import coding_interweave_step
            s = coding_interweave_step(jax.random.fold_in(kc, 1), s)
        theta = jnp.stack([c.mu_fit[0]+c.b0*c.beta[0]*c.nu_fit[0] for c in s.states])
        return s, (theta, jnp.stack([c.beta[0]**2 for c in s.states]),
                   jnp.stack([c.b0**2 for c in s.states]), info)
    with enable_x64(True):
        _, (draws, b2, coding2, info) = jax.jit(lambda: jax.lax.scan(step, state,
            jax.random.split(jax.random.key(9842), 20000)))()
    draws, b2 = np.asarray(draws)[2000:], np.asarray(b2)[2000:]
    np.testing.assert_allclose(draws.mean(0), mean, atol=.035)
    np.testing.assert_allclose(np.cov(draws.T), cov, rtol=.12, atol=.012)
    np.testing.assert_allclose(b2.mean(0), expected_beta2, rtol=.12, atol=.06)
    np.testing.assert_allclose(np.asarray(coding2)[2000:].mean(0), expected_coding2, rtol=.12, atol=.06)
    expected_event = np.sum(mass[(a < 0) & (b > 0)])
    assert abs(np.mean((draws[:, 0] < 0) & (draws[:, 1] > 0))-expected_event) < .03
    assert np.all(np.asarray(info[..., 0]) == 1)
    assert np.all(np.mean(np.asarray(info[..., 1]), axis=0) > .1)
