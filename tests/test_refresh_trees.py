"""Independent Gaussian and observed SUR/probit oracles for whole-tree moves."""
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.special import softmax
from scipy.stats import multivariate_normal

from bartz.grove import traverse_forest
from longbet import LongBetMulti, compute_ess
from longbet._multi_input import normalize_multi_inputs
from longbet._multi_state import init_multi_longbet, split_multi_chain_fields
from longbet._multi_step import multi_step
from longbet._refresh_trees import refresh_trees_step
from longbet._shared_forest import enable_x64
from test_shared_forest import _cfg, _panel


def _tiny_state(binary):
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], np.uint8)
    y = [np.array([0., 1., 0., 1.]) if binary else np.array([-.5, .8, -.1, .4]),
         np.array([.7, .1, -.8, .2])]
    cfg = _cfg(num_trees_pr=1, num_trees_trt=2, num_shared_trees=1,
        max_depth_pr=2, max_depth_trt=2, min_points_per_leaf_pr=1,
        min_points_per_leaf_trt=1, alpha_split_pr=.65, alpha_split_trt=.6,
        sample_beta=False, adaptive_coding=False, random_intercept=True,
        standardize=False)
    norm = normalize_multi_inputs([v.reshape(2, 2) for v in y],
        outcome=['binary' if binary else 'continuous', 'continuous'],
        outcome_names=None, config=cfg)
    state = init_multi_longbet(X_unified=jnp.asarray(X),
        unit_idx=jnp.array([0, 0, 1, 1], jnp.int32),
        time_idx=jnp.array([0, 1, 0, 1], jnp.int32),
        exposure_idx=jnp.array([0, 1, 0, 1], jnp.int32),
        z_vec=jnp.array([0., 1., 0., 1.], jnp.float32),
        max_split_mu=jnp.ones(2, jnp.uint8),
        max_split_nu=jnp.ones(2, jnp.uint8), norm_input=norm, config=cfg)
    children = []
    for m, s in enumerate(state.states):
        mu_offset, nu_offset = jnp.float32(.17-.09*m), jnp.float32(-.23+.12*m)
        s = eqx.tree_at(lambda c: (c.forest.offset, c.forest_nu.offset, c.mu_fit, c.nu_fit),
            s, (mu_offset, nu_offset, jnp.full_like(s.mu_fit, mu_offset),
                jnp.full_like(s.nu_fit, nu_offset)))
        alpha = jnp.float32(-.8 if m == 0 else 1.1)
        beta = jnp.array([.7, -.8], jnp.float32)
        b0, b1 = jnp.float32(.6), jnp.float32(.9)
        w = jnp.where(s.z_vec == 1, b1, b0)*beta[s.exposure_idx]
        response = s.z if binary and m == 0 else s.y
        residual = response-alpha*s.mu_fit-w*s.nu_fit-s.gamma[s.unit_idx]
        children.append(eqx.tree_at(lambda c: (c.alpha, c.beta, c.b0, c.b1,
            c.sigma2, c.error_cov_inv.value, c.sigma_gamma2, c.resid), s,
            (alpha, beta, b0, b1, jnp.float32(1. if m == 0 else .7),
             jnp.float32(1. if m == 0 else 1/.7), jnp.float32(.15+.05*m), residual)))
    state = eqx.tree_at(lambda s: (s.states, s.gamma_loadings), state,
        (tuple(children), jnp.array([[0., 0.], [.55, 0.]], jnp.float32)))
    return state, X, y


def _enumerate(state, X, y, binary):
    """81 configurations; dense observation covariance, no sampler integrals."""
    n = X.shape[1]
    B = np.eye(2)-np.asarray(state.gamma_loadings, float)
    noise = np.linalg.solve(B, np.diag([float(s.sigma2) for s in state.states]))@np.linalg.inv(B.T)
    R = np.kron(noise, np.eye(n))
    response = np.concatenate([np.asarray(s.y) for s in state.states])
    weights = [np.where(np.asarray(s.z_vec) == 1, float(s.b1), float(s.b0))
               *np.asarray(s.beta)[np.asarray(s.exposure_idx)] for s in state.states]
    offset = np.concatenate([float(s.alpha*s.forest.offset)+w*float(s.forest_nu.offset)
                             for s, w in zip(state.states, weights)])
    logps, means, covs, codes, integration_errors = [], [], [], [], []
    for config in itertools.product(range(3), repeat=4):
        designs, logprior = [], 0.
        for m, s in enumerate(state.states):
            local = []
            for j, forest in enumerate((s.forest, s.forest_nu)):
                rule = config[2*m+j]
                membership = np.ones((n, 1)) if rule == 0 else np.eye(2)[X[rule-1]]
                p = float(forest.p_nonterminal[1])
                logprior += np.log(1-p) if rule == 0 else np.log(p/2)
                multiplier = np.full(n, float(s.alpha)) if j == 0 else weights[m]
                local.append(multiplier[:, None]*membership/np.sqrt(float(forest.leaf_prior_cov_inv)))
            # Shared topology stays a stump, but its outcome coefficients and
            # random unit effects participate in every topology integral.
            local.append(weights[m][:, None]/np.sqrt(float(state.shared_forest.prior_precision[m, m])))
            local.append(np.eye(s.N_units)[np.asarray(s.unit_idx)]*np.sqrt(float(s.sigma_gamma2)))
            designs.append(np.column_stack(local))
        A = block_diag(*designs)
        F = A@A.T
        V = R+F
        if binary:
            cmean = offset[:n]+V[:n, n:]@np.linalg.solve(V[n:, n:], response[n:]-offset[n:])
            ccov = V[:n, :n]-V[:n, n:]@np.linalg.solve(V[n:, n:], V[n:, :n])
            sign = 2*y[0]-1
            probs = [multivariate_normal.cdf(np.zeros(n), mean=-sign*cmean,
                cov=sign[:, None]*ccov*sign[None, :], maxpts=1000000,
                abseps=1e-8, releps=1e-8, rng=np.random.default_rng(seed)) for seed in (546, 789)]
            integration_errors.append(abs(probs[0]-probs[1]))
            ll = multivariate_normal.logpdf(response[n:], mean=offset[n:], cov=V[n:, n:])+np.log(np.mean(probs))
        else:
            ll = multivariate_normal.logpdf(response, mean=offset, cov=V)
            means.append(offset+F@np.linalg.solve(V, response-offset))
            covs.append(F-F@np.linalg.solve(V, F))
        logps.append(logprior+ll)
        codes.append(sum(c*3**(3-i) for i, c in enumerate(config)))
    if integration_errors:
        assert max(integration_errors) < 2e-6
    return softmax(logps), np.asarray(means), np.asarray(covs), noise


@pytest.mark.parametrize('binary, capacity', [(False, 6), (False, 4), (True, 6)])
def test_shape_changing_posterior_matches_dense_gaussian_or_observed_probit(binary, capacity):
    state, X, y = _tiny_state(binary)
    expected, means, covariances, noise = _enumerate(state, X, y, binary)
    # With no ordinary sweep in this oracle, capacity4 confines each outcome
    # to at most one split private tree. The optional move must reject any
    # proposed overflow, never evaluate a model with a silently omitted leaf.
    if capacity == 4:
        supported = [all(3+int(c[2*m] > 0)+int(c[2*m+1] > 0) <= capacity
                         for m in range(2)) for c in itertools.product(range(3), repeat=4)]
        expected *= supported
        expected /= expected.sum()
    omega = np.linalg.inv(noise)
    def step(s, key):
        kz, kt = jax.random.split(key)
        if binary:
            child = s.states[0]
            score = omega[0, 0]*child.resid+omega[0, 1]*s.states[1].resid
            mean = child.z-score/omega[0, 0]
            sd = 1/jnp.sqrt(omega[0, 0])
            bound = -mean/sd
            latent = mean+sd*jax.random.truncated_normal(kz,
                jnp.where(y[0] == 1, bound, -jnp.inf),
                jnp.where(y[0] == 1, jnp.inf, bound), dtype=jnp.float64)
            latent = latent.astype(jnp.float32)
            child = eqx.tree_at(lambda c: (c.z, c.resid), child,
                (latent, child.resid+latent-child.z))
            s = eqx.tree_at(lambda t: t.states, s, (child, s.states[1]))
        s, info = refresh_trees_step(kt, s, capacity=capacity)
        code = jnp.int32(0)
        fitted = []
        for child in s.states:
            for forest in (child.forest, child.forest_nu):
                rule = jnp.where(forest.split_tree[0, 1] == 0, 0,
                                 1+forest.var_tree[0, 1].astype(jnp.int32))
                code = 3*code+rule
            w = jnp.where(child.z_vec == 1, child.b1, child.b0)*child.beta[child.exposure_idx]
            fitted.append(child.alpha*child.mu_fit+w*child.nu_fit+child.gamma[child.unit_idx])
        return s, (code, jnp.concatenate(fitted), info[..., 6].sum())
    with enable_x64(True):
        final, (codes, fitted, shapes) = jax.jit(lambda: jax.lax.scan(step,
            state, jax.random.split(jax.random.key(841 if binary else 842), 64000)))()
    codes, fitted = np.asarray(codes)[4000:], np.asarray(fitted)[4000:]
    actual = np.bincount(codes, minlength=81)/len(codes)
    eligible = expected > 0
    np.testing.assert_array_equal(actual[~eligible], 0.)
    ess = np.array([float(compute_ess((codes == c)[None, :], method='bulk'))
                    for c in np.flatnonzero(eligible)])
    mcse = np.sqrt(actual[eligible]*(1-actual[eligible])/ess)
    assert np.max(np.abs(actual[eligible]-expected[eligible])/(mcse+1/len(codes))) < 5
    assert np.sum(np.asarray(shapes)) > 1000
    if not binary:
        mean = expected@means
        cov = np.einsum('k,kij->ij', expected,
            covariances+means[:, :, None]*means[:, None, :])-np.outer(mean, mean)
        np.testing.assert_allclose(fitted.mean(0), mean, atol=.02)
        np.testing.assert_allclose(np.cov(fitted.T), cov, atol=.02, rtol=.1)
    else:
        np.testing.assert_array_equal(final.states[0].sigma2, 1.)


def _errors(state):
    errors = []
    for m, s in enumerate(state.states):
        fits = []
        for forest in (s.forest, s.forest_nu):
            ids = traverse_forest(s.X, forest.var_tree, forest.split_tree)
            fit = forest.offset+forest.leaf_unit*jnp.take_along_axis(forest.leaf_tree, ids, axis=1).sum(0)
            fits.append(fit)
        if state.shared_forest is not None:
            f = state.shared_forest
            ids = traverse_forest(s.X, f.var_tree, f.split_tree)
            shared_fit = jnp.take_along_axis(f.leaf_tree[:, m], ids, axis=1).sum(0)
            fits[1] += shared_fit
            errors.append(jnp.max(jnp.abs(shared_fit-f.fit[m])))
        mu, nu = fits
        y = s.z if s.outcome_type_str == 'binary' else s.y
        w = jnp.where(s.z_vec == 1, s.b1, s.b0)*s.beta[s.exposure_idx]
        residual = jnp.where(s.obs_mask, y-s.alpha*mu-w*nu-s.gamma[s.unit_idx], 0.)
        errors.extend([jnp.max(jnp.abs(mu-s.mu_fit)), jnp.max(jnp.abs(nu-s.nu_fit)),
                       jnp.max(jnp.abs(residual-s.resid))])
    return jnp.stack(errors)


def _assert_caches(state):
    """Rebuild counts, weighted sums and ancestor bounds independently."""
    for child in state.states:
        for name, weight in (('forest', child.prec_scale), ('forest_nu', child.prec_scale_nu)):
            f = getattr(child, name)
            X, maximum = np.asarray(child.X), np.asarray(f.max_split, int)
            for tree, (var, split) in enumerate(zip(np.asarray(f.var_tree), np.asarray(f.split_tree))):
                memberships = np.ones(X.shape[1], int)
                active = [(1, np.zeros(X.shape[0], int), maximum+1)]
                affluent = np.zeros(split.size, bool)
                while active:
                    node, lower, upper = active.pop()
                    if node < split.size and split[node]:
                        variable, cut = int(var[node]), int(split[node])
                        here = memberships == node
                        memberships[here] = 2*node+(X[variable, here] >= cut)
                        left, right = upper.copy(), lower.copy()
                        left[variable], right[variable] = cut, cut
                        active.extend(((2*node, lower.copy(), left), (2*node+1, right, upper.copy())))
                    elif node < split.size:
                        eligible = upper-lower > 1
                        if f.blocked_vars is not None:
                            eligible[np.asarray(f.blocked_vars)] = False
                        threshold = f.min_points_per_decision_node or 0
                        affluent[node] = (eligible.any() and float(f.p_nonterminal[node]) > 0
                                          and np.sum(memberships == node) >= threshold)
                np.testing.assert_array_equal(f.leaf_indices[tree], memberships)
                if f.count_tree is not None:
                    np.testing.assert_array_equal(f.count_tree[tree], np.bincount(memberships,
                        minlength=f.leaf_tree.shape[-1]))
                if f.prec_tree is not None:
                    weights = np.ones(X.shape[1]) if weight is None else np.asarray(weight)
                    expected = np.bincount(memberships, weights=weights, minlength=f.leaf_tree.shape[-1])
                    np.testing.assert_allclose(f.prec_tree[tree], expected, atol=3e-4, rtol=3e-6)
                np.testing.assert_array_equal(f.affluence_tree[tree], affluent)


@pytest.mark.parametrize('sharing', [False, True])
def test_chained_state_missingness_zero_negative_weights_and_following_sweep(sharing):
    x, z, y, types = _panel()
    y['h'][0, 0] = np.nan
    model = LongBetMulti(_cfg(num_chains=2, num_shared_trees=2 if sharing else 0,
                             sample_beta=False)).fit(y, x, z, outcome=types)
    children = []
    for s in model.state.states:
        b0, b1 = jnp.zeros_like(s.b0), -jnp.ones_like(s.b1)
        old_w = jnp.where(s.z_vec == 1, s.b1[:, None], s.b0[:, None])*s.beta[:, s.exposure_idx]
        new_w = jnp.where(s.z_vec == 1, b1[:, None], b0[:, None])*s.beta[:, s.exposure_idx]
        residual = jnp.where(s.obs_mask, s.resid+(old_w-new_w)*s.nu_fit, 0.)
        children.append(eqx.tree_at(lambda c: (c.b0, c.b1, c.resid), s, (b0, b1, residual)))
    old = eqx.tree_at(lambda s: s.states, model.state, tuple(children))
    with enable_x64(True):
        def step(s, key):
            return refresh_trees_step(key, s, capacity=32)
        changed, info = jax.jit(lambda: jax.lax.scan(step, old,
            jax.random.split(jax.random.key(357), 12)))()
        following = multi_step(jax.random.key(358), changed, jnp.int32(2))
        unchanged, skipped = refresh_trees_step(jax.random.key(359), changed, capacity=1)
    assert info.shape == (12, 2, 3, 2, 8)
    assert np.sum(np.asarray(info[..., 6])) > 0
    assert np.all(np.asarray(skipped[..., 0]) == 0)
    for a, b in zip(jax.tree.leaves(changed), jax.tree.leaves(unchanged)):
        np.testing.assert_array_equal(a, b)
    for candidate in (changed, following):
        per, constants = split_multi_chain_fields(candidate)
        errors = jax.vmap(lambda p: _errors(eqx.combine(p, constants)))(per)
        assert np.max(np.asarray(errors)) < 3e-4
        np.testing.assert_array_equal(candidate.states[0].sigma2, 1.)
    for before, after in zip(old.states, changed.states):
        for field in ('alpha', 'beta', 'b0', 'b1', 'z', 'sigma2', 'sigma_gamma2'):
            np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
        for forest in (after.forest, after.forest_nu):
            assert not np.any(np.asarray(forest.to_prune))
            np.testing.assert_array_equal(forest.move_node, 0)
    per, constants = split_multi_chain_fields(changed)
    for chain in range(changed.num_chains):
        _assert_caches(eqx.combine(jax.tree.map(lambda a: a[chain], per), constants))
    if sharing:
        np.testing.assert_array_equal(old.shared_forest.split_tree, changed.shared_forest.split_tree)
