"""Full SUR likelihood tests, including an analytical joint posterior oracle."""
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from longbet import LongBetConfig
from longbet._state import init_longbet
from longbet._step import longbet_single_step
from longbet._sur import conditional_residual


@pytest.mark.parametrize("binary_rows", [0, 2])
def test_conditional_precision_matches_marginal_gaussian(binary_rows):
    """Include empty cells, missing downstream outcomes and non-nested binaries."""
    rng = np.random.default_rng(103)
    M, n = 4, 9
    G = np.tril(rng.normal(0, .6, (M, M)), -1)
    G[:binary_rows] = 0
    v = np.array([.7, 1.3, .4, 2.])
    v[:binary_rows] = 1
    B = np.eye(M)-G
    Sigma = np.linalg.solve(B, np.diag(v)) @ np.linalg.inv(B.T)
    masks = np.arange(M)[:, None] < (np.arange(n) % (M+1))[None, :]
    if binary_rows == 2:
        masks[:, 0] = [False, True, False, False]
    raw = rng.normal(size=(M, n))
    # Masked raw values must not leak into a supported observed likelihood.
    raw[~masks] = 999.
    for m in range(M):
        resid, precision = conditional_residual(
            jnp.array(raw), jnp.array(masks), jnp.array(G), jnp.array(v), m)
        for cell in range(n):
            if not masks[m, cell]:
                assert float(resid[cell]) == 0
                assert float(precision[cell]) == 1
                continue
            idx = np.flatnonzero(masks[:, cell])
            P = np.linalg.inv(Sigma[np.ix_(idx, idx)])
            pos = list(idx).index(m)
            np.testing.assert_allclose(precision[cell], P[pos, pos], rtol=2e-6)
            np.testing.assert_allclose(resid[cell], (P@raw[idx, cell])[pos]/P[pos,pos], rtol=3e-6, atol=2e-6)


def _constant_state(y, observed, variance=1., binary=False, ridge_move=False, **overrides):
    """One constant mu leaf, with nu switched off via a fixed zero trajectory."""
    n = len(y)
    cfg = LongBetConfig(num_trees_pr=1, num_trees_trt=1,
        max_depth_pr=1, max_depth_trt=1,
        min_points_per_leaf_pr=1, min_points_per_leaf_trt=1,
        sample_beta=False, adaptive_coding=False, random_intercept=False,
        ridge_move=ridge_move, standardize=False, num_chains=1,
        outcome="binary" if binary else "continuous")
    cfg = dataclasses.replace(cfg, **overrides)
    st = init_longbet(X_unified=jnp.zeros((1,n),jnp.uint8),
        y=jnp.array(y,jnp.float32), unit_idx=jnp.zeros(n,jnp.int32),
        time_idx=jnp.arange(n), exposure_idx=jnp.zeros(n,jnp.int32),
        z_vec=jnp.zeros(n), obs_mask=jnp.array(observed),
        max_split_mu=jnp.zeros(1,jnp.uint8), max_split_nu=jnp.zeros(1,jnp.uint8),
        config=cfg)
    return eqx.tree_at(lambda s:(s.beta,s.sigma2),st,
        (jnp.zeros_like(st.beta),jnp.float32(variance)))


def test_weighted_tree_conditional_matches_conjugate_normal():
    y = np.array([-1., .2, 1.4, 2., .4, -.3])
    observed = np.array([True,True,False,True,True,True])
    precision = np.array([.2, 3., 1., .6, 4., 2.])
    st = _constant_state(y, observed, variance=1.7)
    # The mu leaf has variance 1.5**2. The conditional innovation variance
    # is NOT the structural 1.7 and must not be multiplied/divided twice.
    expected_var = 1/(1/2.25 + precision[observed].sum())
    expected_mean = expected_var*np.sum(y[observed]*precision[observed])

    @jax.jit
    def run(state):
        def step(s, key):
            updated = longbet_single_step(key, s, jnp.array(precision,jnp.float32))
            return updated, updated.mu_fit[0]
        return jax.lax.scan(step,state,jax.random.split(jax.random.key(322),5000))

    last, draws = run(st)
    np.testing.assert_allclose(np.mean(draws),expected_mean,atol=.025)
    np.testing.assert_allclose(np.var(draws),expected_var,rtol=.06)
    assert float(last.sigma2) == float(st.sigma2)
    # The conditional weights are temporary; preserve the original mask attrs.
    np.testing.assert_array_equal(last.prec_scale, st.prec_scale)
    np.testing.assert_allclose(last.resid, np.where(observed,y-last.mu_fit,0),atol=1e-4)


@pytest.mark.parametrize("missing", [False, True])
def test_actual_tree_kernel_matches_joint_gaussian_posterior(missing):
    """Use the real weighted forest updates, not a stand-in normal sampler.

    Hold Gamma and innovation variances fixed. Each mean is one constant BART
    leaf with a known Gaussian prior, making the joint posterior analytical.
    The old one-way algorithm fails this cross-outcome covariance oracle.
    """
    rng = np.random.default_rng(334)
    M, n = 3, 12
    G = np.array([[0.,0.,0.],[.8,0.,0.],[-.5,.7,0.]])
    v = np.array([.7,1.3,.4])
    B = np.eye(M)-G
    Sigma = np.linalg.solve(B,np.diag(v))@np.linalg.inv(B.T)
    y = rng.multivariate_normal([.3,-.4,.8],Sigma,n).T
    observed = np.ones((M,n),bool)
    if missing:
        observed[2,:4] = False
        observed[1:,:2] = False
        observed[:,0] = False
    y_clean = np.where(observed,y,0)
    states = tuple(_constant_state(y_clean[m],observed[m],v[m]) for m in range(M))
    P = np.eye(M)/2.25
    h = np.zeros(M)
    for cell in range(n):
        idx = np.flatnonzero(observed[:,cell])
        if not len(idx): continue
        Pcell = np.linalg.inv(Sigma[np.ix_(idx,idx)])
        P[np.ix_(idx,idx)] += Pcell
        h[idx] += Pcell@y[idx,cell]
    covariance = np.linalg.inv(P)
    mean = covariance@h

    @jax.jit
    def run(initial):
        def step(current,key):
            states = list(current)
            for m in range(M):
                raw = jnp.stack([s.resid for s in states])
                resid, prec = conditional_residual(raw,jnp.array(observed),jnp.array(G),jnp.array(v),m)
                offset = states[m].resid-resid
                view = eqx.tree_at(lambda s:s.resid,states[m],resid)
                upd = longbet_single_step(jax.random.fold_in(key,m),view,prec)
                states[m] = eqx.tree_at(lambda s:s.resid,upd,upd.resid+offset)
            return tuple(states),jnp.stack([s.mu_fit[0] for s in states])
        return jax.lax.scan(step,initial,jax.random.split(jax.random.key(33),10000))

    _, draws = run(states)
    draws = np.asarray(draws)[1000:]
    np.testing.assert_allclose(draws.mean(0),mean,atol=.025)
    np.testing.assert_allclose(np.cov(draws.T),covariance,rtol=.12,atol=.004)


def test_binary_latent_draw_uses_full_conditional_mean_and_variance():
    """A strong downstream residual changes even the first binary latent draw."""
    n = 10000
    labels = np.tile([0.,1.],n//2)
    binary = _constant_state(labels,np.ones(n,bool),binary=True)
    G = jnp.array([[0.,0.],[.8,0.]])
    raw = jnp.stack([binary.resid,jnp.full(n,.7)])
    resid,precision = conditional_residual(raw,jnp.ones((2,n),bool),G,jnp.array([1.,.4]),0)
    view = eqx.tree_at(lambda s:s.resid,binary,resid)
    updated = longbet_single_step(jax.random.key(862),view,precision)
    latent = np.asarray(updated.z)
    mean = float((binary.z-resid)[0]); sd = float(jax.lax.rsqrt(precision[0]))
    assert sd < 1
    for label in (0.,1.):
        lo,hi = (-np.inf,-mean/sd) if label==0 else (-mean/sd,np.inf)
        expected_mean,expected_var = scipy.stats.truncnorm.stats(lo,hi,loc=mean,scale=sd,moments='mv')
        a = latent[labels==label]
        assert np.all(a<=0) if label==0 else np.all(a>0)
        np.testing.assert_allclose(a.mean(),expected_mean,atol=.03)
        np.testing.assert_allclose(a.var(),expected_var,rtol=.08)
    assert float(updated.sigma2)==1


def test_scalar_treatment_draw_refreshes_changing_precision():
    """Actual scalar kernel, including weights becoming zero and changing units.

    Conditional on the updated mu, a constant nu leaf has a known Gaussian
    law. Standardize its draws by that law, changing the externally fixed beta
    each iteration. Stale cached leaf precision gives the wrong variance.
    """
    y = jnp.array([-1., .2, 1.4, 2., .4, -.3])
    variance = 1.7
    initial = _constant_state(y, np.ones(len(y), bool), variance)
    amplitudes = jnp.array([0., .2, .75, 3.])

    @jax.jit
    def run(state):
        def step(s, inputs):
            key, i = inputs
            w = amplitudes[i % len(amplitudes)]
            # This fixture has fixed b0=b1=1 and one exposure bin.
            s = eqx.tree_at(lambda s: (s.beta, s.sigma2, s.resid), s,
                (jnp.full_like(s.beta, w), jnp.float32(variance),
                 y-s.mu_fit-w*s.nu_fit))
            updated = longbet_single_step(key, s)
            v = 1/(1 + len(y)*w**2/variance)  # nu leaf prior variance = 1
            mean = v*w*jnp.sum(y-updated.mu_fit)/variance
            score = (updated.nu_fit[0]-mean)/jnp.sqrt(v)
            return updated, score
        return jax.lax.scan(step, state,
            (jax.random.split(jax.random.key(882), 8000), jnp.arange(8000)))

    _, scores = run(initial)
    for group in range(len(amplitudes)):
        z = np.asarray(scores)[group::len(amplitudes)]
        np.testing.assert_allclose(z.mean(), 0, atol=.07)
        np.testing.assert_allclose(z.var(), 1, rtol=.09)


@pytest.mark.parametrize("weight", [0., 2e-6, -2e-6, .2])
def test_tiny_weights_preserve_the_actual_treatment_fit(weight):
    """R/w can be millions while a prior leaf update is order one."""
    from bartz.grove import evaluate_forest

    y = jnp.array([-1., .2, 1.4, 2., .4, -.3])
    st = _constant_state(y, np.ones(len(y), bool))
    st = eqx.tree_at(lambda s: s.beta, st, jnp.full_like(st.beta, weight))
    for seed in range(4):
        st = longbet_single_step(jax.random.key(seed), st)
        actual = evaluate_forest(st.X, st.forest_nu, sum_batch_axis=-1)
        np.testing.assert_allclose(st.nu_fit, actual, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(st.resid, y-st.mu_fit-weight*actual, atol=2e-6)


def test_precision_cache_respects_pending_prunes_and_untouched_leaves():
    from longbet._forest_cache import refresh_prec_tree

    st = _constant_state(np.zeros(4), np.ones(4, bool))
    indices = jnp.array([[4, 5, 3, 3], [2, 3, 2, 3]], jnp.uint8)
    st = eqx.tree_at(lambda s: (s.forest.leaf_tree, s.forest.leaf_indices,
        s.forest.to_prune, s.forest.move_node, s.prec_scale), st,
        (jnp.zeros((2, 8)), indices, jnp.array([True, False]),
         jnp.array([2, 1]), jnp.array([1., 2., 0., 4.])),
        is_leaf=lambda x: x is None)
    result = refresh_prec_tree(st)
    expected = np.zeros((2, 8))
    expected[0, [2, 3]] = [3, 4]
    expected[1, [2, 3]] = [1, 6]
    np.testing.assert_array_equal(result.forest.prec_tree, expected)
    np.testing.assert_array_equal(result.forest.leaf_indices, indices)
    np.testing.assert_array_equal(result.forest.to_prune, st.forest.to_prune)
    np.testing.assert_array_equal(result.forest.count_tree, st.forest.count_tree)


def test_full_precision_refresh_never_builds_dense_membership_arrays():
    """Default bartz precision reduction is only memory-safe for two leaves.

    Check traced intermediate shapes, including nested branches: an all-leaf
    refresh must not create any observations-by-leaf-slots dense array.
    This catches the regression even if a small test happens to fit in RAM.
    """
    from longbet._forest_cache import refresh_prec_tree

    n, trees, slots = 2048, 3, 1024
    st = _constant_state(np.zeros(n), np.ones(n, bool))
    indices = jnp.tile(1+jnp.arange(n, dtype=jnp.uint16) % (slots-1), (trees, 1))
    st = eqx.tree_at(lambda s: (s.forest.leaf_tree, s.forest.leaf_indices,
                               s.forest.to_prune, s.forest.move_node), st,
                    (jnp.zeros((trees, slots)), indices,
                     jnp.zeros(trees, bool), jnp.ones(trees, jnp.int32)))
    traced = jax.make_jaxpr(lambda s: refresh_prec_tree(s).forest.prec_tree)(st)

    def shapes(value):
        if hasattr(value, "eqns"):
            for eqn in value.eqns:
                for var in eqn.outvars:
                    yield getattr(var.aval, "shape", ())
                yield from shapes(eqn.params)
        elif hasattr(value, "jaxpr"):
            yield from shapes(value.jaxpr)
        elif isinstance(value, dict):
            for child in value.values():
                yield from shapes(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from shapes(child)

    bound = 8*(trees*n + trees*slots*16)
    for shape in shapes(traced):
        assert np.prod(shape) <= bound, f"Dense cache-refresh intermediate: {shape}"
    result = refresh_prec_tree(st)
    expected = np.bincount(np.asarray(indices[0]), minlength=slots)
    np.testing.assert_allclose(result.forest.prec_tree, np.tile(expected, (trees, 1)))


@pytest.mark.parametrize("coupled", [False, True])
def test_fixed_beta_is_not_rescaled_by_ridge_move(coupled):
    st = _constant_state(np.zeros(4), np.ones(4, bool), ridge_move=True)
    st = eqx.tree_at(lambda s: s.beta, st, jnp.ones_like(st.beta))
    precision = jnp.full(4, 1.3) if coupled else None
    for i in range(20):
        st = longbet_single_step(jax.random.key(i), st, precision)
        np.testing.assert_array_equal(st.beta, 1)


def test_mixed_tree_posterior_matches_observed_likelihood_quadrature():
    """Integrate the observed probit/Gaussian likelihood, not latent draws.

    With fixed covariance and constant mean leaves, a two-dimensional grid
    supplies an independent posterior oracle including a joint sign event.
    This checks repeated binary augmentation plus feedback, not just a single
    truncated-normal draw or a utility fed externally generated samples.
    """
    from scipy.special import log_ndtr

    labels = np.array([0., 0., 1., 0., 1., 1., 0., 1.])
    continuous = np.array([-.8, .4, 1.5, .2, -.2, 1.1, -.5, .8])
    g, v = .8, .4
    G = jnp.array([[0., 0.], [g, 0.]])
    variances = jnp.array([1., v])
    observed = jnp.ones((2, len(labels)), bool)
    initial = (_constant_state(labels, np.ones(len(labels), bool), binary=True),
               _constant_state(continuous, np.ones(len(labels), bool), v))

    # Marginal continuous variance, then binary latent conditional on y_cont.
    marginal_v = v+g*g
    conditional_sd = np.sqrt(v/marginal_v)
    grid = np.linspace(-5, 5, 701)
    a, b = np.meshgrid(grid, grid, indexing="ij")
    log_density = -(a*a+b*b)/(2*2.25)
    for label, value in zip(labels, continuous):
        latent_mean = a+g/marginal_v*(value-b)
        log_density += log_ndtr((2*label-1)*latent_mean/conditional_sd)
        log_density -= (value-b)**2/(2*marginal_v)
    weights = np.exp(log_density-log_density.max())
    weights /= weights.sum()
    assert weights[[0, -1]].sum()+weights[:, [0, -1]].sum() < 1e-10
    expected_mean = np.array([(weights*a).sum(), (weights*b).sum()])
    centered = [a-expected_mean[0], b-expected_mean[1]]
    expected_cov = np.array([[(weights*u*v_).sum() for v_ in centered] for u in centered])
    expected_event = weights[(a < 0) & (b > 0)].sum()

    @jax.jit
    def run(states):
        def step(current, key):
            states = list(current)
            for m in range(2):
                raw = jnp.stack([s.resid for s in states])
                resid, precision = conditional_residual(raw, observed, G, variances, m)
                offset = states[m].resid-resid
                view = eqx.tree_at(lambda s: s.resid, states[m], resid)
                upd = longbet_single_step(jax.random.fold_in(key, m), view, precision)
                states[m] = eqx.tree_at(lambda s: s.resid, upd, upd.resid+offset)
            return tuple(states), jnp.stack([s.mu_fit[0] for s in states])
        return jax.lax.scan(step, states, jax.random.split(jax.random.key(573), 15000))

    _, draws = run(initial)
    draws = np.asarray(draws)[1500:]
    np.testing.assert_allclose(draws.mean(0), expected_mean, atol=.025)
    np.testing.assert_allclose(np.cov(draws.T), expected_cov, rtol=.1, atol=.004)
    np.testing.assert_allclose(np.mean((draws[:, 0] < 0) & (draws[:, 1] > 0)),
                               expected_event, atol=.025)


def test_weighted_scale_gp_coding_and_intercept_conditionals():
    """Check each parameter block's sufficient statistics at its scheduled key."""
    from longbet._gp import sample_beta_gp

    y = jnp.array([-1., .2, 1.4, 2., .4, -.3])
    p = jnp.array([.2, 3., 1., .6, 4., 2.])
    st = _constant_state(y, np.ones(len(y), bool), variance=1.7,
                         sample_alpha=True, sample_beta=True,
                         adaptive_coding=True, random_intercept=True)
    z = jnp.array([0., 0., 0., 1., 1., 1.])
    b_old = jnp.where(z == 1, st.b1, st.b0)
    nu_old = jnp.full(len(y), 1.3)
    st = eqx.tree_at(lambda s: (s.beta, s.nu_fit, s.forest_nu.leaf_tree,
                                s.z_vec, s.resid), st,
                    (jnp.ones_like(st.beta), nu_old,
                     st.forest_nu.leaf_tree.at[:, 1].set(1.3/st.forest_nu.leaf_unit),
                     z, y-b_old*nu_old))
    key = jax.random.key(722)
    keys = jax.random.split(key, 10)
    out = longbet_single_step(key, st, p)

    P_alpha = jnp.sum(p*out.mu_fit**2)+1/st.sigma_alpha**2
    h_alpha = jnp.sum(p*out.mu_fit*(y-b_old*nu_old))
    expected_alpha = h_alpha/P_alpha+jax.random.normal(keys[2], ())/jnp.sqrt(P_alpha)
    np.testing.assert_allclose(out.alpha, expected_alpha, rtol=2e-5)

    partial = y-out.alpha*out.mu_fit
    # nu is updated before beta; alpha still conditions on the previous nu.
    # There is one exposure bin in this conditional-update fixture.
    d = b_old*out.nu_fit
    expected_beta = sample_beta_gp(keys[3], jnp.array([jnp.sum(p*d*d)]),
                                   jnp.array([jnp.sum(p*d*partial)]), st.K_chol)
    np.testing.assert_allclose(out.beta, expected_beta, rtol=2e-5)

    g = out.beta[0]*out.nu_fit
    for label, k in ((0, 5), (1, 6)):
        mask = z == label
        precision = jnp.sum(jnp.where(mask, p*g*g, 0))+1/st.sigma_b**2
        h = jnp.sum(jnp.where(mask, p*g*partial, 0))
        expected = h/precision+jax.random.normal(keys[k], ())/jnp.sqrt(precision)
        np.testing.assert_allclose(out.b0 if label == 0 else out.b1, expected, rtol=2e-5)

    e = partial-jnp.where(z == 1, out.b1, out.b0)*g
    V = 1/(jnp.sum(p)+1/st.sigma_gamma2)
    expected_gamma = V*jnp.sum(p*e)+jax.random.normal(keys[7], (1,))*jnp.sqrt(V)
    np.testing.assert_allclose(out.gamma, expected_gamma, rtol=2e-5)
    assert out.sigma2 == st.sigma2  # the caller owns the structural variance
