"""Dense independent oracle for the joint prognostic-leaf / coding block.

The block draws one prognostic tree's leaf values jointly with ``(b0, b1)``
using an arrowhead factorization.  These tests build the same conditional as a
plain dense Gaussian -- explicit design matrix, explicit matrix inverse -- so
the check does not reuse the Schur derivation under test.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bartz.grove import evaluate_forest

from longbet import LongBetConfig
from longbet._joint_prognostic import (_tree_block_draw, joint_prognostic_step,
                                       joint_prognostic_single_step)
from longbet._multi_input import normalize_multi_inputs
from longbet._multi_state import init_multi_longbet, split_multi_chain_fields
from longbet._multi_step import multi_step
from longbet._shared_forest import enable_x64


def _dense_reference(prec, score, leaf_idx, active, alpha, g, z_vec,
                     theta_cur, b_cur, leaf_prior_prec, b_prior_prec):
    """Mean and covariance of (theta_active, b0, b1) by dense linear algebra."""
    slots = np.flatnonzero(active)
    n = prec.size
    D = np.zeros((n, slots.size + 2))
    for col, ell in enumerate(slots):
        D[:, col] = alpha * (leaf_idx == ell)
    D[:, -2] = np.where(z_vec == 0.0, g, 0.0)
    D[:, -1] = np.where(z_vec == 1.0, g, 0.0)
    lam = np.concatenate([np.full(slots.size, leaf_prior_prec),
                          np.full(2, b_prior_prec)])
    phi_cur = np.concatenate([theta_cur[slots], b_cur])
    P = np.diag(lam) + D.T @ (prec[:, None] * D)
    h = D.T @ score - lam * phi_cur
    cov = np.linalg.inv(P)
    return phi_cur + cov @ h, cov, slots


@pytest.mark.parametrize("degenerate", [False, True])
def test_tree_block_matches_dense_gaussian(degenerate):
    """Empirical moments of actual draws match an explicit dense posterior."""
    rng = np.random.default_rng(4711)
    n, slots_total = 90, 8
    active = np.zeros(slots_total, bool)
    active[[1, 2, 4, 5, 7]] = True
    reachable = np.flatnonzero(active)
    leaf_idx = rng.choice(reachable, size=n)
    prec = rng.gamma(3.0, 0.4, size=n)
    g = rng.normal(size=n)
    z_vec = (rng.random(n) < 0.45).astype(np.float64)
    if degenerate:
        # A cell block with no likelihood information at all: those leaves and
        # the coding coefficients must fall back to their exact priors.
        prec = np.zeros(n)
    score = rng.normal(size=n) * prec
    theta_cur = rng.normal(size=slots_total) * 0.3
    b_cur = np.array([0.4, -0.25])
    leaf_prior_prec, b_prior_prec = 3.5, 2.0

    expected_mean, expected_cov, sl = _dense_reference(
        prec, score, leaf_idx, active, 0.8, g, z_vec, theta_cur, b_cur,
        leaf_prior_prec, b_prior_prec)

    draws = 40000
    with enable_x64(True):
        def one(key):
            theta, b, _, _ = _tree_block_draw(
                key, jnp.asarray(prec), jnp.asarray(score),
                jnp.asarray(leaf_idx), jnp.asarray(active), jnp.float64(0.8),
                jnp.asarray(g), jnp.asarray(z_vec), jnp.asarray(theta_cur),
                jnp.asarray(b_cur), jnp.float64(leaf_prior_prec),
                jnp.float64(b_prior_prec), slots_total)
            return jnp.concatenate([theta[jnp.asarray(sl)], b])
        sample = np.asarray(
            jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(3), draws)))

    got_mean, got_cov = sample.mean(0), np.cov(sample, rowvar=False)
    sd = np.sqrt(np.diag(expected_cov))
    # Monte Carlo error of each mean is sd/sqrt(draws); allow 4.5 of them.
    z = np.abs(got_mean - expected_mean) / (sd / np.sqrt(draws))
    assert z.max() < 4.5, f"worst standardized mean error {z.max():.2f}"
    # Compare the full covariance, scaled so entries are correlation-sized.
    scale = np.outer(sd, sd)
    err = np.abs(got_cov - expected_cov) / scale
    assert err.max() < 0.05, f"worst scaled covariance error {err.max():.4f}"

    if degenerate:
        np.testing.assert_allclose(expected_mean[-2:], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.diag(expected_cov)[-2:],
                                   1.0 / b_prior_prec, rtol=1e-10)


def test_tree_block_couples_leaves_and_coding():
    """The block is not diagonal: leaves and b0 have real posterior covariance.

    A sampler that draws them separately can only move along one axis at a
    time, which is the ridge this block exists to cross.
    """
    rng = np.random.default_rng(99)
    n, slots_total = 120, 6
    active = np.array([False, True, True, False, True, False])
    leaf_idx = rng.choice(np.flatnonzero(active), size=n)
    prec = np.full(n, 2.0)
    g = rng.normal(size=n) + 1.5           # untreated cells carry b0 * g
    z_vec = np.zeros(n)                    # all untreated: S = 0 everywhere
    score = rng.normal(size=n)
    _, cov, sl = _dense_reference(prec, score, leaf_idx, active, 1.0, g, z_vec,
                                  np.zeros(slots_total), np.zeros(2), 1.0, 2.0)
    corr = cov[:sl.size, -2] / np.sqrt(np.diag(cov)[:sl.size] * cov[-2, -2])
    assert np.abs(corr).max() > 0.3


def _small_multi_state(*, adaptive_coding=True, num_chains=1, seed=20):
    rng = np.random.default_rng(seed)
    N, T, P = 8, 5, 3
    y_bin = rng.choice([0.0, 1.0], size=(N, T)).astype(np.float32)
    y_cont = rng.standard_normal((N, T)).astype(np.float32)
    z = np.zeros((N, T), dtype=np.float32)
    z[: N // 2, 3:] = 1.0
    cfg = LongBetConfig(num_trees_pr=4, num_trees_trt=3, max_depth_pr=3,
                        max_depth_trt=3, sur=True, sur_prior_var=1.0,
                        adaptive_coding=adaptive_coding, random_intercept=True)
    norm = normalize_multi_inputs({"bin": y_bin, "cont": y_cont},
                                  outcome={"bin": "binary", "cont": "continuous"},
                                  outcome_names=None, config=cfg)
    exposure = np.where(z == 1, np.cumsum(z, axis=1), 0).astype(np.int32)
    return init_multi_longbet(
        X_unified=jnp.asarray(rng.integers(0, 8, size=(P, N * T)), jnp.uint8),
        unit_idx=jnp.repeat(jnp.arange(N), T),
        time_idx=jnp.tile(jnp.arange(T), N),
        exposure_idx=jnp.asarray(exposure.ravel()),
        z_vec=jnp.asarray(z.ravel()),
        max_split_mu=jnp.full(P, 7, jnp.uint8),
        max_split_nu=jnp.full(P, 7, jnp.uint8),
        norm_input=norm, config=cfg, num_chains=num_chains,
        init_key=jax.random.key(1))


def _advance(state, sweeps=4, key=jax.random.key(7)):
    for sweep in range(sweeps):
        state = multi_step(jax.random.fold_in(key, sweep), state,
                           jnp.int32(sweep))
    return state


def _check_invariants(state):
    for child in state.states:
        mu = evaluate_forest(child.X, child.forest, sum_batch_axis=-1) + child.forest.offset
        nu = evaluate_forest(child.X, child.forest_nu, sum_batch_axis=-1) + child.forest_nu.offset
        np.testing.assert_allclose(child.mu_fit, mu, atol=2e-4)
        fitted = (child.alpha * mu
                  + jnp.where(child.z_vec == 1, child.b1, child.b0)
                  * child.beta[child.exposure_idx] * nu
                  + child.gamma[child.unit_idx])
        response = child.z if child.outcome_type_str == "binary" else child.y
        np.testing.assert_allclose(
            child.resid, jnp.where(child.obs_mask, response - fitted, 0),
            atol=2e-4)


def test_block_preserves_residual_and_fit_invariants():
    """After the block, resid is still y - f and mu_fit still matches the forest."""
    state = _advance(_small_multi_state())
    with enable_x64(True):
        out = joint_prognostic_single_step(jax.random.key(5), state)
    _check_invariants(out)
    # The binary latent variance stays exactly identified at one.
    assert float(out.states[0].sigma2) == 1.0
    # Something actually moved.
    assert not np.allclose(out.states[1].b0, state.states[1].b0)


def test_block_leaves_unreachable_slots_untouched():
    state = _advance(_small_multi_state())
    with enable_x64(True):
        out = joint_prognostic_single_step(jax.random.key(11), state)
    from bartz.grove._grove import is_actual_leaf
    for before, after in zip(state.states, out.states):
        mask = jax.vmap(lambda s: is_actual_leaf(s, add_bottom_level=True))(
            before.forest.split_tree)
        np.testing.assert_array_equal(
            np.asarray(after.forest.leaf_tree)[~np.asarray(mask)],
            np.asarray(before.forest.leaf_tree)[~np.asarray(mask)])


def test_block_is_a_noop_without_adaptive_coding():
    """Fixed coding leaves nothing to block in, so the state must be unchanged."""
    state = _advance(_small_multi_state(adaptive_coding=False))
    with enable_x64(True):
        out = joint_prognostic_single_step(jax.random.key(13), state)
    for before, after in zip(state.states, out.states):
        np.testing.assert_array_equal(after.forest.leaf_tree,
                                      before.forest.leaf_tree)
        np.testing.assert_array_equal(after.resid, before.resid)
        assert float(after.b0) == float(before.b0)
        assert float(after.b1) == float(before.b1)


def test_block_runs_per_chain_and_keeps_axes():
    state = _advance(_small_multi_state(num_chains=3))
    with enable_x64(True):
        out = joint_prognostic_step(jax.random.key(17), state)
    assert out.states[0].b0.shape == (3,)
    assert out.states[0].resid.shape == state.states[0].resid.shape
    # Chains must not be given identical draws.
    assert len(set(np.asarray(out.states[1].b0).round(6).tolist())) == 3
    # Slice with the state's own chain metadata: shared fields have no chain axis.
    import equinox as eqx
    per, shared = split_multi_chain_fields(out)
    for chain in range(3):
        _check_invariants(eqx.combine(
            jax.tree.map(lambda a, c=chain: a[c], per), shared))
