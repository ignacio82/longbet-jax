"""Dense independent oracle for the paired prognostic/treatment leaf block."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bartz.grove import evaluate_forest

from longbet._joint_forest_pair import (_pair_block_draw,
                                        joint_forest_pair_single_step,
                                        joint_forest_pair_step)
from longbet._multi_state import split_multi_chain_fields
from longbet._shared_forest import enable_x64
from test_joint_prognostic import _advance, _small_multi_state


def _dense_pair_reference(prec, score, col_mu, col_nu, w_mu, w_nu,
                          theta_mu, theta_nu, prior_mu, prior_nu, cap):
    """Mean and covariance of (mu leaves, nu leaves) by dense linear algebra."""
    n = prec.size
    keep = np.arange(cap)
    D = np.zeros((n, 2 * cap))
    for c in keep:
        D[:, c] = w_mu * (col_mu == c)
        D[:, cap + c] = w_nu * (col_nu == c)
    lam = np.concatenate([np.full(cap, prior_mu), np.full(cap, prior_nu)])
    phi = np.concatenate([theta_mu[:cap], theta_nu[:cap]])
    P = np.diag(lam) + D.T @ (prec[:, None] * D)
    h = D.T @ score - lam * phi
    cov = np.linalg.inv(P)
    return cov @ h, cov          # returns the DELTA mean, matching the block


@pytest.mark.parametrize("overlap", ["partial", "identical"])
def test_pair_block_matches_dense_gaussian(overlap):
    """Actual draws match an explicitly inverted dense Gaussian posterior."""
    rng = np.random.default_rng(20260907)
    n, cap = 140, 4
    col_mu = rng.integers(0, cap, size=n)
    col_nu = col_mu.copy() if overlap == "identical" else rng.integers(0, cap, size=n)
    prec = rng.gamma(4.0, 0.3, size=n)
    score = rng.normal(size=n) * prec
    w_mu, w_nu = 0.9, rng.normal(size=n)
    theta_mu = np.concatenate([rng.normal(size=cap) * 0.2, [0.0]])
    theta_nu = np.concatenate([rng.normal(size=cap) * 0.2, [0.0]])
    prior_mu, prior_nu = 2.5, 4.0

    exp_mean, exp_cov = _dense_pair_reference(
        prec, score, col_mu, col_nu, w_mu, w_nu, theta_mu, theta_nu,
        prior_mu, prior_nu, cap)

    draws = 40000
    with enable_x64(True):
        def one(key):
            d_mu, d_nu = _pair_block_draw(
                key, jnp.asarray(prec), jnp.asarray(score),
                jnp.asarray(col_mu), jnp.asarray(col_nu), jnp.float64(w_mu),
                jnp.asarray(w_nu), jnp.asarray(theta_mu), jnp.asarray(theta_nu),
                jnp.float64(prior_mu), jnp.float64(prior_nu), cap)
            return jnp.concatenate([d_mu[:cap], d_nu[:cap]])
        sample = np.asarray(
            jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(2), draws)))

    sd = np.sqrt(np.diag(exp_cov))
    z = np.abs(sample.mean(0) - exp_mean) / (sd / np.sqrt(draws))
    assert z.max() < 4.5, f"worst standardized mean error {z.max():.2f}"
    err = np.abs(np.cov(sample, rowvar=False) - exp_cov) / np.outer(sd, sd)
    assert err.max() < 0.05, f"worst scaled covariance error {err.max():.4f}"


def test_pair_block_cross_covariance_is_real():
    """The two forests are posterior-correlated: that is the ridge to cross."""
    rng = np.random.default_rng(5)
    n, cap = 200, 3
    col = rng.integers(0, cap, size=n)
    _, cov = _dense_pair_reference(
        np.full(n, 3.0), rng.normal(size=n), col, col, 1.0,
        np.full(n, 1.2), np.zeros(cap + 1), np.zeros(cap + 1), 1.0, 1.0, cap)
    corr = cov[:cap, cap:] / np.sqrt(np.outer(np.diag(cov)[:cap],
                                              np.diag(cov)[cap:]))
    assert np.abs(corr).max() > 0.5


def _check_invariants(state):
    for child in state.states:
        mu = evaluate_forest(child.X, child.forest, sum_batch_axis=-1) + child.forest.offset
        nu = evaluate_forest(child.X, child.forest_nu, sum_batch_axis=-1) + child.forest_nu.offset
        np.testing.assert_allclose(child.mu_fit, mu, atol=3e-4)
        np.testing.assert_allclose(child.nu_fit, nu, atol=3e-4)
        fitted = (child.alpha * mu
                  + jnp.where(child.z_vec == 1, child.b1, child.b0)
                  * child.beta[child.exposure_idx] * nu
                  + child.gamma[child.unit_idx])
        response = child.z if child.outcome_type_str == "binary" else child.y
        np.testing.assert_allclose(
            child.resid, jnp.where(child.obs_mask, response - fitted, 0),
            atol=3e-4)


def test_pair_block_preserves_invariants_and_moves_both_forests():
    state = _advance(_small_multi_state())
    with enable_x64(True):
        out = joint_forest_pair_single_step(jax.random.key(23), state)
    _check_invariants(out)
    assert float(out.states[0].sigma2) == 1.0
    assert not np.allclose(out.states[1].forest.leaf_tree,
                           state.states[1].forest.leaf_tree)
    assert not np.allclose(out.states[1].forest_nu.leaf_tree,
                           state.states[1].forest_nu.leaf_tree)
    # The coding coefficients are held fixed: b and nu are not jointly Gaussian.
    for before, after in zip(state.states, out.states):
        assert float(after.b0) == float(before.b0)
        assert float(after.b1) == float(before.b1)


def test_pair_block_leaves_unreachable_slots_untouched():
    state = _advance(_small_multi_state())
    with enable_x64(True):
        out = joint_forest_pair_single_step(jax.random.key(29), state)
    from bartz.grove._grove import is_actual_leaf
    for before, after in zip(state.states, out.states):
        for attr in ("forest", "forest_nu"):
            b, a = getattr(before, attr), getattr(after, attr)
            mask = np.asarray(jax.vmap(
                lambda s: is_actual_leaf(s, add_bottom_level=True))(b.split_tree))
            np.testing.assert_array_equal(
                np.asarray(a.leaf_tree)[~mask], np.asarray(b.leaf_tree)[~mask])


def test_pair_block_runs_per_chain():
    import equinox as eqx
    state = _advance(_small_multi_state(num_chains=3))
    with enable_x64(True):
        out = joint_forest_pair_step(jax.random.key(31), state)
    assert out.states[0].resid.shape == state.states[0].resid.shape
    per, shared = split_multi_chain_fields(out)
    for chain in range(3):
        _check_invariants(eqx.combine(
            jax.tree.map(lambda a, c=chain: a[c], per), shared))
