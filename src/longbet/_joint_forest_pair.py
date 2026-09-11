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

"""Experimental exact Gibbs block: one prognostic tree with one treatment tree.

Not enabled by the public sampler.

Why this block exists
---------------------
`_joint_prognostic.py` removes the ridge between the prognostic forest and the
*coding coefficient*, which is the part of the confounding that lives on
untreated cells. A second ridge survives it. On **treated** cells both forests
are active and both can represent the same signal:

    f = alpha*mu(x,t) + b_z*beta(S)*nu(x,S,t) + gamma

Moving a function of `(x, t)` from `mu` into `nu` is nearly free there, and the
reported effect keeps only the `nu` side, so the split again leaks into the
answer. The archived `--treatment-only` variant (which fixes `b0 = 0` and so
removes the *first* ridge entirely) still failed at GMV benefit R-hat 1.3447,
which is the evidence that this second direction exists.

Conditional on the tree topologies, the coding coefficients, the exposure GP and
the variances, **both** forests' leaf values enter the mean linearly, so their
joint conditional is exactly Gaussian. Note the coding coefficients must be held
fixed here: `b_z * beta(S) * nu` is a product of `b` and `nu`, so `b` and the
treatment leaves are *not* jointly Gaussian and must not be put in one block.
This module therefore complements `_joint_prognostic.py` rather than replacing
it; composing the two covers both ridges.

Structure and cost
------------------
Within one tree the leaves partition the cells, so each forest's own leaf block
is diagonal and only the cross block is dense:

    P = [[ Dmu, C  ],
         [ C',  Dnu ]]        Dmu, Dnu diagonal (K), C dense (K x K).

Sampling uses the exact conditional decomposition

    v ~ N(S^-1 (h_nu - C' Dmu^-1 h_mu), S^-1),   S = Dnu - C' Dmu^-1 C
    u ~ N(Dmu^-1 (h_mu - C v), Dmu^-1)   given v

so only a K x K matrix is factorized. Active leaves are packed into ``K``
compact coordinates; a tree with more than ``K`` active leaves simply has its
overflow coordinates held fixed, which is still a valid (smaller) Gibbs block
because the selection depends only on the topology, which the block conditions
on.

Conventions preserved
---------------------
Same as `_joint_prognostic.py`: actual leaves only, offset excluded by
parameterizing the change, `mu_fit`/`nu_fit` and the raw residual updated
together, full SUR precision and score, binary variances untouched, float64
factorization with float32 state, chain axes preserved. Under shared treatment
forests only the *private* ensemble is moved; the shared contribution stays
inside `nu_fit` and is conditioned on.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular

from bartz.grove import traverse_forest
from bartz.grove._grove import is_actual_leaf

from longbet._multi_state import split_multi_chain_fields
from longbet._shared_forest import observed_precision

JOINT_FOREST_PAIR_SEMANTICS = "joint_forest_pair_v1"

#: Compact coordinates retained per tree. Trees under BART's depth prior carry
#: far fewer active leaves than this; overflow coordinates are held fixed.
MAX_PACKED_LEAVES = 48


def _pack(split_tree_j, leaf_idx_j, cap):
    """Map each cell to a compact coordinate in ``[0, cap]``; ``cap`` is a sink.

    Returns the per-cell column, the slot-to-column map and the in-range mask.
    """
    active = is_actual_leaf(split_tree_j, add_bottom_level=True)
    position = jnp.cumsum(active) - 1
    slot_col = jnp.where(active & (position < cap), position, cap)
    return slot_col[leaf_idx_j], slot_col, active


def _pair_block_draw(key, prec, score, col_mu, col_nu, weight_mu, weight_nu,
                     theta_mu, theta_nu, prior_mu, prior_nu, cap):
    """Exact joint draw of one prognostic tree's and one treatment tree's leaves.

    ``theta_*`` are the current per-column leaf values in data units (index
    ``cap`` is the sink and is never written back).
    """
    size = cap + 1
    keep = jnp.arange(size) < cap

    a_mu = weight_mu * prec
    a_nu = weight_nu * prec
    d_mu = jnp.zeros(size, jnp.float64).at[col_mu].add(weight_mu * a_mu)
    d_nu = jnp.zeros(size, jnp.float64).at[col_nu].add(weight_nu * a_nu)
    d_mu = jnp.where(keep, d_mu + prior_mu, 1.0)
    d_nu = jnp.where(keep, d_nu + prior_nu, 1.0)

    cross = jnp.zeros((size, size), jnp.float64).at[col_mu, col_nu].add(
        weight_mu * weight_nu * prec)
    cross = jnp.where(keep[:, None] & keep[None, :], cross, 0.0)

    h_mu = jnp.zeros(size, jnp.float64).at[col_mu].add(weight_mu * score)
    h_nu = jnp.zeros(size, jnp.float64).at[col_nu].add(weight_nu * score)
    h_mu = jnp.where(keep, h_mu - prior_mu * theta_mu, 0.0)
    h_nu = jnp.where(keep, h_nu - prior_nu * theta_nu, 0.0)

    inv_mu = jnp.reciprocal(d_mu)
    schur = jnp.diag(d_nu) - cross.T @ (inv_mu[:, None] * cross)
    rhs = h_nu - cross.T @ (inv_mu * h_mu)
    chol = jnp.linalg.cholesky(schur)
    k_nu, k_mu = random.split(key)
    mean_nu = solve_triangular(
        chol.T, solve_triangular(chol, rhs, lower=True), lower=False)
    d_theta_nu = mean_nu + solve_triangular(
        chol.T, random.normal(k_nu, (size,), jnp.float64), lower=False)
    d_theta_nu = jnp.where(keep, d_theta_nu, 0.0)

    mean_mu = inv_mu * (h_mu - cross @ d_theta_nu)
    d_theta_mu = mean_mu + random.normal(
        k_mu, (size,), jnp.float64) * jnp.sqrt(inv_mu)
    d_theta_mu = jnp.where(keep, d_theta_mu, 0.0)
    return d_theta_mu, d_theta_nu


def _outcome_pairs(key, s, prec, score, resid_m, cap):
    """Cycle prognostic tree j against private treatment tree ``j % J_nu``."""
    unit_mu, unit_nu = s.forest.leaf_unit, s.forest_nu.leaf_unit
    idx_mu = traverse_forest(s.X, s.forest.var_tree, s.forest.split_tree)
    idx_nu = traverse_forest(s.X, s.forest_nu.var_tree, s.forest_nu.split_tree)
    trees_mu = s.forest.split_tree.shape[0]
    trees_nu = s.forest_nu.split_tree.shape[0]

    alpha = s.alpha.astype(jnp.float64)
    w_nu = (jnp.where(s.z_vec == 1.0, s.b1, s.b0)
            * s.beta[s.exposure_idx]).astype(jnp.float64)
    prior_mu = s.forest.leaf_prior_cov_inv.astype(jnp.float64)
    prior_nu = s.forest_nu.leaf_prior_cov_inv.astype(jnp.float64)

    def one_pair(carry, inputs):
        leaf_mu, leaf_nu, mu_fit, nu_fit, resid, sc = carry
        j, k = inputs
        jn = jnp.remainder(j, trees_nu)
        col_mu, slot_mu, act_mu = _pack(s.forest.split_tree[j], idx_mu[j], cap)
        col_nu, slot_nu, act_nu = _pack(
            jnp.take(s.forest_nu.split_tree, jn, axis=0),
            jnp.take(idx_nu, jn, axis=0), cap)

        cur_mu = (jnp.zeros(cap + 1, jnp.float64)
                  .at[slot_mu].set(leaf_mu[j].astype(jnp.float64) * unit_mu))
        cur_nu = (jnp.zeros(cap + 1, jnp.float64).at[slot_nu].set(
            jnp.take(leaf_nu, jn, axis=0).astype(jnp.float64) * unit_nu))

        d_mu, d_nu = _pair_block_draw(
            k, prec, sc, col_mu, col_nu, alpha, w_nu, cur_mu, cur_nu,
            prior_mu, prior_nu, cap)

        delta_mu_cell, delta_nu_cell = d_mu[col_mu], d_nu[col_nu]
        delta_mean = alpha * delta_mu_cell + w_nu * delta_nu_cell
        mu_fit = mu_fit + delta_mu_cell
        nu_fit = nu_fit + delta_nu_cell
        resid = resid - delta_mean
        sc = sc - prec * delta_mean

        new_mu = jnp.where(act_mu, (cur_mu + d_mu)[slot_mu] / unit_mu,
                           leaf_mu[j].astype(jnp.float64))
        new_nu = jnp.where(act_nu, (cur_nu + d_nu)[slot_nu] / unit_nu,
                           jnp.take(leaf_nu, jn, axis=0).astype(jnp.float64))
        leaf_mu = leaf_mu.at[j].set(new_mu.astype(leaf_mu.dtype))
        leaf_nu = leaf_nu.at[jn].set(new_nu.astype(leaf_nu.dtype))
        return (leaf_mu, leaf_nu, mu_fit, nu_fit, resid, sc), None

    carry = (s.forest.leaf_tree, s.forest_nu.leaf_tree,
             s.mu_fit.astype(jnp.float64), s.nu_fit.astype(jnp.float64),
             resid_m, score)
    (leaf_mu, leaf_nu, mu_fit, nu_fit, resid, _), _ = jax.lax.scan(
        one_pair, carry, (jnp.arange(trees_mu), random.split(key, trees_mu)))
    return leaf_mu, leaf_nu, mu_fit, nu_fit, resid


@jax.jit
def joint_forest_pair_single_step(key, state):
    """One pass over outcomes, pairing prognostic and treatment trees."""
    states = list(state.states)
    obs = jnp.stack([s.obs_mask for s in states])
    loadings = (state.gamma_loadings if state.sur_active
                else jnp.zeros_like(state.gamma_loadings))
    omega = observed_precision(
        obs, loadings, jnp.stack([s.sigma2 for s in states]))
    resid_all = jnp.stack([s.resid for s in states]).astype(jnp.float64)

    keys = random.split(key, len(states))
    for m, s in enumerate(states):
        prec = omega[:, m, m]
        score = jnp.einsum('ik,ki->i', omega[:, m, :], resid_all)
        leaf_mu, leaf_nu, mu_fit, nu_fit, resid = _outcome_pairs(
            keys[m], s, prec, score, resid_all[m], MAX_PACKED_LEAVES)
        resid = jnp.where(obs[m], resid, 0.0)
        resid_all = resid_all.at[m].set(resid)
        states[m] = eqx.tree_at(
            lambda s: (s.forest.leaf_tree, s.forest_nu.leaf_tree, s.mu_fit,
                       s.nu_fit, s.resid), s,
            (leaf_mu, leaf_nu, mu_fit.astype(jnp.float32),
             nu_fit.astype(jnp.float32), resid.astype(jnp.float32)))
    return eqx.tree_at(lambda s: s.states, state, tuple(states))


def joint_forest_pair_step(key, state):
    """Chain-aware wrapper."""
    if not state.has_chain_axis:
        return joint_forest_pair_single_step(key, state)
    per, shared = split_multi_chain_fields(state)

    def one(k, p):
        updated = joint_forest_pair_single_step(k, eqx.combine(p, shared))
        return split_multi_chain_fields(updated)[0]

    updated = jax.vmap(one)(random.split(key, state.num_chains), per)
    return eqx.combine(updated, shared)
