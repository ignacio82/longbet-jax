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

"""Experimental exact Gibbs block: prognostic leaves with the coding coefficients.

Not enabled by the public sampler.  The benchmark composes this transition with
an ordinary sweep to test mixing without changing the posterior.

Why this block exists
---------------------
Every untreated cell has exposure ``S = 0``, so the treatment term contributes
``b0 * beta(0) * nu(x, 0, t)`` there -- a function of ``(x, t)``, which is
exactly the argument set of the prognostic forest ``mu(x, t)``.  The data
constrain the *sum* ``alpha*mu + b0*beta(0)*nu(.,0,.)`` on those cells but only
weakly constrain the split, while the reported effect

    tau = b1*beta(S)*nu(x,S,t) - b0*beta(0)*nu(x,0,t)

subtracts one side of it.  An ordinary scan draws ``b0`` with every prognostic
leaf held fixed and each prognostic leaf with ``b0`` held fixed, so it crawls
along that ridge: the chains stay where they started and disagree on the effect
while agreeing on the fitted surface.

Conditional on the tree topologies, the exposure GP, the treatment forest and
the variance parameters, the prognostic leaf values and ``(b0, b1)`` enter the
mean *linearly*, so their joint conditional is exactly Gaussian.  This module
draws one prognostic tree's leaves jointly with ``(b0, b1)``, sweeping over the
trees.  Every such draw is an ordinary Gibbs block, so the target is unchanged.

Structure and cost
------------------
Within one tree the leaves partition the cells, so the leaf-leaf precision block
is diagonal and the system is an arrowhead:

    P = [[ D, A ],
         [ A', B ]]        D diagonal (L), B is 2x2, A is L x 2.

Sampling uses the exact conditional decomposition

    b ~ N(S^-1 (h_b - A' D^-1 h_theta), S^-1),   S = B - A' D^-1 A
    theta | b ~ N(D^-1 (h_theta - A b), D^-1)

which costs ``O(L)`` per tree and inverts only a 2x2 matrix.  No approximation,
jitter, prior change or acceptance threshold is involved.

Conventions preserved
---------------------
* Only *actual* leaves are in the block; unreachable slots never enter the
  model and are left untouched, matching ``_ridge.py``.
* Leaf values are converted to data units with ``leaf_unit`` before use, and the
  forest ``offset`` is excluded from the block by construction: the block is
  parameterized by the change from the current value.
* ``mu_fit`` and the raw full-model residual are updated together, so the
  invariant ``resid = y - f`` holds at the block boundary.
* The likelihood contribution uses the full SUR precision: precision
  ``Omega[m, m]`` and score ``(Omega r)[m]``, so earlier outcomes keep receiving
  downstream feedback.
* Binary structural variances stay exactly one; nothing here rescales them.
* Factorization is in float64; stored state stays float32.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from bartz.grove import traverse_forest
from bartz.grove._grove import is_actual_leaf

from longbet._multi_state import split_multi_chain_fields
from longbet._shared_forest import observed_precision

JOINT_PROGNOSTIC_SEMANTICS = "joint_prognostic_coding_v1"


def _tree_block_draw(key, prec, score, leaf_idx, active, alpha, g, z_vec,
                     theta_cur, b_cur, leaf_prior_prec, b_prior_prec, num_slots):
    """One exact joint draw of (tree leaves, b0, b1); see the module docstring.

    ``prec`` and ``score`` are the per-cell conditional precision ``Omega[m,m]``
    and score ``(Omega r)[m]`` of the *current* residual.  ``theta_cur`` holds
    the tree's leaf values in data units, ``b_cur`` is ``(b0, b1)``.

    Returns the new leaf values, the new ``(b0, b1)`` and the change in the
    outcome's fitted mean, all in float64.
    """
    # Design columns: alpha * leaf indicator, and g restricted to z == 0 / 1.
    c0 = jnp.where(z_vec == 0.0, g, 0.0)
    c1 = jnp.where(z_vec == 1.0, g, 0.0)
    coding = jnp.stack((c0, c1))                       # (2, n)

    # h = D' score - Lambda phi_cur, in the (theta, b) ordering.
    h_theta = (jnp.zeros(num_slots, jnp.float64)
               .at[leaf_idx].add(alpha * score))
    h_theta = jnp.where(active, h_theta - leaf_prior_prec * theta_cur, 0.0)
    h_b = coding @ score - b_prior_prec * b_cur

    # P = Lambda + D' diag(prec) D.
    d_diag = (jnp.zeros(num_slots, jnp.float64)
              .at[leaf_idx].add(jnp.square(alpha) * prec))
    d_diag = jnp.where(active, d_diag + leaf_prior_prec, 1.0)
    cross = jnp.zeros((num_slots, 2), jnp.float64).at[leaf_idx].add(
        (alpha * prec)[:, None] * coding.T)
    cross = jnp.where(active[:, None], cross, 0.0)
    b_block = (coding * prec) @ coding.T + jnp.diag(
        jnp.full(2, b_prior_prec, jnp.float64))

    # Schur complement on the 2x2 coding block; D is diagonal.
    d_inv = jnp.reciprocal(d_diag)
    schur = b_block - cross.T @ (d_inv[:, None] * cross)
    rhs = h_b - cross.T @ (d_inv * h_theta)
    schur_chol = jnp.linalg.cholesky(schur)
    k_b, k_theta = random.split(key)
    mean_b = jax.scipy.linalg.solve_triangular(
        schur_chol.T, jax.scipy.linalg.solve_triangular(
            schur_chol, rhs, lower=True), lower=False)
    db = mean_b + jax.scipy.linalg.solve_triangular(
        schur_chol.T, random.normal(k_b, (2,), jnp.float64), lower=False)

    mean_theta = d_inv * (h_theta - cross @ db)
    dtheta = mean_theta + random.normal(
        k_theta, (num_slots,), jnp.float64) * jnp.sqrt(d_inv)
    dtheta = jnp.where(active, dtheta, 0.0)

    delta_fit = dtheta[leaf_idx]
    delta_mean = alpha * delta_fit + coding.T @ db
    return theta_cur + dtheta, b_cur + db, delta_fit, delta_mean


def _outcome_block(key, s, prec, score_full, resid_m0):
    """Sweep every prognostic tree of outcome ``m``, blocking in (b0, b1).

    ``score_full`` is ``(Omega r)[m]`` for the residual currently in
    ``resid_all``; it is refreshed after each tree because the residual moves.
    """
    leaf_unit = s.forest.leaf_unit
    split_tree = s.forest.split_tree
    var_tree = s.forest.var_tree
    num_trees = split_tree.shape[0]
    leaf_indices = traverse_forest(s.X, var_tree, split_tree)       # (J, n)
    active_all = jax.vmap(lambda t: is_actual_leaf(t, add_bottom_level=True))(
        split_tree)                                                  # (J, slots)
    num_slots = active_all.shape[-1]

    alpha = s.alpha.astype(jnp.float64)
    g = (s.beta[s.exposure_idx] * s.nu_fit).astype(jnp.float64)
    z_vec = s.z_vec
    leaf_prior_prec = (s.forest.leaf_prior_cov_inv).astype(jnp.float64)
    b_prior_prec = jnp.float64(1.0) / jnp.float64(s.sigma_b) ** 2

    def one_tree(carry, inputs):
        leaf_tree, b_cur, mu_fit, resid_m, score = carry
        j, k = inputs
        idx = leaf_indices[j]
        active = active_all[j]
        theta_cur = leaf_tree[j].astype(jnp.float64) * leaf_unit
        theta_new, b_new, delta_fit, delta_mean = _tree_block_draw(
            k, prec, score, idx, active, alpha, g, z_vec, theta_cur, b_cur,
            leaf_prior_prec, b_prior_prec, num_slots)
        leaf_tree = leaf_tree.at[j].set(
            jnp.where(active, theta_new / leaf_unit,
                      leaf_tree[j].astype(jnp.float64)).astype(leaf_tree.dtype))
        mu_fit = mu_fit + delta_fit
        resid_m = resid_m - delta_mean
        # Only outcome m's residual moved, so the score moves by
        # -Omega[m, m] * delta_mean.
        score = score - prec * delta_mean
        return (leaf_tree, b_new, mu_fit, resid_m, score), None

    b_cur = jnp.stack((s.b0, s.b1)).astype(jnp.float64)
    carry = (s.forest.leaf_tree, b_cur, s.mu_fit.astype(jnp.float64),
             resid_m0, score_full)
    keys = random.split(key, num_trees)
    (leaf_tree, b_new, mu_fit, resid_m, _), _ = jax.lax.scan(
        one_tree, carry, (jnp.arange(num_trees), keys))
    return leaf_tree, b_new, mu_fit, resid_m


@jax.jit
def joint_prognostic_single_step(key, state):
    """One pass: for each outcome, block every prognostic tree with (b0, b1)."""
    states = list(state.states)
    obs = jnp.stack([s.obs_mask for s in states])
    loadings = (state.gamma_loadings if state.sur_active
                else jnp.zeros_like(state.gamma_loadings))
    omega = observed_precision(
        obs, loadings, jnp.stack([s.sigma2 for s in states]))   # (n, M, M)
    resid_all = jnp.stack([s.resid for s in states]).astype(jnp.float64)

    keys = random.split(key, len(states))
    for m, s in enumerate(states):
        if not s.adaptive_coding:
            # Without adaptive coding there is no coding coefficient to block
            # in, and bartz already draws the leaves conditionally.
            continue
        prec = omega[:, m, m]
        score = jnp.einsum('ik,ki->i', omega[:, m, :], resid_all)
        leaf_tree, b_new, mu_fit, resid_m = _outcome_block(
            keys[m], s, prec, score, resid_all[m])
        resid_all = resid_all.at[m].set(jnp.where(obs[m], resid_m, 0.0))
        states[m] = eqx.tree_at(
            lambda s: (s.forest.leaf_tree, s.b0, s.b1, s.mu_fit, s.resid), s,
            (leaf_tree, b_new[0].astype(jnp.float32),
             b_new[1].astype(jnp.float32), mu_fit.astype(jnp.float32),
             jnp.where(obs[m], resid_m, 0.0).astype(jnp.float32)))
    return eqx.tree_at(lambda s: s.states, state, tuple(states))


def joint_prognostic_step(key, state):
    """Chain-aware wrapper, mirroring ``_joint_gaussian.joint_gaussian_step``."""
    if not state.has_chain_axis:
        return joint_prognostic_single_step(key, state)
    per, shared = split_multi_chain_fields(state)

    def one(k, p):
        updated = joint_prognostic_single_step(k, eqx.combine(p, shared))
        return split_multi_chain_fields(updated)[0]

    updated = jax.vmap(one)(random.split(key, state.num_chains), per)
    return eqx.combine(updated, shared)
