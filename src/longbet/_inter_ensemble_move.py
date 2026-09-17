# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inter-ensemble residual-transfer Metropolis-Hastings step for LongBet.

On staggered panels where treated and control cohorts follow diverging baseline
trends, the prognostic ensemble ``mu(X_i, t)`` and treatment ensemble
``beta_{S_it} * nu(X_i, t)`` can trade off post-adoption level and calendar-time
slope on treated cells (Z_it = 1). Sequential Gibbs updates update one forest
holding the other fixed, which can trap chains in local modes with elevated
R-hat.

This module implements a joint Metropolis-Hastings step that proposes a
synchronized 2-parameter (level + normalized calendar-time slope) translation
``delta_0 + delta_1 * t_norm`` added to the active leaves of ``mu`` and
subtracted (scaled by ``1 / beta_S``) from the active leaves of ``nu`` on
treated cells. Because the translation vector depends only on fixed tree
topologies and the current GP draw ``beta``, the transformation has unit
Jacobian determinant (``|J| = 1``) and leaves the target posterior distribution
invariant.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32, Int32, Key

from bartz.mcmcstep._step import apply_moves_to_leaf_indices
from longbet._ridge import compute_active_leaf_stats


def _leaf_means(
    ids: Int32[Array, ' n'],
    values: Float32[Array, ' n'],
    mask: Bool[Array, ' n'],
    max_slots: int,
) -> tuple[Float32[Array, ' max_slots'], Float32[Array, ' max_slots']]:
    """Compute per-slot mean of ``values`` and active observation count."""
    v_masked = jnp.where(mask, values, 0.0).astype(jnp.float32)
    m_masked = mask.astype(jnp.float32)
    sum_v = jnp.zeros(max_slots, dtype=jnp.float32).at[ids].add(v_masked)
    cnt = jnp.zeros(max_slots, dtype=jnp.float32).at[ids].add(m_masked)
    mean_v = jnp.where(cnt > 0, sum_v / jnp.maximum(cnt, 1.0), 0.0)
    return mean_v, cnt


def inter_ensemble_transfer_step(
    key: Key[Array, ''],
    forest_mu: Any,
    mu_fit: Float32[Array, ' n'],
    forest_nu: Any,
    nu_fit: Float32[Array, ' n'],
    R: Float32[Array, ' n'],
    alpha: Float32[Array, ''],
    w: Float32[Array, ' n'],
    obs_mask: Bool[Array, ' n'],
    z_vec: Float32[Array, ' n'],
    time_idx: Int32[Array, ' n'],
    T_periods: int,
    sigma2: Float32[Array, ''],
    leaf_prior_cov_inv_mu: Float32[Array, ''],
    leaf_prior_cov_inv_nu: Float32[Array, ''],
    conditional_precision: Float32[Array, ' n'] | None = None,
    temperature: Float32[Array, ''] = jnp.float32(1.0),
    proposal_sigma: float = 0.05,
    min_abs_beta: float = 0.05,
) -> tuple[Any, Float32[Array, ' n'], Any, Float32[Array, ' n'], Float32[Array, ' n']]:
    """Execute one joint level-and-slope transfer MH step between mu and nu leaves.

    Parameters
    ----------
    key
        PRNG key.
    forest_mu, mu_fit
        Current prognostic forest and its fit vector in data units.
    forest_nu, nu_fit
        Current treatment forest and its fit vector in data units.
    R
        Current full-model residual in data units, shape ``(n,)``.
    alpha
        Prognostic scaling factor (scalar).
    w
        Treatment multiplier ``b_z * beta_S`` per cell, shape ``(n,)``.
    obs_mask
        Boolean mask of observed cells, shape ``(n,)``.
    z_vec
        Treatment indicator vector (1.0 treated, 0.0 untreated), shape ``(n,)``.
    time_idx
        Integer calendar period index ``0 .. T_periods - 1``, shape ``(n,)``.
    T_periods
        Total number of calendar periods (static int).
    sigma2
        Current innovation variance (scalar).
    leaf_prior_cov_inv_mu, leaf_prior_cov_inv_nu
        Leaf prior precisions in data units for ``mu`` and ``nu``.
    conditional_precision
        Optional per-cell precision vector for SUR coupling.
    temperature
        Inverse temperature for parallel tempering (default 1.0).
    proposal_sigma
        Standard deviation of the normal proposal for ``(delta_0, delta_1)``.
    min_abs_beta
        Minimum average treatment multiplier ``|w|`` required in a ``nu`` leaf
        for that leaf to participate in the compensating shift.

    Returns
    -------
    forest_mu_new, mu_fit_new, forest_nu_new, nu_fit_new, R_new
    """
    k_prop, k_acc = jax.random.split(key)

    deltas = (
        jax.random.normal(k_prop, shape=(2,), dtype=jnp.float32) * proposal_sigma
    )
    delta_0, delta_1 = deltas[0], deltas[1]

    denom_t = jnp.float32(max(1, T_periods - 1))
    t_norm = (time_idx.astype(jnp.float32) - 0.5 * denom_t) / denom_t

    ids_mu = apply_moves_to_leaf_indices(
        forest_mu.leaf_indices, forest_mu.to_prune, forest_mu.move_node
    )
    ids_nu = apply_moves_to_leaf_indices(
        forest_nu.leaf_indices, forest_nu.to_prune, forest_nu.move_node
    )

    _, _, is_leaf_mu = compute_active_leaf_stats(
        forest_mu.split_tree, forest_mu.leaf_tree
    )
    _, _, is_leaf_nu = compute_active_leaf_stats(
        forest_nu.split_tree, forest_nu.leaf_tree
    )

    max_slots_mu = forest_mu.leaf_tree.shape[-1]
    max_slots_nu = forest_nu.leaf_tree.shape[-1]
    m_mu = jnp.float32(forest_mu.leaf_tree.shape[0])
    m_nu = jnp.float32(forest_nu.leaf_tree.shape[0])

    # 1. Proposed shift on prognostic forest mu (over all observed cells)
    t_mean_mu, cnt_mu = jax.vmap(_leaf_means, in_axes=(0, None, None, None))(
        ids_mu, t_norm, obs_mask, max_slots_mu
    )
    active_mu = is_leaf_mu & (cnt_mu > 0)
    shift_mu_data = jnp.where(
        active_mu, (delta_0 + delta_1 * t_mean_mu) / m_mu, 0.0
    )
    prop_leaf_tree_mu = (
        forest_mu.leaf_tree + shift_mu_data / forest_mu.leaf_unit
    )

    # 2. Proposed compensating shift on treatment forest nu (over treated observed cells)
    trt_obs_mask = obs_mask & (z_vec == 1.0)
    t_mean_nu, cnt_nu = jax.vmap(_leaf_means, in_axes=(0, None, None, None))(
        ids_nu, t_norm, trt_obs_mask, max_slots_nu
    )
    w_mean_nu, _ = jax.vmap(_leaf_means, in_axes=(0, None, None, None))(
        ids_nu, w, trt_obs_mask, max_slots_nu
    )
    active_nu = (
        is_leaf_nu
        & (cnt_nu > 0)
        & (jnp.abs(w_mean_nu) > min_abs_beta)
    )
    safe_w = jnp.where(active_nu, w_mean_nu, 1.0)
    shift_nu_data = jnp.where(
        active_nu,
        -alpha * (delta_0 + delta_1 * t_mean_nu) / (m_nu * safe_w),
        0.0,
    )
    prop_leaf_tree_nu = (
        forest_nu.leaf_tree + shift_nu_data / forest_nu.leaf_unit
    )

    # 3. Evaluate proposed forest fits and full-model residual
    vals_mu_prop = jax.vmap(lambda leaf, index: leaf[index])(
        prop_leaf_tree_mu, ids_mu
    )
    mu_fit_prop = forest_mu.offset + forest_mu.leaf_unit * jnp.sum(
        vals_mu_prop.astype(jnp.float32), axis=0
    )

    vals_nu_prop = jax.vmap(lambda leaf, index: leaf[index])(
        prop_leaf_tree_nu, ids_nu
    )
    nu_fit_prop = forest_nu.offset + forest_nu.leaf_unit * jnp.sum(
        vals_nu_prop.astype(jnp.float32), axis=0
    )

    R_prop = jnp.where(
        obs_mask,
        R - alpha * (mu_fit_prop - mu_fit) - w * (nu_fit_prop - nu_fit),
        0.0,
    )

    # 4. Log-likelihood difference
    sq_diff = jnp.where(obs_mask, jnp.square(R_prop) - jnp.square(R), 0.0)
    if conditional_precision is None:
        log_lik_diff = (-0.5 * jnp.sum(sq_diff) / sigma2) * temperature
    else:
        log_lik_diff = -0.5 * jnp.sum(sq_diff * conditional_precision)

    # 5. Log-prior difference across actual leaves
    curr_mu_data = forest_mu.leaf_tree * forest_mu.leaf_unit
    prop_mu_data = prop_leaf_tree_mu * forest_mu.leaf_unit
    log_prior_diff_mu = -0.5 * leaf_prior_cov_inv_mu * jnp.sum(
        jnp.where(
            is_leaf_mu,
            jnp.square(prop_mu_data) - jnp.square(curr_mu_data),
            0.0,
        )
    )

    curr_nu_data = forest_nu.leaf_tree * forest_nu.leaf_unit
    prop_nu_data = prop_leaf_tree_nu * forest_nu.leaf_unit
    log_prior_diff_nu = -0.5 * leaf_prior_cov_inv_nu * jnp.sum(
        jnp.where(
            is_leaf_nu,
            jnp.square(prop_nu_data) - jnp.square(curr_nu_data),
            0.0,
        )
    )

    log_alpha = log_lik_diff + log_prior_diff_mu + log_prior_diff_nu
    log_u = jnp.log(
        jax.random.uniform(k_acc, shape=(), dtype=jnp.float32)
        + jnp.finfo(jnp.float32).tiny
    )
    accept = log_u < log_alpha

    new_leaf_tree_mu = jnp.where(accept, prop_leaf_tree_mu, forest_mu.leaf_tree)
    forest_mu_new = eqx.tree_at(lambda f: f.leaf_tree, forest_mu, new_leaf_tree_mu)
    mu_fit_new = jnp.where(accept, mu_fit_prop, mu_fit)

    new_leaf_tree_nu = jnp.where(accept, prop_leaf_tree_nu, forest_nu.leaf_tree)
    forest_nu_new = eqx.tree_at(lambda f: f.leaf_tree, forest_nu, new_leaf_tree_nu)
    nu_fit_new = jnp.where(accept, nu_fit_prop, nu_fit)

    R_new = jnp.where(accept, R_prop, R)

    return forest_mu_new, mu_fit_new, forest_nu_new, nu_fit_new, R_new
