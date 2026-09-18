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

"""Exact conjugate subspace Gibbs inter-ensemble residual-transfer step for LongBet.

On staggered panels where treated and control cohorts follow diverging baseline
trends, the prognostic ensemble ``mu(X_i, t)`` and treatment ensemble
``beta_{S_it} * nu(X_i, t)`` can trade off post-adoption level, calendar-time
slope, and curvature on treated cells (Z_it = 1). Sequential Gibbs updates
update one forest holding the other fixed, which can trap chains in local modes
with elevated R-hat.

This module implements an exact conjugate subspace Gibbs step that parameterizes
an orthogonal 4-dimensional transfer subspace delta in R^4 spanning:
  - constant level (1)
  - linear calendar time trend (t_norm)
  - quadratic calendar time curvature (t_norm^2 - 1/12)
  - treatment exposure profile (w_norm)
weighted by each prognostic leaf's squared treated observation fraction (p_trt^2),
so pure untreated control leaves remain untouched. Because both the Gaussian
likelihood (or probit/ordinal latent Gaussian likelihood) and the leaf priors
are quadratic in the leaf values, the conditional posterior distribution
p(delta | Y, trees, beta, sigma^2) is an exact 4-dimensional Gaussian
distribution. Drawing delta from this conditional has 100% acceptance
probability, requires no step-size tuning, and leaves the target joint posterior
distribution strictly invariant.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32, Int32, Key

from bartz.mcmcstep._step import apply_moves_to_leaf_indices
from longbet._ridge import compute_active_leaf_stats


def _leaf_stats_vec(
    ids: Int32[Array, ' n'],
    Phi: Float32[Array, 'n K'],
    mask: Bool[Array, ' n'],
    max_slots: int,
) -> tuple[Float32[Array, 'max_slots K'], Float32[Array, ' max_slots']]:
    """Compute per-slot mean of ``Phi`` (shape ``(n, K)``) and active count."""
    m_f32 = mask.astype(jnp.float32)
    Phi_masked = jnp.where(mask[:, None], Phi, 0.0).astype(jnp.float32)
    sum_Phi = jnp.zeros((max_slots, Phi.shape[1]), dtype=jnp.float32).at[ids].add(Phi_masked)
    cnt = jnp.zeros(max_slots, dtype=jnp.float32).at[ids].add(m_f32)
    mean_Phi = jnp.where(cnt[:, None] > 0, sum_Phi / jnp.maximum(cnt[:, None], 1.0), 0.0)
    return mean_Phi, cnt


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
    unit_idx: Int32[Array, ' n'] | None = None,
    N_units: int | None = None,
    X_unified: Any = None,
) -> tuple[Any, Float32[Array, ' n'], Any, Float32[Array, ' n'], Float32[Array, ' n']]:
    """Execute one exact conjugate subspace Gibbs transfer step between mu and nu leaves."""
    del proposal_sigma, unit_idx, N_units, X_unified
    denom_t = jnp.float32(max(1, T_periods - 1))
    t_norm = (time_idx.astype(jnp.float32) - 0.5 * denom_t) / denom_t
    t_quad = jnp.square(t_norm) - jnp.float32(1.0 / 12.0)
    w_norm = w / jnp.maximum(jnp.max(jnp.abs(w)), 1.0)

    Phi = jnp.stack(
        [
            jnp.ones_like(t_norm),
            t_norm,
            t_quad,
            w_norm,
        ],
        axis=-1,
    )  # shape (n, K=4)
    K = Phi.shape[1]

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

    # 1. Compute leaf basis directions A_mu (data units), shape (m_mu, max_slots_mu, K)
    trt_obs_mask = obs_mask & (z_vec == 1.0)
    Phi_mean_mu, cnt_all_mu = jax.vmap(
        _leaf_stats_vec, in_axes=(0, None, None, None)
    )(ids_mu, Phi, obs_mask, max_slots_mu)
    _, cnt_trt_mu = jax.vmap(
        _leaf_stats_vec, in_axes=(0, None, None, None)
    )(ids_mu, Phi[:, :1], trt_obs_mask, max_slots_mu)

    p_trt_mu = jnp.where(
        cnt_all_mu > 0, cnt_trt_mu / jnp.maximum(cnt_all_mu, 1.0), 0.0
    )
    active_mu = is_leaf_mu & (cnt_trt_mu > 0)
    weight_mu = jnp.where(active_mu, jnp.square(p_trt_mu) / m_mu, 0.0)
    A_mu = weight_mu[:, :, None] * Phi_mean_mu  # (m_mu, max_slots_mu, K)

    # Induced change J_mu in mu_fit per unit of delta: shape (n, K)
    J_mu_per_tree = jax.vmap(lambda a_tree, idx: a_tree[idx])(A_mu, ids_mu)
    J_mu = jnp.sum(J_mu_per_tree, axis=0)  # (n, K)

    # 2. Compute compensating leaf basis directions A_nu (data units), shape (m_nu, max_slots_nu, K)
    target_nu_cell = alpha * J_mu  # (n, K)
    target_mean_nu, cnt_trt_nu = jax.vmap(
        _leaf_stats_vec, in_axes=(0, None, None, None)
    )(ids_nu, target_nu_cell, trt_obs_mask, max_slots_nu)
    w_mean_nu, _ = jax.vmap(
        _leaf_stats_vec, in_axes=(0, None, None, None)
    )(ids_nu, w[:, None], trt_obs_mask, max_slots_nu)
    w_mean_nu = w_mean_nu[:, :, 0]

    active_nu = (
        is_leaf_nu
        & (cnt_trt_nu > 0)
        & (jnp.abs(w_mean_nu) > min_abs_beta)
    )
    safe_w_nu = jnp.where(active_nu, w_mean_nu, 1.0)
    A_nu = jnp.where(
        active_nu[:, :, None],
        -target_mean_nu / (m_nu * safe_w_nu[:, :, None]),
        0.0,
    )  # (m_nu, max_slots_nu, K)

    # Induced change J_nu in nu_fit per unit of delta: shape (n, K)
    J_nu_per_tree = jax.vmap(lambda a_tree, idx: a_tree[idx])(A_nu, ids_nu)
    J_nu = jnp.sum(J_nu_per_tree, axis=0)  # (n, K)

    # Total residual sensitivity H = alpha * J_mu + w * J_nu: shape (n, K)
    H = jnp.where(
        obs_mask[:, None],
        alpha * J_mu + w[:, None] * J_nu,
        0.0,
    )

    # 3. Assemble exact Gaussian conditional precision (Sigma_inv) and linear term (h)
    if conditional_precision is None:
        cell_prec = jnp.where(obs_mask, temperature / sigma2, 0.0)
    else:
        cell_prec = jnp.where(obs_mask, conditional_precision, 0.0)

    H_weighted = H * cell_prec[:, None]
    prec_lik = H.T @ H_weighted  # (K, K)
    h_lik = H_weighted.T @ R  # (K,)

    # Prior contribution from mu leaves
    curr_mu_data = forest_mu.leaf_tree * forest_mu.leaf_unit  # (m_mu, max_slots_mu)
    A_mu_flat = jnp.where(is_leaf_mu[:, :, None], A_mu, 0.0).reshape(-1, K)
    curr_mu_flat = jnp.where(is_leaf_mu, curr_mu_data, 0.0).reshape(-1)
    prec_mu = leaf_prior_cov_inv_mu * (A_mu_flat.T @ A_mu_flat)
    h_mu = -leaf_prior_cov_inv_mu * (A_mu_flat.T @ curr_mu_flat)

    # Prior contribution from nu leaves
    curr_nu_data = forest_nu.leaf_tree * forest_nu.leaf_unit  # (m_nu, max_slots_nu)
    A_nu_flat = jnp.where(is_leaf_nu[:, :, None], A_nu, 0.0).reshape(-1, K)
    curr_nu_flat = jnp.where(is_leaf_nu, curr_nu_data, 0.0).reshape(-1)
    prec_nu = leaf_prior_cov_inv_nu * (A_nu_flat.T @ A_nu_flat)
    h_nu = -leaf_prior_cov_inv_nu * (A_nu_flat.T @ curr_nu_flat)

    prec_total = prec_lik + prec_mu + prec_nu + jnp.eye(K, dtype=jnp.float32) * 1e-5
    h_total = h_lik + h_mu + h_nu

    # 4. Sample delta ~ N(Sigma_delta @ h_total, Sigma_delta)
    L_chol = jnp.linalg.cholesky(prec_total)
    mean_delta = jax.scipy.linalg.cho_solve((L_chol, True), h_total)
    eta = jax.random.normal(key, shape=(K,), dtype=jnp.float32)
    noise_delta = jax.scipy.linalg.solve_triangular(L_chol.T, eta, lower=False)
    delta = mean_delta + noise_delta

    # 5. Apply exact subspace update to leaf trees, forest fits, and residual
    shift_mu_data = jnp.einsum('tsk,k->ts', A_mu, delta)
    new_leaf_tree_mu = forest_mu.leaf_tree + shift_mu_data / forest_mu.leaf_unit
    forest_mu_new = eqx.tree_at(lambda f: f.leaf_tree, forest_mu, new_leaf_tree_mu)
    mu_fit_new = mu_fit + J_mu @ delta

    shift_nu_data = jnp.einsum('tsk,k->ts', A_nu, delta)
    new_leaf_tree_nu = forest_nu.leaf_tree + shift_nu_data / forest_nu.leaf_unit
    forest_nu_new = eqx.tree_at(lambda f: f.leaf_tree, forest_nu, new_leaf_tree_nu)
    nu_fit_new = nu_fit + J_nu @ delta

    R_new = jnp.where(obs_mask, R - H @ delta, 0.0)

    return forest_mu_new, mu_fit_new, forest_nu_new, nu_fit_new, R_new
