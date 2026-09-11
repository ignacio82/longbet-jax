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

"""Global-scale Metropolis move along the beta-nu ridge.

The treatment term enters as a product ``beta_S * nu(X_i, S, t)`` and the
likelihood does not pin down the split: for any positive ``c``,
``(beta, nu) -> (c * beta, nu / c)`` reproduces the fit exactly.  A Metropolis
grow/prune sampler traverses that ridge slowly with local tree moves alone, so
this module gives it an explicit move along the manifold.

Because the likelihood is invariant, the acceptance ratio needs only the two
priors and the Jacobian.  The move costs ``O(S_max^2 + L)`` and can cross the
ridge in a single accepted step.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, UInt

from bartz.grove._grove import is_actual_leaf

from longbet._gp import beta_prior_quadform


def compute_active_leaf_stats(
    split_tree: UInt[Array, 'num_trees half_tree_size'],
    leaf_tree: Float[Array, 'num_trees 2*half_tree_size'],
) -> tuple[Int32[Array, ''], Float[Array, ''], Bool[Array, 'num_trees 2*half_tree_size']]:
    """Count the forest's actual leaves and sum their squares.

    Returns the count, the sum of squared leaf values (in storage units), and
    the boolean mask of actual-leaf slots.  The mask is returned so the caller
    can rescale exactly the values the Jacobian accounts for.
    """
    st_flat = split_tree.reshape(-1, split_tree.shape[-1])
    is_leaf_flat = jax.vmap(lambda s: is_actual_leaf(s, add_bottom_level=True))(st_flat)
    is_leaf = is_leaf_flat.reshape((*split_tree.shape[:-1], is_leaf_flat.shape[-1]))

    num_active = jnp.sum(is_leaf, axis=(-2, -1))
    sum_sq = jnp.sum(jnp.square(leaf_tree) * is_leaf, axis=(-2, -1))
    return num_active, sum_sq, is_leaf


def ridge_scale_step(
    key: Key[Array, ''],
    beta: Float32[Array, ' S_max_plus_1'],
    forest_nu: Any,
    nu_fit: Float32[Array, ' n'],
    K_chol: Float32[Array, 'S_max_plus_1 S_max_plus_1'],
    leaf_prior_cov_inv_nu: Float32[Array, ''],
    proposal_sigma: float = 0.2,
) -> tuple[Float32[Array, ' S_max_plus_1'], Any, Float32[Array, ' n']]:
    """One Metropolis step on ``log c`` for ``(beta, ell) -> (c beta, ell / c)``.

    With ``beta ~ N(0, K_tilde)`` over ``d = S_max + 1`` points and ``L`` active
    leaves ``ell ~ N(0, 1 / leaf_prior_cov_inv)``, the log acceptance ratio is

    .. code::

        -1/2 (c^2 - 1) beta^T K_tilde^-1 beta
        -1/2 leaf_prec (c^-2 - 1) sum(ell^2)
        + (d - L) log c

    the last term being the Jacobian of the map.  The proposal on ``log c`` is
    symmetric, so it drops out.

    Only the *actual leaves* are rescaled, matching the ``L`` counted in the
    Jacobian; unreachable leaf slots are left alone (`bartz` redraws leaf values
    when a node is grown, so their contents never enter the model).

    ``nu_fit`` is rescaled alongside, and the full-model residual is left
    untouched -- it is invariant by construction, which is the point of the move.

    Parameters
    ----------
    key
        PRNG key.
    beta
        Current exposure trajectory, shape ``(S_max + 1,)``.
    forest_nu
        Current treatment forest.
    nu_fit
        Current treatment forest fit, shape ``(n,)``.
    K_chol
        Lower Cholesky factor of the marginalized GP kernel.
    leaf_prior_cov_inv_nu
        Prior precision of the treatment forest's leaves, in data units.
    proposal_sigma
        Standard deviation of the ``log c`` proposal.

    Returns
    -------
    beta_new, forest_nu_new, nu_fit_new
    """
    k_prop, k_acc = jax.random.split(key)

    log_c = jax.random.normal(k_prop, shape=(), dtype=jnp.float32) * proposal_sigma
    c = jnp.exp(log_c)

    # beta^T K_tilde^-1 beta by triangular solve; never forms the precision.
    beta_quad = beta_prior_quadform(beta, K_chol)
    log_p_beta_diff = -0.5 * (jnp.square(c) - 1.0) * beta_quad

    num_active, sum_sq, is_leaf = compute_active_leaf_stats(
        forest_nu.split_tree, forest_nu.leaf_tree
    )
    sum_sq_data = sum_sq * jnp.square(forest_nu.leaf_unit)
    log_p_leaves_diff = (
        -0.5 * leaf_prior_cov_inv_nu * (jnp.reciprocal(jnp.square(c)) - 1.0) * sum_sq_data
    )

    d = beta.shape[-1]
    log_jacobian = (d - num_active.astype(jnp.float32)) * log_c

    log_alpha = log_p_beta_diff + log_p_leaves_diff + log_jacobian
    log_u = jnp.log(
        jax.random.uniform(k_acc, shape=(), dtype=jnp.float32) + jnp.finfo(jnp.float32).tiny
    )
    accept = log_u < log_alpha
    c_eff = jnp.where(accept, c, 1.0)

    beta_new = beta * c_eff
    new_leaf_tree = jnp.where(is_leaf, forest_nu.leaf_tree / c_eff, forest_nu.leaf_tree)
    forest_nu_new = eqx.tree_at(lambda f: f.leaf_tree, forest_nu, new_leaf_tree)
    nu_fit_new = nu_fit / c_eff

    return beta_new, forest_nu_new, nu_fit_new
