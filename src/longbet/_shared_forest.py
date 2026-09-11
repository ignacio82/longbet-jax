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

"""Shared partitions with vector leaves and an explicit grow/prune MH kernel.

Leaves are in physical internal-response units (not bartz storage units).
The rule prior is uniform over globally split-enabled variables, then over
that variable's global cutpoints. Infeasible rules are rejected, not redrawn.
Grow/prune each have probability 1/2, including null proposals at boundaries.
These conventions make every forward/reverse proposal probability explicit.
The depth prior is conditioned on the fixed observed-design minimum leaf size.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, random
try:
    from jax import enable_x64
except ImportError:  # JAX < 0.8
    from jax.experimental import enable_x64
from bartz._jaxext import field
from bartz.grove._grove import is_actual_leaf
from bartz.mcmcstep._axes import CHAIN_AXIS

SHARED_SAMPLER_SEMANTICS = "shared_private_sur_v1"


class SharedTreatmentForest(eqx.Module):
    leaf_tree: jax.Array = field(chains=CHAIN_AXIS)  # (*C, J, M, L)
    var_tree: jax.Array = field(chains=CHAIN_AXIS)   # (*C, J, L/2)
    split_tree: jax.Array = field(chains=CHAIN_AXIS)
    leaf_indices: jax.Array = field(chains=CHAIN_AXIS, data=-1)
    fit: jax.Array = field(chains=CHAIN_AXIS, data=-1)  # (*C, M, n)
    grow_prop_count: jax.Array = field(chains=CHAIN_AXIS)
    grow_acc_count: jax.Array = field(chains=CHAIN_AXIS)
    prune_prop_count: jax.Array = field(chains=CHAIN_AXIS)
    prune_acc_count: jax.Array = field(chains=CHAIN_AXIS)
    max_split: jax.Array = field()
    prior_precision: jax.Array = field()  # (M,M), diagonal in this version
    split_alpha: float = field(static=True)
    split_beta: float = field(static=True)
    min_points: int = field(static=True)


def init_shared_forest(config, M, n, max_split):
    J, L = config.num_shared_trees, 2**config.max_depth_trt
    var_dtype = jnp.uint8 if max_split.size <= 256 else jnp.uint16 if max_split.size <= 65536 else jnp.uint32
    return SharedTreatmentForest(
        leaf_tree=jnp.zeros((J, M, L), jnp.float32),
        var_tree=jnp.zeros((J, L//2), var_dtype),
        split_tree=jnp.zeros((J, L//2), jnp.uint8),
        leaf_indices=jnp.ones((J, n), jnp.int32),
        fit=jnp.zeros((M, n), jnp.float32),
        grow_prop_count=jnp.int32(0), grow_acc_count=jnp.int32(0),
        prune_prop_count=jnp.int32(0), prune_acc_count=jnp.int32(0),
        max_split=max_split,
        prior_precision=jnp.eye(M, dtype=jnp.float32) * J / config.shared_variance_fraction,
        split_alpha=config.alpha_split_trt, split_beta=config.beta_split_trt,
        min_points=config.min_points_per_leaf_trt,
    )


def observed_precision(observed, loadings, variances):
    """(n,M,M) precision for supported masks; Gamma=0 permits arbitrary masks."""
    B = jnp.eye(loadings.shape[0], dtype=jnp.float64) - loadings.astype(jnp.float64)
    weights = observed.astype(B.dtype) / variances[:, None]
    return jnp.einsum("km,ki,kl->iml", B, weights, B)


def vector_leaf_posterior(ids, gram, score, prior_precision, num_leaves):
    """Leaf precision, mean, Cholesky and log integrated likelihood contribution.

    gram_i = D_i' Omega_i D_i, score_i = D_i' Omega_i r_i.
    Scatter reductions allocate O(n M^2 + L M^2), never O(n L M^2).
    The response-only quadratic cancels between trees and is omitted.
    """
    M = prior_precision.shape[0]
    # Accumulate/factor in float64: near-collinear SUR equations can lose the
    # positive prior contribution in float32. Do not add an arbitrary jitter
    # that would silently change the leaf prior. This local context does not
    # change the scalar sampler's dtype policy or RNG schedule. The MCMC
    # driver keeps X64 enabled through BOTH tracing and lowering, not just
    # inside this function (a nested context is insufficient in JAX).
    P = jnp.zeros((num_leaves, M, M), jnp.float64).at[ids].add(gram.astype(jnp.float64))
    P = (P + jnp.swapaxes(P, -1, -2)) * .5 + prior_precision
    h = jnp.zeros((num_leaves, M), jnp.float64).at[ids].add(score.astype(jnp.float64))
    chol = jnp.linalg.cholesky(P)
    solve = lambda L, b: jax.scipy.linalg.cho_solve((L, True), b)
    mean = jax.vmap(solve)(chol, h)
    logdet = 2 * jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
    prior_logdet = jnp.linalg.slogdet(prior_precision)[1]
    log_integral = .5 * (prior_logdet - logdet + jnp.sum(h * mean, axis=-1))
    return P, mean, chol, log_integral


def _node_masks(split):
    leaves = is_actual_leaf(split, add_bottom_level=True)
    H = split.shape[-1]
    nodes = jnp.arange(H)
    growable = leaves[:H] & (nodes > 0)
    prunable = (split != 0) & leaves[2*nodes] & leaves[2*nodes+1]
    return leaves, growable, prunable


def _choose(key, mask):
    # An empty set produces a harmless placeholder; the proposal is rejected.
    logits = jnp.where(mask, 0., -jnp.inf)
    logits = jnp.where(jnp.any(mask), logits, jnp.zeros_like(logits))
    return random.categorical(key, logits).astype(jnp.int32)


def grow_log_ratio(alpha, beta, depth, bottom_children, n_grow, n_prune_new,
                   num_variables, num_cutpoints):
    """Separate tree-prior and reverse/forward log ratios for a grow move."""
    p = alpha / (1. + depth)**beta
    pc = jnp.where(bottom_children, 0., alpha / (2. + depth)**beta)
    log_rule_count = jnp.log(num_variables) + jnp.log(num_cutpoints)
    log_prior = jnp.log(p) + 2*jnp.log1p(-pc) - jnp.log1p(-p) - log_rule_count
    log_proposal = jnp.log(n_grow) + log_rule_count - jnp.log(n_prune_new)
    return log_prior, log_proposal


def shared_tree_step(key, leaves, var, split, ids, partial, gram, omega, w,
                     observed_any, forest, X):
    """A collapsed topology MH move followed by all conditional vector leaves."""
    k_kind, k_node, k_var, k_cut, k_acc, k_leaf = random.split(key, 6)
    L, H = leaves.shape[-1], split.shape[-1]
    old_mask, growable, prunable = _node_masks(split)
    grow = random.bernoulli(k_kind)
    candidates = jnp.where(grow, growable, prunable)
    node = _choose(k_node, candidates)
    variable = _choose(k_var, forest.max_split > 0)
    cut = random.randint(k_cut, (), 1, jnp.maximum(forest.max_split[variable], 1)+1)
    variable = jnp.where(grow, variable, var[node]).astype(var.dtype)
    new_var = var.at[node].set(jnp.where(grow, variable, 0))
    new_split = split.at[node].set(jnp.where(grow, cut, 0).astype(split.dtype))
    go_right = X[variable] >= cut
    proposed_ids = jnp.where(grow,
        jnp.where(ids == node, 2*node + go_right.astype(ids.dtype), ids),
        jnp.where((ids == 2*node) | (ids == 2*node+1), node, ids))
    new_mask, new_growable, new_prunable = _node_masks(new_split)
    counts = jnp.zeros(L, jnp.int32).at[proposed_ids].add(observed_any.astype(jnp.int32))
    valid = jnp.any(candidates) & jnp.any(forest.max_split > 0)
    valid &= (~grow) | ((counts[2*node] >= forest.min_points) &
                        (counts[2*node+1] >= forest.min_points))

    score = w.T * jnp.einsum("imk,ki->im", omega, partial)
    old = vector_leaf_posterior(ids, gram, score, forest.prior_precision, L)
    new = vector_leaf_posterior(proposed_ids, gram, score, forest.prior_precision, L)
    log_likelihood = jnp.sum(jnp.where(new_mask, new[3], 0)) - jnp.sum(jnp.where(old_mask, old[3], 0))
    depth = jnp.floor(jnp.log2(jnp.maximum(node, 1)))
    ng = jnp.where(grow, jnp.sum(growable), jnp.sum(new_growable))
    np_ = jnp.where(grow, jnp.sum(new_prunable), jnp.sum(prunable))
    lp, lq = grow_log_ratio(forest.split_alpha, forest.split_beta, depth,
                           2*node >= H, jnp.maximum(ng, 1), jnp.maximum(np_, 1),
                           jnp.maximum(jnp.sum(forest.max_split > 0), 1),
                           jnp.maximum(forest.max_split[variable], 1))
    log_ratio = log_likelihood + jnp.where(grow, lp+lq, -lp-lq)
    accept = valid & (jnp.log(random.uniform(k_acc)) < log_ratio)
    ids = jnp.where(accept, proposed_ids, ids)
    var = jnp.where(accept, new_var, var)
    split = jnp.where(accept, new_split, split)
    mask = jnp.where(accept, new_mask, old_mask)
    mean, chol = jnp.where(accept, new[1], old[1]), jnp.where(accept, new[2], old[2])
    noise = random.normal(k_leaf, mean.shape)
    draw = mean + jax.vmap(lambda l, e: jax.scipy.linalg.solve_triangular(l.T, e, lower=False))(chol, noise)
    leaves = jnp.where(mask[:, None], draw, 0.).T.astype(jnp.float32)
    counters = jnp.array([grow, grow & accept, ~grow, (~grow) & accept], jnp.int32)
    return leaves, var, split, ids, counters


def shared_forest_step(key, forest, states, loadings):
    """Backfit shared trees against current raw residuals and all outcomes."""
    raw = jnp.stack([s.resid for s in states])
    observed = jnp.stack([s.obs_mask for s in states])
    variances = jnp.stack([s.sigma2 for s in states])
    w = jnp.stack([jnp.where(s.z_vec == 1, s.b1, s.b0) *
                   s.beta[s.exposure_idx] for s in states])
    omega = observed_precision(observed, loadings, variances)
    gram = omega * w.T[:, :, None] * w.T[:, None, :]
    observed_any = jnp.any(observed, axis=0)

    def tree_step(resid, args):
        k, leaves, var, split, ids = args
        partial = resid + w * leaves[:, ids]
        updated = shared_tree_step(k, leaves, var, split, ids, partial, gram,
                                   omega, w, observed_any, forest, states[0].X)
        lv, _, _, idx, _ = updated
        resid = jnp.where(observed, partial - w * lv[:, idx], 0.)
        return resid, updated

    resid, (leaves, var, split, ids, counters) = lax.scan(tree_step, raw,
        (random.split(key, forest.leaf_tree.shape[0]), forest.leaf_tree,
         forest.var_tree, forest.split_tree, forest.leaf_indices))
    fit = jnp.sum(jax.vmap(lambda lv, idx: lv[:, idx])(leaves, ids), axis=0)
    counts = counters.sum(0, dtype=jnp.int32)
    updated = eqx.tree_at(lambda f: (f.leaf_tree, f.var_tree, f.split_tree,
        f.leaf_indices, f.fit, f.grow_prop_count, f.grow_acc_count,
        f.prune_prop_count, f.prune_acc_count), forest,
        (leaves, var, split, ids, fit, *counts))
    new_states = [eqx.tree_at(lambda s: (s.resid, s.nu_fit), st,
        (resid[m], st.nu_fit + fit[m] - forest.fit[m])) for m, st in enumerate(states)]
    return updated, new_states


def shared_ridge_step(key, forest, states):
    """Scale both shared and private leaves in each outcome's beta-nu move.

    Current leaf priors are diagonal, so this column-wise prior ratio is exact.
    Both ensembles' active dimensions enter the Jacobian. Raw fits are invariant.
    """
    from longbet._gp import beta_prior_quadform
    from longbet._ridge import compute_active_leaf_stats
    mask_shared = jax.vmap(lambda s: is_actual_leaf(s, add_bottom_level=True))(forest.split_tree)
    ls = jnp.sum(mask_shared)
    states = list(states)
    for m, st in enumerate(states):
        if not (st.ridge_move and st.sample_beta):
            continue
        kp, ka = random.split(random.fold_in(key, m))
        log_c = st.ridge_proposal_sigma * random.normal(kp, dtype=jnp.float32)
        c = jnp.exp(log_c)
        lp, sum_sq, private_mask = compute_active_leaf_stats(st.forest_nu.split_tree, st.forest_nu.leaf_tree)
        q_private = st.leaf_prior_cov_inv_nu * st.forest_nu.leaf_unit**2 * sum_sq
        q_shared = forest.prior_precision[m, m] * jnp.sum(mask_shared * forest.leaf_tree[:, m, :]**2)
        q_beta = beta_prior_quadform(st.beta, st.K_chol)
        log_ratio = -.5*(c*c-1)*q_beta - .5*(c**-2-1)*(q_private+q_shared)
        log_ratio += (st.beta.shape[-1] - lp - ls)*log_c
        scale = jnp.where(jnp.log(random.uniform(ka, dtype=jnp.float32)) < log_ratio, c, 1.)
        states[m] = eqx.tree_at(lambda s: (s.beta, s.forest_nu.leaf_tree, s.nu_fit), st,
            (st.beta*scale, jnp.where(private_mask, st.forest_nu.leaf_tree/scale,
                                     st.forest_nu.leaf_tree), st.nu_fit/scale))
        forest = eqx.tree_at(lambda f: (f.leaf_tree, f.fit), forest,
            (forest.leaf_tree.at[:, m, :].divide(scale), forest.fit.at[m].divide(scale)))
    return forest, states
