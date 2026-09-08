"""Experimental whole-tree independence moves; not used by the public driver.

Draw a complete private tree from its original geometric BART prior, including
its shape. Reject unsupported candidates once, without conditioning individual
rules or terminal probabilities on sample counts. For supported trees, the
independence proposal and tree-prior factors cancel exactly. Acceptance uses
the outcome's likelihood with ALL prognostic/private/shared leaves and unit
intercepts integrated. All those coefficients are immediately redrawn after
acceptance or rejection, under the retained topology.

The optional compact-system restriction rejects proposed overflow and is an
identity at current overflow. Thus this kernel is reversible on each side of
the capacity boundary; ordinary transitions retain access to the full model.
X64 must remain enabled during tracing and execution; stored states stay f32.
"""
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from bartz.grove import traverse_forest
from bartz.grove._grove import is_actual_leaf
from bartz.mcmcstep._moves import split_range

from longbet._collapsed_exposure import (
    MAX_ENSEMBLE_LEAVES, _pack, collapsed_statistics, collapsed_draw,
)
from longbet._multi_state import split_multi_chain_fields
from longbet._shared_forest import observed_precision


REFRESH_TREES_SEMANTICS = 'refresh_trees_v1'


def _rule_ranges(var, split, max_split, blocked_vars, node):
    lo, hi = jax.vmap(lambda v: split_range(var, split, max_split, node, v))(
        jnp.arange(max_split.size))
    eligible = hi > lo
    if blocked_vars is not None:
        eligible &= ~jnp.any(jnp.arange(max_split.size)[:, None] ==
                             blocked_vars[None, :], axis=1)
    return lo, hi, eligible


def draw_prior_tree(key, var_template, split_template, max_split,
                    blocked_vars, log_s, p_nonterminal):
    """Draw the unrestricted geometric prior; templates supply shapes/dtypes.

    Only reachable non-bottom nodes enter the breadth-first queue. Leaves are
    forced terminal when ancestor restrictions exhaust the available rules or
    the original maximum depth is reached. Sample-count limits are deliberately
    absent here: callers reject the entire candidate if those limits fail.
    """
    var, split = jnp.zeros_like(var_template), jnp.zeros_like(split_template)
    size = split.size
    if size <= 1:
        return var, split
    logits = jnp.zeros(max_split.size, jnp.float64) if log_s is None else log_s.astype(jnp.float64)
    queue = jnp.zeros(size, jnp.int32).at[0].set(1)

    def body(carry):
        v, sp, pending, head, tail = carry
        node = pending[head]
        kn, kv, kc = random.split(random.fold_in(key, node), 3)
        lo, hi, eligible = _rule_ranges(v, sp, max_split, blocked_vars, node)
        grow = random.bernoulli(kn, p_nonterminal[node].astype(jnp.float64)) & eligible.any()
        # Safe dummy draws at forced terminals never enter the returned tree.
        safe_logits = jnp.where(eligible.any(), jnp.where(eligible, logits, -jnp.inf), 0.)
        variable = random.categorical(kv, safe_logits)
        cut = random.randint(kc, (), lo[variable], jnp.maximum(hi[variable], lo[variable]+1))
        v = v.at[node].set(jnp.where(grow, variable, 0).astype(v.dtype))
        sp = sp.at[node].set(jnp.where(grow, cut, 0).astype(sp.dtype))

        def append(args):
            q, end = args
            q = q.at[end].set(2*node).at[end+1].set(2*node+1)
            return q, end+2

        pending, tail = jax.lax.cond(grow & (2*node < size), append,
                                   lambda args: args, (pending, tail))
        return v, sp, pending, head+1, tail

    var, split, _, _, _ = jax.lax.while_loop(lambda c: c[3] < c[4], body,
        (var, split, queue, jnp.int32(0), jnp.int32(1)))
    return var, split


def tree_support(var, split, X, min_leaf, min_decision):
    """Check reachable count support, using ALL design cells regardless of y."""
    size = 2*split.size
    ids = traverse_forest(X, var[None, :], split[None, :])[0]
    counts = jnp.bincount(ids, length=size)
    active = is_actual_leaf(split, add_bottom_level=True)
    supported = jnp.all(jnp.where(active, counts >= min_leaf, True))
    node_counts = counts
    for depth in range(split.size.bit_length()-2, -1, -1):
        lo, hi = 2**depth, 2**(depth+1)
        node_counts = node_counts.at[lo:hi].add(
            node_counts[2*lo:2*hi].reshape(-1, 2).sum(axis=1))
    supported &= jnp.all(jnp.where(split != 0,
                                   node_counts[:split.size] >= min_decision, True))
    return supported, ids, counts


def _affluence(var, split, counts, forest, capacity):
    """Exact growable-leaf mask for a retained tree fitting the ensemble cap."""
    leaves = is_actual_leaf(split, add_bottom_level=True)[:split.size]
    nodes = jnp.nonzero(leaves, size=capacity, fill_value=0)[0]
    eligible = jax.vmap(lambda node: _rule_ranges(var, split, forest.max_split,
                                                 forest.blocked_vars, node)[2])(nodes)
    threshold = 0 if forest.min_points_per_decision_node is None else forest.min_points_per_decision_node
    growable = (nodes != 0) & eligible.any(axis=1) & (forest.p_nonterminal[nodes] > 0)
    growable &= counts[nodes] >= threshold
    return jnp.zeros(split.size, bool).at[nodes].set(growable)


def _refresh_forest(forest, leaves, X, precision_weights):
    """Synchronize materialized membership and caches after the joint draw."""
    ids = traverse_forest(X, forest.var_tree, forest.split_tree)
    forest = eqx.tree_at(lambda f: (f.leaf_tree, f.leaf_indices, f.to_prune, f.move_node),
        forest, (leaves, ids.astype(forest.leaf_indices.dtype),
                 jnp.zeros_like(forest.to_prune), jnp.zeros_like(forest.move_node)))
    if forest.count_tree is not None:
        counts = jax.vmap(lambda index: jnp.bincount(index, length=leaves.shape[-1]))(ids)
        forest = eqx.tree_at(lambda f: f.count_tree, forest, counts.astype(forest.count_tree.dtype))
    if forest.prec_tree is not None:
        weights = jnp.ones(X.shape[-1], jnp.float32) if precision_weights is None else precision_weights
        sums = jax.vmap(lambda index: jnp.zeros(leaves.shape[-1], jnp.float32).at[index].add(weights))(ids)
        forest = eqx.tree_at(lambda f: f.prec_tree, forest, sums)
    fitted = forest.offset+forest.leaf_unit*jnp.sum(
        jnp.take_along_axis(leaves, ids, axis=1).astype(jnp.float32), axis=0)
    return forest, fitted


def _outcome_refresh(key, state, m, precision, score, treatment, capacity):
    child, shared = state.states[m], state.shared_forest
    current_pack = _pack(child, shared, m, capacity)
    current_count = current_pack[5]

    def move(_):
        kj, kp, ka, kd = random.split(key, 4)
        forest = child.forest_nu if treatment else child.forest
        j = random.randint(kj, (), 0, forest.leaf_tree.shape[0])
        var, split = draw_prior_tree(kp, forest.var_tree[j], forest.split_tree[j],
            forest.max_split, forest.blocked_vars, forest.log_s, forest.p_nonterminal)
        supported, _, counts = tree_support(var, split, child.X,
            0 if forest.min_points_per_leaf is None else forest.min_points_per_leaf,
            0 if forest.min_points_per_decision_node is None else forest.min_points_per_decision_node)
        candidate_forest = eqx.tree_at(lambda f: (f.var_tree, f.split_tree), forest,
            (forest.var_tree.at[j].set(var), forest.split_tree.at[j].set(split)))
        if treatment:
            candidate_child = eqx.tree_at(lambda s: s.forest_nu, child, candidate_forest)
        else:
            candidate_child = eqx.tree_at(lambda s: s.forest, child, candidate_forest)
        candidate_pack = _pack(candidate_child, shared, m, capacity)
        candidate_count = candidate_pack[5]
        old_w = jnp.where(child.z_vec == 1, child.b1, child.b0).astype(jnp.float64)*child.beta[child.exposure_idx]
        alpha = child.alpha.astype(jnp.float64)
        old_mean = alpha*child.mu_fit+old_w*child.nu_fit+child.gamma[child.unit_idx]
        pseudo = score/jnp.where(precision > 0, precision, 1.)+old_mean
        unit_sd = jnp.sqrt(child.sigma_gamma2).astype(jnp.float64) if child.random_intercept else jnp.float64(0.)
        fixed = alpha*child.forest.offset+old_w*child.forest_nu.offset
        if not child.random_intercept:
            fixed += child.gamma[child.unit_idx]
        response = jnp.where(child.obs_mask, pseudo-fixed, 0.)

        def statistics(packed):
            base, is_trt = packed[:2]
            design = base*jnp.where(is_trt[None, :], old_w[:, None], alpha)
            return collapsed_statistics(design, response, precision, child.unit_idx,
                                        unit_sd, child.N_units)

        old_stats = statistics(current_pack)
        # Never use a truncated candidate integral for either the target or draw.
        valid = supported & (candidate_count <= capacity)
        proposed_stats = jax.lax.cond(valid, lambda _: statistics(candidate_pack),
                                      lambda _: old_stats, operand=None)
        log_ratio = jnp.where(valid, proposed_stats[-1]-old_stats[-1], -jnp.inf)
        accepted = valid & jnp.isfinite(log_ratio) & (jnp.log(random.uniform(ka, dtype=jnp.float64)) < log_ratio)
        stats = jax.tree.map(lambda a, b: jnp.where(accepted, b, a), old_stats, proposed_stats)
        # Shape changes can renumber every compact column after this tree. All
        # candidate membership, packed indices, priors AND physical values move
        # together; reusing old compact metadata here would corrupt the draw.
        retained_pack = jax.tree.map(lambda a, b: jnp.where(accepted, b, a),
                                    current_pack[:-1], candidate_pack[:-1])
        _, _, leaf_sd, packed, values, _ = retained_pack
        sizes = current_pack[-1]
        x, units = collapsed_draw(kd, stats)
        values = values.at[packed].set(leaf_sd*x, mode='drop')
        mu_leaves = (values[:sizes[0]].reshape(child.forest.leaf_tree.shape)/child.forest.leaf_unit).astype(child.forest.leaf_tree.dtype)
        nu_leaves = (values[sizes[0]:sizes[0]+sizes[1]].reshape(child.forest_nu.leaf_tree.shape)/child.forest_nu.leaf_unit).astype(child.forest_nu.leaf_tree.dtype)
        retained_var = jnp.where(accepted, var, forest.var_tree[j])
        retained_split = jnp.where(accepted, split, forest.split_tree[j])
        # Candidate affluence only matters on acceptance; accepted candidates fit
        # capacity, so every actual leaf is represented by the compact mask.
        candidate_affluence = _affluence(var, split, counts, forest, capacity)
        forest = eqx.tree_at(lambda f: (f.var_tree, f.split_tree, f.affluence_tree), forest,
            (forest.var_tree.at[j].set(retained_var), forest.split_tree.at[j].set(retained_split),
             forest.affluence_tree.at[j].set(jnp.where(accepted, candidate_affluence, forest.affluence_tree[j]))))
        mu_forest = child.forest if treatment else forest
        nu_forest = forest if treatment else child.forest_nu
        mu_forest, mu_fit = _refresh_forest(mu_forest, mu_leaves, child.X, child.prec_scale)
        nu_forest, nu_fit = _refresh_forest(nu_forest, nu_leaves, child.X, child.prec_scale_nu)
        shared_new = shared
        if shared is not None:
            shared_values = values[sum(sizes[:2]):].reshape(shared.leaf_tree[:, m, :].shape).astype(jnp.float32)
            shared_fit = jnp.sum(jnp.take_along_axis(shared_values, shared.leaf_indices, axis=1), axis=0)
            shared_new = eqx.tree_at(lambda f: (f.leaf_tree, f.fit), shared,
                (shared.leaf_tree.at[:, m, :].set(shared_values), shared.fit.at[m].set(shared_fit)))
            nu_fit += shared_fit
        gamma_new = (unit_sd*units).astype(jnp.float32) if child.random_intercept else child.gamma
        new_mean = alpha*mu_fit+old_w*nu_fit+gamma_new[child.unit_idx]
        residual = jnp.where(child.obs_mask, child.resid.astype(jnp.float64)+old_mean-new_mean, 0.).astype(jnp.float32)
        updated_child = eqx.tree_at(lambda s: (s.forest, s.forest_nu, s.gamma,
                                              s.mu_fit, s.nu_fit, s.resid), child,
            (mu_forest, nu_forest, gamma_new, mu_fit, nu_fit, residual))
        children = list(state.states)
        children[m] = updated_child
        updated = eqx.tree_at(lambda s: (s.states, s.shared_forest), state,
                              (tuple(children), shared_new), is_leaf=lambda x: x is None)
        old_split = (child.forest_nu if treatment else child.forest).split_tree[j]
        old_var = (child.forest_nu if treatment else child.forest).var_tree[j]
        shape_changed = jnp.any((old_split != 0) != (split != 0))
        root_changed = (old_split[1] != split[1]) | ((split[1] != 0) & (old_var[1] != var[1]))
        info = jnp.array([1., accepted, log_ratio, current_count, candidate_count,
                          supported, accepted & shape_changed, accepted & root_changed], jnp.float64)
        return updated, info

    return jax.lax.cond(current_count <= capacity, move,
        lambda _: (state, jnp.array([0., 0., 0., current_count, 0., 0., 0., 0.], jnp.float64)),
        operand=None)


@partial(jax.jit, static_argnames=('capacity',))
def refresh_trees_single_step(key, state, *, capacity=MAX_ENSEMBLE_LEAVES):
    observed = jnp.stack([s.obs_mask for s in state.states])
    loadings = state.gamma_loadings if state.sur_active else jnp.zeros_like(state.gamma_loadings)
    omega = observed_precision(observed, loadings, jnp.stack([s.sigma2 for s in state.states]))
    diagnostics = []
    for m in range(state.M):
        outcome_info = []
        for kind, treatment in enumerate((False, True)):
            raw = jnp.stack([s.resid for s in state.states]).astype(jnp.float64)
            score = jnp.einsum('ik,ki->i', omega[:, m, :], raw)
            state, info = _outcome_refresh(random.fold_in(random.fold_in(key, m), kind),
                state, m, omega[:, m, m], score, treatment, capacity)
            outcome_info.append(info)
        diagnostics.append(jnp.stack(outcome_info))
    return state, jnp.stack(diagnostics)


def refresh_trees_step(key, state, *, capacity=MAX_ENSEMBLE_LEAVES):
    """One private tree per forest/outcome; diagnostics have (..., M, 2, 8)."""
    if not isinstance(capacity, int) or capacity < 1:
        raise ValueError('capacity must be a positive integer')
    if not state.has_chain_axis:
        return refresh_trees_single_step(key, state, capacity=capacity)
    per, constants = split_multi_chain_fields(state)
    keys = random.split(key, state.num_chains)
    def one(k, p):
        updated, info = refresh_trees_single_step(k, eqx.combine(p, constants), capacity=capacity)
        return split_multi_chain_fields(updated)[0], info
    updated, info = jax.vmap(one)(keys, per)
    return eqx.combine(updated, constants), info
