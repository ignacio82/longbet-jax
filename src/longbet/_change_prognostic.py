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

"""Benchmark-only root change/swap moves for prognostic trees.

Keep a tree's shape fixed, propose a root rule or swap it with an internal
descendant, integrate that tree's leaves, then redraw them immediately. The
full ancestor-restricted bartz rule prior and forced-terminal probabilities
are evaluated, including descendants affected by the changed root. Counts
restrict the reachable support but do not renormalize the splitting prior.
Three equally weighted kernels: global root rule, adjacent root cutpoint,
and root/descendant swap. Component-wise MH ratios preserve the same target.
"""
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
from bartz.grove import traverse_forest
from bartz.grove._grove import is_actual_leaf
from bartz.mcmcstep._moves import split_range, choose_variable, choose_split
from longbet._multi_state import split_multi_chain_fields
from longbet._shared_forest import observed_precision


def rule_geometry(var, split, max_split, blocked, log_s, pnt, capacity=64):
    """Exact restricted-rule log prior, plus growable-leaf mask (no counts)."""
    size = 2*split.size
    leaves = is_actual_leaf(split, add_bottom_level=True)
    internal = jnp.pad(split != 0, (0, split.size))
    active = leaves | internal
    nodes = jnp.nonzero(active, size=capacity, fill_value=size)[0]
    valid = nodes < size
    safe_nodes = jnp.minimum(nodes, size-1)
    def ranges(node):
        return jax.vmap(lambda v: split_range(var, split, max_split, node, v))(
            jnp.arange(max_split.size))
    lo, hi = jax.vmap(ranges)(safe_nodes)
    eligible = hi > lo
    if blocked is not None:
        allowed = ~jnp.any(jnp.arange(max_split.size)[:, None] == blocked[None, :], axis=1)
        eligible &= allowed[None, :]
    ls = jnp.zeros(max_split.size, jnp.float64) if log_s is None else log_s.astype(jnp.float64)
    normalizer = logsumexp(jnp.where(eligible, ls, -jnp.inf), axis=1)
    variables = var.at[safe_nodes].get(mode='fill', fill_value=0).astype(jnp.int32)
    cuts = split.at[safe_nodes].get(mode='fill', fill_value=0).astype(jnp.int32)
    l, h = lo[jnp.arange(capacity), variables], hi[jnp.arange(capacity), variables]
    admissible = eligible.any(axis=1)
    nt = pnt[safe_nodes].astype(jnp.float64)*admissible
    rule = ls[variables]-normalizer-jnp.log(jnp.maximum(h-l, 1))
    rule_valid = (cuts >= l) & (cuts < h) & eligible[jnp.arange(capacity), variables]
    contribution = jnp.where(internal[safe_nodes],
        jnp.where(rule_valid, jnp.log(nt)+rule, -jnp.inf), jnp.log1p(-nt))
    log_prior = jnp.where(jnp.sum(active) <= capacity,
                          jnp.where(valid, contribution, 0.).sum(), -jnp.inf)
    growable = jnp.zeros(size, bool).at[nodes].set(
        valid & leaves[safe_nodes] & admissible & (nt > 0), mode='drop')
    return log_prior, growable[:split.size], jnp.sum(active)


def root_move(key, var, split, leaves, X, precision, score, alpha, prior_precision,
              max_split, blocked, log_s, pnt, min_leaf, min_decision, capacity=64):
    """One tree; score includes the current tree, so it is a partial score."""
    kk, kv, kc, ks, ka, kl = random.split(key, 6)
    kind = random.randint(kk, (), 0, 3)
    variable, _ = choose_variable(kv, var, split, max_split, jnp.int32(1), blocked, log_s)
    cut, _, _ = choose_split(kc, variable, var, split, max_split, jnp.int32(1))
    adjacent = split[1].astype(jnp.int32)+jnp.where(random.bernoulli(kc), 1, -1)
    descendants = (split != 0) & (jnp.arange(split.size) > 1)
    other = random.categorical(ks, jnp.where(descendants, 0., -jnp.inf))
    proposed_var = var.at[1].set(jnp.where(kind == 0, variable,
        jnp.where(kind == 2, var[other], var[1])).astype(var.dtype))
    proposed_split = split.at[1].set(jnp.where(kind == 0, cut,
        jnp.where(kind == 2, split[other], adjacent)).astype(split.dtype))
    proposed_var = proposed_var.at[other].set(jnp.where(kind == 2, var[1], proposed_var[other]))
    proposed_split = proposed_split.at[other].set(jnp.where(kind == 2, split[1], proposed_split[other]))
    geometry = lambda v, sp: rule_geometry(v, sp, max_split, blocked, log_s, pnt, capacity)
    old_prior, _, count = geometry(var, split)
    new_prior, _, _ = geometry(proposed_var, proposed_split)
    size = leaves.size
    def stats(v, sp):
        ids = traverse_forest(X, v[None, :], sp[None, :])[0]
        w = jnp.zeros(size, jnp.float64).at[ids].add(precision*alpha**2)
        h = jnp.zeros(size, jnp.float64).at[ids].add(score*alpha)
        P = prior_precision+w
        active = is_actual_leaf(sp, add_bottom_level=True)
        counts = jnp.bincount(ids, length=size)
        supported = jnp.all(jnp.where(active, counts >= min_leaf, True))
        # Preserve support reachable by ordinary grow/prune, without making
        # count thresholds change the normalization of the splitting prior.
        node_counts = counts
        for depth in range(split.size.bit_length()-2, -1, -1):
            lo, hi = 2**depth, 2**(depth+1)
            node_counts = node_counts.at[lo:hi].add(node_counts[2*lo:2*hi].reshape(-1, 2).sum(axis=1))
        supported &= jnp.all(jnp.where(split != 0, node_counts[:split.size] >= min_decision, True))
        log_lik = .5*jnp.where(active, jnp.log(prior_precision/P)+h*h/P, 0.).sum()
        return ids, counts, P, h, jnp.where(supported, log_lik, -jnp.inf)
    old_stats, new_stats = stats(var, split), stats(proposed_var, proposed_split)
    ls = jnp.zeros(max_split.size, jnp.float64) if log_s is None else log_s.astype(jnp.float64)
    log_q_ratio = ls[var[1]]-ls[proposed_var[1]]
    log_q_ratio += jnp.log(max_split[proposed_var[1]].astype(jnp.float64))-jnp.log(max_split[var[1]].astype(jnp.float64))
    ratio = new_prior-old_prior+new_stats[-1]-old_stats[-1]+jnp.where(kind == 0, log_q_ratio, 0.)
    allowed = (split[1] != 0) & (count <= capacity)
    allowed &= (kind != 2) | descendants.any()
    allowed &= (kind != 1) | ((adjacent > 0) & (adjacent <= max_split[var[1]]))
    accepted = allowed & jnp.isfinite(ratio) & (jnp.log(random.uniform(ka, dtype=jnp.float64)) < ratio)
    v = jnp.where(accepted, proposed_var, var)
    sp = jnp.where(accepted, proposed_split, split)
    ids, counts, P, h, _ = jax.tree.map(lambda a, b: jnp.where(accepted, b, a), old_stats, new_stats)
    active = is_actual_leaf(sp, add_bottom_level=True)
    draw = h/P+random.normal(kl, leaves.shape, jnp.float64)/jnp.sqrt(P)
    values = jnp.where(active, draw, leaves)
    _, affluent, _ = geometry(v, sp)
    affluent &= counts[:split.size] >= min_decision
    info = jnp.array([allowed, accepted, jnp.where(allowed, ratio, 0.),
        accepted & ((v[1] != var[1]) | (sp[1] != split[1])), kind], jnp.float64)
    return v, sp, values, ids, counts, affluent, info


@partial(jax.jit, static_argnames=('capacity',))
def change_prognostic_single_step(key, state, capacity=64):
    observed = jnp.stack([s.obs_mask for s in state.states])
    G = state.gamma_loadings if state.sur_active else jnp.zeros_like(state.gamma_loadings)
    omega = observed_precision(observed, G, jnp.stack([s.sigma2 for s in state.states]))
    infos = []
    for m in range(state.M):
        s, f = state.states[m], state.states[m].forest
        raw = jnp.stack([s.resid for s in state.states]).astype(jnp.float64)
        score = jnp.einsum('ik,ki->i', omega[:, m, :], raw)
        precision = omega[:, m, m]
        def execute(carry, j):
            forest, residual, score, fit = carry
            old_ids = traverse_forest(s.X, forest.var_tree[j][None, :], forest.split_tree[j][None, :])[0]
            old_values = forest.leaf_tree[j].astype(jnp.float64)*forest.leaf_unit
            old_fit = old_values[old_ids]
            partial_score = score+precision*s.alpha*old_fit
            v, sp, values, ids, counts, affluent, info = root_move(random.fold_in(random.fold_in(key, m), j),
                forest.var_tree[j], forest.split_tree[j], old_values, s.X, precision, partial_score,
                jnp.float64(s.alpha), jnp.float64(forest.leaf_prior_cov_inv), forest.max_split,
                forest.blocked_vars, forest.log_s, forest.p_nonterminal,
                0 if forest.min_points_per_leaf is None else forest.min_points_per_leaf,
                0 if forest.min_points_per_decision_node is None else forest.min_points_per_decision_node,
                capacity)
            stored = (values/forest.leaf_unit).astype(forest.leaf_tree.dtype)
            new_fit = stored[ids].astype(jnp.float64)*forest.leaf_unit
            delta = new_fit-old_fit
            residual = jnp.where(s.obs_mask, residual-s.alpha*delta, 0.)
            score -= precision*s.alpha*delta
            fit += delta
            forest = eqx.tree_at(lambda f: (f.var_tree, f.split_tree, f.leaf_tree,
                f.leaf_indices, f.to_prune, f.move_node, f.affluence_tree), forest,
                (forest.var_tree.at[j].set(v), forest.split_tree.at[j].set(sp),
                 forest.leaf_tree.at[j].set(stored), forest.leaf_indices.at[j].set(ids),
                 forest.to_prune.at[j].set(False), forest.move_node.at[j].set(0),
                 forest.affluence_tree.at[j].set(affluent)))
            if forest.count_tree is not None:
                forest = eqx.tree_at(lambda f: f.count_tree, forest,
                                      forest.count_tree.at[j].set(counts.astype(forest.count_tree.dtype)))
            if forest.prec_tree is not None:
                w = s.prec_scale if s.prec_scale is not None else jnp.ones(s.y.shape, jnp.float32)
                sums = jnp.zeros(forest.leaf_tree.shape[-1], jnp.float32).at[ids].add(w)
                forest = eqx.tree_at(lambda f: f.prec_tree, forest, forest.prec_tree.at[j].set(sums))
            return (forest, residual, score, fit), info
        def step(carry, j):
            active_nodes = 2*jnp.count_nonzero(carry[0].split_tree[j])+1
            return jax.lax.cond(active_nodes <= capacity, lambda c: execute(c, j),
                                lambda c: (c, jnp.zeros(5, jnp.float64)), carry)
        (forest, residual, _, fit), info = jax.lax.scan(step,
            (f, s.resid.astype(jnp.float64), score, s.mu_fit.astype(jnp.float64)),
            jnp.arange(f.leaf_tree.shape[0]))
        child = eqx.tree_at(lambda s: (s.forest, s.resid, s.mu_fit), s,
                            (forest, residual.astype(jnp.float32), fit.astype(jnp.float32)))
        children = list(state.states); children[m] = child
        state = eqx.tree_at(lambda s: s.states, state, tuple(children))
        infos.append(info)
    return state, jnp.stack(infos)


def change_prognostic_step(key, state, capacity=64):
    if not state.has_chain_axis:
        return change_prognostic_single_step(key, state, capacity)
    per, constants = split_multi_chain_fields(state)
    def one(k, p):
        updated, info = change_prognostic_single_step(k, eqx.combine(p, constants), capacity)
        return split_multi_chain_fields(updated)[0], info
    new, info = jax.vmap(one)(random.split(key, state.num_chains), per)
    return eqx.combine(new, constants), info
