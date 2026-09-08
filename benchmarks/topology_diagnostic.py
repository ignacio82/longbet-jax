"""Benchmark-only topology instrumentation and a fixed-topology control.

The control samples a CONDITIONAL posterior at one learned topology, selected
without truth labels. It is not an estimator of the original model posterior.
Only topology is copied from reference chain 0; fresh chains retain their own
initial parameters and zero initial leaves. No dependency files are modified.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bartz.grove import traverse_forest


def save_final_topology(state, path):
    # Preserve a full final state for future conditional-density calculations;
    # effects/rule archives alone cannot reconstruct latent values and leaves.
    # This is an internal Equinox checkpoint, not a public LongBet fit archive.
    eqx.tree_serialise_leaves(path.parent / 'final_state.eqx', state)
    arrays = {}
    for m, child in enumerate(state.states):
        for name in ('forest', 'forest_nu'):
            forest = getattr(child, name)
            for field in ('var_tree', 'split_tree', 'affluence_tree'):
                arrays[f'{m}_{name}_{field}'] = np.asarray(getattr(forest, field))
    if state.shared_forest is not None:
        for field in ('var_tree', 'split_tree'):
            arrays[f'shared_{field}'] = np.asarray(getattr(state.shared_forest, field))
    np.savez_compressed(path, **arrays)


def initialize_common_topology(state, path):
    """Install chain 0's rules and membership in fresh zero-leaf chains."""
    with np.load(path, allow_pickle=False) as source:
        def rules(template, prefix):
            values = []
            for field in ('var_tree', 'split_tree'):
                target = getattr(template, field)
                original = source[f'{prefix}{field}']
                if original.ndim != 3 or original.shape[1:] != target.shape[-2:]:
                    raise ValueError('reference topology needs compatible chained forests')
                values.append(jnp.broadcast_to(jnp.asarray(original[0], target.dtype), target.shape))
            return values

        children = []
        for m, child in enumerate(state.states):
            forests = []
            for name in ('forest', 'forest_nu'):
                forest = getattr(child, name)
                if np.any(np.asarray(forest.leaf_tree)):
                    raise ValueError('common topology must be installed in fresh zero-leaf states')
                prefix = f'{m}_{name}_'
                var, split = rules(forest, prefix)
                ids = jax.vmap(lambda v, s: traverse_forest(child.X, v, s))(var, split)
                ids = ids.astype(forest.leaf_indices.dtype)
                count = forest.count_tree
                if count is not None:
                    count = jax.vmap(jax.vmap(lambda i: jnp.bincount(
                        i, length=forest.leaf_tree.shape[-1])))(ids).astype(count.dtype)
                affluent = jnp.broadcast_to(jnp.asarray(source[prefix+'affluence_tree'][0]),
                                            forest.affluence_tree.shape)
                forest = eqx.tree_at(lambda f: (f.var_tree, f.split_tree, f.leaf_indices,
                    f.to_prune, f.move_node, f.count_tree, f.affluence_tree), forest,
                    (var, split, ids, jnp.zeros_like(forest.to_prune),
                     jnp.zeros_like(forest.move_node), count, affluent),
                    is_leaf=lambda x: x is None)
                forests.append(forest)
            children.append(eqx.tree_at(lambda s: (s.forest, s.forest_nu), child, tuple(forests)))
        state = eqx.tree_at(lambda s: s.states, state, tuple(children))
        if state.shared_forest is not None:
            shared = state.shared_forest
            if np.any(np.asarray(shared.leaf_tree)):
                raise ValueError('shared leaves must be zero at initialization')
            var, split = rules(shared, 'shared_')
            ids = jax.vmap(lambda v, s: traverse_forest(state.states[0].X, v, s))(var, split)
            shared = eqx.tree_at(lambda f: (f.var_tree, f.split_tree, f.leaf_indices),
                shared, (var, split, ids.astype(shared.leaf_indices.dtype)))
            state = eqx.tree_at(lambda s: s.shared_forest, state, shared)
    return state


def install_fixed_topology_hooks():
    """Veto topology proposals; preserve the existing conditional leaf draws.

    Invoke only in a dedicated benchmark process BEFORE tracing any sweeps.
    The returned originals let tests restore the process-local hooks.
    """
    from bartz.mcmcstep import _step as bartz
    from longbet import _shared_forest as shared
    original_propose, original_shared = bartz.propose_moves, shared.shared_tree_step

    def propose(key, forest):
        moves = original_propose(key, forest)
        return eqx.tree_at(lambda m: m.allowed, moves, jnp.zeros_like(moves.allowed))

    def shared_tree(key, leaves, var, split, ids, partial, gram, omega, w,
                    observed_any, forest, X):
        # A zero rule-support mask makes `valid` false; the existing kernel
        # then samples the leaves of the unchanged tree from its full SUR law.
        blocked = eqx.tree_at(lambda f: f.max_split, forest, jnp.zeros_like(forest.max_split))
        return original_shared(key, leaves, var, split, ids, partial, gram,
                               omega, w, observed_any, blocked, X)

    bartz.propose_moves, shared.shared_tree_step = propose, shared_tree
    return original_propose, original_shared


def summarize_topology(multi, output):
    """Save compact raw rules plus chain-wise histograms and proposal counters."""
    result, arrays = {}, {}
    for m, trace in enumerate(multi.trace.traces):
        name = multi.outcome_names[multi.order[m]]
        result[name] = {}
        for label, tr in (('mu', trace.mu_trace), ('nu', trace.nu_trace)):
            var, split = np.asarray(tr.var_tree), np.asarray(tr.split_tree)
            active = split != 0
            arrays[f'{name}_{label}_var'] = var
            arrays[f'{name}_{label}_split'] = split
            hist = np.stack([(active & (var == p)).sum((-1, -2))
                             for p in range(multi.state.states[m].X.shape[0])], axis=-1)
            # Reachable internal nodes are exactly the nonzero split slots.
            rules = np.where(active, (var.astype(np.int32)+1)*65536+split, 0)
            changes = (rules[:, 1:] != rules[:, :-1]).sum((-1, -2))
            item = {'split_histogram_chain_means': hist.mean(1).tolist(),
                    'leaves_chain_means': (active.sum((-1, -2))+split.shape[-2]).mean(1).tolist(),
                    'changed_rules_between_saves_chain_means': changes.mean(1).tolist(),
                    'root_changes_per_chain': (rules[:, 1:, :, 1] != rules[:, :-1, :, 1]).sum((1, 2)).tolist()}
            for kind in ('grow', 'prune'):
                prop = np.asarray(getattr(tr, kind+'_prop_count'))
                acc = np.asarray(getattr(tr, kind+'_acc_count'))
                arrays[f'{name}_{label}_{kind}_proposed'] = prop
                arrays[f'{name}_{label}_{kind}_accepted'] = acc
                item[kind+'_acceptance_by_chain'] = np.divide(
                    acc.sum(1), prop.sum(1), out=np.full(prop.shape[0], np.nan),
                    where=prop.sum(1)>0).tolist()
            result[name][label] = item
    np.savez_compressed(output / 'topology.npz', **arrays)
    return result
