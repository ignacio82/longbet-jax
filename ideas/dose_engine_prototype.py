"""Feasibility prototype of the vector-leaf GROW/PRUNE kernel (plan section 5).

NOT production code. It exists to prove, before anyone spends days on it, that
the plan's central engineering bet works:

* tree proposals, the transition/prior ratio and all heap bookkeeping are taken
  unchanged from ``bartz`` internals (``_propose_moves``, ``complete_ratio``,
  ``apply_grow_to_indices``, ``apply_moves_to_split_trees``,
  ``apply_moves_to_affluence_trees``), so the tree prior is bartz's by
  construction; and
* only the likelihood part is new: Gaussian *regression* leaves with a
  ``p``-vector of coefficients and a full within-unit error precision.

``python ideas/dose_engine_prototype.py`` runs one tree (with and without the
minimum-units veto) and a two-tree forest on a tiny panel and compares the visited
tree structures with the exact enumerated posteriors from ``dose_reference.py``.
``--mutants`` runs deliberately broken kernels to show the test can fail.
Measured on 2026-09-17 (CPU, x64): total variation 0.010, 0.008 and 0.012 for the
three correct runs; 0.136 without the log-determinant and 0.369 with the quadratic
form halved. Wall time: 6 s for 60,000 one-tree sweeps, 26 s for 200,000 two-tree
sweeps including the 484-forest exact calculation.

Deliberate simplifications (the production engine must not copy them):
per-unit ``p x p`` information matrices are materialized (use the Kronecker
factored statistics of plan 5.1 instead), there is one chain, one forest, a
fixed error precision, no CHANGE move, and no trace object.
"""

from __future__ import annotations

import collections
import sys
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import lax, random

from bartz.mcmcstep import make_p_nonterminal
from bartz.mcmcstep._moves import _propose_moves
from bartz.mcmcstep._step import (
    apply_grow_to_indices,
    apply_moves_to_affluence_trees,
    apply_moves_to_split_trees,
    complete_ratio,
)
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dose_reference as ref  # noqa: E402


def leaf_log_ml(A, b, kappa):
    """Integrated log likelihood of a node; 0 for an empty node (plan 5.1)."""
    p = b.shape[-1]
    chol = jnp.linalg.cholesky(kappa * jnp.eye(p) + A)
    half = jax.scipy.linalg.solve_triangular(chol, b, lower=True)
    return 0.5 * p * jnp.log(kappa) - jnp.sum(jnp.log(jnp.diagonal(chol))) + 0.5 * half @ half


def draw_leaf(z, A, b, kappa):
    p = b.shape[-1]
    chol = jnp.linalg.cholesky(kappa * jnp.eye(p) + A)
    mean = jax.scipy.linalg.cho_solve((chol, True), b)
    return mean + jax.scipy.linalg.solve_triangular(chol.T, z, lower=False)


@partial(jax.jit, static_argnames=("min_units_per_leaf",))
def sweep(key, forest, R, H, P, X, contributes, p_nonterminal, max_split, kappa,
          min_units_per_leaf=None):
    """One GROW/PRUNE pass over every tree, then a redraw of all its leaves.

    ``forest`` is a dict of ``var_tree``, ``split_tree``, ``affluence_tree``
    (``(M, half)``), ``leaf_indices`` (``(M, N)``) and ``coef`` (``(M, 2*half, p)``).
    ``R`` is the full residual ``(N, q)``; ``H`` the whitened unit designs ``(N, q, p)``.
    """
    num_trees, half = forest["var_tree"].shape
    tree_size = 2 * half
    k_prop, k_leaf = random.split(key)

    moves = _propose_moves(
        random.split(k_prop, num_trees), forest["var_tree"], forest["split_tree"],
        forest["affluence_tree"], max_split, None, p_nonterminal[:half], None,
    )
    grown_indices = apply_grow_to_indices(moves, forest["leaf_indices"], X)

    # per-unit information and the pieces that do not depend on the residual
    info = jnp.einsum("nqa,qr,nrb->nab", H, P, H)                  # (N, p, p)
    w = contributes.astype(R.dtype)

    def node_counts(idx, nodes):
        return jnp.stack([jnp.sum(w * (idx == nodes[c])) for c in (0, 1)])

    counts_lr = jax.vmap(node_counts)(grown_indices, moves.lrt_nodes)
    counts_lrt = jnp.concatenate([counts_lr, counts_lr.sum(axis=1, keepdims=True)], axis=1)
    lrt_affluent = (moves.lrt_nodes < half) & moves.lrt_growable
    allowed = moves.allowed
    if min_units_per_leaf is not None:
        allowed &= jnp.all(counts_lrt[:, :2] >= min_units_per_leaf, axis=1)
    moves = replace(moves, lrt_affluent=lrt_affluent, allowed=allowed)
    moves = complete_ratio(moves, p_nonterminal)

    normals = random.normal(k_leaf, (num_trees, tree_size, H.shape[-1]))

    # bartz's `adapt_leaf_trees_to_grow_indices`: a grow was pre-applied to the
    # indices, so the two new children must carry the value of the leaf they came
    # from. Every heap slot is redrawn each sweep, so an unused child slot holds a
    # prior draw, not its parent's value; skipping this step corrupts the residual.
    def adapt(coef, lrt, grow):
        target = jnp.where(grow, lrt[:2], tree_size)
        return coef.at[target].set(coef[lrt[2]], mode="drop")

    coef_in = jax.vmap(adapt)(forest["coef"], moves.lrt_nodes, moves.grow)

    def one_tree(R, xs):
        idx, coef, lrt, grow, ok, log_tp, logu, z = xs
        left, right, parent = lrt[0], lrt[1], lrt[2]
        # residual with this tree's contribution added back. `coef` mirrors a
        # leaf's value onto its not-yet-existing children, so indexing with the
        # grown indices is correct whatever the move.
        fit_old = jnp.einsum("nqa,na->nq", H, coef[idx])
        R_plus = R + fit_old
        score = jnp.einsum("nqa,qr,nr->na", H, P, R_plus) * w[:, None]   # (N, p)

        def stats(mask):
            m = mask.astype(R.dtype) * w
            return jnp.einsum("n,nab->ab", m, info), jnp.einsum("n,na->a", m, score)

        A_l, b_l = stats(idx == left)
        A_r, b_r = stats(idx == right)
        log_lk = (leaf_log_ml(A_l, b_l, kappa) + leaf_log_ml(A_r, b_r, kappa)
                  - leaf_log_ml(A_l + A_r, b_l + b_r, kappa))
        log_ratio = log_tp + log_lk
        log_ratio = jnp.where(grow, log_ratio, -log_ratio)
        acc = ok & (logu <= log_ratio)
        to_prune = acc ^ grow

        # final membership: undo the pre-applied grow, or apply the accepted prune
        in_children = (idx >> 1).astype(jnp.int32) == parent
        idx_final = jnp.where(to_prune & in_children, parent.astype(idx.dtype), idx)

        # redraw every leaf of the final tree from its exact conditional
        onehot = (idx_final[:, None] == jnp.arange(tree_size)[None, :]).astype(R.dtype) * w[:, None]
        A_all = jnp.einsum("nt,nab->tab", onehot, info)
        b_all = jnp.einsum("nt,na->ta", onehot, score)
        coef_new = jax.vmap(draw_leaf, in_axes=(0, 0, 0, None))(z, A_all, b_all, kappa)
        # keep bartz's mirroring invariant: children slots of the move carry the
        # parent's value whenever the parent is (again) a leaf
        mirror = jnp.where(to_prune, jnp.stack([left, right]), tree_size)
        coef_new = coef_new.at[mirror].set(coef_new[parent], mode="drop")

        fit_new = jnp.einsum("nqa,na->nq", H, coef_new[idx_final])
        return R_plus - fit_new, (idx_final, coef_new, acc, to_prune)

    xs = (grown_indices, coef_in, moves.lrt_nodes, moves.grow, moves.allowed,
          moves.log_trans_prior_ratio, moves.logu, normals)
    R, (leaf_indices, coef, acc, to_prune) = lax.scan(one_tree, R, xs)

    moves = replace(moves, acc=acc, to_prune=to_prune)
    new_forest = dict(
        var_tree=moves.var_tree,
        split_tree=apply_moves_to_split_trees(forest["split_tree"], moves),
        affluence_tree=apply_moves_to_affluence_trees(forest["affluence_tree"], moves),
        leaf_indices=leaf_indices,
        coef=coef,
    )
    return new_forest, R


def init_forest(num_trees, max_depth, N, p, index_dtype=jnp.uint8):
    half = 2 ** (max_depth - 1)
    affluence = jnp.zeros((num_trees, half), bool).at[:, 1].set(max_depth > 1)
    return dict(
        var_tree=jnp.zeros((num_trees, half), jnp.uint8),
        split_tree=jnp.zeros((num_trees, half), jnp.uint8),
        affluence_tree=affluence,
        leaf_indices=jnp.ones((num_trees, N), index_dtype),
        coef=jnp.zeros((num_trees, 2 * half, p)),
    )


def main(num_trees=1, n_sweeps=60_000, burn=2_000, seed=20260917, min_units_per_leaf=None):
    rng = np.random.default_rng(seed)
    fx = ref._fixture(rng, n=14, T=4, K=4)
    N, q = fx["n"], fx["q"]
    max_split, max_depth, alpha, beta = np.array([2, 1], np.uint8), 3, 0.5, 1.0
    X = np.vstack([rng.integers(0, 3, N), rng.integers(0, 2, N)]).astype(np.uint8)
    contributes = np.ones(N, bool)
    contributes[:3] = False                                  # three never-treated units
    kappa = num_trees / 1.0 ** 2                             # tau = 1
    H_np = np.stack([ref.unit_design(fx["h_w"][i], fx["E_w"][fx["cohort"][i]]) for i in range(N)])
    H_np[~contributes] = 0.0                                 # a never-treated unit has no treatment design
    # data with real signal, so the posterior differs visibly from the prior
    theta_true = {0: rng.standard_normal(H_np.shape[-1]), 1: rng.standard_normal(H_np.shape[-1])}
    chol_V = np.linalg.cholesky(fx["V"])
    v = np.stack([H_np[i] @ theta_true[int(X[1, i])] + chol_V @ rng.standard_normal(q) for i in range(N)])

    trees = ref.enumerate_trees(max_split, max_depth)
    if min_units_per_leaf is None:
        tuples, exact = ref.exact_forest_posterior(
            trees, num_trees, X, H_np, contributes, fx["P"], v, kappa, max_split, max_depth, alpha, beta)
        exact = {tuple(trees[t].rules for t in combo): pr for combo, pr in zip(tuples, exact)}
    else:
        assert num_trees == 1, "the veto oracle is single-tree"
        post = ref.exact_tree_posterior(
            trees, X, fx["h_w"], fx["cohort"], contributes, fx["E_w"], fx["P"], v, kappa,
            max_split, max_depth, alpha, beta, min_units_per_leaf=min_units_per_leaf)
        exact = {(t.rules,): pr for t, pr in zip(trees, post)}

    # bartz's convention: p_nonterminal is indexed by heap node, not by depth, is 0
    # on the last level, and its first `half` entries double as `p_propose_grow`
    depth_of = np.floor(np.log2(np.maximum(np.arange(2 ** max_depth), 1))).astype(int)
    p_depth = np.r_[np.asarray(make_p_nonterminal(max_depth, alpha, beta)), 0.0]
    p_nt = jnp.asarray(np.where(np.arange(2 ** max_depth) == 0, p_depth[0], p_depth[depth_of]))

    H = jnp.asarray(H_np)
    args = (jnp.asarray(fx["P"]), jnp.asarray(X), jnp.asarray(contributes), p_nt,
            jnp.asarray(max_split), kappa)

    def total_fit(forest):
        per_tree = jax.vmap(lambda coef, idx: jnp.einsum("nqa,na->nq", H, coef[idx]))(
            forest["coef"], forest["leaf_indices"])
        return per_tree.sum(axis=0)

    @jax.jit
    def run(key, forest, R):
        def body(carry, _):
            forest, R, key = carry
            key, sub = random.split(key)
            forest, R = sweep(sub, forest, R, H, *args, min_units_per_leaf=min_units_per_leaf)
            fit = total_fit(forest)
            return (forest, R, key), (forest["var_tree"], forest["split_tree"], fit,
                                      jnp.max(jnp.abs(R + fit - v)))
        return lax.scan(body, (forest, R, key), None, length=n_sweeps)[1]

    forest = init_forest(num_trees, max_depth, N, H.shape[-1])
    var, split, fits, invariant = map(np.asarray, run(random.key(seed), forest, jnp.asarray(v)))

    def identity(vv, ss):      # stale var_tree entries survive at leaves: read rules through split > 0
        return tuple(tuple((n, int(vv[m][n]), int(ss[m][n])) for n in range(1, ss.shape[1]) if ss[m][n] > 0)
                     for m in range(num_trees))

    counts = collections.Counter(identity(vv, ss) for vv, ss in zip(var[burn:], split[burn:]))
    total = sum(counts.values())
    assert set(counts) <= set(exact), "the sampler visited a forest outside the enumerated support"
    print(f"[M={num_trees}, veto={min_units_per_leaf}] {len(exact)} forests in the exact posterior, "
          f"{len(counts)} visited; max residual-invariant violation {invariant.max():.2e}")
    tv = 0.5 * sum(abs(counts.get(r, 0) / total - pr) for r, pr in exact.items())
    for rules, pr in sorted(exact.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  exact {pr:.4f}  sampled {counts.get(rules, 0) / total:.4f}  {rules}")
    print(f"  total variation distance to the exact posterior over tree structures: {tv:.4f}")
    return tv


def run_mutation_arms():
    """Show that the enumerated-posterior test can fail, and how to build a valid mutation arm.

    ``jax.jit`` caches traces by the *identity of the wrapped Python function*.
    Rebinding ``leaf_log_ml`` and wrapping the same ``sweep`` in a new ``jax.jit``
    silently reuses the unpatched trace, and the "broken" arm reproduces the correct
    result to every digit (measured here: TV 0.0102 three times). A mutation arm is
    only valid if it traces a NEW function object, and if the test asserts that the
    arm really changed the answer.
    """
    import types

    module = sys.modules[__name__]
    original_ml, original_sweep = module.leaf_log_ml, module.sweep

    def no_logdet(A, b, kappa):
        chol = jnp.linalg.cholesky(kappa * jnp.eye(b.shape[-1]) + A)
        half = jax.scipy.linalg.solve_triangular(chol, b, lower=True)
        return 0.5 * half @ half

    def halved_quadratic(A, b, kappa):
        p = b.shape[-1]
        chol = jnp.linalg.cholesky(kappa * jnp.eye(p) + A)
        half = jax.scipy.linalg.solve_triangular(chol, b, lower=True)
        return 0.5 * p * jnp.log(kappa) - jnp.sum(jnp.log(jnp.diagonal(chol))) + 0.25 * half @ half

    results = {}
    try:
        for name, fn in (("correct", original_ml), ("missing log-determinant", no_logdet),
                         ("quadratic form halved", halved_quadratic)):
            module.leaf_log_ml = fn
            f = original_sweep.__wrapped__
            fresh = types.FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
            fresh.__kwdefaults__ = f.__kwdefaults__
            module.sweep = jax.jit(fresh, static_argnames=("min_units_per_leaf",))
            results[name] = float(main(1))
    finally:
        module.leaf_log_ml, module.sweep = original_ml, original_sweep
    print({k: round(v, 4) for k, v in results.items()})
    ok = results["correct"] < 0.03 and min(results["missing log-determinant"],
                                           results["quadratic form halved"]) > 0.08
    print("MUTATION ARMS", "DETECTED" if ok else "NOT DETECTED")
    return ok


if __name__ == "__main__":
    if "--mutants" in sys.argv:
        sys.exit(0 if run_mutation_arms() else 1)
    results = {
        "one tree": main(1),
        "one tree, min 3 units per leaf": main(1, min_units_per_leaf=3),
        "two trees (joint oracle)": main(2, n_sweeps=200_000),
    }
    ok = results["one tree"] < 0.03 and results["one tree, min 3 units per leaf"] < 0.03 \
        and results["two trees (joint oracle)"] < 0.05
    print({k: round(float(v), 4) for k, v in results.items()})
    print("PROTOTYPE", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
