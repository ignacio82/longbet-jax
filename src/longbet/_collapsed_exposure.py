"""Experimental collapsed exposure/coding moves; not used by the public driver.

At fixed topologies, coding, SUR parameters and augmented responses, integrate
ALL prognostic/treatment leaves and unit intercepts for one outcome. Propose
the exposure GP with a prior-reversible pCN step, accept with the marginal
likelihood ratio, then immediately redraw all integrated coefficients (also on
rejection). This crosses the beta/nu product ridge without treating that product
as jointly Gaussian. The target posterior and every prior are unchanged.
Alternatively update coding, or coding and beta jointly, by elliptical slice
sampling under the same integral. The coding-only variant caches the exact
quadratic in (b0, b1), including unit-induced cross-stratum dependence.
The tree variant proposes one prognostic root per outcome using a reversible
tree-prior transition, then accepts with the ALL-forest/unit marginal likelihood
ratio. It therefore allows other trees to compensate for a changed partition.

Private and shared treatment leaves both enter the integral. The shared leaf
prior is diagonal in the supported model, so an outcome-wise draw conditions
on the other outcomes through the full SUR conditional score and precision.
Binary latent values and their identifying variances stay fixed in this move.

All occupied leaves must fit in the compact system. Overflow causes an identity
transition, selected only by the conditioned-on topology; no leaf is dropped
from the model. Unreachable storage slots are preserved. X64 is required through
tracing and execution, with float32 stored state as in the other experiments.
"""
from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular

from bartz.grove import traverse_forest
from bartz.grove._grove import is_actual_leaf
from longbet._multi_state import split_multi_chain_fields
from longbet._shared_forest import observed_precision

MAX_ENSEMBLE_LEAVES = 192
COLLAPSED_EXPOSURE_SEMANTICS = 'collapsed_exposure_v3'


def _solve(chol, rhs):
    return solve_triangular(chol.T, solve_triangular(chol, rhs, lower=True), lower=False)


def collapsed_statistics(design, response, precision, unit, unit_sd, n_units):
    """Integrate x~N(0,I), g~N(0,I), mean = A x + unit_sd*g[unit].

    Return Schur Cholesky, coefficient mean, unit precision/score/cross term,
    and log marginal likelihood excluding the fixed observation normalization.
    Missing cells have zero precision. No observation-by-observation covariance
    matrix is formed. Unit coordinates are eliminated using diagonal solves.
    """
    A, y, w, sd = [jnp.asarray(v, jnp.float64)
                    for v in (design, response, precision, unit_sd)]
    weighted = w[:, None]*A
    cross = sd*jnp.zeros((n_units, A.shape[1]), jnp.float64).at[unit].add(weighted)
    pu = 1 + sd**2*jnp.zeros(n_units, jnp.float64).at[unit].add(w)
    hu = sd*jnp.zeros(n_units, jnp.float64).at[unit].add(w*y)
    Q = jnp.eye(A.shape[1], dtype=jnp.float64) + A.T@weighted - cross.T@(cross/pu[:, None])
    h = A.T@(w*y) - cross.T@(hu/pu)
    chol = jnp.linalg.cholesky((Q+Q.T)*.5)
    mean = _solve(chol, h)
    quad = jnp.sum(w*y*y) - jnp.sum(hu*hu/pu) - h@mean
    logdet = jnp.log(pu).sum() + 2*jnp.log(jnp.diag(chol)).sum()
    return chol, mean, pu, hu, cross, -.5*(quad+logdet)


def collapsed_draw(key, stats):
    """Draw all whitened leaves and immediately redraw the integrated units."""
    chol, mean, pu, hu, cross, _ = stats
    k_leaf, k_unit = random.split(key)
    x = mean + solve_triangular(chol.T, random.normal(k_leaf, mean.shape, jnp.float64), lower=False)
    g = (hu-cross@x)/pu + random.normal(k_unit, pu.shape, jnp.float64)/jnp.sqrt(pu)
    return x, g


def coding_sufficient_statistics(base, is_trt, alpha, beta, treatment,
                                  response, offset, precision, unit, unit_sd, n_units):
    """Cache the exact unit-integrated quadratic for arbitrary (b0, b1).

    H contains unscaled leaf columns, the response, and the treatment offset.
    Its two treatment strata have column multipliers d0 and d1. Their four
    projected Gram blocks suffice to evaluate each coding proposal without
    revisiting the observation panel. Cross-stratum blocks arise from units.
    """
    base, beta, response, precision, unit_sd = [jnp.asarray(v, jnp.float64)
        for v in (base, beta, response, precision, unit_sd)]
    A = base*jnp.where(is_trt[None, :], beta[:, None], alpha)
    H = jnp.column_stack((A, response, beta*offset))
    pu = 1+unit_sd**2*jnp.zeros(n_units, jnp.float64).at[unit].add(precision)
    cross, gram = [], []
    for treated in (False, True):
        w = precision*((treatment == 1) == treated)
        weighted = w[:, None]*H
        C = unit_sd*jnp.zeros((n_units, H.shape[1]), jnp.float64).at[unit].add(weighted)
        cross.append(C)
        gram.append(H.T@weighted-C.T@(C/pu[:, None]))
    between = -cross[0].T@(cross[1]/pu[:, None])
    return tuple(gram), between, tuple(cross), pu, is_trt


def coding_statistics(coefficients, sufficient):
    """Same outputs as collapsed_statistics, using the cached coding quadratic."""
    gram, between, cross, pu, is_trt = sufficient
    d0, d1 = [jnp.concatenate((jnp.where(is_trt, b, 1.), jnp.array([1., -b])))
              for b in coefficients]
    off = between*d0[:, None]*d1[None, :]
    G = gram[0]*jnp.outer(d0, d0)+gram[1]*jnp.outer(d1, d1)+off+off.T
    C = cross[0]*d0+cross[1]*d1
    p = is_trt.size
    Q = jnp.eye(p, dtype=jnp.float64)+G[:p, :p]
    h = G[:p, p:].sum(axis=1)
    chol = jnp.linalg.cholesky((Q+Q.T)*.5)
    mean = _solve(chol, h)
    quad = G[p:, p:].sum()-h@mean
    logdet = jnp.log(pu).sum()+2*jnp.log(jnp.diag(chol)).sum()
    return chol, mean, pu, C[:, p:].sum(axis=1), C[:, :p], -.5*(quad+logdet)


def elliptical_slice(key, current, direction, statistics):
    """Exact Gaussian-prior elliptical slice step with an uncapped bracket.

    direction is an independent draw from the same zero-mean Gaussian prior.
    Every rejected angle shrinks the bracket containing angle zero. Never
    truncate the search or return an unaccepted proposal. The caller must
    immediately redraw the coefficients marginalized by statistics.
    """
    ku, kt, kr = random.split(key, 3)
    old_stats = statistics(current)
    threshold = old_stats[-1]+jnp.log(random.uniform(ku, dtype=jnp.float64))
    theta = random.uniform(kt, dtype=jnp.float64)*2*jnp.pi
    initial = (theta, theta-2*jnp.pi, theta, current, old_stats, jnp.int32(0), jnp.bool_(False))

    def body(carry):
        angle, lo, hi, _, _, evaluations, _ = carry
        proposal = current*jnp.cos(angle)+direction*jnp.sin(angle)
        stats = statistics(proposal)
        accepted = jnp.isfinite(stats[-1]) & (stats[-1] >= threshold)
        lo = jnp.where(angle < 0, angle, lo)
        hi = jnp.where(angle > 0, angle, hi)
        next_angle = random.uniform(random.fold_in(kr, evaluations), dtype=jnp.float64,
                                     minval=lo, maxval=hi)
        return next_angle, lo, hi, proposal, stats, evaluations+1, accepted

    _, _, _, proposal, stats, evaluations, _ = jax.lax.while_loop(
        lambda c: ~c[-1], body, initial)
    return proposal, stats, stats[-1]-old_stats[-1], evaluations+1


def _pack(s, shared, m, capacity):
    """Build a whitened ensemble design from actual leaves, in data units."""
    forests = [(s.forest, False), (s.forest_nu, True)]
    masks, values, ids, trees, sds, treatment = [], [], [], [], [], []
    node_offset, tree_offset = 0, 0
    sizes = []
    for f, is_trt in forests:
        J, L = f.leaf_tree.shape
        masks.append(jax.vmap(lambda t: is_actual_leaf(t, add_bottom_level=True))(f.split_tree).ravel())
        values.append(f.leaf_tree.ravel().astype(jnp.float64)*f.leaf_unit)
        membership = traverse_forest(s.X, f.var_tree, f.split_tree)
        ids.append(membership.astype(jnp.int32) + jnp.arange(J)[:, None]*L + node_offset)
        trees.append(jnp.repeat(jnp.arange(J)+tree_offset, L))
        sds.append(jnp.full(J*L, 1/jnp.sqrt(f.leaf_prior_cov_inv), jnp.float64))
        treatment.append(jnp.full(J*L, is_trt))
        sizes.append(J*L)
        node_offset += J*L
        tree_offset += J
    if shared is not None:
        J, _, L = shared.leaf_tree.shape
        masks.append(jax.vmap(lambda t: is_actual_leaf(t, add_bottom_level=True))(shared.split_tree).ravel())
        values.append(shared.leaf_tree[:, m, :].ravel().astype(jnp.float64))
        ids.append(shared.leaf_indices.astype(jnp.int32)+jnp.arange(J)[:, None]*L+node_offset)
        trees.append(jnp.repeat(jnp.arange(J)+tree_offset, L))
        sds.append(jnp.full(J*L, 1/jnp.sqrt(shared.prior_precision[m, m]), jnp.float64))
        treatment.append(jnp.ones(J*L, bool))
        sizes.append(J*L)
        node_offset += J*L
    mask, values = jnp.concatenate(masks), jnp.concatenate(values)
    packed = jnp.nonzero(mask, size=capacity, fill_value=node_offset)[0]
    valid = packed < node_offset
    which = jnp.concatenate(trees).at[packed].get(mode='fill', fill_value=0)
    sd = jnp.concatenate(sds).at[packed].get(mode='fill', fill_value=0.)
    is_trt = jnp.concatenate(treatment).at[packed].get(mode='fill', fill_value=False)
    base = (jnp.concatenate(ids)[which].T == packed[None, :])*sd[None, :]
    base = jnp.where(valid[None, :], base, 0.)
    return base, is_trt, sd, packed, values, jnp.sum(mask), tuple(sizes)


def _outcome_move(key, state, m, precision, score, *, capacity, proposal_scale, proposal='pcn'):
    s, shared = state.states[m], state.shared_forest
    base, is_trt, leaf_sd, packed, values, count, sizes = _pack(s, shared, m, capacity)
    observed = s.obs_mask
    beta = s.beta.astype(jnp.float64)
    coefficients = jnp.array([s.b0, s.b1], jnp.float64)
    coding = jnp.where(s.z_vec == 1, s.b1, s.b0).astype(jnp.float64)
    old_w = coding*beta[s.exposure_idx]
    alpha = s.alpha.astype(jnp.float64)
    old_mean = alpha*s.mu_fit + old_w*s.nu_fit + s.gamma[s.unit_idx]
    # Conditional response given the other outcomes; raw residuals never get
    # replaced in persistent state. Zero-precision cells contribute nothing.
    pseudo = score/jnp.where(precision > 0, precision, 1.) + old_mean
    unit_sd = jnp.sqrt(s.sigma_gamma2).astype(jnp.float64) if s.random_intercept else jnp.float64(0.)
    fixed = alpha*s.forest.offset
    if not s.random_intercept:
        fixed = fixed+s.gamma[s.unit_idx]

    def statistics(b, bc, design_base=base):
        w = jnp.where(s.z_vec == 1, bc[1], bc[0])*b[s.exposure_idx]
        A = design_base*jnp.where(is_trt[None, :], w[:, None], alpha)
        y = jnp.where(observed, pseudo-fixed-w*s.forest_nu.offset, 0.)
        return collapsed_statistics(A, y, precision, s.unit_idx, unit_sd, s.N_units)

    def move(_):
        k_propose, k_acc, k_draw = random.split(key, 3)
        bc_new = coefficients
        forest = s.forest
        if proposal == 'coding' and s.adaptive_coding:
            sufficient = coding_sufficient_statistics(base, is_trt, alpha,
                beta[s.exposure_idx], s.z_vec, jnp.where(observed, pseudo-fixed, 0.),
                s.forest_nu.offset, precision, s.unit_idx, unit_sd, s.N_units)
            direction = jnp.float64(s.sigma_b)*random.normal(k_propose, (2,), jnp.float64)
            bc_new, stats, log_ratio, evaluations = elliptical_slice(k_acc, coefficients,
                direction, lambda bc: coding_statistics(bc, sufficient))
            proposed, accepted = beta, jnp.bool_(True)
        elif proposal == 'joint' and (s.sample_beta or s.adaptive_coding):
            kb, kc = random.split(k_propose)
            direction_beta = s.K_chol.astype(jnp.float64)@random.normal(kb, beta.shape, jnp.float64)
            direction_coding = jnp.float64(s.sigma_b)*random.normal(kc, (2,), jnp.float64)
            def joint_stats(v):
                return statistics(v[:-2] if s.sample_beta else beta,
                                  v[-2:] if s.adaptive_coding else coefficients)
            result, stats, log_ratio, evaluations = elliptical_slice(k_acc,
                jnp.concatenate((beta, coefficients)),
                jnp.concatenate((direction_beta, direction_coding)), joint_stats)
            proposed = result[:-2] if s.sample_beta else beta
            bc_new = result[-2:] if s.adaptive_coding else coefficients
            accepted = jnp.bool_(True)
        elif proposal == 'tree':
            from longbet._change_prognostic import root_move
            kj, kt = random.split(k_propose)
            j = random.randint(kj, (), 0, forest.leaf_tree.shape[0])
            # A zero-likelihood root MH step is reversible under the actual
            # tree prior (including its support). Use that entire transition
            # as the proposal, so the second-stage MH ratio is just L'/L.
            result = root_move(kt, forest.var_tree[j], forest.split_tree[j],
                forest.leaf_tree[j].astype(jnp.float64)*forest.leaf_unit,
                s.X, jnp.zeros_like(precision), jnp.zeros_like(score), alpha,
                jnp.float64(forest.leaf_prior_cov_inv), forest.max_split,
                forest.blocked_vars, forest.log_s, forest.p_nonterminal,
                0 if forest.min_points_per_leaf is None else forest.min_points_per_leaf,
                0 if forest.min_points_per_decision_node is None else forest.min_points_per_decision_node)
            var, split, _, _, _, affluent, _ = result
            candidate = eqx.tree_at(lambda f: (f.var_tree, f.split_tree), forest,
                (forest.var_tree.at[j].set(var), forest.split_tree.at[j].set(split)))
            candidate_child = eqx.tree_at(lambda s: s.forest, s, candidate)
            candidate_base = _pack(candidate_child, shared, m, capacity)[0]
            current_stats = statistics(beta, coefficients)
            proposal_stats = statistics(beta, coefficients, candidate_base)
            log_ratio = proposal_stats[-1]-current_stats[-1]
            different = jnp.any(var != forest.var_tree[j]) | jnp.any(split != forest.split_tree[j])
            accepted = different & (jnp.log(random.uniform(k_acc, dtype=jnp.float64)) < log_ratio)
            stats = jax.tree.map(lambda a, b: jnp.where(accepted, b, a), current_stats, proposal_stats)
            forest = eqx.tree_at(lambda f: (f.var_tree, f.split_tree, f.affluence_tree), forest,
                (jnp.where(accepted, candidate.var_tree, forest.var_tree),
                 jnp.where(accepted, candidate.split_tree, forest.split_tree),
                 forest.affluence_tree.at[j].set(jnp.where(accepted, affluent, forest.affluence_tree[j]))))
            proposed, evaluations = beta, jnp.int32(2)
        else:
            rho = jnp.float64(proposal_scale if proposal == 'pcn' else 0.)
            proposed = jnp.sqrt(1-rho**2)*beta + rho*(s.K_chol.astype(jnp.float64) @
                random.normal(k_propose, beta.shape, jnp.float64))
            if not s.sample_beta:
                proposed = beta
            current_stats, proposal_stats = statistics(beta, coefficients), statistics(proposed, coefficients)
            log_ratio = proposal_stats[-1]-current_stats[-1]
            accepted = jnp.log(random.uniform(k_acc, dtype=jnp.float64)) < log_ratio
            stats = jax.tree.map(lambda a, b: jnp.where(accepted, b, a), current_stats, proposal_stats)
            evaluations = jnp.int32(2)
        x, units = collapsed_draw(k_draw, stats)
        beta_new = jnp.where(accepted, proposed, beta).astype(jnp.float32)
        bc_new = bc_new.astype(jnp.float32)
        # Fill indices are out of bounds: scatter drops them, preserving all
        # unreachable leaf slots and never letting dummy coordinates enter f.
        new_values = values.at[packed].set(leaf_sd*x, mode='drop')
        mu_values = new_values[:sizes[0]].reshape(s.forest.leaf_tree.shape)
        nu_values = new_values[sizes[0]:sizes[0]+sizes[1]].reshape(s.forest_nu.leaf_tree.shape)
        mu_leaves = (mu_values/s.forest.leaf_unit).astype(s.forest.leaf_tree.dtype)
        nu_leaves = (nu_values/s.forest_nu.leaf_unit).astype(s.forest_nu.leaf_tree.dtype)
        mu_idx = traverse_forest(s.X, forest.var_tree, forest.split_tree)
        nu_idx = traverse_forest(s.X, s.forest_nu.var_tree, s.forest_nu.split_tree)
        mu_fit = s.forest.offset+s.forest.leaf_unit*jnp.sum(jnp.take_along_axis(mu_leaves, mu_idx, axis=1), axis=0)
        nu_fit = s.forest_nu.offset+s.forest_nu.leaf_unit*jnp.sum(jnp.take_along_axis(nu_leaves, nu_idx, axis=1), axis=0)
        shared_new = shared
        if shared is not None:
            shared_values = new_values[sum(sizes[:2]):].reshape(shared.leaf_tree[:, m, :].shape).astype(jnp.float32)
            shared_fit = jnp.sum(jnp.take_along_axis(shared_values, shared.leaf_indices, axis=1), axis=0)
            shared_new = eqx.tree_at(lambda f: (f.leaf_tree, f.fit), shared,
                (shared.leaf_tree.at[:, m, :].set(shared_values), shared.fit.at[m].set(shared_fit)))
            nu_fit = nu_fit+shared_fit
        gamma_new = (unit_sd*units).astype(jnp.float32) if s.random_intercept else s.gamma
        forest = eqx.tree_at(lambda f: f.leaf_tree, forest, mu_leaves)
        if proposal == 'tree':
            # Membership is now authoritative for every tree; clear pending
            # prunes together with it and refresh both bartz cache variants.
            forest = eqx.tree_at(lambda f: (f.leaf_indices, f.to_prune, f.move_node), forest,
                (mu_idx.astype(forest.leaf_indices.dtype), jnp.zeros_like(forest.to_prune),
                 jnp.zeros_like(forest.move_node)))
            if forest.count_tree is not None:
                counts = jax.vmap(lambda ids: jnp.bincount(ids, length=mu_leaves.shape[-1]))(mu_idx)
                forest = eqx.tree_at(lambda f: f.count_tree, forest, counts.astype(forest.count_tree.dtype))
            if forest.prec_tree is not None:
                weights = s.prec_scale if s.prec_scale is not None else jnp.ones(s.y.shape, jnp.float32)
                sums = jax.vmap(lambda ids: jnp.zeros(mu_leaves.shape[-1], jnp.float32).at[ids].add(weights))(mu_idx)
                forest = eqx.tree_at(lambda f: f.prec_tree, forest, sums)
        new_mean = s.alpha*mu_fit+jnp.where(s.z_vec == 1, bc_new[1], bc_new[0])*beta_new[s.exposure_idx]*nu_fit+gamma_new[s.unit_idx]
        resid = jnp.where(observed, s.resid.astype(jnp.float64)+old_mean-new_mean, 0.).astype(jnp.float32)
        child = eqx.tree_at(lambda t: (t.beta, t.gamma, t.forest,
            t.forest_nu.leaf_tree, t.mu_fit, t.nu_fit, t.resid, t.b0, t.b1), s,
            (beta_new, gamma_new, forest, nu_leaves, mu_fit, nu_fit, resid, bc_new[0], bc_new[1]))
        children = list(state.states)
        children[m] = child
        new_state = eqx.tree_at(lambda t: (t.states, t.shared_forest), state,
                               (tuple(children), shared_new), is_leaf=lambda x: x is None)
        return new_state, jnp.array([1., accepted, log_ratio, count, evaluations], jnp.float64)

    return jax.lax.cond(count <= capacity, move,
        lambda _: (state, jnp.array([0., 0., 0., count, 0.], jnp.float64)), operand=None)


@partial(jax.jit, static_argnames=('capacity', 'proposal_scale', 'proposal'))
def collapsed_exposure_single_step(key, state, *, capacity=MAX_ENSEMBLE_LEAVES, proposal_scale=.3, proposal='pcn'):
    observed = jnp.stack([s.obs_mask for s in state.states])
    loadings = state.gamma_loadings if state.sur_active else jnp.zeros_like(state.gamma_loadings)
    omega = observed_precision(observed, loadings, jnp.stack([s.sigma2 for s in state.states]))
    diagnostics = []
    for m in range(state.M):
        raw = jnp.stack([s.resid for s in state.states]).astype(jnp.float64)
        score = jnp.einsum('ik,ki->i', omega[:, m, :], raw)
        state, info = _outcome_move(random.fold_in(key, m), state, m, omega[:, m, m], score,
                                   capacity=capacity, proposal_scale=proposal_scale, proposal=proposal)
        diagnostics.append(info)
    return state, jnp.stack(diagnostics)


def collapsed_exposure_step(key, state, *, capacity=MAX_ENSEMBLE_LEAVES, proposal_scale=.3, proposal='pcn'):
    """All chains; diagnostics: attempt, accept, log ratio, leaves, evaluations."""
    if proposal not in ('pcn', 'coding', 'joint', 'tree'):
        raise ValueError('proposal must be pcn, coding, joint or tree')
    if not state.has_chain_axis:
        return collapsed_exposure_single_step(key, state, capacity=capacity, proposal_scale=proposal_scale, proposal=proposal)
    per, constants = split_multi_chain_fields(state)
    keys = random.split(key, state.gamma_loadings.shape[0])
    def one(k, p):
        return collapsed_exposure_single_step(k, eqx.combine(p, constants),
            capacity=capacity, proposal_scale=proposal_scale, proposal=proposal)
    # Returning a whole vmapped state would replicate shared fields. Split it
    # again inside vmap, as the package's other experimental blocks do.
    def run(k, p):
        out, info = one(k, p)
        return split_multi_chain_fields(out)[0], info
    updated, info = jax.vmap(run)(keys, per)
    return eqx.combine(updated, constants), info
