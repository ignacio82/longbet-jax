"""``LongBetState`` and initialization for the composed LongBet model.

The state subclasses ``bartz.mcmcstep.State`` so that ``X``, ``y`` and
``config`` (mesh, reduction strategies, sharding) are shared with the
prognostic forest rather than duplicated, and so the axis annotations `bartz`
reads are inherited.

Chain handling
--------------
`bartz` supports ``init(num_chains=k)``, but its weighted-observation path does
**not**: ``_compute_count_or_prec_tree`` assumes ``prec_scale`` carries no chain
axis (its second axis is the multi-outcome axis, not chains). LongBet's
treatment forest has per-observation weights ``w = b_Z * beta_S`` that differ
across chains by construction, so ``init(num_chains=k)`` cannot be used for it.

Chains are therefore run with ``jax.vmap`` over single-chain states. To avoid
replicating the design matrix -- which dominates device memory on a large panel
-- only the leaves `bartz` itself marks with a chain axis are replicated;
``X``, ``y``, the panel index vectors, the observation mask and the kernel
factor stay shared and are closed over by the vmapped function.  Which leaves
those are is read from `bartz`'s own field metadata, not hardcoded, so it cannot
drift from upstream (see :func:`chain_filter_spec`).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float32, Int32, UInt

from bartz._jaxext import field
from bartz.mcmcstep import Forest, State, Wishart, init, make_p_nonterminal
from bartz.mcmcstep._axes import CHAIN_AXIS

from longbet._config import LongBetConfig
from longbet._gp import build_kernel_matrix, kernel_cholesky
from longbet._ordinal import full_cutpoints, prepare_ordinal, sample_ordinal_latents
from longbet._shared_forest import enable_x64

#: Which leaves carry a chain axis is **not** hardcoded here.  ``bartz``
#: annotates it on its own dataclass fields (``field(chains=CHAIN_AXIS)``), and
#: :func:`chain_filter_spec` reads that metadata, so LongBet's notion of a chain
#: axis cannot drift from upstream's.  ``Forest.leaf_tree`` is per chain while
#: ``Forest.leaf_unit`` is not, for instance, and getting that distinction wrong
#: silently breaks ``evaluate_trace``.  LongBet's own extra fields declare it the
#: same way in :class:`LongBetState` below.


class LongBetState(State):
    """The full MCMC state for LongBet.

    Inherits ``X``, ``y``, ``z``, ``binary_indices``, ``resid``, ``resid_unit``,
    ``resid_eff_scale``, ``resid_inexact_integral``, ``error_cov_inv``,
    ``error_scale``, ``prec_scale``, ``inv_sdev_scale``, ``inv_sdev_unit``,
    ``n_non_missing``, ``sum_diag_prec_scale``, ``forest`` (mu) and ``config``
    from ``bartz.mcmcstep.State``.

    ``resid`` holds the **full-model residual in data units**, not the scaled
    single-forest residual `bartz` stores there. Conversion happens only at the
    boundary of a `bartz` call (see ``longbet._step``).
    """

    # --- treatment forest nu and its observation scales ---------------------
    forest_nu: Forest = field()
    resid_nu: Float32[Array, '*chains n'] = field(chains=CHAIN_AXIS, data=-1)
    prec_scale_nu: Float32[Array, '*chains n'] | None = field(
        chains=CHAIN_AXIS, data=-1
    )
    inv_sdev_scale_nu: Float32[Array, '*chains n'] | None = field(
        chains=CHAIN_AXIS, data=-1
    )
    inv_sdev_unit_nu: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    resid_unit_nu: Float32[Array, ''] = field()
    leaf_prior_cov_inv_nu: Float32[Array, ''] = field()

    # --- panel indices and masks (M = N * T cells, unit-major) --------------
    unit_idx: Int32[Array, ' n'] = field(data=-1)
    time_idx: Int32[Array, ' n'] = field(data=-1)
    exposure_idx: Int32[Array, ' n'] = field(data=-1)
    z_vec: Float32[Array, ' n'] = field(data=-1)
    obs_mask: Bool[Array, ' n'] = field(data=-1)
    #: Number of observed cells per unit; precomputed since it never changes.
    unit_counts: Float32[Array, ' N'] = field()

    # --- panel dimensions (static) ------------------------------------------
    N_units: int = field(static=True)
    T_periods: int = field(static=True)
    S_max: int = field(static=True)

    # --- model parameters ----------------------------------------------------
    beta: Float32[Array, '*chains S_max_plus_1'] = field(chains=CHAIN_AXIS)
    gamma: Float32[Array, '*chains N'] = field(chains=CHAIN_AXIS)
    alpha: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    b0: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    b1: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    sigma2: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    sigma_gamma2: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    cutpoints: Float32[Array, '*chains K_free'] = field(chains=CHAIN_AXIS)

    # --- cached forest fits, in their own units ------------------------------
    mu_fit: Float32[Array, '*chains n'] = field(chains=CHAIN_AXIS, data=-1)
    nu_fit: Float32[Array, '*chains n'] = field(chains=CHAIN_AXIS, data=-1)

    # --- GP kernel Cholesky factor over 0..S_max (shared, float32) -----------
    K_chol: Float32[Array, 'S_max_plus_1 S_max_plus_1'] = field()

    # --- fields with defaults ------------------------------------------------
    resid_eff_scale_nu: Any = field(chains=CHAIN_AXIS, default=None)
    resid_inexact_integral_nu: Any = field(chains=CHAIN_AXIS, default=None)
    sample_alpha: bool = field(static=True, default=False)
    sample_beta: bool = field(static=True, default=True)
    adaptive_coding: bool = field(static=True, default=True)
    random_intercept: bool = field(static=True, default=True)
    ridge_move: bool = field(static=True, default=True)
    ridge_proposal_sigma: float = field(static=True, default=0.2)
    gamma_prior_a: float = field(static=True, default=1.0)
    gamma_prior_b: float = field(static=True, default=0.1)
    sigma_prior_a: float = field(static=True, default=0.0)
    sigma_prior_b: float = field(static=True, default=0.0)
    sigma_b: float = field(static=True, default=0.7071067811865476)
    sigma_alpha: float = field(static=True, default=1.0)
    outcome_type_str: str = field(static=True, default="continuous")
    num_categories: int = field(static=True, default=0)
    cutpoint_prior_scale: float = field(static=True, default=5.0)

    @property
    def has_chain_axis(self) -> bool:
        """Whether this state carries a leading chain axis."""
        return self.beta.ndim > 1

    @property
    def num_chains(self) -> int:
        """Number of chains represented by this state."""
        return self.beta.shape[0] if self.has_chain_axis else 1


def _leaf_is_chained(root: Any, path: tuple[Any, ...]) -> bool:
    """Whether the leaf at ``path`` carries a chain axis.

    Walks the path through the actual objects and reads each dataclass field's
    ``chains`` metadata; the innermost annotation wins.  A field with no
    annotation is shared, which is `bartz`'s convention (``X``, ``resid_unit``
    and ``Forest.leaf_unit`` are shared; ``resid``, ``Forest.leaf_tree`` and
    ``Wishart.value`` are not).
    """
    obj = root
    chained = False
    for entry in path:
        name = getattr(entry, "name", None)
        if name is None:  # a sequence index: keep the enclosing decision
            idx = getattr(entry, "idx", None)
            if idx is not None and isinstance(obj, (tuple, list)) and idx < len(obj):
                obj = obj[idx]
            continue
        try:
            fields_by_name = {f.name: f for f in dataclasses.fields(obj)}
        except TypeError:  # not a dataclass; nothing more to learn
            return chained
        fld = fields_by_name.get(name)
        if fld is None:
            return chained
        chained = fld.metadata.get("chains", None) == CHAIN_AXIS
        obj = getattr(obj, name)
    return chained


def chain_filter_spec(state: LongBetState) -> Any:
    """Boolean pytree marking the leaves that carry a chain axis.

    Used with :func:`equinox.partition` to split a chained state into the part
    ``jax.vmap`` should map over and the part it should close over.
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(state)
    flags = [_leaf_is_chained(state, path) for path, _ in leaves_with_path]
    return jax.tree_util.tree_unflatten(treedef, flags)


def chained_field_names(state: LongBetState) -> set[str]:
    """Top-level field names that contain at least one chain-axis leaf."""
    names: set[str] = set()
    for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]:
        if _leaf_is_chained(state, path):
            top = next((getattr(e, "name") for e in path if hasattr(e, "name")), None)
            if top is not None:
                names.add(top)
    return names


def split_chain_fields(state: LongBetState) -> tuple[Any, Any]:
    """Split into ``(per_chain, shared)`` pytrees."""
    return eqx.partition(state, chain_filter_spec(state))


def broadcast_to_chains(
    state: LongBetState,
    num_chains: int,
    key: Any = None,
) -> LongBetState:
    """Replicate the chain-axis fields of a single-chain state.

    ``X``, ``y``, the index vectors and the kernel factor are left untouched, so
    a ``k``-chain fit costs ``k`` copies of the parameters and residuals rather
    than ``k`` copies of the design matrix.

    When ``key`` is given the chains are additionally **overdispersed**: each
    one's scalar and vector parameters are drawn from their priors rather than
    started at a common point. This matters for what R-hat means. Chains started
    at identical values and separated only by their random streams can agree
    with each other while all of them sit in the same unvisited region, so R-hat
    understates non-convergence; the statistic is designed for starts that are
    dispersed relative to the target.

    Only quantities that leave the initial fit unchanged are dispersed freely --
    at initialisation both forests are empty, so ``beta``, ``b0``, ``b1`` and the
    variances do not enter the fitted values at all. ``gamma`` does, so the
    residual is corrected for it and the model's invariant
    ``resid == y - fitted`` continues to hold on the first sweep.
    """
    if num_chains <= 1:
        return state
    if state.has_chain_axis:
        raise ValueError("state already carries a chain axis")

    per, shared = split_chain_fields(state)
    per = jax.tree.map(lambda x: jnp.broadcast_to(x, (num_chains, *x.shape)), per)
    state = eqx.combine(per, shared)
    if key is None:
        return state
    return _overdisperse_chains(state, key, num_chains)


def _overdisperse_chains(
    state: LongBetState, key: Any, num_chains: int
) -> LongBetState:
    """Draw each chain's starting parameters from their priors."""
    k_beta, k_gamma, k_sig, k_siggam, k_b = jax.random.split(key, 5)

    # beta ~ N(0, K_tilde), via the same Cholesky factor the sampler uses.
    if state.sample_beta:
        eta = jax.random.normal(
            k_beta, (num_chains, state.K_chol.shape[-1]), jnp.float32
        )
        beta = jnp.einsum('ij,cj->ci', state.K_chol, eta)
    else:
        beta = state.beta

    # The variances start from their priors when those are proper, and from a
    # lognormal spread around 1 under the improper reference prior, which has no
    # scale of its own to draw from.
    def _inv_gamma_or_spread(k, a, b):
        if a > 0 and b > 0:
            return (b / jnp.maximum(jax.random.gamma(k, a, (num_chains,)), 1e-30)).astype(
                jnp.float32
            )
        return jnp.exp(jax.random.normal(k, (num_chains,), jnp.float32)).astype(jnp.float32)

    if state.outcome_type_str in ("binary", "ordinal"):
        sigma2 = jnp.ones((num_chains,), dtype=jnp.float32)
    else:
        sigma2 = _inv_gamma_or_spread(k_sig, state.sigma_prior_a, state.sigma_prior_b)

    if state.random_intercept:
        sigma_gamma2 = _inv_gamma_or_spread(
            k_siggam, state.gamma_prior_a, state.gamma_prior_b
        )
        # gamma_i | sigma_gamma^2 ~ N(0, sigma_gamma^2); it enters the fit, so the
        # residual has to follow it.
        gamma = jax.random.normal(
            k_gamma, (num_chains, state.N_units), jnp.float32
        ) * jnp.sqrt(sigma_gamma2)[:, None]
        resid = state.resid - jnp.where(
            state.obs_mask, gamma[:, state.unit_idx], 0.0
        )
    else:
        sigma_gamma2 = state.sigma_gamma2
        gamma = state.gamma
        resid = state.resid

    # b0, b1 ~ N(their starting value, sigma_b): a spread around the coding the
    # user asked for, not a different model.
    if state.adaptive_coding:
        b_noise = jax.random.normal(k_b, (2, num_chains), jnp.float32) * state.sigma_b
        b0 = state.b0 + b_noise[0]
        b1 = state.b1 + b_noise[1]
    else:
        b0 = state.b0
        b1 = state.b1

    state = eqx.tree_at(
        lambda s: (s.beta, s.gamma, s.sigma2, s.sigma_gamma2, s.b0, s.b1, s.resid),
        state,
        (beta, gamma, sigma2, sigma_gamma2, b0, b1, resid),
    )
    if state.num_categories > 2:
        # Ordinal-only streams; preserve the five legacy initialization keys.
        gap_key = jax.random.fold_in(key, 8101)
        latent_keys = jax.random.split(jax.random.fold_in(key, 8102), num_chains)
        gaps = jnp.diff(jnp.concatenate((jnp.zeros((num_chains, 1)),
                                         state.cutpoints), axis=1), axis=1)
        gaps *= jnp.exp(.25 * jax.random.normal(gap_key, gaps.shape, jnp.float32))
        cutpoints = jnp.cumsum(gaps, axis=1)
        mean = (state.alpha[:, None] * state.mu_fit
                + jnp.where(state.z_vec == 1, b1[:, None], b0[:, None])
                * beta[:, state.exposure_idx] * state.nu_fit
                + gamma[:, state.unit_idx])
        z = jax.vmap(lambda k, m, cp, old: sample_ordinal_latents(
            k, state.y, m, 1., cp, state.obs_mask, old))(
                latent_keys, mean, cutpoints, state.z)
        state = eqx.tree_at(lambda s: (s.cutpoints, s.z, s.resid), state,
                            (cutpoints, z, jnp.where(state.obs_mask, z - mean, 0.)))
    return state


@enable_x64(False)
def init_longbet(
    *,
    X_unified: UInt[Array, 'p n'],
    y: Float32[Array, ' n'],
    unit_idx: Int32[Array, ' n'],
    time_idx: Int32[Array, ' n'],
    exposure_idx: Int32[Array, ' n'],
    z_vec: Float32[Array, ' n'],
    obs_mask: Bool[Array, ' n'],
    max_split_mu: UInt[Array, ' p'],
    max_split_nu: UInt[Array, ' p'],
    config: LongBetConfig,
    offset: float = 0.0,
    num_chains: int | None = None,
    chain_key: Any = None,
    mesh: Any = None,
    **kwargs: Any,
) -> LongBetState:
    """Build the initial ``LongBetState``.

    Parameters
    ----------
    X_unified
        Unified binned predictor matrix, shape ``(p, n)``, shared by both
        forests. Which columns each forest may split on is controlled by its own
        ``max_split`` vector, not by a second copy of the matrix.
    y
        Response vector of length ``n`` (already standardized for a continuous
        outcome; raw 0/1 for a binary one). Masked cells must be finite.
    unit_idx, time_idx, exposure_idx, z_vec, obs_mask
        Panel indices and indicators, length ``n``.
    max_split_mu, max_split_nu
        Per-forest split bounds, length ``p``. A zero entry blocks that column.
    config
        Sampler configuration.
    offset
        Additive offset handed to `bartz`. For a binary outcome this is the
        probit intercept ``Phi^-1(rate)``; for a standardized continuous outcome
        it is 0.
    num_chains
        Number of chains to replicate the parameter block into.
    chain_key
        PRNG key used to overdisperse the chains' starting points. Without it
        every chain starts identically, which weakens R-hat.
    mesh
        Optional JAX device mesh.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    offset = float(offset)
    unit_idx = jnp.asarray(unit_idx, dtype=jnp.int32)
    time_idx = jnp.asarray(time_idx, dtype=jnp.int32)
    exposure_idx = jnp.asarray(exposure_idx, dtype=jnp.int32)
    z_vec = jnp.asarray(z_vec, dtype=jnp.float32)
    obs_mask = jnp.asarray(obs_mask, dtype=jnp.bool_)

    n = y.shape[0]
    N_units = int(jnp.max(unit_idx)) + 1
    T_periods = int(jnp.max(time_idx)) + 1
    S_max = int(jnp.max(exposure_idx))

    p_nonterminal_mu = make_p_nonterminal(
        config.max_depth_pr, config.alpha_split_pr, config.beta_split_pr
    )
    p_nonterminal_nu = make_p_nonterminal(
        config.max_depth_trt, config.alpha_split_trt, config.beta_split_trt
    )

    filter_splitless_mu = int(jnp.sum(max_split_mu == 0))
    filter_splitless_nu = int(jnp.sum(max_split_nu == 0))

    # Standard BART leaf-variance priors on a standardized response.
    sigma_mu = 3.0 / (2.0 * math.sqrt(config.num_trees_pr))
    sigma_nu = 3.0 / (3.0 * math.sqrt(config.num_trees_trt))
    leaf_prior_cov_inv_mu = jnp.array(1.0 / sigma_mu**2, dtype=jnp.float32)
    leaf_prior_cov_inv_nu = jnp.array(1.0 / sigma_nu**2, dtype=jnp.float32)

    # Adaptive coding: XBCF's antisymmetric start. When it is off, b0 = b1 = 1,
    # as specified by the public API. Treatment-only coding (b0 = 0) changes
    # both the model and its counterfactual contrast.
    if config.adaptive_coding:
        b0_init, b1_init = -0.5, 0.5
    else:
        b0_init, b1_init = 1.0, 1.0

    beta_init = jnp.ones(S_max + 1, dtype=jnp.float32)
    b_z_init = jnp.where(z_vec == 1.0, b1_init, b0_init)
    w_init = b_z_init * beta_init[exposure_idx]

    mask_nu_init = obs_mask & (jnp.abs(w_init) > 1e-6)
    error_scale_nu = jnp.reciprocal(jnp.where(mask_nu_init, jnp.abs(w_init), 1.0))

    is_binary = config.outcome == "binary" or (
        config.outcome == "ordinal" and config.num_categories == 2)
    is_ordered = config.outcome == "ordinal" and config.num_categories > 2
    cutpoints = jnp.empty((0,), jnp.float32)
    working_y = y
    if config.outcome == "ordinal":
        prepared = prepare_ordinal(np.where(np.asarray(obs_mask), np.asarray(y), np.nan),
                                   config.num_categories)
        y = jnp.asarray(prepared.labels)
        cutpoints = jnp.asarray(prepared.cutpoints)
        if is_ordered:
            full = full_cutpoints(cutpoints)
            labels = y.astype(jnp.int32)
            lo, hi = full[labels], full[labels + 1]
            # A finite interior working response, never category labels passed
            # through bartz's binary branch. No hidden initialization RNG.
            lo_f = jnp.where(jnp.isfinite(lo), lo, hi - 2.)
            hi_f = jnp.where(jnp.isfinite(hi), hi, lo + 2.)
            working_y = jnp.where(obs_mask, lo_f / 2 + hi_f / 2, 0.)
        else:
            working_y = y

    def _err_cov() -> Wishart:
        # A fresh object per call: bartz's init donates and deletes its buffers.
        return Wishart(nu=jnp.array(3.0), rate=jnp.array(3.0), value=jnp.array(1.0))

    # X and y are copied because bartz's init donates and deletes its buffers.
    state_mu = init(
        X=jnp.copy(X_unified),
        y=jnp.copy(working_y),
        outcome_type="binary" if is_binary else "continuous",
        offset=offset,
        max_split=jnp.copy(max_split_mu),
        num_trees=config.num_trees_pr,
        p_nonterminal=p_nonterminal_mu,
        leaf_prior_cov_inv=leaf_prior_cov_inv_mu,
        error_cov_inv=None if is_binary else _err_cov(),
        missing=~obs_mask,
        min_points_per_leaf=config.min_points_per_leaf_pr,
        filter_splitless_vars=filter_splitless_mu,
        num_chains=None,
        mesh=mesh,
        **kwargs,
    )

    state_nu = init(
        X=jnp.copy(X_unified),
        y=jnp.copy(working_y),
        outcome_type="continuous",
        offset=0.0,
        max_split=jnp.copy(max_split_nu),
        num_trees=config.num_trees_trt,
        p_nonterminal=p_nonterminal_nu,
        leaf_prior_cov_inv=leaf_prior_cov_inv_nu,
        error_cov_inv=_err_cov(),
        error_scale=error_scale_nu,
        missing=~mask_nu_init,
        min_points_per_leaf=config.min_points_per_leaf_trt,
        filter_splitless_vars=filter_splitless_nu,
        num_chains=None,
        mesh=mesh,
        **kwargs,
    )

    # Disable bartz's own error-variance draw on *both* views: sigma^2 is drawn
    # once per sweep from the full-model residual. Leaving it on either view
    # would draw it twice from two different conditionals.
    def _freeze_variance(st: State) -> State:
        frozen = eqx.tree_at(
            lambda w: (w.nu, w.rate),
            st.error_cov_inv,
            (None, None),
            is_leaf=lambda x: x is None,
        )
        return eqx.tree_at(lambda s: s.error_cov_inv, st, frozen)

    state_mu = _freeze_variance(state_mu)
    state_nu = _freeze_variance(state_nu)

    # GP kernel: build and factor in float64, hand only the Cholesky to JAX.
    K_tilde = build_kernel_matrix(
        s=np.arange(S_max + 1, dtype=np.float64),
        sig_knl=config.sig_knl,
        lambda_knl=config.lambda_knl,
        kernel_type=config.kernel_type,
        sigma_m=config.sigma_m,
        gp_constant_mean=config.gp_constant_mean,
        jitter=config.gp_jitter,
    )
    K_chol = jnp.asarray(kernel_cholesky(K_tilde), dtype=jnp.float32)

    init_sigma_gamma2 = float(config.gamma_prior_b / max(config.gamma_prior_a, 1e-4))
    unit_counts = (
        jnp.zeros((N_units,), dtype=jnp.float32)
        .at[unit_idx]
        .add(obs_mask.astype(jnp.float32))
    )

    # The initial full-model residual is whatever bartz's own init left on the
    # prognostic view, read back into data units. For a standardized continuous
    # outcome that is y - offset; for a binary outcome the latent z starts at
    # the offset and the residual starts at zero. Deriving it rather than
    # writing y - offset keeps the binary path correct.
    resid_init = state_mu.resid * state_mu.resid_unit
    resid_init = jnp.where(obs_mask, resid_init, 0.0).astype(jnp.float32)

    state = LongBetState(
        _chain_anchor=state_mu._chain_anchor,
        X=state_mu.X,
        y=y if is_ordered else state_mu.y,
        z=working_y if is_ordered else state_mu.z,
        binary_indices=state_mu.binary_indices,
        resid=resid_init,
        resid_unit=state_mu.resid_unit,
        resid_eff_scale=state_mu.resid_eff_scale,
        resid_inexact_integral=state_mu.resid_inexact_integral,
        error_cov_inv=state_mu.error_cov_inv,
        error_scale=state_mu.error_scale,
        prec_scale=state_mu.prec_scale,
        inv_sdev_scale=state_mu.inv_sdev_scale,
        inv_sdev_unit=state_mu.inv_sdev_unit,
        n_non_missing=state_mu.n_non_missing,
        sum_diag_prec_scale=state_mu.sum_diag_prec_scale,
        forest=state_mu.forest,
        config=state_mu.config,
        forest_nu=state_nu.forest,
        resid_nu=state_nu.resid,
        prec_scale_nu=state_nu.prec_scale,
        inv_sdev_scale_nu=state_nu.inv_sdev_scale,
        inv_sdev_unit_nu=state_nu.inv_sdev_unit,
        resid_unit_nu=state_nu.resid_unit,
        leaf_prior_cov_inv_nu=state_nu.forest.leaf_prior_cov_inv,
        unit_idx=unit_idx,
        time_idx=time_idx,
        exposure_idx=exposure_idx,
        z_vec=z_vec,
        obs_mask=obs_mask,
        unit_counts=unit_counts,
        N_units=N_units,
        T_periods=T_periods,
        S_max=S_max,
        beta=beta_init,
        gamma=jnp.zeros(N_units, dtype=jnp.float32),
        alpha=jnp.array(1.0, dtype=jnp.float32),
        b0=jnp.array(b0_init, dtype=jnp.float32),
        b1=jnp.array(b1_init, dtype=jnp.float32),
        sigma2=jnp.array(1.0, dtype=jnp.float32),
        sigma_gamma2=jnp.array(init_sigma_gamma2, dtype=jnp.float32),
        cutpoints=cutpoints,
        # Include the forest offset: prediction includes it, and the alpha
        # conditional must scale the same prognostic mean as the likelihood.
        mu_fit=jnp.full(n, offset, dtype=jnp.float32),
        nu_fit=jnp.zeros(n, dtype=jnp.float32),
        K_chol=K_chol,
        resid_eff_scale_nu=state_nu.resid_eff_scale,
        resid_inexact_integral_nu=state_nu.resid_inexact_integral,
        sample_alpha=config.sample_alpha,
        sample_beta=config.sample_beta,
        adaptive_coding=config.adaptive_coding,
        random_intercept=config.random_intercept,
        ridge_move=config.ridge_move,
        ridge_proposal_sigma=config.ridge_proposal_sigma,
        gamma_prior_a=config.gamma_prior_a,
        gamma_prior_b=config.gamma_prior_b,
        sigma_prior_a=config.sigma_prior_a,
        sigma_prior_b=config.sigma_prior_b,
        sigma_b=config.sigma_b,
        sigma_alpha=config.sigma_alpha,
        outcome_type_str=config.outcome,
        num_categories=(config.num_categories if config.outcome == "ordinal"
                        else 2 if is_binary else 0),
        cutpoint_prior_scale=config.cutpoint_prior_scale,
    )

    if is_ordered:
        latent = working_y
        if chain_key is not None and (num_chains is None or num_chains <= 1):
            latent = sample_ordinal_latents(jax.random.fold_in(chain_key, 8103),
                y, state.mu_fit, 1., cutpoints, obs_mask, working_y)
        state = eqx.tree_at(lambda s: (s.z, s.resid), state,
                            (latent, jnp.where(obs_mask, latent - state.mu_fit, 0.)))

    if num_chains is not None and num_chains > 1:
        state = broadcast_to_chains(state, int(num_chains), key=chain_key)
    return state
