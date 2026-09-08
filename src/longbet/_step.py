"""One full Gibbs sweep for LongBet.

Two invariants govern this module.

**Units.** ``bartz`` stores its residual scaled: ``resid_unit * resid == y -
sum(trees)``, and ``resid_unit`` differs between the two forests because their
leaf priors and tree counts differ.  LongBet therefore carries the full-model
residual ``R`` in **data units** at all times and converts only at the boundary
of a ``bartz`` call, via :func:`_load` and :func:`_read`. Forest fits are summed
from cached leaf memberships, without traversing trees again. Differencing
residuals after division by a small treatment multiplier loses leaf updates
to cancellation; those errors would then enter the GP and coding conditionals.

**Chains.** :func:`longbet_single_step` always operates on a single chain.
Multiple chains are run by :func:`longbet_step`, which splits the state into its
per-chain and shared parts and vmaps over the former only, so the design matrix
is never replicated (see ``longbet._state``).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, Float32, Key

from bartz.mcmcstep import State, Wishart
from bartz.mcmcstep import step as bartz_step
from bartz.mcmcstep._step import step_z

from longbet._gp import sample_beta_gp
from longbet._forest_cache import refresh_prec_tree, current_forest_fit
from longbet._shared_forest import enable_x64
from longbet._ridge import ridge_scale_step
from longbet._scales import compute_treatment_scale_attrs
from longbet._state import LongBetState, chain_filter_spec

#: Below this weight a cell carries no information about ``nu`` and is dropped
#: from the treatment forest for the sweep.
WEIGHT_EPS = 1e-6


def _load(
    r: Float32[Array, ' n'], resid_unit: Float32[Array, '']
) -> Float32[Array, ' n']:
    """Convert a residual from data units to bartz's state units."""
    return (r / resid_unit).astype(jnp.float32)


def _read(
    resid: Float32[Array, ' n'], resid_unit: Float32[Array, '']
) -> Float32[Array, ' n']:
    """Convert a residual from bartz's state units to data units."""
    return resid * resid_unit


def _sample_inv_gamma(
    key: Key[Array, ''],
    shape: Float32[Array, ''],
    rate: Float32[Array, ''],
) -> Float32[Array, '']:
    """Draw from Inv-Gamma(shape, rate) via a Gamma draw."""
    g = jnp.maximum(random.gamma(key, shape, dtype=jnp.float32), 1e-30)
    return (rate / g).astype(jnp.float32)


@partial(jax.jit, static_argnames=('shared_treatment',))
@enable_x64(False)
def longbet_single_step(
    key: Key[Array, ''],
    state: LongBetState,
    conditional_precision: Float32[Array, ' n'] | None = None,
    *, shared_treatment: bool = False,
) -> LongBetState:
    """Execute one Gibbs sweep on a single chain.

    Jitted at the boundary. ``bartz.mcmcstep.step`` donates its argument's
    buffers, so an un-jitted sweep would delete arrays the caller still holds --
    including ``X``, which is shared between chains. Inside ``jit`` the donation
    is an XLA aliasing hint and stays contained.

    Sweep order (plan section 4.9)::

        z (binary only) -> mu -> alpha -> nu -> beta -> b0,b1
                        -> gamma -> sigma_gamma^2 -> sigma^2 -> ridge move

    Any order is valid provided each draw conditions on current values; this one
    keeps ``R`` consistent with the least bookkeeping and puts the variance draws
    last so they see the fully updated residual.  In particular ``gamma`` is drawn
    *after* the treatment fit, so a treated unit's post-adoption periods do not
    drag its baseline upward.

    When ``conditional_precision`` is supplied, ``state.resid`` is the
    conditional (pseudo-response) residual from the full SUR likelihood.
    Every mean/latent update uses that cell-specific precision. The structural
    innovation variance ``state.sigma2`` is then LEFT UNCHANGED: its caller
    must update it from the structural innovation, not this pseudo-residual.
    None selects the scalar likelihood with its existing random-key schedule.
    Both paths refresh leaf caches when their observation weights change.
    With ``shared_treatment=True``, ``nu_fit`` includes the caller's shared
    ensemble while ``forest_nu`` holds only private trees. Backfitting changes
    the private term; GP/coding/intercepts use the total. The caller must run
    the ridge move on BOTH ensembles, so the scalar ridge move is skipped.
    """
    keys = random.split(key, 10)

    R = state.resid  # full-model residual, data units, shape (n,)
    obs_mask = state.obs_mask
    obs_mask_f32 = obs_mask.astype(jnp.float32)
    sigma2 = state.sigma2
    alpha = state.alpha
    is_binary = state.outcome_type_str == "binary"

    # Keep the global innovation precision factor and express conditional
    # weights relative to it. These temporary, chain-dependent attributes must
    # not replace the shared baseline observation mask in the returned state.
    conditional_attrs = None
    if conditional_precision is not None:
        conditional_attrs, _ = compute_treatment_scale_attrs(
            jnp.sqrt(conditional_precision * sigma2), obs_mask, eps=0.0
        )

    def weighted_sum(value):
        if conditional_precision is None:
            return jnp.sum(value) / sigma2
        return jnp.sum(jnp.where(obs_mask, value * conditional_precision, 0.0))

    # ------------------------------------------------------------------
    # 0. Probit latent z (binary outcomes only)
    # ------------------------------------------------------------------
    if is_binary:
        # step_z reconstructs the mean as z - resid * resid_unit, so it recovers
        # alpha*mu + b*beta*nu + gamma correctly without knowing the model has
        # more terms in it -- provided the residual it sees is the full-model one.
        view_z = State(
            _chain_anchor=state._chain_anchor,
            X=state.X,
            y=state.y,
            z=state.z,
            binary_indices=state.binary_indices,
            resid=_load(R, state.resid_unit),
            resid_unit=state.resid_unit,
            resid_eff_scale=state.resid_eff_scale,
            resid_inexact_integral=state.resid_inexact_integral,
            error_cov_inv=Wishart(nu=None, rate=None, value=jnp.float32(1.0)),
            # Binary structural sigma2 is 1, but its conditional latent
            # variance given continuous outcomes is generally smaller than 1.
            error_scale=(state.error_scale if conditional_attrs is None else
                         lax.rsqrt(conditional_precision)),
            prec_scale=(state.prec_scale if conditional_attrs is None else
                        conditional_attrs.prec_scale),
            inv_sdev_scale=(state.inv_sdev_scale if conditional_attrs is None else
                            conditional_attrs.inv_sdev_scale),
            inv_sdev_unit=(state.inv_sdev_unit if conditional_attrs is None else
                           conditional_attrs.inv_sdev_unit),
            n_non_missing=state.n_non_missing,
            sum_diag_prec_scale=(state.sum_diag_prec_scale if conditional_attrs is None else
                                 conditional_attrs.sum_diag_prec_scale),
            forest=state.forest,
            config=state.config,
        )
        view_z = step_z(keys[0], view_z)
        current_z = view_z.z
        R = R + jnp.where(obs_mask, current_z - state.z, 0.0)
    else:
        current_z = state.z

    # ------------------------------------------------------------------
    # 1. Prognostic forest mu
    # ------------------------------------------------------------------
    # Target is Y - b*beta*nu - gamma = alpha*mu + eps, so the forest sees the
    # response divided by alpha and a precision multiplied by alpha^2. Because
    # alpha is a scalar it folds into the error precision rather than into the
    # per-observation prec_scale.
    alpha_safe = jnp.where(jnp.abs(alpha) < 1e-6, 1.0, alpha)

    view_mu = State(
        _chain_anchor=state._chain_anchor,
        X=state.X,
        y=state.y,
        # The full-model latent response was sampled above. bartz.step would
        # sample it AGAIN after the tree update if z were non-None, corrupting
        # fit-by-residual-differencing and leaving state.z inconsistent with R.
        z=None,
        binary_indices=None,
        resid=_load(R / alpha_safe, state.resid_unit),
        resid_unit=state.resid_unit,
        resid_eff_scale=state.resid_eff_scale,
        resid_inexact_integral=state.resid_inexact_integral,
        error_cov_inv=Wishart(
            nu=None, rate=None, value=jnp.square(alpha_safe) / sigma2
        ),
        error_scale=None,
        prec_scale=(state.prec_scale if conditional_attrs is None else
                    conditional_attrs.prec_scale),
        inv_sdev_scale=(state.inv_sdev_scale if conditional_attrs is None else
                        conditional_attrs.inv_sdev_scale),
        inv_sdev_unit=(state.inv_sdev_unit if conditional_attrs is None else
                       conditional_attrs.inv_sdev_unit),
        n_non_missing=state.n_non_missing,
        sum_diag_prec_scale=(state.sum_diag_prec_scale if conditional_attrs is None else
                             conditional_attrs.sum_diag_prec_scale),
        forest=state.forest,
        config=state.config,
    )

    if conditional_attrs is not None:
        view_mu = refresh_prec_tree(view_mu)
    view_mu = bartz_step(keys[1], view_mu)
    mu_fit = current_forest_fit(view_mu.forest)
    R = jnp.where(obs_mask, R-alpha*(mu_fit-state.mu_fit), 0.)

    # ------------------------------------------------------------------
    # 2. Prognostic scale alpha
    # ------------------------------------------------------------------
    alpha_new = alpha
    if state.sample_alpha:
        u = jnp.where(obs_mask, mu_fit, 0.0)
        e = jnp.where(obs_mask, R + alpha * mu_fit, 0.0)
        prec = weighted_sum(jnp.square(u)) + 1.0 / state.sigma_alpha**2
        mean = weighted_sum(u * e) / prec
        alpha_new = mean + random.normal(keys[2], (), dtype=jnp.float32) * lax.rsqrt(prec)
        R = jnp.where(obs_mask, R - (alpha_new - alpha) * mu_fit, 0.0)

    # ------------------------------------------------------------------
    # 3. Treatment forest nu
    # ------------------------------------------------------------------
    b_z = jnp.where(state.z_vec == 1.0, state.b1, state.b0)
    beta_expanded = state.beta[state.exposure_idx]
    w = b_z * beta_expanded
    attrs_nu, mask_nu = compute_treatment_scale_attrs(w, obs_mask, eps=WEIGHT_EPS)
    if conditional_precision is not None:
        # Drop cells by their actual treatment multiplier, NOT by a rescaled
        # threshold that changes when the conditional precision changes.
        attrs_nu, _ = compute_treatment_scale_attrs(
            w * jnp.sqrt(conditional_precision * sigma2), mask_nu, eps=0.0
        )
    w_safe = jnp.where(mask_nu, w, 1.0)
    resid_nu_data = jnp.where(mask_nu, R / w_safe, 0.0)

    view_nu = State(
        _chain_anchor=state._chain_anchor,
        X=view_mu.X,
        y=view_mu.y,
        z=None,
        binary_indices=None,
        resid=_load(resid_nu_data, state.resid_unit_nu),
        resid_unit=state.resid_unit_nu,
        resid_eff_scale=state.resid_eff_scale_nu,
        resid_inexact_integral=state.resid_inexact_integral_nu,
        error_cov_inv=Wishart(nu=None, rate=None, value=jnp.reciprocal(sigma2)),
        error_scale=None,
        prec_scale=attrs_nu.prec_scale,
        inv_sdev_scale=attrs_nu.inv_sdev_scale,
        inv_sdev_unit=attrs_nu.inv_sdev_unit,
        n_non_missing=attrs_nu.n_non_missing,
        sum_diag_prec_scale=attrs_nu.sum_diag_prec_scale,
        forest=state.forest_nu,
        config=view_mu.config,
    )

    view_nu = refresh_prec_tree(view_nu)
    private_before = (current_forest_fit(view_nu.forest) if shared_treatment
                      else state.nu_fit)
    view_nu = bartz_step(keys[4], view_nu)
    private_after = current_forest_fit(view_nu.forest)
    private_delta = private_after - private_before
    nu_fit = state.nu_fit + private_delta if shared_treatment else private_after
    forest_nu = view_nu.forest

    # Apply the physical fit delta even on zero/negligible-weight cells. Avoid
    # subtracting large R/w values to recover a small leaf change.
    R = jnp.where(obs_mask, R-w*private_delta, 0.0)

    # ------------------------------------------------------------------
    # 4. GP trajectory beta_S
    # ------------------------------------------------------------------
    d_vec = b_z * nu_fit  # treatment factor with beta removed
    r_partial = R + d_vec * state.beta[state.exposure_idx]

    d_obs = jnp.where(obs_mask, d_vec, 0.0)
    r_obs = jnp.where(obs_mask, r_partial, 0.0)

    A = (
        jnp.zeros(state.S_max + 1, dtype=jnp.float32)
        .at[state.exposure_idx]
        .add(jnp.square(d_obs) / sigma2 if conditional_precision is None else
             jnp.square(d_obs) * conditional_precision)
    )
    C = (
        jnp.zeros(state.S_max + 1, dtype=jnp.float32)
        .at[state.exposure_idx]
        .add(d_obs * r_obs / sigma2 if conditional_precision is None else
             d_obs * r_obs * conditional_precision)
    )

    if state.sample_beta:
        beta_new = sample_beta_gp(keys[3], A, C, state.K_chol)
    else:
        beta_new = state.beta

    beta_expanded = beta_new[state.exposure_idx]
    R = jnp.where(obs_mask, r_partial - d_vec * beta_expanded, 0.0)

    # ------------------------------------------------------------------
    # 5. Adaptive coding b0, b1
    # ------------------------------------------------------------------
    b0_new, b1_new = state.b0, state.b1
    if state.adaptive_coding:
        g = beta_expanded * nu_fit
        r_b = R + b_z * g
        g_obs = jnp.where(obs_mask, g, 0.0)
        r_b_obs = jnp.where(obs_mask, r_b, 0.0)
        prior_prec = 1.0 / state.sigma_b**2

        g0 = jnp.where(state.z_vec == 0.0, g_obs, 0.0)
        prec_b0 = weighted_sum(jnp.square(g0)) + prior_prec
        mean_b0 = weighted_sum(g0 * r_b_obs) / prec_b0
        b0_new = mean_b0 + random.normal(keys[5], (), dtype=jnp.float32) * lax.rsqrt(prec_b0)

        g1 = jnp.where(state.z_vec == 1.0, g_obs, 0.0)
        prec_b1 = weighted_sum(jnp.square(g1)) + prior_prec
        mean_b1 = weighted_sum(g1 * r_b_obs) / prec_b1
        b1_new = mean_b1 + random.normal(keys[6], (), dtype=jnp.float32) * lax.rsqrt(prec_b1)

        b_z_new = jnp.where(state.z_vec == 1.0, b1_new, b0_new)
        R = jnp.where(obs_mask, r_b - b_z_new * g, 0.0)

    # ------------------------------------------------------------------
    # 6. Unit intercepts gamma_i and sigma_gamma^2
    # ------------------------------------------------------------------
    gamma_new, sigma_gamma2_new = state.gamma, state.sigma_gamma2
    if state.random_intercept:
        e = R + state.gamma[state.unit_idx]
        e_sum = (
            jnp.zeros(state.N_units, dtype=jnp.float32)
            .at[state.unit_idx]
            .add(jnp.where(obs_mask, e, 0.0))
        )
        if conditional_precision is not None:
            e_sum = jnp.zeros(state.N_units, dtype=jnp.float32).at[state.unit_idx].add(
                jnp.where(obs_mask, e * conditional_precision, 0.0)
            )
            unit_precision = jnp.zeros(state.N_units, dtype=jnp.float32).at[state.unit_idx].add(
                jnp.where(obs_mask, conditional_precision, 0.0)
            )
        else:
            e_sum = e_sum / sigma2
            unit_precision = state.unit_counts / sigma2
        V = jnp.reciprocal(
            unit_precision + jnp.reciprocal(state.sigma_gamma2)
        )
        gamma_new = V * e_sum + random.normal(
            keys[7], (state.N_units,), dtype=jnp.float32
        ) * jnp.sqrt(V)
        R = jnp.where(obs_mask, e - gamma_new[state.unit_idx], 0.0)

        sigma_gamma2_new = _sample_inv_gamma(
            keys[8],
            jnp.float32(state.gamma_prior_a + state.N_units / 2.0),
            state.gamma_prior_b + 0.5 * jnp.sum(jnp.square(gamma_new)),
        )

    # ------------------------------------------------------------------
    # 7. Error variance sigma^2
    # ------------------------------------------------------------------
    if conditional_precision is not None:
        sigma2_new = sigma2
    elif is_binary:
        sigma2_new = jnp.float32(1.0)
    else:
        m_obs = jnp.sum(obs_mask_f32)
        sigma2_new = _sample_inv_gamma(
            keys[9],
            state.sigma_prior_a + m_obs / 2.0,
            state.sigma_prior_b + 0.5 * jnp.sum(jnp.square(jnp.where(obs_mask, R, 0.0))),
        )

    # ------------------------------------------------------------------
    # 8. Metropolis move along the beta-nu scale ridge
    # ------------------------------------------------------------------
    if state.ridge_move and state.sample_beta and not shared_treatment:
        beta_new, forest_nu, nu_fit = ridge_scale_step(
            random.fold_in(key, 888),
            beta_new,
            forest_nu,
            nu_fit,
            state.K_chol,
            state.leaf_prior_cov_inv_nu,
            proposal_sigma=state.ridge_proposal_sigma,
        )

    # ------------------------------------------------------------------
    # Assemble the updated state
    # ------------------------------------------------------------------
    # error_cov_inv records the *sampled* error precision so the saved trace
    # carries a meaningful value; the sweep itself always reads state.sigma2.
    error_cov_inv = eqx.tree_at(
        lambda w_: w_.value,
        state.error_cov_inv,
        jnp.reciprocal(sigma2_new).astype(jnp.float32),
    )

    return eqx.tree_at(
        lambda s: (
            s.X,
            s.y,
            s.config,
            s.z,
            s.resid,
            s.forest,
            s.resid_unit,
            s.resid_eff_scale,
            s.resid_inexact_integral,
            s.error_cov_inv,
            s.prec_scale,
            s.inv_sdev_scale,
            s.inv_sdev_unit,
            s.n_non_missing,
            s.sum_diag_prec_scale,
            s.forest_nu,
            s.resid_nu,
            s.resid_unit_nu,
            s.resid_eff_scale_nu,
            s.resid_inexact_integral_nu,
            s.prec_scale_nu,
            s.inv_sdev_scale_nu,
            s.inv_sdev_unit_nu,
            s.leaf_prior_cov_inv_nu,
            s.alpha,
            s.beta,
            s.b0,
            s.b1,
            s.gamma,
            s.sigma2,
            s.sigma_gamma2,
            s.mu_fit,
            s.nu_fit,
        ),
        state,
        (
            view_nu.X,
            view_nu.y,
            view_nu.config,
            current_z,
            R,
            view_mu.forest,
            view_mu.resid_unit,
            view_mu.resid_eff_scale,
            view_mu.resid_inexact_integral,
            error_cov_inv,
            state.prec_scale if conditional_attrs is not None else view_mu.prec_scale,
            state.inv_sdev_scale if conditional_attrs is not None else view_mu.inv_sdev_scale,
            state.inv_sdev_unit if conditional_attrs is not None else view_mu.inv_sdev_unit,
            view_mu.n_non_missing,
            state.sum_diag_prec_scale if conditional_attrs is not None else view_mu.sum_diag_prec_scale,
            forest_nu,
            view_nu.resid,
            view_nu.resid_unit,
            view_nu.resid_eff_scale,
            view_nu.resid_inexact_integral,
            view_nu.prec_scale,
            view_nu.inv_sdev_scale,
            view_nu.inv_sdev_unit,
            forest_nu.leaf_prior_cov_inv,
            alpha_new,
            beta_new,
            b0_new,
            b1_new,
            gamma_new,
            sigma2_new,
            sigma_gamma2_new,
            mu_fit,
            nu_fit,
        ),
        is_leaf=lambda x: x is None,
    )


@jax.jit
def longbet_step(key: Key[Array, ''], state: LongBetState) -> LongBetState:
    """Execute one Gibbs sweep, across chains if the state carries a chain axis.

    Chains are mapped with ``jax.vmap`` over the per-chain partition of the
    state only; ``X``, ``y``, the panel indices and the GP kernel factor are
    closed over, so a ``k``-chain fit does not hold ``k`` copies of the design
    matrix.  ``bartz``'s own ``num_chains`` cannot be used here because its
    weighted-observation path does not accept a per-chain ``prec_scale``, and
    LongBet's treatment weights differ across chains by construction.
    """
    if not state.has_chain_axis:
        return longbet_single_step(key, state)

    spec = chain_filter_spec(state)
    per_chain, shared = eqx.partition(state, spec)
    keys = random.split(key, state.num_chains)

    def one(k: Key[Array, ''], p: Any) -> Any:
        out = longbet_single_step(k, eqx.combine(p, shared))
        # Return only the per-chain part, so vmap does not add a chain axis to
        # the shared arrays.
        return eqx.filter(out, spec)

    out_per = jax.vmap(one, in_axes=(0, 0))(keys, per_chain)
    return eqx.combine(out_per, shared)
