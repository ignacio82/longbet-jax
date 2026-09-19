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

from longbet._change_internal_move import change_internal_step
from longbet._change_move import change_step
from longbet._regrow_move import regrow_step
from longbet._gp import sample_beta_gp
from longbet._ordinal import sample_cutpoints_marginalized, sample_ordinal_latents
from longbet._forest_cache import refresh_prec_tree, current_forest_fit
from longbet._x64 import enable_x64
from longbet._inter_ensemble_move import (
    inter_ensemble_beta_gamma_step,
    inter_ensemble_transfer_step,
)
from longbet._ridge import ridge_scale_step
from longbet._scales import compute_treatment_scale_attrs
from longbet._trend_basis import (
    horseshoe_col_scales,
    sample_horseshoe,
    trend_block_step,
)
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


def _tempered_single_step(key: Key[Array, ''], state: LongBetState) -> LongBetState:
    """One sweep against ``p(theta) L(theta)^beta`` with ``beta = state.temperature``.

    Every mean, latent and coefficient update runs with the per-cell conditional
    precision ``beta / sigma2``; a continuous outcome's ``sigma2`` is then drawn
    from its tempered conditional ``IG(a + beta m / 2, b + beta SSR / 2)``. See
    ``longbet._tempering``.
    """
    beta = jnp.asarray(state.temperature, dtype=jnp.float32)
    prec = jnp.where(state.obs_mask, beta / state.sigma2, 0.0).astype(jnp.float32)
    k_step, k_sig = random.split(key)
    new = longbet_single_step(k_step, state, prec)
    if state.num_categories < 2:
        m_obs = jnp.sum(state.obs_mask.astype(jnp.float32))
        ssr = jnp.sum(jnp.square(jnp.where(new.obs_mask, new.resid, 0.0)))
        sigma2_new = _sample_inv_gamma(
            k_sig,
            jnp.float32(state.sigma_prior_a) + 0.5 * beta * m_obs,
            jnp.float32(state.sigma_prior_b) + 0.5 * beta * ssr,
        )
        error_cov_inv = eqx.tree_at(
            lambda w: w.value, new.error_cov_inv, jnp.reciprocal(sigma2_new).astype(jnp.float32)
        )
        new = eqx.tree_at(lambda s: (s.sigma2, s.error_cov_inv), new, (sigma2_new, error_cov_inv))
    return new


#: Proposal standard deviation of ``log c`` in the beta-nu scale ridge move.
RIDGE_PROPOSAL_SIGMA = 0.2


@enable_x64()
@jax.jit
@enable_x64(False)
def longbet_single_step(
    key: Key[Array, ''],
    state: LongBetState,
    conditional_precision: Float32[Array, ' n'] | None = None,
) -> LongBetState:
    """Execute one Gibbs sweep on a single chain.

    Jitted at the boundary. ``bartz.mcmcstep.step`` donates its argument's
    buffers, so an un-jitted sweep would delete arrays the caller still holds --
    including ``X``, which is shared between chains. Inside ``jit`` the donation
    is an XLA aliasing hint and stays contained.

    Sweep order (plan section 4.9)::

        z (binary only) -> mu -> nu -> beta -> gamma -> sigma_gamma^2
                        -> ridge move -> subspace transfers
                        -> (c, gamma, beta) joint draw -> sigma^2

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
    """
    keys = random.split(key, 10)

    R = state.resid  # full-model residual, data units, shape (n,)
    obs_mask = state.obs_mask
    obs_mask_f32 = obs_mask.astype(jnp.float32)
    sigma2 = state.sigma2
    alpha = state.alpha
    is_binary = state.num_categories == 2
    is_ordered = state.num_categories > 2
    cutpoints = state.cutpoints

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
    if is_ordered:
        # Distinct ordinal-only fold-in tags leave random.split(key, 10) and
        # every legacy binary/continuous transition unchanged.
        sd = 1. if conditional_precision is None else lax.rsqrt(conditional_precision)
        mean = state.z - R
        cutpoints = sample_cutpoints_marginalized(
            random.fold_in(key, 8112), cutpoints, state.y, mean, sd, obs_mask, state.cutpoint_prior_scale
        )
        current_z = sample_ordinal_latents(
            random.fold_in(key, 8111), state.y, mean, sd, cutpoints, obs_mask, state.z
        )
        R = jnp.where(obs_mask, R + current_z - state.z, 0.)
    elif is_binary:
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
    view_mu = change_step(random.fold_in(key, 7101), view_mu)
    view_mu = change_internal_step(random.fold_in(key, 7105), view_mu)
    view_mu = regrow_step(random.fold_in(key, 7103), view_mu, 1)
    mu_fit = current_forest_fit(view_mu.forest)
    R = jnp.where(obs_mask, R-alpha*(mu_fit-state.mu_fit), 0.)

    # ------------------------------------------------------------------
    # 2. Prognostic scale alpha
    # ------------------------------------------------------------------
    alpha_new = alpha  # fixed at one; kept as a state field for the archive layout

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
    private_before = state.nu_fit
    view_nu = bartz_step(keys[4], view_nu)
    view_nu = change_step(random.fold_in(key, 7102), view_nu)
    view_nu = change_internal_step(random.fold_in(key, 7106), view_nu)
    view_nu = regrow_step(random.fold_in(key, 7104), view_nu, 1)
    private_after = current_forest_fit(view_nu.forest)
    private_delta = private_after - private_before
    nu_fit = private_after
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
    # 5. Coding scales: fixed at b0 = 0, b1 = 1 (treatment-only coding)
    # ------------------------------------------------------------------
    b0_new, b1_new = state.b0, state.b1

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
    # 7. Metropolis move along the beta-nu scale ridge
    # ------------------------------------------------------------------
    if state.sample_beta:
        beta_new, forest_nu, nu_fit = ridge_scale_step(
            random.fold_in(key, 888),
            beta_new,
            forest_nu,
            nu_fit,
            state.K_chol,
            state.leaf_prior_cov_inv_nu,
            proposal_sigma=RIDGE_PROPOSAL_SIGMA,
        )

    # ------------------------------------------------------------------
    # 8. Inter-ensemble residual-transfer Metropolis move (mu <-> gamma, beta, nu)
    # ------------------------------------------------------------------
    if state.use_inter_ensemble_move:
        temp = (
            jnp.float32(1.0)
            if state.temperature is None
            else jnp.asarray(state.temperature, dtype=jnp.float32)
        )
        view_mu_forest, mu_fit, gamma_new, beta_new, R = inter_ensemble_beta_gamma_step(
            random.fold_in(key, 887),
            view_mu.forest,
            mu_fit,
            gamma_new,
            beta_new,
            R,
            alpha_new,
            nu_fit,
            b_z,
            obs_mask,
            state.z_vec,
            state.time_idx,
            state.T_periods,
            state.unit_idx,
            state.N_units,
            state.exposure_idx,
            state.S_max,
            sigma2,
            view_mu.forest.leaf_prior_cov_inv,
            sigma_gamma2_new,
            state.K_chol,
            state.random_intercept,
            state.sample_beta,
            conditional_precision=conditional_precision,
            temperature=temp,
            X_unified=state.X,
            y_vec=state.y,
        )
        view_mu = eqx.tree_at(lambda v: v.forest, view_mu, view_mu_forest)

        w_transfer = b_z * beta_new[state.exposure_idx]
        view_mu_forest, mu_fit, forest_nu, nu_fit, R = inter_ensemble_transfer_step(
            random.fold_in(key, 889),
            view_mu.forest,
            mu_fit,
            forest_nu,
            nu_fit,
            R,
            alpha_new,
            w_transfer,
            obs_mask,
            state.z_vec,
            state.time_idx,
            state.T_periods,
            sigma2,
            view_mu.forest.leaf_prior_cov_inv,
            state.leaf_prior_cov_inv_nu,
            conditional_precision=conditional_precision,
            temperature=temp,
            unit_idx=state.unit_idx,
            N_units=state.N_units,
            X_unified=state.X,
            y_vec=state.y,
        )
        view_mu = eqx.tree_at(lambda v: v.forest, view_mu, view_mu_forest)

    # ------------------------------------------------------------------
    # 8b. Smooth trend block, drawn jointly with gamma and beta
    # ------------------------------------------------------------------
    # The forests are held fixed here, and conditional on them the model is
    # linear-Gaussian in (c, gamma, beta), so this draw is from their exact
    # joint conditional: acceptance one, no tuning. Drawing the three together
    # rather than in sequence is what removes the ridge between the baseline
    # trend, the unit levels and the exposure profile -- a coordinate-wise
    # sweep through blocks this dependent crawls along it. See
    # ``longbet._trend_basis``.
    trend_coef_new = state.trend_coef
    trend_local2_new = state.trend_local2
    trend_local_aux_new = state.trend_local_aux
    trend_global2_new = state.trend_global2
    trend_global_aux_new = state.trend_global_aux
    if state.use_trend_block:
        temp_tb = (
            jnp.float32(1.0)
            if state.temperature is None
            else jnp.asarray(state.temperature, dtype=jnp.float32)
        )
        # Under the horseshoe the interaction columns' prior scale is the
        # current tau * lambda_j; under the ridge it is the fixed vector. Either
        # way the block below is the same exact Gaussian draw -- the horseshoe
        # is conjugate precisely so that it does not have to change.
        if state.use_trend_horseshoe:
            col_scale = horseshoe_col_scales(
                jnp.asarray(state.trend_prior_scale_vec, dtype=jnp.float32),
                state.trend_num_period,
                state.trend_local2,
                state.trend_global2,
            )
        else:
            col_scale = jnp.asarray(
                state.trend_prior_scale_vec, dtype=jnp.float32
            )
        trend_coef_new, gamma_new, beta_new, R = trend_block_step(
            random.fold_in(key, 9311),
            state.trend_coef,
            gamma_new,
            beta_new,
            R,
            state.trend_design,
            b_z * nu_fit,
            obs_mask,
            state.unit_idx,
            state.N_units,
            state.exposure_idx,
            sigma2,
            sigma_gamma2_new,
            state.K_chol,
            col_scale,
            state.random_intercept,
            state.sample_beta,
            conditional_precision=conditional_precision,
            temperature=temp_tb,
            trend_gram=state.trend_gram,
            trend_unit_sum=state.trend_unit_sum,
        )
        if state.use_trend_horseshoe:
            # Scales given the coefficients just drawn. Ordering matters only
            # in that both directions are valid Gibbs sweeps; drawing the
            # scales last means the state carries scales consistent with the
            # coefficients it also carries.
            (
                trend_local2_new,
                trend_local_aux_new,
                trend_global2_new,
                trend_global_aux_new,
            ) = sample_horseshoe(
                random.fold_in(key, 9313),
                trend_coef_new[state.trend_num_period:],
                state.trend_local2,
                state.trend_local_aux,
                state.trend_global2,
                state.trend_global_aux,
                state.trend_hs_global_scale,
            )
        if state.random_intercept:
            # gamma moved, so its variance is redrawn from the conditional that
            # matches the values now in the state.
            sigma_gamma2_new = _sample_inv_gamma(
                random.fold_in(key, 9312),
                jnp.float32(state.gamma_prior_a + state.N_units / 2.0),
                state.gamma_prior_b + 0.5 * jnp.sum(jnp.square(gamma_new)),
            )

    # ------------------------------------------------------------------
    # 9. Error variance sigma^2
    # ------------------------------------------------------------------
    if conditional_precision is not None:
        sigma2_new = sigma2
    elif is_binary or is_ordered:
        sigma2_new = jnp.float32(1.0)
    else:
        m_obs = jnp.sum(obs_mask_f32)
        sigma2_new = _sample_inv_gamma(
            keys[9],
            state.sigma_prior_a + m_obs / 2.0,
            state.sigma_prior_b + 0.5 * jnp.sum(jnp.square(jnp.where(obs_mask, R, 0.0))),
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
            s.cutpoints,
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
            s.trend_coef,
            s.trend_local2,
            s.trend_local_aux,
            s.trend_global2,
            s.trend_global_aux,
        ),
        state,
        (
            view_nu.X,
            view_nu.y,
            view_nu.config,
            current_z,
            cutpoints,
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
            trend_coef_new,
            trend_local2_new,
            trend_local_aux_new,
            trend_global2_new,
            trend_global_aux_new,
        ),
        is_leaf=lambda x: x is None,
    )


@jax.jit
@enable_x64()
def longbet_step(key: Key[Array, ''], state: LongBetState) -> LongBetState:
    """Execute one Gibbs sweep, across chains if the state carries a chain axis.

    Chains are mapped with ``jax.vmap`` over the per-chain partition of the
    state only; ``X``, ``y``, the panel indices and the GP kernel factor are
    closed over, so a ``k``-chain fit does not hold ``k`` copies of the design
    matrix.  ``bartz``'s own ``num_chains`` cannot be used here because its
    weighted-observation path does not accept a per-chain ``prec_scale``, and
    LongBet's treatment weights differ across chains by construction.
    """
    sweep = _tempered_single_step if state.is_tempered else longbet_single_step
    if not state.has_chain_axis:
        return sweep(key, state)

    spec = chain_filter_spec(state)
    per_chain, shared = eqx.partition(state, spec)
    keys = random.split(key, state.num_chains)

    def one(k: Key[Array, ''], p: Any) -> Any:
        out = sweep(k, eqx.combine(p, shared))
        # Return only the per-chain part, so vmap does not add a chain axis to
        # the shared arrays.
        return eqx.filter(out, spec)

    out_per = jax.vmap(one, in_axes=(0, 0))(keys, per_chain)
    return eqx.combine(out_per, shared)
