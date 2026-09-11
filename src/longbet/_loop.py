"""MCMC driver for LongBet: a ``lax.while_loop`` over preallocated traces."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, Bool, Float32, Int32, Key

from bartz.mcmcloop._loop import _empty_trace, _set
from bartz.mcmcloop._trace import BurninTrace, MainTrace
from bartz.mcmcstep import State

from longbet._state import LongBetState
from longbet._step import _load, longbet_step


class LongBetTrace(eqx.Module):
    """Posterior draws saved during the main sampling phase.

    Parameter arrays are shaped ``(*chains, n_save, ...)``; the forest traces
    follow ``bartz``'s own layout, which is the same convention.
    """

    mu_trace: MainTrace
    nu_trace: MainTrace
    beta: Float32[Array, '...']
    gamma: Float32[Array, '...']
    b0: Float32[Array, '...']
    b1: Float32[Array, '...']
    alpha: Float32[Array, '...']
    sigma2: Float32[Array, '...']
    sigma_gamma2: Float32[Array, '...']
    cutpoints: Float32[Array, '...'] | None = None


class LongBetBurninTrace(eqx.Module):
    """Diagnostic traces saved during burn-in."""

    mu_trace: BurninTrace
    nu_trace: BurninTrace


class RunLongBetResult(NamedTuple):
    """Return value of :func:`run_longbet_mcmc`."""

    final_state: LongBetState
    main_trace: LongBetTrace
    burnin_trace: LongBetBurninTrace | None


class _MCMCCarry(eqx.Module):
    """Carry for the ``lax.while_loop``."""

    state: LongBetState
    key: Key[Array, '']
    i_total: Int32[Array, '']
    mu_burnin: BurninTrace
    nu_burnin: BurninTrace
    mu_main: MainTrace
    nu_main: MainTrace
    beta_trace: Float32[Array, '...']
    gamma_trace: Float32[Array, '...']
    b0_trace: Float32[Array, '...']
    b1_trace: Float32[Array, '...']
    alpha_trace: Float32[Array, '...']
    sigma2_trace: Float32[Array, '...']
    sigma_gamma2_trace: Float32[Array, '...']
    cutpoints_trace: Float32[Array, '...'] | None


def _make_views(state: LongBetState) -> tuple[State, State]:
    """Build plain ``State`` views of the two forests, for the trace machinery.

    Only the forest, the error precision and the move counters are read by
    ``MainTrace.from_state`` / ``BurninTrace.from_state``, but the residuals are
    converted into each forest's own units anyway so the views are internally
    consistent if upstream ever starts reading them.
    """
    view_mu = State(
        _chain_anchor=state._chain_anchor,
        X=state.X,
        y=state.y,
        z=state.z,
        binary_indices=state.binary_indices,
        resid=_load(state.resid, state.resid_unit),
        resid_unit=state.resid_unit,
        resid_eff_scale=state.resid_eff_scale,
        resid_inexact_integral=state.resid_inexact_integral,
        error_cov_inv=state.error_cov_inv,
        error_scale=state.error_scale,
        prec_scale=state.prec_scale,
        inv_sdev_scale=state.inv_sdev_scale,
        inv_sdev_unit=state.inv_sdev_unit,
        n_non_missing=state.n_non_missing,
        sum_diag_prec_scale=state.sum_diag_prec_scale,
        forest=state.forest,
        config=state.config,
    )
    view_nu = State(
        _chain_anchor=state._chain_anchor,
        X=state.X,
        y=state.y,
        z=None,
        binary_indices=None,
        resid=state.resid_nu,
        resid_unit=state.resid_unit_nu,
        resid_eff_scale=state.resid_eff_scale_nu,
        resid_inexact_integral=state.resid_inexact_integral_nu,
        error_cov_inv=state.error_cov_inv,
        error_scale=None,
        prec_scale=state.prec_scale_nu,
        inv_sdev_scale=state.inv_sdev_scale_nu,
        inv_sdev_unit=state.inv_sdev_unit_nu,
        n_non_missing=state.n_non_missing,
        sum_diag_prec_scale=state.sum_diag_prec_scale,
        forest=state.forest_nu,
        config=state.config,
    )
    return view_mu, view_nu


def _set_param(
    trace: Float32[Array, '...'],
    index: Int32[Array, ''],
    val: Float32[Array, '...'],
    sample_axis: int,
) -> Float32[Array, '...']:
    """Write ``val`` at ``index`` along ``sample_axis``; out-of-range is dropped."""
    ndindex = (slice(None),) * sample_axis + (index, ...)
    return trace.at[ndindex].set(val, mode='drop')


def run_longbet_mcmc(
    key: Key[Array, ''],
    state: LongBetState,
    n_burn: int,
    n_save: int,
    n_skip: int = 1,
    inner_loop_length: int | None = None,
    callback: Callable[[int, int, LongBetState], None] | None = None,
) -> RunLongBetResult:
    """Run the LongBet sampler.

    The thinning convention matches ``bartz`` exactly: the total number of
    iterations is ``n_burn + n_skip * n_save``, and ``n_skip = 1`` means no
    thinning.

    Parameters
    ----------
    key
        PRNG key.
    state
        Initial state, possibly carrying a chain axis.
    n_burn
        Burn-in sweeps, discarded.
    n_save
        Posterior draws to retain.
    n_skip
        Thinning interval.
    inner_loop_length
        Sweeps per outer dispatch. Bounds compile time and lets ``callback``
        run between batches. The loop bound is passed as a traced value, so
        varying it does **not** trigger recompilation.
    callback
        Optional host callable ``(outer_step, n_outer, state)`` run between
        batches.
    """
    n_burn_i, n_save_i, n_skip_i = int(n_burn), int(n_save), int(n_skip)
    n_iters = n_burn_i + n_skip_i * n_save_i

    chains = state.beta.shape[:-1]
    sample_axis = 1 if chains else 0

    view_mu_init, view_nu_init = _make_views(state)

    carry = _MCMCCarry(
        state=state,
        key=key,
        i_total=jnp.int32(0),
        mu_burnin=_empty_trace(n_burn_i, view_mu_init, BurninTrace),
        nu_burnin=_empty_trace(n_burn_i, view_nu_init, BurninTrace),
        mu_main=_empty_trace(n_save_i, view_mu_init, MainTrace),
        nu_main=_empty_trace(n_save_i, view_nu_init, MainTrace),
        beta_trace=jnp.zeros((*chains, n_save_i, state.S_max + 1), jnp.float32),
        gamma_trace=jnp.zeros((*chains, n_save_i, state.N_units), jnp.float32),
        b0_trace=jnp.zeros((*chains, n_save_i), jnp.float32),
        b1_trace=jnp.zeros((*chains, n_save_i), jnp.float32),
        alpha_trace=jnp.zeros((*chains, n_save_i), jnp.float32),
        sigma2_trace=jnp.zeros((*chains, n_save_i), jnp.float32),
        sigma_gamma2_trace=jnp.zeros((*chains, n_save_i), jnp.float32),
        cutpoints_trace=(jnp.zeros((*chains, n_save_i, state.num_categories - 2), jnp.float32)
                         if state.outcome_type_str == "ordinal" else None),
    )

    inner_length = n_iters if inner_loop_length is None else max(1, int(inner_loop_length))
    n_outer = max(1, -(-n_iters // inner_length))
    noop_idx = jnp.iinfo(jnp.int32).max

    def run_inner_batch(c: _MCMCCarry, i_target: Int32[Array, '']) -> _MCMCCarry:
        def cond_fn(carry_in: _MCMCCarry) -> Bool[Array, '']:
            return carry_in.i_total < i_target

        def body_fn(carry_in: _MCMCCarry) -> _MCMCCarry:
            key_step, key_next = random.split(carry_in.key)
            new_state = longbet_step(key_step, carry_in.state)
            i = carry_in.i_total

            is_burn = i < n_burn_i
            burnin_idx = jnp.where(is_burn, i, noop_idx)

            i_from_burn = i - n_burn_i
            is_save = (~is_burn) & (((i_from_burn + 1) % n_skip_i) == 0)
            main_idx = jnp.where(is_save, i_from_burn // n_skip_i, noop_idx)

            v_mu, v_nu = _make_views(new_state)

            return _MCMCCarry(
                state=new_state,
                key=key_next,
                i_total=i + 1,
                mu_burnin=_set(carry_in.mu_burnin, burnin_idx, BurninTrace.from_state(v_mu)),
                nu_burnin=_set(carry_in.nu_burnin, burnin_idx, BurninTrace.from_state(v_nu)),
                mu_main=_set(carry_in.mu_main, main_idx, MainTrace.from_state(v_mu)),
                nu_main=_set(carry_in.nu_main, main_idx, MainTrace.from_state(v_nu)),
                beta_trace=_set_param(carry_in.beta_trace, main_idx, new_state.beta, sample_axis),
                gamma_trace=_set_param(carry_in.gamma_trace, main_idx, new_state.gamma, sample_axis),
                b0_trace=_set_param(carry_in.b0_trace, main_idx, new_state.b0, sample_axis),
                b1_trace=_set_param(carry_in.b1_trace, main_idx, new_state.b1, sample_axis),
                alpha_trace=_set_param(carry_in.alpha_trace, main_idx, new_state.alpha, sample_axis),
                sigma2_trace=_set_param(carry_in.sigma2_trace, main_idx, new_state.sigma2, sample_axis),
                sigma_gamma2_trace=_set_param(
                    carry_in.sigma_gamma2_trace, main_idx, new_state.sigma_gamma2, sample_axis
                ),
                cutpoints_trace=(_set_param(carry_in.cutpoints_trace, main_idx,
                                           new_state.cutpoints, sample_axis)
                                 if carry_in.cutpoints_trace is not None else None),
            )

        return lax.while_loop(cond_fn, body_fn, c)

    # i_target is traced, not static: a different batch bound must not recompile.
    jitted_inner = jax.jit(run_inner_batch)

    for outer_step in range(n_outer):
        i_target = min((outer_step + 1) * inner_length, n_iters)
        carry = jitted_inner(carry, jnp.int32(i_target))
        if callback is not None:
            callback(outer_step, n_outer, carry.state)

    main_trace = LongBetTrace(
        mu_trace=carry.mu_main,
        nu_trace=carry.nu_main,
        beta=carry.beta_trace,
        gamma=carry.gamma_trace,
        b0=carry.b0_trace,
        b1=carry.b1_trace,
        alpha=carry.alpha_trace,
        sigma2=carry.sigma2_trace,
        sigma_gamma2=carry.sigma_gamma2_trace,
        cutpoints=carry.cutpoints_trace,
    )
    burnin_trace = (
        LongBetBurninTrace(mu_trace=carry.mu_burnin, nu_trace=carry.nu_burnin)
        if n_burn_i > 0
        else None
    )
    return RunLongBetResult(carry.state, main_trace, burnin_trace)
