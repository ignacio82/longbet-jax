"""Common-population, natural-scale posterior encouragement contrasts.

This module reduces posterior draws inside the forest evaluation loop. It never
uses the treated-only ATT to represent the full study population. No sampler
conditionals are changed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pandas as pd
from scipy.special import ndtr

from longbet._encourage import _critical, _validate
from longbet._model import _eval_block, derive_exposure, resolve_device
from longbet._multi_model import LongBetMulti
from longbet._shared_forest import SHARED_SAMPLER_SEMANTICS
from longbet._summary import BlockAccumulator, PosteriorSummary, choose_block_size
from longbet._sur import SAMPLER_SEMANTICS


@dataclass(frozen=True)
class EncouragementDraws:
    """Aligned reduced-form draws and the population over which they average.

    ``draws[name]`` has axes ``(group, horizon, chain, retained_draw)``.
    ``cell_draws[name]``, when requested, has axes ``(unit, period, draw)``;
    its last axis is chain-major. Cell effects before encouragement are zero.
    ``weights[g]`` sum to one over the study units in group ``g``. All outcomes
    use these same weights, including units in the observed control arm.
    """

    draws: dict[str, np.ndarray]
    group_labels: tuple[str, ...]
    group_counts: np.ndarray
    weights: np.ndarray
    periods: np.ndarray
    period_indices: np.ndarray
    horizons: np.ndarray
    cell_draws: dict[str, np.ndarray] | None
    cell_summaries: dict[str, PosteriorSummary]
    standardization: str
    provenance: str

    @property
    def effects(self) -> np.ndarray:
        """Equivalent array with axes ``(chain, draw, group, horizon, outcome)``."""
        return np.stack(tuple(self.draws.values()), axis=-1).transpose(2, 3, 0, 1, 4)


def _target_weights(
    n_units: int, groups: Any, weights: Any,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Normalize fixed, nonnegative unit weights within each baseline group."""
    if weights is None:
        unit_weights = np.ones(n_units, dtype=np.float64)
    else:
        unit_weights = np.asarray(weights, dtype=np.float64)
        if (unit_weights.shape != (n_units,) or
                not np.all(np.isfinite(unit_weights)) or np.any(unit_weights < 0)):
            raise ValueError("weights must be a finite nonnegative vector of length N.")
        # Scale first: the sum of valid finite weights can otherwise overflow.
        if unit_weights.max(initial=0) > 0:
            unit_weights = unit_weights / unit_weights.max()
    if unit_weights.sum() <= 0:
        raise ValueError("weights must have a positive total.")

    masks = [np.ones(n_units, dtype=bool)]
    labels = ["all"]
    if groups is not None:
        raw_groups = np.asarray(groups)
        if raw_groups.shape != (n_units,) or np.any(pd.isna(raw_groups)):
            raise ValueError("groups must be a length-N vector of nonmissing baseline labels.")
        try:
            values = list(pd.unique(raw_groups))
        except TypeError as exc:
            raise ValueError("groups must contain scalar baseline labels.") from exc
        for value in values:
            label = str(value)
            if label in labels:
                raise ValueError("group labels must be unique as strings and cannot be 'all'.")
            labels.append(label)
            masks.append(raw_groups == value)

    counts = np.asarray([mask.sum() for mask in masks], dtype=np.int64)
    target = np.stack([unit_weights * mask for mask in masks])
    totals = target.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("every group must have positive total weight.")
    return tuple(labels), counts, target / totals[:, None]


def standardize_encouragement(
    model: LongBetMulti,
    x: Any,
    z: Any,
    t: Any = None,
    *,
    groups: Any = None,
    weights: Any = None,
    summary_only: bool = True,
    alpha: float = 0.05,
    block_size: int | None = None,
    standardization: str = "conditional",
    x_trt: Any = None,
    x_tv: Any = None,
    x_trt_tv: Any = None,
    ps: Any = None,
) -> EncouragementDraws:
    """Average encouragement contrasts over the same population for each draw.

    The prediction population must be the fitted study units, in fitted row
    order, with the original calendar vector. ``z`` describes their observed
    single-wave assignment. Both counterfactual schedules apply to *every*
    study unit: encouragement from the common start versus never encouraged.
    Return one aggregate per observed post-encouragement period, retaining gaps
    in the exposure clock. There is no forecast or treatment-history inversion.
    Supplied calendar labels are preserved, and must replay the fitted float32
    calendar and exposure clock. When omitted, whole-unit offsets are recovered
    from the stored calendar within float32 rounding tolerance.

    ``groups`` are fixed baseline labels; the overall population is returned as
    ``'all'`` followed by the groups in first-appearance order. ``weights`` are
    fixed nonnegative unit weights normalized within each group, and must not
    be chosen using post-encouragement uptake or outcomes. Adjusting covariates
    requires baseline or otherwise causally justified inputs.

    ``standardization='conditional'`` holds the fitted unit-intercept draw for
    each observed unit fixed under both assignments. For binary outcomes the
    contrast is ``Phi(mu0 + tau) - Phi(mu0)``. ``'population'`` instead integrates
    over a fresh normal unit intercept using each draw of its variance:
    ``Phi(eta1 / sqrt(1 + variance)) - Phi(eta0 / sqrt(1 + variance))``.
    This is a different, population-marginal model estimand, not the finite
    study-unit estimand of the design-based reference. Continuous contrasts are
    identical under these choices because the additive unit effect cancels.

    Under ``summary_only=True`` only blocks of cells times draws are evaluated,
    along with the small group/horizon draw arrays and per-cell summaries. No
    full unit/period/draw cube is allocated or cached. Set it to False to retain
    natural-scale cell draws for validation or custom analyses.
    """
    if not isinstance(model, LongBetMulti) or model.fits is None:
        raise RuntimeError("model must be a fitted LongBetMulti.")
    if model.sampler_semantics not in (SAMPLER_SEMANTICS, SHARED_SAMPLER_SEMANTICS):
        raise ValueError("Encouragement contrasts require current joint sampler semantics.")
    _critical(alpha)
    if standardization == "observed":
        standardization = "conditional"
    if standardization not in ("conditional", "population"):
        raise ValueError("standardization must be 'conditional' or 'population'.")
    if block_size is not None and (
        isinstance(block_size, (bool, np.bool_)) or
        not isinstance(block_size, (int, np.integer)) or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer.")
    fitted_t = np.asarray(model.fitted_t_, dtype=np.float32)
    if t is None:
        # LongBet persists the calendar in float32. A fractional origin then
        # gives apparent noninteger gaps (for example 1.1f - 0.1f), although
        # the original study used whole-unit offsets. Reconstruct only offsets
        # compatible with rounding each stored endpoint; retain the fitted
        # origin rather than inventing an integer origin or changing labels.
        fitted64 = fitted_t.astype(np.float64)
        offsets = fitted64 - fitted64[0]
        whole_offsets = np.rint(offsets)
        ulp = np.abs(np.spacing(fitted_t).astype(np.float64))
        rounding_tolerance = (ulp + ulp[0]) / 2
        rounding_tolerance += np.finfo(np.float64).eps * np.maximum(1, np.abs(offsets))
        if np.any(np.abs(offsets - whole_offsets) > rounding_tolerance):
            raise ValueError("the fitted calendar must have whole-unit gaps for encouragement standardization.")
        t = fitted64[0] + whole_offsets
        supplied_calendar = False
    else:
        supplied_calendar = True
    panel = _validate(z, z, t)
    n_units, n_periods = panel.z.shape
    if (n_units, n_periods) != (model.N_, model.T_):
        raise ValueError("standardization requires all fitted study units and periods in order.")
    if supplied_calendar and not np.array_equal(panel.t.astype(np.float32), fitted_t):
        raise ValueError("t must equal the fitted calendar vector for study-unit standardization.")
    if not np.array_equal(derive_exposure(panel.z, panel.t), derive_exposure(panel.z, fitted_t)):
        raise ValueError("t must preserve the fitted exposure clock for study-unit standardization.")
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2 or x_np.shape[0] != n_units or not np.all(np.isfinite(x_np)):
        raise ValueError("x must be a finite (N, P) matrix in fitted study-unit order.")
    labels, counts, target = _target_weights(n_units, groups, weights)

    # Include controls at the encouraged exposure in the counterfactual design.
    z_all = np.zeros_like(panel.z)
    z_all[:, panel.start:] = 1
    exposure = derive_exposure(z_all, panel.t).ravel().astype(np.int32)
    if exposure.max() > model.S_max_:
        raise ValueError("the counterfactual encouragement horizon exceeds the fitted horizon.")
    unit_idx = np.repeat(np.arange(n_units, dtype=np.int32), n_periods)
    time_idx = np.tile(np.arange(n_periods, dtype=np.int32), n_units)
    n_cells = n_units * n_periods
    n_horizons = n_periods - panel.start
    draws: dict[str, np.ndarray] = {}
    cell_draws: dict[str, np.ndarray] | None = None if summary_only else {}
    cell_summaries: dict[str, PosteriorSummary] = {}
    chain_shape = None

    for name, child in zip(model.outcome_names, model.fits):
        if child.trace is None or child.design_ is None:
            raise RuntimeError("every outcome must contain fitted posterior traces and design.")
        raw = child._raw_inputs(
            x=x_np, x_trt=x_trt, x_tv=x_tv, x_trt_tv=x_trt_tv, ps=ps,
            time_idx=time_idx, exposure_idx=exposure, N=n_units, T=n_periods,
        )
        raw_zero = dict(raw, s=np.zeros_like(raw["s"]))
        factual = np.asarray(child.design_.build(raw), dtype=np.uint8)
        control = np.asarray(child.design_.build(raw_zero), dtype=np.uint8)

        def flat(value: Any) -> np.ndarray:
            arr = np.asarray(value)
            return arr.reshape(-1, *arr.shape[2:]) if child._chained else arr

        beta = flat(child.trace.beta)
        n_draws = beta.shape[0]
        n_chains = child.config.num_chains
        shape = (n_chains, n_draws // n_chains)
        if n_draws % n_chains or (chain_shape is not None and shape != chain_shape):
            raise ValueError("joint outcomes must have aligned chain and retained-draw axes.")
        chain_shape = shape
        b0, b1 = flat(child.trace.b0)[:, None], flat(child.trace.b1)[:, None]
        alpha_draws = flat(child.trace.alpha)[:, None]
        gamma = flat(child.trace.gamma)
        if gamma.shape != (n_draws, n_units):
            raise ValueError("unit-intercept draws must match fitted study units in order.")
        denom: Any = 1.0
        if standardization == "population" and child.config.random_intercept:
            variance = flat(child.trace.sigma_gamma2)
            if np.any(variance < 0) or not np.all(np.isfinite(variance)):
                raise ValueError("unit-intercept variances must be finite and nonnegative.")
            denom = np.sqrt(1 + variance[:, None] * child.sdy**2)

        width = min(n_cells, block_size or choose_block_size(n_draws, n_cells))
        sums = np.zeros((len(labels), n_horizons, n_draws), dtype=np.float64)
        acc = BlockAccumulator(n_cells, alpha)
        full = None if summary_only else np.empty((n_draws, n_cells), dtype=np.float64)
        s_visible = any(b.name == "s" and b.nu_visible for b in child.design_.blocks)
        device = child.device_ if child.device_ is not None else resolve_device(child.config.device)
        with jax.default_device(device):
            for lo in range(0, n_cells, width):
                hi = min(lo + width, n_cells)
                sl = slice(lo, hi)
                nu1 = _eval_block(factual, lo, hi, width, child.trace.nu_trace)
                nu0 = (_eval_block(control, lo, hi, width, child.trace.nu_trace)
                       if s_visible else nu1)
                tau = (b1 * beta[:, exposure[sl]] * nu1 - b0 * beta[:, :1] * nu0) * child.sdy
                if child.config.outcome == "binary":
                    mu = _eval_block(factual, lo, hi, width, child.trace.mu_trace)
                    intercept: Any = gamma[:, unit_idx[sl]] if standardization == "conditional" else 0.0
                    eta0 = (alpha_draws * mu + b0 * beta[:, :1] * nu0 + intercept) * child.sdy + child.meany
                    natural = ndtr((eta0 + tau) / denom) - ndtr(eta0 / denom)
                else:
                    natural = tau.astype(np.float64)
                if not np.all(np.isfinite(natural)):
                    raise FloatingPointError(f"non-finite posterior encouragement contrasts for {name!r}.")
                post = time_idx[sl] >= panel.start
                natural[:, ~post] = 0.0
                for g, weights_g in enumerate(target):
                    weighted = natural[:, post].T * weights_g[unit_idx[sl][post], None]
                    np.add.at(sums[g], time_idx[sl][post] - panel.start, weighted)
                acc.update(lo, hi, natural)
                if full is not None:
                    full[:, sl] = natural
        draws[name] = sums.reshape(len(labels), n_horizons, *chain_shape)
        cell_summaries[name] = PosteriorSummary(
            *(value.reshape(n_units, n_periods) for value in acc.result())
        )
        if cell_draws is not None:
            cell_draws[name] = full.T.reshape(n_units, n_periods, n_draws)

    return EncouragementDraws(
        draws=draws, group_labels=labels, group_counts=counts, weights=target,
        periods=panel.t[panel.start:].copy(),
        period_indices=np.arange(panel.start, n_periods),
        horizons=panel.exposure[panel.start:].copy(),
        cell_draws=cell_draws, cell_summaries=cell_summaries,
        standardization=standardization, provenance=model.provenance,
    )
