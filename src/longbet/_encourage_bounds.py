"""Partial identification bounds for binary encouragement designs.

Implements sharp bounds on principal-stratum treatment effects under no-defiers
(monotonicity) and sensitivity limits on direct encouragement risk differences
(Manski / Balke--Pearl / StochTree IV bounds).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IdentificationBoundsResult:
    """Sharp partial identification bounds on principal-stratum causal effects.

    ``table`` contains summary statistics and posterior quantiles for each estimand:
    ``complier_encouragement``, ``treatment_z0``, and ``treatment_z1``.
    ``bounds_draws`` contains raw posterior draws of lower and upper bound endpoints.
    ``metadata`` records assumptions, sensitivity parameters, and compatibility flags.
    """

    table: pd.DataFrame
    bounds_draws: dict[str, np.ndarray]
    metadata: dict[str, Any]


def validate_binary_inputs(
    y: Any, d: Any, z: Any, x: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate binary single-horizon vectors and partition into baseline covariate cells."""
    y = np.asarray(y, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if not (y.shape == d.shape == z.shape) or len(y) < 4:
        raise ValueError("y, d, and z must be aligned 1D binary vectors with length >= 4.")
    if not all(np.isin(v, [0, 1]).all() for v in (y, d, z)):
        raise ValueError("y, d, and z must contain finite binary values (0 or 1).")
    if not 0 < z.sum() < len(z):
        raise ValueError("Both randomized arms (Z=0 and Z=1) must be present.")
    if x is None:
        x_mat = np.ones((len(y), 1), dtype=float)
    else:
        x_mat = np.asarray(x, dtype=float)
        if x_mat.ndim == 1:
            x_mat = x_mat[:, None]
        if len(x_mat) != len(y) or not np.isfinite(x_mat).all():
            raise ValueError("x must be a finite baseline covariate array aligned with y.")
    grid, inverse, counts = np.unique(x_mat, axis=0, return_inverse=True, return_counts=True)
    weights = counts / len(y)
    return y, d, z, x_mat, grid, inverse, weights


def _counts(
    y: np.ndarray, d: np.ndarray, z: np.ndarray, inverse: np.ndarray, n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    uptake = np.zeros((n_groups, 2, 2))  # group, Z, D
    outcomes = np.zeros((n_groups, 2, 2, 2))  # group, D, Z, Y
    for g, yy, dd, zz in zip(inverse, y.astype(int), d.astype(int), z.astype(int)):
        uptake[g, zz, dd] += 1
        outcomes[g, dd, zz, yy] += 1
    return uptake, outcomes


def ordered_beta(
    rng: np.random.Generator,
    a: np.ndarray,
    b: np.ndarray,
    size: int,
    *,
    max_candidates: int = 2000000,
) -> tuple[np.ndarray, int]:
    """Exact rejection sampling for (p0, p1) conditional on p1 >= p0."""
    kept, total, attempted = [], 0, 0
    while total < size and attempted < max_candidates:
        batch = min(max(2 * (size - total), 256), 65536, max_candidates - attempted)
        proposed = rng.beta(a, b, size=(batch, 2))
        accepted = proposed[proposed[:, 1] >= proposed[:, 0]]
        kept.append(accepted)
        total += len(accepted)
        attempted += batch
    if total < size:
        raise RuntimeError(
            f"Ordered Beta rejection budget exhausted: {total}/{size} accepted in {attempted} candidates."
        )
    return np.concatenate(kept, axis=0)[:size], attempted


def fit_binary_cells(
    y: Any,
    d: Any,
    z: Any,
    x: Any = None,
    *,
    seed: int = 0,
    chains: int = 4,
    draws: int = 1000,
    first_stage: str = "monotone",
    prior: float = 1.0,
    max_candidates: int = 2000000,
) -> dict[str, Any]:
    """Draw exact Dirichlet/Beta posterior samples for discrete cells."""
    y_arr, d_arr, z_arr, x_mat, grid, inverse, weights = validate_binary_inputs(y, d, z, x)
    if first_stage not in ("unrestricted", "monotone", "null"):
        raise ValueError("first_stage must be 'unrestricted', 'monotone', or 'null'.")
    if not np.isfinite(prior) or prior <= 0:
        raise ValueError("prior must be positive.")
    for val in (chains, draws, max_candidates):
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ValueError("chains, draws, and max_candidates must be positive integers.")
    uptake, outcomes = _counts(y_arr, d_arr, z_arr, inverse, len(grid))
    rng = np.random.default_rng(seed)
    p = np.empty((chains, draws, len(grid), 2))
    for g, count in enumerate(uptake):
        if first_stage == "monotone":
            val, _ = ordered_beta(
                rng, count[:, 1] + prior, count[:, 0] + prior,
                chains * draws, max_candidates=max_candidates,
            )
            p[..., g, :] = val.reshape(chains, draws, 2)
        elif first_stage == "null":
            val = rng.beta(count[:, 1].sum() + prior, count[:, 0].sum() + prior, size=(chains, draws))
            p[..., g, :] = val[..., None]
        else:
            p[..., g, :] = rng.beta(count[:, 1] + prior, count[:, 0] + prior, size=(chains, draws, 2))
    q = rng.beta(
        outcomes[..., 1] + prior, outcomes[..., 0] + prior,
        size=(chains, draws, len(grid), 2, 2),
    )
    r = p * q[..., 1, :] + (1 - p) * q[..., 0, :]
    return dict(
        p=p, q=q, r=r, weights=weights, grid=grid,
        uptake_counts=uptake, outcome_counts=outcomes,
        first_stage=first_stage, prior=prior, seed=seed,
    )


def identification_bounds(
    p: Any,
    q: Any,
    weights: Any,
    *,
    delta: float | tuple[float, float, float] | np.ndarray = 1.0,
) -> dict[str, Any]:
    """Calculate sharp bounds under no-defiers and direct encouragement risk limits.

    ``delta`` is a scalar or 3-tuple ``(always, never, complier)``, each in [0, 1].
    Returns the joint encouragement/adoption complier contrast and treatment effects at
    fixed Z=0 and Z=1 separately, weighted across baseline covariate cells.
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    if p_arr.shape[-1:] != (2,) or q_arr.shape != (*p_arr.shape[:-1], 2, 2):
        raise ValueError("Require p(..., G, Z) and q(..., G, D, Z).")
    if (not np.isfinite(p_arr).all() or not np.isfinite(q_arr).all()
            or np.any((p_arr < 0) | (p_arr > 1)) or np.any((q_arr < 0) | (q_arr > 1))):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if w_arr.shape != (p_arr.shape[-2],) or np.any(w_arr < 0) or not np.isfinite(w_arr).all() or not np.isclose(w_arr.sum(), 1):
        raise ValueError("weights must be nonnegative and sum to 1.")
    if np.any(p_arr[..., 1] < p_arr[..., 0]):
        raise ValueError("Principal-stratum bounds require the no-defiers restriction (p_1 >= p_0).")
    delta_arr = np.broadcast_to(np.asarray(delta, dtype=float), (3,))
    if not np.isfinite(delta_arr).all() or np.any((delta_arr < 0) | (delta_arr > 1)):
        raise ValueError("delta bounds must lie in [0, 1].")
    da, dn, dc = delta_arr
    pa, pc, pn = p_arr[..., 0], p_arr[..., 1] - p_arr[..., 0], 1 - p_arr[..., 1]
    t11 = p_arr[..., 1] * q_arr[..., 1, 1]
    t00 = (1 - p_arr[..., 0]) * q_arr[..., 0, 0]
    la, ua = np.maximum(0, q_arr[..., 1, 0] - da), np.minimum(1, q_arr[..., 1, 0] + da)
    ln, un = np.maximum(0, q_arr[..., 0, 1] - dn), np.minimum(1, q_arr[..., 0, 1] + dn)
    l11, u11 = np.maximum(0, t11 - pa * ua), np.minimum(pc, t11 - pa * la)
    l00, u00 = np.maximum(0, t00 - pn * un), np.minimum(pc, t00 - pn * ln)
    compatible = np.all(((l11 <= u11 + 1e-14) & (l00 <= u00 + 1e-14)) | (w_arr == 0), axis=-1)
    share = np.sum(w_arr * pc, axis=-1)
    defined = share > 0
    available = compatible & defined
    l10, u10 = np.maximum(0, l11 - pc * dc), np.minimum(pc, u11 + pc * dc)
    l01, u01 = np.maximum(0, l00 - pc * dc), np.minimum(pc, u00 + pc * dc)
    result = dict(
        complier_share=share,
        compatible=compatible,
        defined=defined,
        available=available,
    )
    for name, low, high in (
        ("complier_encouragement", l11 - u00, u11 - l00),
        ("treatment_z0", l10 - u00, u10 - l00),
        ("treatment_z1", l11 - u01, u11 - l01),
    ):
        result[name] = np.stack([
            np.divide(np.sum(w_arr * v, axis=-1), share,
                      out=np.full_like(share, np.nan), where=available)
            for v in (low, high)
        ], axis=-1)
    return result


def encouragement_bounds(
    y: Any,
    d: Any,
    z: Any,
    x: Any = None,
    *,
    delta: float | tuple[float, float, float] | np.ndarray = 1.0,
    seed: int = 0,
    chains: int = 4,
    draws: int = 1000,
    prior: float = 1.0,
    alpha: float = 0.05,
) -> IdentificationBoundsResult:
    """Estimate principal-stratum bounds on causal effects with sensitivity limits.

    Estimates sharp Manski/Balke--Pearl bounds on three estimands:
    1. ``complier_encouragement``: Effect of encouragement and adoption among compliers,
       $Y(1, 1) - Y(0, 0)$. When delta=0, this collapses to the Wald ratio.
    2. ``treatment_z0``: Effect of adoption holding encouragement fixed at 0, $Y(1, 0) - Y(0, 0)$.
    3. ``treatment_z1``: Effect of adoption holding encouragement fixed at 1, $Y(1, 1) - Y(0, 1)$.

    ``delta`` specifies maximum direct effect on outcome probabilities:
    ``delta=1`` leaves direct effects unrestricted.
    ``delta=0`` imposes strict exclusion.
    """
    fit = fit_binary_cells(
        y, d, z, x, seed=seed, chains=chains, draws=draws,
        first_stage="monotone", prior=prior,
    )
    bounds = identification_bounds(fit["p"], fit["q"], fit["weights"], delta=delta)
    rows = []
    quantiles = [alpha / 2, 0.5, 1 - alpha / 2]
    for estimand in ("complier_encouragement", "treatment_z0", "treatment_z1"):
        end_pts = bounds[estimand]  # (chains, draws, 2)
        lower_draws = end_pts[..., 0].ravel()
        upper_draws = end_pts[..., 1].ravel()
        avail = bounds["available"].ravel()
        n_avail = int(np.sum(avail))
        if n_avail > 0:
            lo_q = np.nanquantile(lower_draws[avail], quantiles)
            hi_q = np.nanquantile(upper_draws[avail], quantiles)
            rows.append(dict(
                estimand=estimand,
                lower_median=float(lo_q[1]),
                upper_median=float(hi_q[1]),
                lower_ci_lower=float(lo_q[0]),
                lower_ci_upper=float(lo_q[2]),
                upper_ci_lower=float(hi_q[0]),
                upper_ci_upper=float(hi_q[2]),
                available_fraction=float(n_avail / len(avail)),
                compatible_fraction=float(np.mean(bounds["compatible"])),
            ))
        else:
            rows.append(dict(
                estimand=estimand,
                lower_median=np.nan,
                upper_median=np.nan,
                lower_ci_lower=np.nan,
                lower_ci_upper=np.nan,
                upper_ci_lower=np.nan,
                upper_ci_upper=np.nan,
                available_fraction=0.0,
                compatible_fraction=float(np.mean(bounds["compatible"])),
            ))
    table = pd.DataFrame(rows)
    return IdentificationBoundsResult(
        table=table,
        bounds_draws=bounds,
        metadata=dict(
            delta=delta,
            prior=prior,
            chains=chains,
            draws=draws,
            seed=seed,
            assumption="monotonicity (no defiers) + bounded direct effect delta",
            exclusion_imposed=bool(np.all(np.asarray(delta) == 0)),
        ),
    )
