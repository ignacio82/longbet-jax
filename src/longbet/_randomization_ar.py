"""Randomization AR tests for single-wave randomized encouragement experiments.

For each candidate effect beta, permutes assignment under the exact or Monte Carlo
randomization distribution while holding W = Y - beta * D fixed.
Under the sharp null W_i(1) = W_i(0), the test is finite-sample exact for complete
randomization. It does not invert across unmeasured grid values or extrapolate tails.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, islice
from math import comb
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

from longbet._encourage import _validate


@dataclass(frozen=True)
class RandomizationARResult:
    """Evaluated AR p-values, statistics, and randomization metadata across a grid.

    ``table`` has columns ``period``, ``horizon``, ``beta``, ``statistic``,
    ``p_value``, ``accepted``, ``exceedances``, ``mc_p_lower``, and ``mc_p_upper``.
    ``metadata`` records the randomization scheme, evaluated assignments, and assumptions.
    """

    table: pd.DataFrame
    metadata: dict[str, Any]


def _statistics(w: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Absolute studentized difference, HC0 denominator; 0/0=0, a/0=inf."""
    a = np.asarray(assignments, dtype=bool)
    n1 = a.sum(axis=1)[:, None]
    n0 = w.shape[0] - n1
    m1 = a @ w / n1
    m0 = (~a) @ w / n0
    v1 = np.sum(a[..., None] * (w[None] - m1[:, None])**2, axis=1) / n1**2
    v0 = np.sum((~a)[..., None] * (w[None] - m0[:, None])**2, axis=1) / n0**2
    numerator = np.abs(m1 - m0)
    denominator = np.sqrt(v1 + v0)
    out = np.divide(numerator, denominator, out=np.full_like(numerator, np.inf),
                    where=denominator > 0)
    out[(denominator == 0) & (numerator == 0)] = 0
    return out


def randomization_ar(
    y: Any,
    d: Any,
    z: Any,
    t: Any = None,
    *,
    beta: Any,
    alpha: float = 0.05,
    method: str = "auto",
    permutations: int = 9999,
    seed: int = 0,
    max_enumerations: int = 100000,
    design: str = "complete",
    block_size: int = 128,
) -> RandomizationARResult:
    """Evaluate pointwise AR p-values under the complete randomization distribution.

    ``beta`` is a grid of candidate effect values to test.
    ``method='exact'`` enumerates every permutation with the observed arm counts.
    ``'monte_carlo'`` samples assignments independently with replacement and uses
    ``(1 + exceedances) / (1 + permutations)``.
    ``'auto'`` chooses exact enumeration when total assignments <= ``max_enumerations``.

    ``mc_p_lower/upper`` are 95% Clopper--Pearson bounds on the ideal enumeration
    tail probability, not uncertainty in the causal effect parameter.
    """
    if design != "complete":
        raise ValueError("Only complete individual randomization is implemented here.")
    panel = _validate(z, d, t)
    outcome = np.asarray(y, dtype=float)
    if outcome.shape != panel.d.shape or not np.isfinite(outcome).all():
        raise ValueError("y must be a finite panel aligned with d and z.")
    grid = np.atleast_1d(np.asarray(beta, dtype=float))
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("beta must be a nonempty finite one-dimensional grid.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    if method not in ("auto", "exact", "monte_carlo"):
        raise ValueError("method must be auto, exact, or monte_carlo.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer so grid values reuse assignments.")
    for name, value in (("permutations", permutations), ("max_enumerations", max_enumerations),
                        ("block_size", block_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    n, n1 = len(outcome), int(panel.assigned.sum())
    if min(n1, n - n1) < 2:
        raise ValueError("Studentization requires at least two units in each arm.")
    total = comb(n, n1)
    chosen = ("exact" if total <= max_enumerations else "monte_carlo") if method == "auto" else method
    if chosen == "exact" and total > max_enumerations:
        raise ValueError("Assignment support exceeds max_enumerations; choose Monte Carlo explicitly.")
    count = total if chosen == "exact" else permutations
    rows = []
    for candidate in grid:
        with np.errstate(over="ignore", invalid="ignore"):
            w = outcome[:, panel.start:] - candidate * panel.d[:, panel.start:]
        if not np.isfinite(w).all():
            raise ValueError("The transformed outcome overflows at a supplied beta.")
        scale = np.max(np.abs(w), axis=0)
        w = w / np.where(scale > 0, scale, 1.)
        w -= w.mean(axis=0)
        observed = _statistics(w, panel.assigned[None])[0]
        exceedances = np.zeros(w.shape[1], dtype=int)
        rng = np.random.default_rng(seed)
        support = combinations(range(n), n1) if chosen == "exact" else None
        for offset in range(0, count, block_size):
            size = min(block_size, count - offset)
            assignments = np.zeros((size, n), dtype=bool)
            selected = (list(islice(support, size)) if support is not None
                        else [rng.choice(n, n1, replace=False) for _ in range(size)])
            for j, indices in enumerate(selected):
                assignments[j, list(indices)] = True
            statistic = _statistics(w, assignments)
            tolerance = np.where(np.isfinite(observed), 1e-12 * np.maximum(1, observed), 0.)
            exceedances += np.sum(statistic >= observed - tolerance, axis=0)
        if chosen == "exact":
            pvalues = exceedances / count
            lower = upper = pvalues
        else:
            pvalues = (1 + exceedances) / (1 + count)
            lower = np.where(exceedances == 0, 0.,
                             beta_distribution.ppf(.025, exceedances, count - exceedances + 1))
            upper = np.where(exceedances == count, 1.,
                             beta_distribution.ppf(.975, exceedances + 1, count - exceedances))
        for h in range(w.shape[1]):
            rows.append(dict(
                period=panel.t[panel.start + h],
                horizon=panel.exposure[panel.start + h],
                beta=candidate,
                statistic=observed[h],
                p_value=pvalues[h],
                accepted=bool(pvalues[h] > alpha),
                exceedances=int(exceedances[h]),
                mc_p_lower=lower[h],
                mc_p_upper=upper[h],
            ))
    return RandomizationARResult(pd.DataFrame(rows), {
        **panel.metadata(),
        "randomization_method": chosen,
        "assignment_support": float(total) if total > 2**53 else int(total),
        "evaluated_assignments": int(count),
        "seed": seed if chosen == "monte_carlo" else None,
        "alpha": alpha,
        "null": "sharp horizon-specific Y(1)-beta*D(1)=Y(0)-beta*D(0)",
        "finite_sample_valid_under_sharp_null": True,
        "finite_sample_exact_for_heterogeneous_late": False,
        "simultaneous": False,
        "inversion": "evaluated grid only",
        "between_grid_points": "unknown",
        "left_tail": "unknown",
        "right_tail": "unknown",
        "monte_carlo_formula": "(1+exceedances)/(1+permutations)" if chosen == "monte_carlo" else None,
        "mc_interval_target": "ideal enumeration tail probability",
        "mc_interval_level": 0.95,
    })
