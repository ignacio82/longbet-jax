"""Fixed-parameter repeated-sampling checks for longitudinal IV.

The filename is retained for compatibility; this is NOT simulation-based
calibration (parameters are not drawn from the fitted models' joint prior).
Continuous sharp-null beta and a prespecified binary complier risk difference
are different targets and are reported separately. AR summaries describe only
evaluated grid points, with no interpolation or inference about the tails.
Experimental hazard working intervals are optional and never causal intervals.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from longbet._coupled_hazard_iv import CoupledHazardConfig, CoupledHazardIV
from longbet._encourage import encouragement_effects

try:
    from .encouragement_bounds import encouragement_bounds
    from .randomization_ar import randomization_ar
except (ImportError, ValueError):
    from encouragement_bounds import encouragement_bounds
    from randomization_ar import randomization_ar


@dataclass(frozen=True)
class CoveragePanel:
    y: np.ndarray
    d: np.ndarray
    z: np.ndarray
    untreated_y: np.ndarray
    d_z0: np.ndarray
    d_z1: np.ndarray
    true_beta: float
    binary_threshold: float
    binary_complier_target: float | None


@dataclass(frozen=True)
class ScenarioCoverageSummary:
    scenario: str
    instrument_effect: float
    mean_first_stage_f: float
    true_beta: float
    ar_coverage: float
    ar_coverage_mcse: float
    ar_mean_accepted_grid_points: float
    ar_mean_grid_components: float
    ar_grid_boundary_acceptance_rate: float
    ar_empty_grid_rate: float
    ar_tail_status: str
    fieller_coverage: float
    fieller_coverage_mcse: float
    fieller_unbounded_rate: float
    binary_threshold: float
    mean_binary_complier_target: float | None
    binary_target_defined_replications: int
    binary_median_bounds_containment: float | None
    binary_posterior_outer_interval_coverage: float | None
    binary_mean_compatible_fraction: float
    coupled_working_interval_coverage: float | None
    coupled_causal_inference_supported: bool
    coupled_mean_null_relevance_fraction: float | None
    failures: dict[str, int]
    replications: int


def generate_coverage_panel(
    n: int = 100,
    t_len: int = 5,
    instrument_effect: float = 0.8,
    true_beta: float = 1.0,
    rho: float = 0.3,
    sigma_y: float = 1.0,
    seed: int = 42,
    binary_threshold: float = 0.0,
) -> CoveragePanel:
    """Generate fixed potential panels, then completely randomize half the units.

    Shared latent adoption errors define monotone potential adoption paths under
    both assignments. ``Y_it(d) = beta*d + eps_y,it`` obeys a sharp constant effect
    null for the continuous outcome. The binary outcome uses a threshold fixed
    before observing data; its true horizon-one complier contrast is calculated
    directly from both potential outcomes, and is undefined with no compliers.
    Potentials are regenerated across replications (superpopulation coverage).
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 4:
        raise ValueError("n must be an integer >= 4.")
    if isinstance(t_len, bool) or not isinstance(t_len, int) or t_len < 2:
        raise ValueError("t_len must be an integer >= 2.")
    if (not np.isfinite([instrument_effect, true_beta, rho, sigma_y, binary_threshold]).all()
            or instrument_effect < 0 or abs(rho) >= 1 or sigma_y <= 0):
        raise ValueError("Require finite parameters, nonnegative instrument_effect, abs(rho)<1 and sigma_y>0.")
    rng = np.random.default_rng(seed)
    covariance = [[sigma_y**2, rho * sigma_y], [rho * sigma_y, 1.0]]
    errors = rng.multivariate_normal([0, 0], covariance, size=(n, t_len))
    eps_y, eps_d = errors[:, :, 0], errors[:, :, 1]
    potential_d = []
    for assignment in (0, 1):
        path = np.zeros((n, t_len))
        for s in range(1, t_len):
            path[:, s] = np.maximum(path[:, s - 1],
                                    (-0.6 + instrument_effect * assignment + eps_d[:, s]) > 0)
        potential_d.append(path)
    d0, d1 = potential_d
    assigned = np.zeros(n, dtype=bool)
    assigned[rng.choice(n, n // 2, replace=False)] = True
    z = np.zeros((n, t_len))
    z[assigned, 1:] = 1
    d = np.where(assigned[:, None], d1, d0)
    y = true_beta * d + eps_y
    compliers = d1[:, 1] > d0[:, 1]
    contrast = ((eps_y[:, 1] + true_beta > binary_threshold).astype(float)
                - (eps_y[:, 1] > binary_threshold).astype(float))
    target = float(contrast[compliers].mean()) if compliers.any() else None
    return CoveragePanel(y, d, z, eps_y, d0, d1, float(true_beta), float(binary_threshold), target)


def interval_contains(lower: float, upper: float, target: float) -> bool:
    """Coverage of the actual reported interval, without padding or tolerance."""
    return bool(not np.isnan(lower) and not np.isnan(upper) and lower <= target <= upper)


def summarize_ar_grid(table: pd.DataFrame, target: float) -> dict[str, Any]:
    """Keep all contiguous runs of accepted *grid points* and unknown tails.

    Runs do not assert acceptance between their points. No confidence-set width
    can be inferred from a finite grid, including when its endpoints are rejected.
    The truth must have been evaluated exactly, rather than at a nearby surrogate.
    """
    rows = table.sort_values("beta")
    grid = rows["beta"].to_numpy()
    accepted = rows["accepted"].to_numpy(dtype=bool)
    indices = np.flatnonzero(grid == target)
    if not len(grid) or len(indices) != 1 or len(np.unique(grid)) != len(grid):
        raise ValueError("Require a nonempty unique grid that evaluates the exact target once.")
    starts = np.flatnonzero(accepted & ~np.r_[False, accepted[:-1]])
    stops = np.flatnonzero(accepted & ~np.r_[accepted[1:], False])
    return {
        "covers_target": bool(accepted[indices[0]]),
        "accepted_grid_points": int(accepted.sum()),
        "grid_runs": [(float(grid[a]), float(grid[b])) for a, b in zip(starts, stops)],
        "left_boundary_accepted": bool(accepted[0]),
        "right_boundary_accepted": bool(accepted[-1]),
        "left_tail": "unknown", "right_tail": "unknown",
        "between_grid_points": "unknown",
    }


def _mcse(rate: float, replications: int) -> float:
    return float(np.sqrt(rate * (1 - rate) / replications))


def evaluate_coverage(
    replications: int = 20,
    n: int = 80,
    t_len: int = 4,
    seed: int = 100,
    *,
    permutations: int = 999,
    bounds_draws: int = 1000,
    include_experimental: bool = False,
    coupled_sweeps: int = 400,
    coupled_burnin: int = 200,
) -> list[ScenarioCoverageSummary]:
    """Run repeated-sampling checks, retaining method failures in denominators.

    Strength labels refer to assigned hazard coefficients, not promised F ranges.
    Binary endpoint priors are Beta(1,1) with monotone uptake. Median bounds are
    identification-region estimates; the outer posterior endpoint interval is
    evaluated separately without claiming nominal frequentist calibration.
    Experimental working-model interval containment is an optional diagnostic,
    not evidence of validated causal coverage. Its priors are fixed at defaults,
    and this experiment does not establish MCMC convergence.
    """
    if isinstance(replications, bool) or not isinstance(replications, int) or replications < 1:
        raise ValueError("replications must be a positive integer.")
    scenarios = [("null", 0.0), ("small", 0.1), ("medium", 0.5), ("large", 1.2)]
    summaries = []
    true_beta, threshold = 1.0, 0.0
    beta_grid = np.unique(np.r_[np.linspace(-3, 5, 41), true_beta])
    expected_failures = (ValueError, RuntimeError, np.linalg.LinAlgError)
    for name, effect in scenarios:
        ar_covers = fieller_covers = coupled_covers = 0
        binary_median_covers = binary_outer_covers = 0
        ar_points, ar_components, ar_boundaries, ar_empty = [], [], [], []
        f_stats, fieller_unbounded, binary_targets, compatible, null_relevance = [], [], [], [], []
        failures = dict(ar=0, fieller=0, binary_bounds=0, coupled_working=0)
        for rep in range(replications):
            rep_seed = seed + 37 * rep
            panel = generate_coverage_panel(n=n, t_len=t_len, instrument_effect=effect,
                                            true_beta=true_beta, seed=rep_seed, binary_threshold=threshold)
            y, d, z = panel.y, panel.d, panel.z
            try:
                ar = randomization_ar(y, d, z, beta=beta_grid, alpha=.05, method="monte_carlo",
                                      permutations=permutations, seed=rep_seed)
                grid_summary = summarize_ar_grid(ar.table[ar.table.horizon == 1], true_beta)
                ar_covers += grid_summary["covers_target"]
                ar_points.append(grid_summary["accepted_grid_points"])
                ar_components.append(len(grid_summary["grid_runs"]))
                ar_boundaries.append(grid_summary["left_boundary_accepted"] or grid_summary["right_boundary_accepted"])
                ar_empty.append(grid_summary["accepted_grid_points"] == 0)
            except expected_failures as exc:
                failures["ar"] += 1
                warnings.warn(f"{name} replication {rep}: AR failed: {exc}", RuntimeWarning)
            try:
                ref = encouragement_effects(y, d, z, alpha=.05)
                row = ref[ref.horizon == 1].iloc[0]
                first_stage, se = float(row.itt_d), float(row.itt_d_se)
                f_stats.append((first_stage / se)**2 if se > 0 else (np.inf if first_stage else 0.0))
                if row.wald_set_type == "unavailable":
                    failures["fieller"] += 1
                else:
                    fieller_covers += any(interval_contains(row[f"wald_lower_{k}"], row[f"wald_upper_{k}"], true_beta)
                                          for k in (1, 2))
                    fieller_unbounded.append(row.wald_set_type in ("all_real", "disjoint", "half_line"))
            except expected_failures as exc:
                failures["fieller"] += 1
                warnings.warn(f"{name} replication {rep}: Fieller failed: {exc}", RuntimeWarning)
            if panel.binary_complier_target is not None:
                target = panel.binary_complier_target
                binary_targets.append(target)
                try:
                    binary = encouragement_bounds((y[:, 1] > threshold).astype(float), d[:, 1], z[:, 1],
                                                  delta=1., prior=1., draws=bounds_draws, chains=2, seed=rep_seed)
                    row = binary.table[binary.table.estimand == "complier_encouragement"].iloc[0]
                    compatible.append(float(row.compatible_fraction))
                    if not np.isfinite([row.lower_median, row.upper_median]).all():
                        failures["binary_bounds"] += 1
                    binary_median_covers += interval_contains(row.lower_median, row.upper_median, target)
                    binary_outer_covers += interval_contains(row.lower_ci_lower, row.upper_ci_upper, target)
                except expected_failures as exc:
                    failures["binary_bounds"] += 1
                    warnings.warn(f"{name} replication {rep}: binary bounds failed: {exc}", RuntimeWarning)
            if include_experimental:
                try:
                    config = CoupledHazardConfig(num_sweeps=coupled_sweeps, num_burnin=coupled_burnin, seed=rep_seed)
                    working = CoupledHazardIV(config).fit(y, d, z)
                    coupled_covers += interval_contains(*working.beta_ci, true_beta)
                    null_relevance.append(working.metadata["null_relevance_fraction"])
                except expected_failures as exc:
                    failures["coupled_working"] += 1
                    warnings.warn(f"{name} replication {rep}: experimental working fit failed: {exc}", RuntimeWarning)
        ar_rate, fieller_rate = ar_covers / replications, fieller_covers / replications
        mean = lambda values: float(np.mean(values)) if values else np.nan
        summaries.append(ScenarioCoverageSummary(
            scenario=name, instrument_effect=effect, mean_first_stage_f=mean(f_stats), true_beta=true_beta,
            ar_coverage=ar_rate, ar_coverage_mcse=_mcse(ar_rate, replications),
            ar_mean_accepted_grid_points=mean(ar_points), ar_mean_grid_components=mean(ar_components),
            ar_grid_boundary_acceptance_rate=mean(ar_boundaries), ar_empty_grid_rate=mean(ar_empty),
            ar_tail_status="unknown outside evaluated grid; between-grid acceptance also unknown",
            fieller_coverage=fieller_rate, fieller_coverage_mcse=_mcse(fieller_rate, replications),
            fieller_unbounded_rate=mean(fieller_unbounded), binary_threshold=threshold,
            mean_binary_complier_target=mean(binary_targets) if binary_targets else None,
            binary_target_defined_replications=len(binary_targets),
            binary_median_bounds_containment=binary_median_covers / len(binary_targets) if binary_targets else None,
            binary_posterior_outer_interval_coverage=binary_outer_covers / len(binary_targets) if binary_targets else None,
            binary_mean_compatible_fraction=mean(compatible),
            coupled_working_interval_coverage=coupled_covers / replications if include_experimental else None,
            coupled_causal_inference_supported=False,
            coupled_mean_null_relevance_fraction=mean(null_relevance) if include_experimental else None,
            failures=failures, replications=replications,
        ))
    return summaries


def generate_sbc_panel(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deprecated compatibility wrapper; this generator is not an SBC experiment."""
    warnings.warn("Use generate_coverage_panel; fixed-parameter coverage is not SBC.", DeprecationWarning, stacklevel=2)
    panel = generate_coverage_panel(*args, **kwargs)
    return panel.y, panel.d, panel.z


def evaluate_sbc(*args: Any, **kwargs: Any) -> list[ScenarioCoverageSummary]:
    """Deprecated compatibility alias for evaluate_coverage."""
    warnings.warn("Use evaluate_coverage; this experiment is not SBC.", DeprecationWarning, stacklevel=2)
    return evaluate_coverage(*args, **kwargs)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fixed-parameter IV repeated-sampling coverage checks (not SBC)")
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bounds-draws", type=int, default=1000)
    parser.add_argument("--include-experimental", action="store_true")
    parser.add_argument("--coupled-sweeps", type=int, default=400)
    parser.add_argument("--coupled-burnin", type=int, default=200)
    parser.add_argument("--output", default="benchmarks/encouragement/coverage_report.json")
    args = parser.parse_args()
    options = vars(args).copy()
    out_path = Path(options.pop("output"))
    summaries = evaluate_coverage(**options)
    records = [asdict(summary) for summary in summaries]
    report = {
        "experiment": "fixed-parameter superpopulation coverage; not SBC",
        "options": options,
        "estimands": {
            "continuous": "constant current-adoption effect beta, at horizon one for AR and Fieller",
            "binary": "finite-population complier risk difference at horizon one, threshold fixed at zero",
            "experimental": "working-model coefficient interval containment; causal interval withheld",
        },
        "priors": {"binary_cell_beta": [1, 1], "binary_uptake_restriction": "monotone",
                   "binary_direct_effect_bound": 1., "experimental_hazard_defaults": asdict(CoupledHazardConfig())},
        "limitations": ["AR tails and values between grid points are unknown.",
                        "Posterior endpoint intervals have no asserted nominal frequentist coverage.",
                        "Monte Carlo SEs describe finite replication noise; they are not coverage guarantees.",
                        "Experimental hazard chain convergence is not established by this report.",
                        "Method failures count as noncoverage; descriptive set statistics use successful evaluations."],
        "scenarios": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    print(pd.DataFrame(records).to_string(index=False))
