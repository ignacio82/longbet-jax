"""Repeated randomization, finite-population covariance and coverage study.

Run from the repository root with ``python benchmarks/encouragement/reference_matrix.py``.
Every scenario fixes both potential paths and repeats independently seeded complete
assignment. Horizons share units and are never treated as independent replications.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import traceback

import numpy as np

from longbet import encouragement_effects
from dgp import SCENARIOS, make_population


def portable(value):
    """Strict JSON: undefined values become null, infinities remain labeled."""
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return portable(value.item())
    if isinstance(value, (tuple, list, np.ndarray)):
        return [portable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return (float(value) if np.isfinite(value) else
                None if np.isnan(value) else "Infinity" if value > 0 else "-Infinity")
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def environment():
    versions = {}
    for name in ("numpy", "scipy", "pandas", "jax", "jaxlib", "bartz", "longbet"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed-as-distribution"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "git_commit": commit, "git_dirty": dirty,
            "created_utc": datetime.now(timezone.utc).isoformat()}


def rate(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    estimate = np.divide(np.nansum(values, axis=0), count,
                         out=np.full(count.shape, np.nan), where=count > 0)
    variance = np.divide(estimate * (1 - estimate), count,
                         out=np.full(count.shape, np.nan), where=count > 0)
    return {"rate": estimate, "mcse": np.sqrt(variance),
            "replications": count}


def run_setting(name, n, replications, seed, output):
    population = make_population(n, name, seed=seed)
    truth = population.truth
    exact = population.covariance_truth()
    h = len(truth["itt_y"])
    seeds = np.random.SeedSequence(seed + 1).generate_state(replications)
    estimates = np.full((replications, 2 * h), np.nan)
    covariance = np.full((replications, 2 * h, 2 * h), np.nan)
    coverage = np.full((replications, h, 3), np.nan)
    undefined, negative, unavailable = [np.zeros((replications, h)) for _ in range(3)]
    set_types = {}
    failures = []
    for r, assignment_seed in enumerate(seeds):
        try:
            data = population.assign(int(assignment_seed))
            table = encouragement_effects(data["y"], data["d"], data["z"], data["t"])
            estimate = np.r_[table.itt_y, table.itt_d]
            values = np.c_[data["y"][:, population.start:], data["d"][:, population.start:]]
            arms = data["assignment"]
            v = (np.cov(values[arms], rowvar=False, ddof=1) / arms.sum()
                 + np.cov(values[~arms], rowvar=False, ddof=1) / (~arms).sum())
            # Also verifies the scalar API's within-horizon cross covariance.
            np.testing.assert_allclose(np.diag(v)[:h], table.itt_y_se**2, atol=1e-12)
            np.testing.assert_allclose(np.diag(v)[h:], table.itt_d_se**2, atol=1e-12)
            np.testing.assert_allclose(v[np.arange(h), h + np.arange(h)], table.itt_y_d_cov,
                                       atol=1e-12)
            estimates[r], covariance[r] = estimate, v
            for j, row in enumerate(table.to_dict("records")):
                for k, quantity in enumerate(("itt_y", "itt_d")):
                    coverage[r, j, k] = (row[f"{quantity}_lower"] <= truth[quantity][j]
                                         <= row[f"{quantity}_upper"])
                if np.isfinite(truth["wald"][j]):
                    coverage[r, j, 2] = any(row[f"wald_lower_{k}"] <= truth["wald"][j]
                                           <= row[f"wald_upper_{k}"] for k in (1, 2))
                undefined[r, j] = not np.isfinite(row["wald"])
                negative[r, j] = row["itt_d"] < 0
                unavailable[r, j] = row["wald_set_type"] == "unavailable"
                kind = row["wald_set_type"]
                set_types[kind] = set_types.get(kind, 0) + 1
        except Exception as exc:
            failures.append({"replication": r, "assignment_seed": int(assignment_seed),
                             "exception": repr(exc), "traceback": traceback.format_exc()})
            undefined[r], negative[r], unavailable[r] = np.nan, np.nan, np.nan
            estimates[r], covariance[r], coverage[r] = np.nan, np.nan, np.nan
    successful = np.isfinite(estimates).all(axis=1)
    if not successful.any():
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / f"{name}_n{n}_failed_population.npz",
                            **population.archive(int(seeds[0])), assignment_seeds=seeds)
        (output / f"{name}_n{n}_failures.json").write_text(json.dumps(portable({
            "scenario": asdict(population.scenario), "n": n, "data_seed": seed,
            "environment": environment(), "failures": failures}), indent=2))
        raise RuntimeError(f"Every replication failed for {name}/N={n}: {failures[0]}")
    draws, variances = estimates[successful], covariance[successful]
    observed_covariance = np.cov(draws, rowvar=False, ddof=1)
    mean_covariance = variances.mean(axis=0)
    # Nonparametric MCSE for empirical covariance (asymptotic; assignments IID).
    products = (draws - draws.mean(axis=0))[:, :, None] * (draws - draws.mean(axis=0))[:, None, :]
    covariance_mcse = products.std(axis=0, ddof=1) / np.sqrt(len(draws))
    joint_truth = np.r_[truth["itt_y"], truth["itt_d"]]
    simultaneous = {}
    for k, quantity in enumerate(("itt_y", "itt_d", "wald")):
        eligible = np.isfinite(coverage[:, :, k]).all(axis=1)
        simultaneous[quantity] = rate(coverage[eligible, :, k].all(axis=1)) if eligible.any() else None
    output.mkdir(parents=True, exist_ok=True)
    archive = f"{name}_n{n}.npz"
    np.savez_compressed(output / archive, **population.archive(int(seeds[0])),
                        assignment_seeds=seeds, estimates=estimates,
                        neyman_covariances=covariance, coverage=coverage)
    return {
        "scenario": asdict(population.scenario), "assumption_failures": population.scenario.assumption_failures,
        "n": n, "data_seed": seed, "assignment_seed_sequence": seed + 1,
        "replications": replications, "successful_replications": int(successful.sum()),
        "truth": truth, "conditional_mean_truth": population.conditional_mean_truth,
        "realized_type_counts": {k: int((population.compliance_type == k).sum())
                                 for k in np.unique(population.compliance_type)},
        "pointwise_coverage": {q: rate(coverage[:, :, k])
                               for k, q in enumerate(("itt_y", "itt_d", "wald"))},
        "simultaneous_coverage_of_pointwise_intervals": simultaneous,
        "undefined_sample_ratio": rate(undefined), "negative_sample_first_stage": rate(negative),
        "unavailable_inference": rate(unavailable),
        "confidence_set_counts_across_dependent_horizons": set_types,
        "bias": draws.mean(axis=0) - joint_truth,
        "bias_mcse": draws.std(axis=0, ddof=1) / np.sqrt(len(draws)),
        "covariance_variable_order": [f"{q}_h{j + 1}" for q in ("y", "d") for j in range(h)],
        "exact_covariance": exact, "empirical_covariance": observed_covariance,
        "empirical_covariance_mcse": covariance_mcse,
        "mean_estimated_neyman_covariance": mean_covariance,
        "mean_neyman_covariance_mcse": variances.std(axis=0, ddof=1) / np.sqrt(len(draws)),
        "failures": failures, "raw_archive": archive,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reference_matrix_results")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--sizes", nargs="+", type=int, default=[80, 400, 1200])
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("At least two replications are required.")
    settings = [(s, n) for s in args.scenarios for n in args.sizes]
    manifest = {"environment": environment(), "method": "normal_ar",
                "nominal_pointwise_coverage": .95,
                "target": "fixed_finite_population_complete_randomization",
                "covariance_target": "exact finite-population covariance; Neyman estimate is conservative",
                "settings": [], "replications": args.replications}
    args.output.mkdir(parents=True, exist_ok=True)
    for j, (s, n) in enumerate(settings):
        result = run_setting(s, n, args.replications, args.seed + 1000 * j, args.output)
        name = f"{s}_n{n}.json"
        (args.output / name).write_text(json.dumps(portable(result), indent=2, allow_nan=False) + "\n")
        manifest["settings"].append(name)
        (args.output / "manifest.json").write_text(json.dumps(portable(manifest), indent=2) + "\n")
        print(f"{s} N={n}: {result['successful_replications']}/{args.replications}", flush=True)


if __name__ == "__main__":
    main()
