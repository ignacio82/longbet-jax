"""Resumable reduced-form comparison on independently generated experiments.

This is a calibration harness, not a declaration that the model is calibrated.
``--limit`` bounds newly attempted fits. Existing successful runs are replayable
from NPZ archives; failed runs retain inputs, configuration, traceback and seeds.
SUR-disabled fits have separate independent forests and independent intercepts;
SUR-enabled fits couple observation innovations but still have independent
unit-intercept priors. Neither is a correlated-intercept model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import warnings

import numpy as np

from dgp import SCENARIOS, make_population, population_standardized_truth
from reference_matrix import environment, portable, rate


VARIANTS = ("joint_lpm", "separate_lpm", "joint_probit", "separate_probit")


def draw_diagnostics(draws, alpha=.05, *, ratio=False):
    """Diagnose the transformed quantities themselves, including ratio tails."""
    from longbet._diagnostics import compute_ess, compute_rhat
    import arviz as az
    result = []
    # Input H,C,K; never treat horizons as chains or exchange independent fits.
    for value in draws:
        flat = value.ravel()
        finite = np.isfinite(flat).all()
        if finite and np.ptp(flat) > 0:
            ess_bulk = float(compute_ess(value, method="bulk"))
            ess_tail = float(compute_ess(value, method="tail", prob=alpha / 2))
            rhat = float(compute_rhat(value))
            quantile_mcse = {f"mcse_{name}": float(az.mcse(value, method="quantile", prob=p))
                             for name, p in (("lower", alpha / 2), ("median", .5), ("upper", 1 - alpha / 2))}
            if not ratio:
                ess_mean = float(compute_ess(value, method="mean"))
                mcse = float(az.mcse(value, method="mean"))
        else:
            ess_bulk = ess_tail = ess_mean = rhat = mcse = np.nan
            quantile_mcse = {f"mcse_{name}": np.nan for name in ("lower", "median", "upper")}
        diagnostic = {"ess_bulk": ess_bulk, "ess_tail": ess_tail,
                       "rhat": rhat, **quantile_mcse,
                       "convergence_checks_pass": bool(ess_bulk >= 400 and ess_tail >= 400
                                                        and rhat <= 1.01),
                       "undefined_fraction": float(np.mean(~np.isfinite(flat)))}
        if not ratio:
            diagnostic.update(ess_mean=ess_mean, mcse_mean=mcse)
        result.append(diagnostic)
    return result


def summarize_draws(values, targets, *, alpha=.05):
    y, d = values["outcome"][0], values["takeup"][0]  # H,C,K
    ratio = np.divide(y, d, out=np.full_like(y, np.nan), where=d != 0)
    summaries = {}
    for quantity, draws in (("itt_y", y), ("itt_d", d), ("wald", ratio)):
        finite = np.isfinite(draws).all(axis=(1, 2))
        q = np.full((3, len(draws)), np.nan)
        if finite.any():
            q[:, finite] = np.quantile(draws[finite].reshape(finite.sum(), -1),
                                      [alpha / 2, .5, 1 - alpha / 2], axis=1)
        # No clipping, sign restriction, or conditioning away undefined ratios.
        summaries[quantity] = {"lower": q[0], "median": q[1], "upper": q[2],
                               "diagnostics": draw_diagnostics(draws, alpha, ratio=quantity == "wald"),
                               "coverage": {name: ((q[0] <= truth[quantity])
                                                   & (truth[quantity] <= q[2])).astype(float)
                                            for name, truth in targets.items()}}
        for name, truth in targets.items():
            summaries[quantity]["coverage"][name][~np.isfinite(truth[quantity])] = np.nan
        if quantity != "wald":
            summaries[quantity]["mean"] = draws.mean(axis=(1, 2))
    joint = np.concatenate([y, d], axis=0).reshape(2 * len(y), -1)
    summaries["posterior_itt_covariance"] = np.cov(joint, ddof=1)
    summaries["negative_denominator_probability"] = (d < 0).mean(axis=(1, 2))
    return summaries, {"itt_y": y, "itt_d": d, "wald": ratio}


def aggregate(output):
    """Across independently generated populations; avoid counting horizons as runs."""
    groups = {}
    for file in sorted(output.glob("*/result.json")):
        try:
            result = json.loads(file.read_text())
        except json.JSONDecodeError:
            # Another resumable worker may be publishing this independent run.
            # A final aggregation after all workers finish includes it.
            continue
        key = (json.dumps(result["scenario"], sort_keys=True), result["n"],
               result["variant"], result["config_id"])
        groups.setdefault(key, []).append(result)
    summary = []
    for (scenario_json, n, variant, config_id), attempts in groups.items():
        scenario = json.loads(scenario_json)
        runs = [r for r in attempts if r["status"] == "complete"]
        finished = [r for r in attempts if r["status"] != "running"]
        entry = {"scenario": scenario["name"], "scenario_parameters": scenario,
                 "n": n, "variant": variant, "config_id": config_id,
                 "independent_datasets": len(runs), "standardizations": {},
                 "failed_runs": sum(r["status"] == "failed" for r in attempts),
                 "pending_runs": sum(r["status"] == "running" for r in attempts),
                 "failure_rate": rate([r["status"] == "failed" for r in finished]) if finished else None}
        if not runs:
            summary.append(entry)
            continue
        for target in ("conditional", "population"):
            results = [run["standardizations"][target] for run in runs]
            mean = [np.r_[r["itt_y"]["mean"], r["itt_d"]["mean"]] for r in results]
            truth_name = "conditional_mean" if target == "conditional" else "population_marginal"
            truth = [np.r_[run["targets"][truth_name]["itt_y"], run["targets"][truth_name]["itt_d"]]
                     for run in runs]
            errors = np.asarray(mean) - np.asarray(truth)
            covariance_draws = np.asarray([r["posterior_itt_covariance"] for r in results])
            covariance = covariance_draws.mean(axis=0)
            centered = errors - errors.mean(axis=0)
            products = centered[:, :, None] * centered[:, None, :]
            coverage = {}
            for q in ("itt_y", "itt_d", "wald"):
                v = np.asarray([r[q]["coverage"][truth_name] for r in results], dtype=float)
                count = np.isfinite(v).sum(axis=0)
                p = np.divide(np.nansum(v, axis=0), count, out=np.full(count.shape, np.nan), where=count > 0)
                se = np.sqrt(np.divide(p * (1 - p), count, out=np.full(count.shape, np.nan), where=count > 0))
                coverage[q] = {"rate": p, "mcse": se, "datasets": count}
            entry["standardizations"][target] = {
                "target": truth_name, "coverage": coverage,
                "mean_error": errors.mean(axis=0),
                "empirical_error_covariance": np.cov(errors, rowvar=False, ddof=1) if len(runs) > 1 else None,
                "empirical_error_covariance_mcse": products.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else None,
                "mean_posterior_itt_covariance": covariance,
                "mean_posterior_covariance_mcse": covariance_draws.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else None,
                "convergence_failure_rate": {
                    q: rate([[not d["convergence_checks_pass"] for d in r[q]["diagnostics"]] for r in results])
                    for q in ("itt_y", "itt_d", "wald")},
                "posterior_ratio_interval_unavailable": rate([
                    [d["undefined_fraction"] > 0 for d in r["wald"]["diagnostics"]] for r in results]),
                "posterior_ratio_withheld_if_convergence_required": rate([
                    [d["undefined_fraction"] > 0 or not d["convergence_checks_pass"]
                     for d in r["wald"]["diagnostics"]] for r in results]),
                "mean_undefined_ratio_draw_fraction": np.mean([
                    [d["undefined_fraction"] for d in r["wald"]["diagnostics"]] for r in results], axis=0),
                "covariance_order": "Y horizons then D horizons; no pooling across time",
                "calibration_status": "not_established",
            }
        summary.append(entry)
    (output / "summary.json").write_text(json.dumps(portable(summary), indent=2, allow_nan=False) + "\n")


def refresh_archive_diagnostics(output):
    """Recompute transformed diagnostics from saved draws without refitting.

    This makes diagnostic-method improvements replayable without changing the
    likelihood, chain lengths, targets, coverage indicators or posterior draws.
    """
    updated = 0
    for file in sorted(output.glob("*/result.json")):
        record = json.loads(file.read_text())
        if record["status"] != "complete" or record.get("diagnostics_version") == 2:
            continue
        with np.load(file.parent / "effect_draws.npz", allow_pickle=False) as draws:
            for standardization in ("conditional", "population"):
                for quantity in ("itt_y", "itt_d", "wald"):
                    stats = record["standardizations"][standardization][quantity]
                    stats["diagnostics"] = draw_diagnostics(draws[f"{standardization}_{quantity}"],
                                                            ratio=quantity == "wald")
                    stats.pop("mean_mcse_interpretation", None)
        record["diagnostics_version"] = 2
        temporary = file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(portable(record), indent=2, allow_nan=False) + "\n")
        temporary.replace(file)
        updated += 1
    aggregate(output)
    return updated


def run(args):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpus])
    # Affinity is bounded before JAX creates worker threads.
    import jax
    from longbet import LongBetConfig, LongBetMulti
    from longbet._encourage_predict import standardize_encouragement
    from longbet import encouragement_effects

    args.output.mkdir(parents=True, exist_ok=True)
    attempted = 0
    for scenario in args.scenarios:
        for dataset in range(args.datasets):
            scenario_idx = list(SCENARIOS).index(scenario)
            data_seed = args.seed + 100_000 * scenario_idx + 100 * dataset
            assignment_seed, sampler_seed = data_seed + 1, data_seed + 50
            population = make_population(args.n, scenario, seed=data_seed)
            data = population.assign(assignment_seed)
            marginal = population_standardized_truth(population, draws=args.integration_draws,
                                                     seed=data_seed + 2)
            targets = {"finite_population": population.truth,
                       "conditional_mean": population.conditional_mean_truth,
                       "population_marginal": marginal}
            for variant in args.variants:
                config = LongBetConfig(
                    num_chains=args.chains, num_burnin=args.burnin * args.chain_multiplier,
                    num_sweeps=args.draws * args.chain_multiplier, n_skip=1,
                    num_trees_pr=args.trees, num_trees_trt=args.trees,
                    max_depth_pr=3, max_depth_trt=3, num_cutpoints=16,
                    sigma_prior_a=2, sigma_prior_b=1, random_intercept=True,
                    sur=variant.startswith("joint"), num_shared_trees=0,
                )
                cfg = asdict(config)
                config_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
                run_id = f"{scenario}_n{args.n}_d{dataset}_{variant}_{config_id}_seed{data_seed}"
                folder = args.output / run_id
                if (folder / "result.json").exists():
                    previous = json.loads((folder / "result.json").read_text())
                    if previous["status"] == "complete" or not args.retry_failures:
                        continue
                if args.limit is not None and attempted >= args.limit:
                    aggregate(args.output)
                    return
                attempted += 1
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "data.npz", **population.archive(assignment_seed))
                record = {"run_id": run_id, "config": cfg, "config_id": config_id,
                          "environment": environment(), "scenario": asdict(population.scenario),
                          "assumption_failures": population.scenario.assumption_failures,
                          "dataset": dataset, "n": args.n, "variant": variant,
                          "data_seed": data_seed, "assignment_seed": assignment_seed,
                          "sampler_seed": sampler_seed, "targets": targets,
                          "unit_intercept_prior": "independent across outcomes",
                          "innovation_coupling": "triangular SUR" if config.sur else "independent",
                          "calibration_status": "not_established", "status": "running",
                          "diagnostics_version": 2}
                (folder / "result.json").write_text(json.dumps(portable(record), indent=2))
                started = time.monotonic()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model = LongBetMulti(config).fit(
                            {"outcome": data["y"], "takeup": data["d"]},
                            x=data["x"], z=data["z"], t=data["t"],
                            outcome={"outcome": "binary" if population.scenario.binary else "continuous",
                                     "takeup": "binary" if variant.endswith("probit") else "continuous"},
                            key=jax.random.key(sampler_seed),
                        )
                        jax.block_until_ready(model.trace)
                        # Binary SUR rows have zero loadings to preserve the
                        # marginal probit scale; all-binary pairs are uncoupled
                        # here because shared trees are disabled.
                        coupled = bool(model.sur_active and any(
                            model.outcome[k] == "continuous" for k in model.order[1:]))
                        record["innovation_coupling"] = (
                            "triangular SUR" if coupled else
                            "none: both equations binary" if all(v == "binary" for v in model.outcome)
                            else "independent")
                        record["requested_sur"] = bool(config.sur)
                        record["fit_seconds"] = time.monotonic() - started
                        model.save(folder / "model.npz")
                        reference = encouragement_effects(data["y"], data["d"], data["z"], data["t"])
                        record["reference"] = reference.to_dict("records")
                        record["standardizations"] = {}
                        raw = {}
                        for target in ("conditional", "population"):
                            result = standardize_encouragement(model, data["x"], data["z"], data["t"],
                                                               standardization=target, block_size=128)
                            stats, draws = summarize_draws(result.draws, targets)
                            record["standardizations"][target] = stats
                            raw.update({f"{target}_{q}": v for q, v in draws.items()})
                        np.savez_compressed(folder / "effect_draws.npz", **raw)
                        record["warnings"] = sorted(set(str(w.message) for w in caught))
                    record["status"] = "complete"
                except Exception as exc:
                    record.update(status="failed", exception=repr(exc), traceback=traceback.format_exc())
                record["total_seconds"] = time.monotonic() - started
                (folder / "result.json").write_text(json.dumps(portable(record), indent=2, allow_nan=False) + "\n")
                print(f"{run_id}: {record['status']} ({record['total_seconds']:.1f}s)", flush=True)
                aggregate(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "model_results")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=["strong", "weak", "binary", "correlated_serial_heteroskedastic"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--datasets", type=int, default=2)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--burnin", type=int, default=400)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--trees", type=int, default=5)
    parser.add_argument("--chain-multiplier", type=int, default=1)
    parser.add_argument("--integration-draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpus", type=int, default=3)
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if min(args.datasets, args.n, args.chains, args.draws, args.trees,
           args.chain_multiplier, args.cpus) < 1 or args.burnin < 0:
        parser.error("Counts must be positive and burnin nonnegative.")
    run(args)


if __name__ == "__main__":
    main()
