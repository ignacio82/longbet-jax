"""Resumable four-chain direct-effect repair comparison; see the repair protocol.

Run each variant in its own process to keep compilation and BLAS timings honest.
Existing attempts, including failures, are preserved. --refresh summarizes saved
draws without refitting. Short windows are prefixes of the full retained chain.
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

from dgp import Scenario, make_population

VARIANTS = ("current", "current_stump", "parametric", "direct_smooth", "direct_blocked", "direct_joint")
QUANTITIES = ("itt_y", "itt_d", "wald")


def write_json(path, record):
    from reference_matrix import portable
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(portable(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize_windows(raw, targets, fit_seconds):
    from model_calibration import summarize_draws
    targets = {name: {q: np.asarray(truth[q], dtype=float) for q in QUANTITIES}
               for name, truth in targets.items()}
    total = raw["itt_y"].shape[-1]
    if total < 8 or total % 2:
        raise ValueError("Retained draw count must be even and at least eight.")
    windows = {}
    for label, size in (("half", total // 2), ("full", total)):
        values = {"outcome": raw["itt_y"][None, ..., :size],
                  "takeup": raw["itt_d"][None, ..., :size]}
        summaries, _ = summarize_draws(values, targets)
        windows[label] = {"retained_per_chain": size, "quantities": summaries}
        if label == "full":
            for q in QUANTITIES:
                for diagnostic in summaries[q]["diagnostics"]:
                    diagnostic["bulk_ess_per_fit_second"] = diagnostic["ess_bulk"] / fit_seconds
                    diagnostic["tail_ess_per_fit_second"] = diagnostic["ess_tail"] / fit_seconds
    growth = {}
    for q in QUANTITIES:
        growth[q] = {kind: np.array([d[kind] for d in windows["full"]["quantities"][q]["diagnostics"]])
                     / np.array([d[kind] for d in windows["half"]["quantities"][q]["diagnostics"]])
                     for kind in ("ess_bulk", "ess_tail")}
    return {"windows": windows, "ess_growth": growth,
            "all_effect_checks_pass": all(d["convergence_checks_pass"]
                for q in QUANTITIES for d in windows["full"]["quantities"][q]["diagnostics"]),
            "ess_growth_screen_pass": all(np.all(v >= 1.5) for g in growth.values() for v in g.values())}


def model_config(variant, *, chains, burnin, draws):
    if variant == "parametric":
        from parametric_candidate import ParametricConfig
        return ParametricConfig()
    if variant.startswith("direct"):
        from direct_smooth_candidate import ForestConfig
        return ForestConfig(joint_baseline_intercept=variant in ("direct_blocked", "direct_joint"),
                            joint_effect_leaves=variant == "direct_joint")
    from longbet import LongBetConfig
    stump = variant == "current_stump"
    return LongBetConfig(num_chains=chains, num_burnin=burnin,
        num_sweeps=draws, n_skip=1, num_trees_pr=5, num_trees_trt=5,
        max_depth_pr=2 if stump else 3, max_depth_trt=2 if stump else 3,
        num_cutpoints=8, sigma_prior_a=3, sigma_prior_b=2,
        gamma_prior_a=3, gamma_prior_b=2, random_intercept=True,
        sur=False, num_shared_trees=0)


def fit_current(data, start, *, seed, chains, burnin, draws, stump, folder):
    import jax
    from longbet import LongBetMulti
    from longbet._encourage_predict import standardize_encouragement
    config = model_config("current_stump" if stump else "current", chains=chains, burnin=burnin, draws=draws)
    started = time.monotonic()
    model = LongBetMulti(config).fit({"outcome": data["y"], "takeup": data["d"]},
        x=data["x"], z=data["z"], t=data["t"],
        outcome={"outcome": "continuous", "takeup": "continuous"},
        key=jax.random.key(seed))
    jax.block_until_ready(model.trace)
    fit_seconds = time.monotonic() - started
    model.save(folder / "model.npz")
    result = standardize_encouragement(model, data["x"], data["z"], data["t"], block_size=80)
    return result.draws, {}, asdict(config), fit_seconds


def run(args):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, args.cpu_ids)
    # Imports that initialize JAX come after the affinity restriction.
    from reference_matrix import environment
    from longbet import encouragement_effects
    scenario = (Scenario(name="clean_strong", unit_correlation=0,
                         serial_correlation=0, heteroskedasticity=0)
                if args.scenario == "clean_strong" else Scenario())
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name("dgp.py")]
    source_paths += [Path(__file__).with_name(f"{name}_candidate.py")
                    for name in ("parametric", "direct_smooth")]
    sources = {str(p.relative_to(Path(__file__).resolve().parents[2])):
               hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.exists()}
    for dataset in range(args.datasets):
        if args.dataset_index is not None and dataset != args.dataset_index:
            continue
        data_seed = args.seed + 100 * dataset
        assignment_seed = data_seed + 1
        population = make_population(args.n, scenario, seed=data_seed)
        data = population.assign(assignment_seed)
        targets = {"finite_population": population.truth,
                   "conditional_mean": population.conditional_mean_truth}
        for sampler in range(args.sampler_seeds):
            if args.sampler_index is not None and sampler != args.sampler_index:
                continue
            seed = data_seed + 50 + 10000 * sampler
            for variant in args.variants:
                run_id = (f"{scenario.name}_n{args.n}_d{dataset}_s{sampler}_{variant}"
                          f"_c{args.chains}_b{args.burnin}_k{args.draws}_seed{args.seed}")
                folder = args.output / run_id
                file = folder / "result.json"
                if file.exists():
                    record = json.loads(file.read_text())
                    if args.refresh and record["status"] == "complete":
                        with np.load(folder / "effect_draws.npz", allow_pickle=False) as archive:
                            raw = dict(archive)
                        record.update(summarize_windows(raw, record["targets"], record["fit_seconds"]))
                        write_json(file, record)
                    continue
                if args.refresh:
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "data.npz", **population.archive(assignment_seed))
                config_object = model_config(variant, chains=args.chains, burnin=args.burnin, draws=args.draws)
                record = {"run_id": run_id, "variant": variant, "scenario": asdict(scenario),
                    "n": args.n, "dataset": dataset, "sampler": sampler,
                    "data_seed": data_seed, "assignment_seed": assignment_seed, "sampler_seed": seed,
                    "chains": args.chains, "burnin": args.burnin, "draws": args.draws, "config": asdict(config_object),
                    "target": "all_original_study_units", "targets": targets,
                    "environment": environment(), "source_sha256": sources,
                    "calibration_status": "not_established", "status": "running"}
                write_json(file, record)
                started = time.monotonic()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        kwargs = dict(seed=seed, chains=args.chains, burnin=args.burnin, draws=args.draws)
                        if variant.startswith("current"):
                            values, parameters, config, fit_seconds = fit_current(data, population.start,
                                **kwargs, stump=variant == "current_stump", folder=folder)
                        else:
                            if variant == "parametric":
                                from parametric_candidate import fit_candidate
                            else:
                                from direct_smooth_candidate import fit_candidate
                            fit_start = time.monotonic()
                            values, parameters = fit_candidate(data, population.start, config=config_object, **kwargs)
                            fit_seconds = time.monotonic() - fit_start
                            config = asdict(config_object)
                            np.savez_compressed(folder / "parameters.npz", **parameters)
                        raw = {"itt_y": values["outcome"][0], "itt_d": values["takeup"][0]}
                        raw["wald"] = np.divide(raw["itt_y"], raw["itt_d"],
                            out=np.full_like(raw["itt_y"], np.nan), where=raw["itt_d"] != 0)
                        np.savez_compressed(folder / "effect_draws.npz", **raw)
                        record.update(config=config, fit_seconds=fit_seconds,
                                      **summarize_windows(raw, targets, fit_seconds))
                        record["reference"] = encouragement_effects(data["y"], data["d"], data["z"], data["t"]).to_dict("records")
                        record["warnings"] = sorted(set(str(w.message) for w in caught))
                    record["status"] = "complete"
                except Exception as exc:
                    record.update(status="failed", exception=repr(exc), traceback=traceback.format_exc())
                record["total_seconds"] = time.monotonic() - started
                write_json(file, record)
                print(f"{run_id}: {record['status']} {record['total_seconds']:.1f}s "
                      f"checks={record.get('all_effect_checks_pass')}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "repair_results")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    p.add_argument("--scenario", choices=("clean_strong", "strong"), default="clean_strong")
    p.add_argument("--datasets", type=int, default=2)
    p.add_argument("--sampler-seeds", type=int, default=2)
    p.add_argument("--dataset-index", type=int, help="Run one dataset index for distributed workers.")
    p.add_argument("--sampler-index", type=int, help="Run one sampler index for distributed workers.")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--burnin", type=int, default=1000)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20270909)
    p.add_argument("--cpu-ids", nargs="+", type=int, default=[8, 9, 10, 11])
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    if args.chains < 4 or args.draws < 8 or args.draws % 2 or args.burnin < 0:
        p.error("Require >=4 chains, even draws >=8, nonnegative burnin.")
    if min(args.datasets, args.sampler_seeds, args.n) < 1:
        p.error("Counts must be positive.")
    if args.dataset_index is not None and not 0 <= args.dataset_index < args.datasets:
        p.error("dataset-index must be within the configured dataset range.")
    if args.sampler_index is not None and not 0 <= args.sampler_index < args.sampler_seeds:
        p.error("sampler-index must be within the configured sampler range.")
    run(args)


if __name__ == "__main__":
    main()
