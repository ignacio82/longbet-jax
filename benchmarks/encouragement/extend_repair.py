"""Extend every matching direct-forest run from its saved RNG and sampler state.

Original folders are immutable. New folders contain the exact original prefixes
plus the added draws, with summed measured fit time and explicit parent records.
This is continuation, not a restart or a selective replacement of failing seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from direct_smooth_candidate import ForestConfig, fit_candidate
from repair_comparison import summarize_windows, write_json


def concatenate_parameters(old, new, chains, old_draws, added_draws):
    """Concatenate trace axes; retain latest RNG state and static design metadata."""
    combined = {}
    for name, value in new.items():
        prior = old[name]
        if (value.ndim >= 2 and prior.shape[:2] == (chains, old_draws)
                and value.shape[:2] == (chains, added_draws)):
            combined[name] = np.concatenate([prior, value], axis=1)
            np.testing.assert_array_equal(combined[name][:, :old_draws], prior)
        else:
            combined[name] = value
    return combined


def run(args):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, args.cpu_ids)
    from reference_matrix import environment
    attempted = 0
    for file in sorted(args.output.glob("*/result.json")):
        original = json.loads(file.read_text())
        if (original["status"] != "complete" or original["variant"] != args.variant
                or original["scenario"]["name"] != args.scenario or original["draws"] != args.from_draws):
            continue
        if getattr(args, "dataset_index", None) is not None and original["dataset"] != args.dataset_index:
            continue
        if getattr(args, "sampler_index", None) is not None and original["sampler"] != args.sampler_index:
            continue
        parent_id = original["run_id"]
        run_id = parent_id.replace(f"_k{args.from_draws}_", f"_k{args.to_draws}_")
        if run_id == parent_id:
            raise ValueError("Parent identifier does not encode the retained draw count.")
        folder = args.output / run_id
        if (folder / "result.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        record = {**original, "run_id": run_id, "draws": args.to_draws, "status": "running"}
        for key in ("windows", "ess_growth", "all_effect_checks_pass", "ess_growth_screen_pass",
                    "parameter_diagnostics", "all_parameter_checks_pass", "missing_parameter_diagnostics"):
            record.pop(key, None)
        record["continuation"] = dict(parent_run=parent_id, original_draws=args.from_draws,
            additional_draws=args.to_draws - args.from_draws, original_burnin=original["burnin"],
            additional_burnin=0, environment=environment(), exact_prefix_preserved=True,
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), Path(__file__).with_name("direct_smooth_candidate.py"))})
        write_json(folder / "result.json", record)
        started = time.monotonic()
        try:
            shutil.copy2(file.parent / "data.npz", folder / "data.npz")
            with np.load(file.parent / "data.npz", allow_pickle=False) as archive:
                data = dict(archive)
            with np.load(file.parent / "parameters.npz", allow_pickle=False) as archive:
                old = dict(archive)
            added = args.to_draws - args.from_draws
            fit_start = time.monotonic()
            _, new = fit_candidate(data, int(old["start"]), seed=original["sampler_seed"],
                chains=original["chains"], burnin=0, draws=added,
                config=ForestConfig(**original["config"]), resume=old)
            elapsed = time.monotonic() - fit_start
            parameters = concatenate_parameters(old, new, original["chains"], args.from_draws, added)
            np.savez_compressed(folder / "parameters.npz", **parameters)
            raw = {q: parameters["itt"][..., j].transpose(2, 0, 1)
                   for j, q in enumerate(("itt_y", "itt_d"))}
            raw["wald"] = np.divide(raw["itt_y"], raw["itt_d"], out=np.full_like(raw["itt_y"], np.nan),
                                     where=raw["itt_d"] != 0)
            with np.load(file.parent / "effect_draws.npz", allow_pickle=False) as archive:
                for q in raw:
                    np.testing.assert_array_equal(raw[q][..., :args.from_draws], archive[q])
            np.savez_compressed(folder / "effect_draws.npz", **raw)
            record["fit_seconds"] = original["fit_seconds"] + elapsed
            record["continuation"]["additional_fit_seconds"] = elapsed
            record.update(summarize_windows(raw, record["targets"], record["fit_seconds"]))
            record["status"] = "complete"
        except Exception as exc:
            record.update(status="failed", exception=repr(exc), traceback=traceback.format_exc())
        record["continuation"]["additional_total_seconds"] = time.monotonic() - started
        record["total_seconds"] = original["total_seconds"] + record["continuation"]["additional_total_seconds"]
        write_json(folder / "result.json", record)
        attempted += 1
        print(f"{run_id}: {record['status']} ({record['continuation']['additional_total_seconds']:.1f}s additional)", flush=True)
    print(f"Attempted {attempted} continuations; existing attempts preserved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "repair_results")
    p.add_argument("--variant", choices=("direct_smooth", "direct_blocked", "direct_joint"), default="direct_joint")
    p.add_argument("--scenario", choices=("clean_strong", "strong"), default="clean_strong")
    p.add_argument("--from-draws", type=int, default=2000)
    p.add_argument("--to-draws", type=int, default=4000)
    p.add_argument("--dataset-index", type=int)
    p.add_argument("--sampler-index", type=int)
    p.add_argument("--cpu-ids", type=int, nargs="+", default=[8, 9, 10, 11])
    args = p.parse_args()
    if args.from_draws < 8 or args.to_draws <= args.from_draws or args.to_draws % 2:
        p.error("Require from-draws>=8 and a larger even to-draws.")
    run(args)
