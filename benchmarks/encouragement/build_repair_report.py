"""Replay effect/parameter diagnostics and publish the bounded repair results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from repair_comparison import QUANTITIES, summarize_windows, write_json


def parameter_diagnostics(parameters):
    from longbet._diagnostics import compute_ess, compute_rhat
    result = {}
    parameters = dict(parameters)
    if "rule_feature" in parameters:
        features = np.unique(parameters["rule_feature"])
        for forest in ("baseline", "effect"):
            feature = parameters["rule_feature"][parameters[f"{forest}_rules"]]
            parameters[f"{forest}_rule_usage"] = np.stack([(feature == f).sum(axis=-1) for f in features], axis=-1)
    for name in ("sigma2", "gamma2", "gamma", "baseline_probe", "effect_probe", "baseline_rule_usage", "effect_rule_usage"):
        if name not in parameters:
            continue
        raw = np.asarray(parameters[name])
        flat = raw.reshape(*raw.shape[:2], -1)
        stats = []
        for k in range(flat.shape[-1]):
            value = flat[..., k]
            stats.append(dict(index=list(np.unravel_index(k, raw.shape[2:])),
                rhat=float(compute_rhat(value)),
                ess_bulk=float(compute_ess(value, method="bulk")),
                ess_tail=float(compute_ess(value, method="tail", prob=.025))))
        result[name] = {"shape_after_chain_draw": raw.shape[2:], "cells": stats,
                       "max_rhat": max(s["rhat"] for s in stats),
                       "min_bulk_ess": min(s["ess_bulk"] for s in stats),
                       "min_tail_ess": min(s["ess_tail"] for s in stats),
                       "checks_pass": all(s["rhat"] <= 1.01 and s["ess_bulk"] >= 400
                                          and s["ess_tail"] >= 400 for s in stats)}
    return result


def number(value, decimals=0):
    return "—" if value is None else f"{value:.{decimals}f}"


def extreme(values, kind):
    values = np.asarray(values, dtype=float)
    return float(getattr(np, kind)(values)) if np.isfinite(values).all() else None


def run(output):
    records = []
    for file in sorted(output.glob("*/result.json")):
        record = json.loads(file.read_text())
        if record["status"] == "complete":
            with np.load(file.parent / "effect_draws.npz", allow_pickle=False) as archive:
                raw = dict(archive)
            record.update(summarize_windows(raw, record["targets"], record["fit_seconds"]))
            path = file.parent / "parameters.npz"
            record["parameter_diagnostics"] = {}
            if path.exists():
                with np.load(path, allow_pickle=False) as parameters:
                    record["parameter_diagnostics"] = parameter_diagnostics(parameters)
            elif record["variant"].startswith("current"):
                from longbet import LongBetMulti
                model = LongBetMulti.load(file.parent / "model.npz")
                parameters = {name: np.stack([np.asarray(getattr(t, attr))
                    for t in model.trace.traces], axis=-1)
                    for name, attr in (("sigma2", "sigma2"), ("gamma2", "sigma_gamma2"), ("gamma", "gamma"))}
                record["parameter_diagnostics"] = parameter_diagnostics(parameters)
            required = {"sigma2", "gamma2", "gamma"}
            if not record["variant"].startswith("current"):
                required.update(("baseline_probe", "effect_probe"))
            if record["variant"].startswith("direct"):
                required.update(("baseline_rule_usage", "effect_rule_usage"))
            record["missing_parameter_diagnostics"] = sorted(required - record["parameter_diagnostics"].keys())
            record["all_parameter_checks_pass"] = (not record["missing_parameter_diagnostics"] and all(
                item["checks_pass"] for item in record["parameter_diagnostics"].values()))
            write_json(file, record)
        records.append(record)

    groups = {}
    for r in records:
        identity = json.dumps({"scenario": r["scenario"], "config": r.get("config", {})}, sort_keys=True)
        config_id = hashlib.sha256(identity.encode()).hexdigest()[:12]
        key = (r["scenario"]["name"], r["n"], r["chains"], r["burnin"], r["draws"], r["variant"], config_id)
        groups.setdefault(key, []).append(r)
    summary = []
    seed_agreement = []
    for key, attempts in groups.items():
        runs = [r for r in attempts if r["status"] == "complete"]
        if not runs:
            summary.append(dict(scenario=key[0], n=key[1], chains=key[2], burnin=key[3], draws=key[4], variant=key[5], config_id=key[6],
                runs=0, datasets=0, pending=sum(r["status"] == "running" for r in attempts),
                failed=sum(r["status"] == "failed" for r in attempts), effect_passes=0, parameter_passes=0,
                growth_passes=0, **{name: None for name in ("max_rhat", "min_bulk_ess", "min_tail_ess",
                    "min_ess_growth", "median_ess_growth", "min_bulk_ess_per_second", "median_fit_seconds")}))
            continue
        diagnostics = [d for r in runs for q in QUANTITIES
                       for d in r["windows"]["full"]["quantities"][q]["diagnostics"]]
        growth = [v for r in runs for q in QUANTITIES for kind in ("ess_bulk", "ess_tail")
                  for v in r["ess_growth"][q][kind]]
        for dataset in sorted({r["data_seed"] for r in runs}):
            replicas = sorted([r for r in runs if r["data_seed"] == dataset], key=lambda r: r["sampler"])
            if len(replicas) != 2:
                continue
            for q in QUANTITIES:
                left, right = [r["windows"]["full"]["quantities"][q] for r in replicas]
                center, mcse = ("median", "mcse_median") if q == "wald" else ("mean", "mcse_mean")
                error = np.asarray(left[center], dtype=float) - np.asarray(right[center], dtype=float)
                se = np.sqrt(np.array([d[mcse] for d in left["diagnostics"]], dtype=float)**2
                             + np.array([d[mcse] for d in right["diagnostics"]], dtype=float)**2)
                seed_agreement.append(dict(scenario=key[0], variant=key[5], data_seed=dataset,
                    n=key[1], chains=key[2], burnin=key[3], draws=key[4], config_id=key[6],
                    quantity=q, center=center, difference=error, combined_mcse=se,
                    available=np.isfinite(error) & np.isfinite(se) & (se > 0),
                    difference_in_mcse=np.divide(error, se, out=np.full_like(error, np.nan), where=se > 0)))
        summary.append(dict(scenario=key[0], n=key[1], chains=key[2], burnin=key[3], draws=key[4], variant=key[5], config_id=key[6],
            runs=len(runs), datasets=len({r["data_seed"] for r in runs}),
            pending=sum(r["status"] == "running" for r in attempts),
            failed=sum(r["status"] == "failed" for r in attempts),
            effect_passes=sum(r["all_effect_checks_pass"] for r in runs),
            parameter_passes=sum(r.get("all_parameter_checks_pass", False) for r in runs),
            growth_passes=sum(r["ess_growth_screen_pass"] for r in runs),
            max_rhat=extreme([d["rhat"] for d in diagnostics], "max"),
            min_bulk_ess=extreme([d["ess_bulk"] for d in diagnostics], "min"),
            min_tail_ess=extreme([d["ess_tail"] for d in diagnostics], "min"),
            min_ess_growth=extreme(growth, "min"), median_ess_growth=extreme(growth, "median"),
            min_bulk_ess_per_second=extreme([d["bulk_ess_per_fit_second"] for d in diagnostics], "min"),
            median_fit_seconds=float(np.median([r["fit_seconds"] for r in runs]))))
    write_json(output / "summary.json", summary)
    write_json(output / "seed-agreement.json", seed_agreement)
    passing = [s for s in summary if s["runs"] >= 4 and s["datasets"] >= 2
               and s["effect_passes"] == s["parameter_passes"] == s["growth_passes"] == s["runs"]
               and s["failed"] == s["pending"] == 0]
    lines = ["# Direct encouragement model repair comparison", "",
        "These results evaluate mixing on a bounded experiment. They do not establish",
        "frequentist calibration or authorize a production default change. See the",
        "[protocol](../../docs/encouragement-repair-protocol.md).", ""]
    if passing:
        lines += ["The following completed settings pass every effect, parameter and ESS-growth",
                  "screen in all four runs:", ""]
        lines += [f"- `{s['variant']}`, `{s['scenario']}`, {s['draws']:,} retained draws per chain."
                  for s in passing]
        lines += ["", "Passing these screens supports further calibration work for the named",
            "settings. The original forest, baseline-only block, and shorter-chain",
            "failures remain below.", ""]
    lines += [
        "Every run uses four dispersed chains and 1,000 warmup sweeps. Retained",
        "draw counts are shown per chain; extensions preserve their shorter prefixes.",
        "There are two independent datasets and two sampler seeds per dataset in each",
        "completed setting. Both ITTs and the raw Wald ratio are checked at all four",
        "post-encouragement horizons in six-period panels with 80 study units. The",
        "tail ESS refers to the 2.5% and 97.5% quantiles. Failed runs and",
        "diagnostic failures are retained; repeated seeds do not count as new datasets.", "",
        "| Scenario | Model | Draws | Effect passes | Failed / pending | Max R-hat | Min bulk ESS | Min tail ESS | Median fit seconds | Min bulk ESS/sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['scenario']} | {s['variant']} | {s['draws']} | {s['effect_passes']}/{s['runs']} | "
            f"{s['failed']} / {s['pending']} | {number(s['max_rhat'], 4)} | {number(s['min_bulk_ess'])} | {number(s['min_tail_ess'])} | "
            f"{number(s['median_fit_seconds'], 1)} | {number(s['min_bulk_ess_per_second'], 1)} |")
    lines += ["", "## Other posterior diagnostics and longer windows", "",
        "The first half of retained draws is compared with its enclosing full",
        "window. The 1.5-fold ESS growth flag is a noisy screening criterion. Parameter",
        "checks include every unit intercept, both equation variances, and, for the",
        "new candidates, aggregate baseline/effect functions at three fixed original",
        "covariate rows. Direct-forest checks also include counts of root trees and",
        "each split feature across the forest, which do not depend on tree labels.",
        "Individual labeled tree components are archived for audit;",
        "exchangeability means their identities are not substantive estimands.", "",
        "| Scenario | Model | Draws | Parameter passes | Growth passes | Min ESS growth | Median ESS growth |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['scenario']} | {s['variant']} | {s['draws']} | {s['parameter_passes']}/{s['runs']} | "
            f"{s['growth_passes']}/{s['runs']} | {number(s['min_ess_growth'], 2)} | {number(s['median_ess_growth'], 2)} |")
    lines += ["", "`seed-agreement.json` compares independent sampler seeds on each fixed",
        "dataset using ITT posterior means and Wald medians, with their combined",
        "Monte Carlo SEs. These checks are meaningful only for adequately mixed",
        "draws; they do not turn two datasets into four calibration replications."]
    lines += ["", "## Interpretation and limitations", "",
        "The current-stump ablation limits tree depth while retaining the current",
        "product/coding model. New and old models share variance hyperparameters,",
        "but their function priors and topology updates differ. These experiments",
        "therefore compare complete modeling/sampling choices; they cannot isolate a",
        "single causal explanation for a performance difference. New forests contain",
        "sampled stumps with smooth time-vector leaves and cannot express general",
        "covariate interactions. The parametric model has linear baseline covariates.", "",
        "Matching average pointwise function variance does not match the prior",
        "variance of an all-unit mean contrast, which also depends on covariance",
        "across units. The unblocked and blocked direct forests do share exactly the",
        "same statistical model: their differences are exact joint Gaussian updates",
        "of baseline leaves (`direct_blocked`) or both forests' leaves (`direct_joint`)",
        "and unit intercepts given the sampled topologies.", "",
        "All models here use independent Gaussian equations, including an LPM working",
        "likelihood for binary uptake. They do not model a correlated latent binary",
        "system or an absorbing adoption hazard. Better mixing does not repair",
        "likelihood misspecification, establish interval coverage, or identify a ratio",
        "when its population denominator is zero. No probability-range restriction or",
        "denominator clipping has been introduced.", "",
        "Fit timings include JAX compilation when incurred; later fits in a process",
        "may reuse cached compilation. Candidate fit timing includes integrated",
        "effect/probe evaluation and construction of parameter arrays; old-model",
        "timing excludes its separate standardization step. All fit timings exclude",
        "NPZ serialization and post-fit diagnostic calculations. Total wall times",
        "are also retained. Backend and bookkeeping differences limit comparisons. Raw",
        "draws, full candidate states, current-model archives, source hashes, exact",
        "seeds, configurations, truths and warnings accompany each result locally.",
        "Compact JSON files and this report are reproducible from the scripts.", "",
        "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/encouragement/repair_comparison.py",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/encouragement/extend_repair.py",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/encouragement/extend_repair.py --from-draws 4000 --to-draws 8000",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/encouragement/repair_comparison.py --variants parametric --scenario strong",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/encouragement/repair_comparison.py --variants direct_joint --scenario strong --draws 8000",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/encouragement/build_repair_report.py",
        "```", ""]
    (output.parent / "repair-report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "repair_results")
    args = parser.parse_args()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, [8, 9, 10, 11])
    run(args.output)
