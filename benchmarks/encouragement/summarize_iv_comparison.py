"""Summarize the prespecified IV comparison without discarding failed fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from iv_comparison import clean_json


def number(value):
    return np.nan if value is None else float(value)


def contains(row: dict, truth: float) -> bool:
    """Check both components, including infinite endpoints and the empty set."""
    if not np.isfinite(truth):
        return False
    return any(number(row.get(lo)) <= truth <= number(row.get(hi))
               for lo, hi in (("lower", "upper"), ("lower2", "upper2")))


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return np.nan, np.nan
    z = norm.ppf(.975)
    p = successes / total
    denominator = 1 + z*z / total
    center = (p + z*z/(2*total)) / denominator
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return center-half, center+half


def paired_rmse_ratio(candidate, comparator, seed=20260910, draws=20000):
    """Resample datasets; all their horizons remain together."""
    candidate, comparator = np.asarray(candidate), np.asarray(comparator)
    valid = np.isfinite(candidate).all(axis=1) & np.isfinite(comparator).all(axis=1)
    if not valid.all() or not len(candidate):
        return dict(available=False, complete_pairs=int(valid.sum()), total_pairs=len(valid))
    a, b = np.mean(candidate**2, axis=1), np.mean(comparator**2, axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), (draws, len(a)))
    ratio = np.sqrt(a.mean()/b.mean())
    bootstrap = np.sqrt(a[indices].mean(axis=1)/b[indices].mean(axis=1))
    lo, hi = np.quantile(bootstrap, [.025, .975])
    return dict(available=True, complete_pairs=len(a), total_pairs=len(a),
                rmse_ratio=ratio, lower=lo, upper=hi,
                practical_reduction=bool(ratio <= .9), credible_advantage=bool(hi < 1))


def summarize(directory: Path):
    manifest = json.loads((directory/"manifest.json").read_text())
    expected = manifest["arguments"]["replications"]
    rows, fits = [], []
    for scenario in manifest["arguments"]["scenarios"]:
        for replicate in range(expected):
            path = directory/"runs"/f"{scenario}_{replicate:04d}.json"
            if not path.exists():
                raise ValueError(f"Missing evaluation run: {path}")
            run = json.loads(path.read_text())
            for method, result in run["methods"].items():
                completed = result["status"] == "completed"
                lookup = {(r["quantity"], r["horizon"]): r for r in result.get("rows", [])}
                passed = completed and len(lookup) == 12 and all(
                    r.get("diagnostics_passed", True) for r in lookup.values())
                fits.append(dict(scenario=scenario, replicate=replicate, method=method,
                                 completed=completed, diagnostics_passed=passed,
                                 seconds=result["seconds"]))
                for quantity, truths in run["truth"].items():
                    for horizon, value in enumerate(truths, 1):
                        truth = number(value)
                        row = lookup.get((quantity, horizon), {})
                        estimate = number(row.get("estimate"))
                        lo, hi = number(row.get("lower")), number(row.get("upper"))
                        lo2, hi2 = number(row.get("lower2")), number(row.get("upper2"))
                        width = hi-lo
                        if not np.isnan(lo2) and not np.isnan(hi2):
                            width += hi2-lo2
                        rows.append(dict(scenario=scenario, replicate=replicate, method=method,
                            quantity=quantity, horizon=horizon, truth=truth, estimate=estimate,
                            error=estimate-truth, covered=contains(row, truth), width=width,
                            available=bool(row), set_type=row.get("set_type", "unavailable"),
                            diagnostics_passed=bool(row.get("diagnostics_passed", bool(row)))))
    observations, fit_table = pd.DataFrame(rows), pd.DataFrame(fits)
    pointwise = []
    for keys, group in observations.groupby(["scenario", "method", "quantity", "horizon"], sort=False):
        defined = group[np.isfinite(group.truth)]
        total = len(defined)
        covered = int(defined.covered.sum())
        lo, hi = wilson(covered, total)
        errors = defined.error.to_numpy()
        pointwise.append(dict(zip(("scenario", "method", "quantity", "horizon"), keys)) | dict(
            datasets=len(group), defined_targets=total, available=int(defined.available.sum()),
            finite_estimates=int(np.isfinite(errors).sum()),
            rmse=float(np.sqrt(np.mean(errors**2))) if total else np.nan,
            bias=float(np.mean(errors)) if total else np.nan,
            covered=covered, coverage=covered/total if total else np.nan,
            coverage_lower=lo, coverage_upper=hi,
            median_width=float(defined.width.median()) if total else np.nan,
            width_available=int(defined.width.notna().sum()),
            width_missing=int(defined.width.isna().sum()),
            empty_sets=int((defined.set_type == "empty").sum()),
            unbounded=int(np.isinf(defined.width).sum()),
            diagnostics_passed=int(group.diagnostics_passed.sum())))
    pointwise = pd.DataFrame(pointwise)
    aggregate, comparisons, decisions = [], [], []
    for (scenario, method, quantity), group in observations.groupby(["scenario", "method", "quantity"], sort=False):
        subset = pointwise[(pointwise.scenario == scenario) & (pointwise.method == method) &
                           (pointwise.quantity == quantity)]
        selected_fits = fit_table[(fit_table.scenario == scenario) & (fit_table.method == method)]
        errors = group.loc[np.isfinite(group.truth), "error"].to_numpy()
        joint = group.groupby("replicate").covered.all()
        aggregate.append(dict(scenario=scenario, method=method, quantity=quantity,
            rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
            min_coverage=float(subset.coverage.min()), max_coverage=float(subset.coverage.max()),
            joint_coverage=float(joint.mean()) if len(errors) else np.nan,
            median_width=float(group.width.median()),
            width_available=int(group.width.notna().sum()),
            width_missing=int(group.width.isna().sum()),
            empty_sets=int((group.set_type == "empty").sum()),
            unbounded_sets=int(np.isinf(group.width).sum()),
            completed=int(selected_fits.completed.sum()), datasets=len(selected_fits),
            diagnostic_fits=int(selected_fits.diagnostics_passed.sum()),
            median_seconds=float(selected_fits.seconds.median())))
    for scenario in ("linear", "smooth", "abrupt"):
        data = observations[(observations.scenario == scenario) & (observations.quantity == "wald")]
        if data.empty:
            continue
        errors = {method: g.pivot(index="replicate", columns="horizon", values="error").to_numpy()
                  for method, g in data.groupby("method")}
        for comparator in ("linear_ancova", "spline_ancova"):
            comparisons.append(dict(scenario=scenario, comparator=comparator,
                                    **paired_rmse_ratio(errors["longbet_direct"], errors[comparator])))
        relevant = [r for r in comparisons if r["scenario"] == scenario]
        coverage = pointwise[(pointwise.scenario == scenario) & (pointwise.method == "longbet_direct")]
        diagnostics = fit_table[(fit_table.scenario == scenario) & (fit_table.method == "longbet_direct")]
        demonstrated_deficit = bool((coverage.coverage_upper < .9).any())
        all_pointwise_at_least_90 = bool((coverage.coverage >= .9).all())
        diagnostic_rate = float(diagnostics.diagnostics_passed.mean())
        accuracy = all(r.get("credible_advantage", False) and r.get("practical_reduction", False)
                       for r in relevant)
        decisions.append(dict(scenario=scenario, accuracy_criterion=accuracy,
            diagnostic_rate=diagnostic_rate, demonstrated_coverage_below_90=demonstrated_deficit,
            all_pointwise_coverage_at_least_90=all_pointwise_at_least_90,
            continuation_supported=accuracy and diagnostic_rate >= .95 and
                                   not demonstrated_deficit and all_pointwise_at_least_90))
    report = dict(protocol="comparison-protocol.md", monte_carlo_datasets_per_scenario=expected,
                  width_summary=("Median over reported sets with computable endpoint widths; "
                                 "missing and empty sets are counted separately. Undefined causal "
                                 "targets remain undefined even when a method reports a finite set."),
                  aggregate=aggregate, paired_comparisons=comparisons, decisions=decisions,
                  continuation_supported=any(d["continuation_supported"] for d in decisions))
    observations.to_csv(directory/"observations.csv", index=False)
    fit_table.to_csv(directory/"fits.csv", index=False)
    pointwise.to_csv(directory/"pointwise.csv", index=False)
    pd.DataFrame(aggregate).to_csv(directory/"aggregate.csv", index=False)
    (directory/"summary.json").write_text(json.dumps(clean_json(report), indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    print(json.dumps(clean_json(result), indent=2))
