"""Independently diagnose complete chapter runs, including joint decision events.

Uses ArviZ directly on saved chain-major draws. This is a convergence/recovery
audit of one fixed dataset, not a coverage or calibration study.

python benchmarks/summarize_multi_runs.py --data chapter-input.npz \
    --output comparison.json --plot traces.png run-directory [run-directory ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import arviz as az
import numpy as np


def diagnose(values):
    values = np.asarray(values, dtype=np.float64)
    result = {"mean": float(values.mean()),
              "interval": np.quantile(values, [.025, .975]).tolist(),
              "chain_means": values.mean(-1).tolist(),
              "constant": bool(np.all(values == values.flat[0]))}
    for name, method in (("ess_bulk", "bulk"), ("ess_tail", "tail"), ("ess_mean", "mean")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = az.ess(values, method=method,
                           **({"prob": (.025, .975)} if method == "tail" else {}))
        result[name] = float(value) if np.isfinite(value) and not result["constant"] else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, value in (("rhat", az.rhat(values)), ("mcse_mean", az.mcse(values, method="mean"))):
            result[name] = float(value) if np.isfinite(value) and not result["constant"] else None
    result["passes"] = bool(result["rhat"] is not None and result["rhat"] <= 1.01
                             and all(result[k] is not None and result[k] >= 400
                                     for k in ("ess_bulk", "ess_tail")))
    return result


def audit_run(directory, data):
    report = json.loads((directory / "report.json").read_text())
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    if report["mode"] != "multi" or report["data_sha256"] != digest:
        raise ValueError(f"{directory}: expected a joint fit on the specified data")
    settings = report["settings"]
    chains, retained = settings["num_chains"], settings["num_sweeps"]
    if chains < 4:
        raise ValueError("At least four original chains are required for this audit")
    with np.load(data, allow_pickle=False) as source:
        n = source["x"].shape[0]
        groups = {"good": (source["listings"] >= 80) & (source["fulfilled"] == 1),
                  "bad": (source["listings"] <= 25) & (source["fulfilled"] == 0)}
        truths = {name: np.asarray(source["truth_" + name], dtype=float)
                  for name in ("gmv", "hours", "complaint")}
    with np.load(directory / "draws.npz", allow_pickle=False) as archive:
        effects = {name: np.asarray(archive[name], dtype=float) for name in truths}
    if not all(value.shape == (n, chains * retained) and np.isfinite(value).all()
               for value in effects.values()):
        raise ValueError(f"{directory}: invalid or incomplete effect arrays")
    results = {"directory": str(directory.resolve()), "data_sha256": digest,
               "python_source_sha256": report["python_source_sha256"],
               "settings": settings, "seed": report["base_seed"],
               "experimental_settings": {key: value for key, value in report.items()
                   if key.startswith("experimental_") or key in
                   ("collapsed_every", "collapsed_capacity", "collapsed_scale", "collapsed_proposal")},
               "fit_seconds": report["fit_seconds"], "effects": {}, "joint_events": {}}
    traces = []
    for outcome, effect in effects.items():
        truth = truths[outcome] if outcome == "complaint" else np.expm1(truths[outcome])
        for label, mask in groups.items():
            values = effect[mask].mean(0).reshape(chains, retained)
            target = float(truth[mask].mean())
            summary = diagnose(values)
            summary.update(truth=target, n=int(mask.sum()),
                           p_correct_sign=float(np.mean(values * np.sign(target) > 0)),
                           last_half=diagnose(values[:, retained // 2:]))
            if outcome != "complaint":
                summary["log_effect"] = diagnose(np.log1p(effect[mask]).mean(0).reshape(chains, retained))
            results["effects"][f"{outcome}/{label}"] = summary
            traces.append((f"{outcome}/{label}", values, target))
    # Sellers jointly satisfying thresholds within each aligned draw.
    event = ((effects["gmv"] >= .05) & (effects["hours"] <= -.1)
             & (effects["complaint"] <= .01))
    for label, mask in groups.items():
        results["joint_events"][f"seller_share/{label}"] = diagnose(
            event[mask].mean(0).reshape(chains, retained))
        # A distinct estimand: do the group's average effects satisfy thresholds?
        means = {key: value[mask].mean(0) for key, value in effects.items()}
        group_event = ((means["gmv"] >= .05) & (means["hours"] <= -.1)
                       & (means["complaint"] <= .01))
        results["joint_events"][f"group_mean_event/{label}"] = diagnose(
            group_event.reshape(chains, retained))
    results["effect_gates_passed"] = sum(v["passes"] for v in results["effects"].values())
    return results, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    results, traces = {}, []
    for directory in args.runs:
        result, trace = audit_run(directory, args.data)
        results[str(directory)] = result
        traces.append((directory.name, trace))
    payload = {"qualification": "Fixed-data convergence/recovery only; failed gates invalidate inference. "
               "Constant decision indicators do not establish convergence. Runtimes are contended.",
               "tail_probabilities": [.025, .975], "runs": results}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(6, len(traces), figsize=(6 * len(traces), 12), squeeze=False)
        for column, (label, rows) in enumerate(traces):
            for row, (name, values, truth) in enumerate(rows):
                ax = axes[row, column]
                for chain, sequence in enumerate(values):
                    ax.plot(sequence, lw=.7, alpha=.75, label=f"chain {chain+1}")
                ax.axhline(truth, color="black", ls="--", lw=1, label="truth")
                ax.set_title(f"{label}: {name}", fontsize=9)
                ax.set_xlabel("Retained draw within chain")
        axes[0, 0].legend(fontsize=7, ncol=3)
        fig.suptitle("Observed sampler output — inspect diagnostics before inference")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
