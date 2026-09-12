"""Matched evaluation of repaired LongBet reduced forms against adjusted IV.

See comparison-protocol.md. Results retain failures, full confidence-set topology,
all pointwise outcomes, Monte Carlo diagnostics, and hashes of evaluated sources.
This is repeated fixed-parameter coverage, not Bayesian simulation-based calibration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
from importlib.metadata import version
import subprocess
import time
import traceback

import numpy as np
from scipy.stats import norm

from longbet import DirectSmoothConfig, LongBetDirectSmooth
from longbet._encourage import _wald_set, encouragement_effects

SCENARIOS = ("linear", "smooth", "abrupt", "weak", "zero", "serial_heteroskedastic")
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def generate_panel(scenario: str, seed: int, n: int = 160) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    periods, start = 6, 2
    x = rng.uniform(-1, 1, (n, 3))
    u = rng.normal(size=n)
    order = np.argsort(u + 0.6 * np.sin(np.pi * x[:, 0]))
    share = 0 if scenario == "zero" else (0.075 if scenario == "weak" else 0.55)
    n_always, n_complier = round(.15 * n), round(share * n)
    always = np.zeros(n, bool)
    complier = np.zeros(n, bool)
    always[order[-n_always:]] = True
    if n_complier:
        complier[order[-n_always-n_complier:-n_always]] = True
    d0, d1 = np.zeros((n, periods)), np.zeros((n, periods))
    d0[always, start:] = 1
    d1[always | complier, start:] = 1
    tt = np.arange(periods)
    if scenario == "linear":
        baseline = x[:, :1] * (2 + .25 * tt) + 1.5 * x[:, 1:2]
        response = np.broadcast_to(2 + .4 * x[:, :1], (n, periods)).copy()
    else:
        baseline = (2 * np.sin(np.pi * x[:, :1]) * (1 + .3 * tt)
                    + 1.5 * (x[:, 1:2]**2 - 1/3) * (1 + .2 * tt)
                    + 1.5 * (x[:, 2:3] > 0) * np.sin(tt / 2))
        response = 1.5 + .4 * tt + .7 * np.tanh(x[:, :1])
        if scenario == "abrupt":
            response = np.array([0, 0, 0, 3, 0, 3])[None, :] + .7 * np.tanh(x[:, :1])
    noise = rng.normal(size=(n, periods))
    if scenario == "serial_heteroskedastic":
        for j in range(1, periods):
            noise[:, j] = .7 * noise[:, j-1] + np.sqrt(1-.7**2) * noise[:, j]
        noise *= (.5 + 1.5 * np.abs(x[:, :1]))
    baseline += 3 * u[:, None] + .5 * rng.normal(size=(n, 1)) + .25 * tt + noise
    y0, y1 = baseline + response * d0, baseline + response * d1
    # Fixed potential outcomes precede complete randomization.
    assignment = rng.permutation(np.repeat([0, 1], n // 2)).astype(bool)
    z = assignment[:, None] * (tt >= start)[None, :]
    d = np.where(assignment[:, None], d1, d0)
    y = np.where(assignment[:, None], y1, y0)
    truth_y = (y1-y0).mean(axis=0)[start:]
    truth_d = (d1-d0).mean(axis=0)[start:]
    truth_ratio = np.divide(truth_y, truth_d, out=np.full_like(truth_y, np.nan), where=truth_d > 0)
    return dict(y=y, d=d, z=z.astype(float), x=x, t=tt+1, assignment=assignment,
                start=start, truth=dict(itt_y=truth_y, itt_d=truth_d, wald=truth_ratio))


def feature_matrix(data: dict, spline: bool = False) -> np.ndarray:
    x = data["x"]
    baseline = data["y"][:, :data["start"]].mean(axis=1)
    baseline = (baseline - baseline.mean()) / baseline.std()
    columns = [np.ones(len(x)), baseline]
    for column in x.T:
        columns.append(column)
        if spline:
            columns.extend([column**2, column**3])
            columns.extend(np.maximum(column-k, 0)**3 for k in (-.5, 0, .5))
    return np.column_stack(columns)


def adjusted_contrasts(data: dict, spline: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Arm-specific standardized regressions and joint HC2 covariance.

    The influence weights are the exact linear-regression standardization weights.
    Separate arms preserve treatment/covariate interactions. Unit observations are
    independent within each horizon; outcome and adoption residuals remain paired.
    """
    design = feature_matrix(data, spline=spline)
    target = design.mean(axis=0)
    response = np.stack([data["y"], data["d"]], axis=-1)[:, data["start"]:]
    estimate = np.zeros(response.shape[1:])
    covariance = np.zeros((response.shape[1], 2, 2))
    for arm, sign in ((False, -1), (True, 1)):
        idx = data["assignment"] == arm
        a = design[idx]
        if np.linalg.matrix_rank(a) != a.shape[1] or len(a) <= a.shape[1] + 2:
            raise ValueError("Adjusted comparator design is underidentified.")
        inverse = np.linalg.pinv(a)
        weights = target @ inverse
        leverage = np.einsum("ij,ji->i", a, inverse)
        observed = response[idx]
        coefficients = np.einsum("qn,nhk->qhk", inverse, observed)
        residual = observed - np.einsum("nq,qhk->nhk", a, coefficients)
        estimate += sign * np.einsum("n,nhk->hk", weights, observed)
        weighted = residual * (weights / np.sqrt(1-leverage))[:, None, None]
        covariance += np.einsum("nhj,nhk->hjk", weighted, weighted)
    return estimate, covariance


def reference_rows(estimate: np.ndarray, cov: np.ndarray) -> list[dict]:
    rows = []
    critical = norm.ppf(.975)
    for h, (effect, variance) in enumerate(zip(estimate, cov), 1):
        for k, name in enumerate(("itt_y", "itt_d")):
            se = float(np.sqrt(max(variance[k, k], 0)))
            rows.append(dict(horizon=h, quantity=name, estimate=effect[k],
                             lower=effect[k]-critical*se, upper=effect[k]+critical*se,
                             set_type="bounded", lower2=np.nan, upper2=np.nan))
        result = _wald_set(effect[0], effect[1], variance[0, 0], variance[1, 1], variance[0, 1], critical)
        rows.append(dict(horizon=h, quantity="wald",
                         estimate=effect[0]/effect[1] if effect[1] != 0 else np.nan,
                         lower=result["wald_lower_1"], upper=result["wald_upper_1"],
                         lower2=result["wald_lower_2"], upper2=result["wald_upper_2"],
                         set_type=result["wald_set_type"]))
    return rows


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj): return None
        if np.isposinf(obj): return "Infinity"
        if np.isneginf(obj): return "-Infinity"
        return float(obj)
    return obj


def run_one(task: dict) -> str:
    path = Path(task["output"]) / "runs" / f'{task["scenario"]}_{task["replicate"]:04d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_panel(task["scenario"], task["seed"], n=task["n"])
    record = {**task, "truth": data["truth"], "methods": {}}
    for method in ("unadjusted", "linear_ancova", "spline_ancova", "longbet_direct"):
        started = time.perf_counter()
        try:
            if method == "unadjusted":
                ref = encouragement_effects(data["y"], data["d"], data["z"], data["t"])
                estimate = ref[["itt_y", "itt_d"]].to_numpy()
                cov = np.zeros((len(ref), 2, 2))
                cov[:, 0, 0] = ref.itt_y_se**2
                cov[:, 1, 1] = ref.itt_d_se**2
                cov[:, 0, 1] = cov[:, 1, 0] = ref.itt_y_d_cov
                rows = reference_rows(estimate, cov)
            elif method.endswith("ancova"):
                rows = reference_rows(*adjusted_contrasts(data, spline=method == "spline_ancova"))
            else:
                config = DirectSmoothConfig(baseline_trees=4, effect_trees=4,
                    correlated_intercepts=task.get("correlated_intercepts", False))
                fit = LongBetDirectSmooth(config).fit(
                    data["y"], data["d"], data["z"], data["x"], data["t"],
                    chains=4, burnin=task["burnin"], draws=task["draws"], seed=task["seed"]+10000000)
                pred = fit.predict()
                table = pred.effects()
                rows = []
                for row in table.itertuples():
                    rows.append(dict(horizon=row.horizon, quantity=row.quantity,
                                     estimate=row.posterior_median, lower=row.posterior_lower,
                                     upper=row.posterior_upper, lower2=np.nan, upper2=np.nan,
                                     set_type="posterior_equal_tail", ess_bulk=row.ess_bulk,
                                     ess_tail=row.ess_tail, rhat=row.rhat,
                                     diagnostics_passed=row.diagnostics_passed,
                                     posterior_status=row.posterior_status))
                record["longbet_config"] = asdict(config)
            record["methods"][method] = dict(status="completed", rows=rows)
        except Exception as exc:
            record["methods"][method] = dict(status="failed", error=str(exc), traceback=traceback.format_exc())
        record["methods"][method]["seconds"] = time.perf_counter()-started
    path.write_text(json.dumps(clean_json(record), indent=2)+"\n")
    return f'{task["scenario"]}/{task["replicate"]} '+", ".join(
        f'{k}: {v["status"]} ({v["seconds"]:.1f}s)' for k, v in record["methods"].items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=2026091100)
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--correlated-intercepts", action="store_true",
                        help="Separate covariance-model sensitivity analysis; not the primary model.")
    args = parser.parse_args()
    if args.n < 100 or args.n % 2 or min(args.replications, args.draws, args.workers) < 1:
        parser.error("Require even n >= 100 and positive replication/draw/worker counts.")
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(__file__).with_name("comparison-protocol.md"),
                    *sorted((SOURCE_ROOT/"src"/"longbet").glob("*.py"))]
    hashes = {str(p.relative_to(SOURCE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    manifest = dict(arguments={**vars(args), "output": str(args.output)}, source_sha256=hashes,
                    python=platform.python_version(), numpy=np.__version__,
                    packages={name: version(name) for name in ("scipy", "jax", "arviz", "bartz", "pandas")},
                    git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip())
    manifest_path = args.output/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output manifest differs; use a fresh output directory to avoid mixing sources/configurations.")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    tasks=[]
    for scenario in args.scenarios:
        for replicate in range(args.replications):
            path=args.output/"runs"/f"{scenario}_{replicate:04d}.json"
            if not path.exists():
                tasks.append(dict(scenario=scenario, replicate=replicate,
                                  seed=args.seed+100000*SCENARIOS.index(scenario)+replicate,
                                  n=args.n, burnin=args.burnin, draws=args.draws, output=str(args.output),
                                  correlated_intercepts=args.correlated_intercepts))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(run_one,t) for t in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
