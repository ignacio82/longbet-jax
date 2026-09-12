"""Matched cross-fitted encouragement-IV comparison.

The statistical protocol lives in orthogonal-protocol.md.  All nuisance learners
receive identical account features, folds, inner validation observations, and
reduced-form inference.  No learner sees potential-outcome truths while fitting.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from iv_comparison import (
    SCENARIOS as SHORT_SCENARIOS, SOURCE_ROOT, adjusted_contrasts, clean_json,
    generate_panel as generate_short_panel, reference_rows,
)

SCENARIOS = (*SHORT_SCENARIOS, "long_smooth", "long_abrupt")
CORRECTED_METHODS = ("crossfit_spline", "crossfit_extratrees", "crossfit_longbet")
REFERENCE_METHODS = ("unadjusted", "linear_ancova", "spline_ancova")
NULL_MOMENT_BETAS = (-2., 0., 2., 10.)


def generate_panel(scenario: str, seed: int, n: int | None = None) -> dict:
    """Fixed potential outcomes, then balanced complete randomization.

    The six original designs are unchanged. Long-panel outcomes depend on
    *current* adoption, so the time-specific Wald target has a CACE interpretation
    under the deliberately satisfied exclusion and monotonicity assumptions.
    The design does not claim to identify exposure-duration effects.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    n = (240 if scenario.startswith("long_") else 160) if n is None else n
    if n < 40 or n % 2:
        raise ValueError("Require an even number of at least 40 accounts.")
    if scenario in SHORT_SCENARIOS:
        return generate_short_panel(scenario, seed, n=n)
    rng = np.random.default_rng(seed)
    start, horizons = 6, 16
    periods = start + horizons
    x = rng.uniform(-1, 1, (n, 3))
    u = rng.normal(size=n)
    readiness = u + .7*np.sin(np.pi*x[:, 0]*x[:, 1]) + .4*x[:, 2]
    order = np.argsort(readiness)
    n_always, n_accelerated, n_encouraged = round(.2*n), round(.45*n), round(.25*n)
    always = order[-n_always:]
    accelerated = order[-n_always-n_accelerated:-n_always]
    encouraged = order[-n_always-n_accelerated-n_encouraged:-n_always-n_accelerated]
    a0 = np.full(n, horizons + 100, dtype=int)
    a1 = a0.copy()
    a0[always] = 1 + np.floor(12*expit(-readiness[always] + rng.normal(size=n_always))).astype(int)
    a1[always] = a0[always]
    a0[accelerated] = 7 + np.floor(13*expit(-.4*readiness[accelerated]
        + .5*x[accelerated, 0]*x[accelerated, 2] + rng.normal(size=n_accelerated))).astype(int)
    a1[accelerated] = 1 + np.floor((a0[accelerated]-2)*expit(-1
        + .7*x[accelerated, 0]*x[accelerated, 1] - .3*u[accelerated]
        + rng.normal(size=n_accelerated))).astype(int)
    a1[encouraged] = 1
    h = np.arange(1-start, horizons+1)
    d0 = (h[None, :] >= a0[:, None]).astype(float)
    d1 = (h[None, :] >= a1[:, None]).astype(float)
    tt = np.arange(periods)
    scaled_time = tt / (periods-1)
    interaction = np.sin(np.pi*x[:, :1]*x[:, 1:2])
    baseline = (2*np.sin(np.pi*x[:, :1])*(1 + scaled_time)
        + 2*interaction*(.5 + 2*scaled_time)
        + 1.5*(x[:, 1:2]**2 - 1/3)*(1 + .5*np.sin(tt/4))
        + 1.2*(x[:, 2:3] > 0)*np.cos(tt/3)
        + 3*u[:, None] + .15*tt)
    response = 2 + .08*np.maximum(h, 0) + .6*np.sin(np.maximum(h, 0)/4) + .7*interaction
    if scenario == "long_abrupt":
        blocks = np.array([1., 4., .5, 3.])
        effect_time = blocks[np.minimum(np.maximum(h-1, 0)//4, 3)]
        response = effect_time[None, :] + .7*interaction
        baseline += 1.5*(x[:, :1]*x[:, 1:2] > .15)*((h >= 5) - .7*(h >= 12))
    noise = rng.normal(size=(n, periods))
    for j in range(1, periods):
        noise[:, j] = .6*noise[:, j-1] + .8*noise[:, j]
    baseline += noise*(.6 + .6*np.abs(x[:, 2:3]))
    y0, y1 = baseline + response*d0, baseline + response*d1
    assignment = rng.permutation(np.repeat([0, 1], n//2)).astype(bool)
    z = assignment[:, None]*(h >= 1)[None, :]
    y = np.where(assignment[:, None], y1, y0)
    d = np.where(assignment[:, None], d1, d0)
    truth_y, truth_d = (y1-y0).mean(axis=0)[start:], (d1-d0).mean(axis=0)[start:]
    truth_ratio = np.divide(truth_y, truth_d, out=np.full_like(truth_y, np.nan), where=truth_d > 0)
    return dict(y=y, d=d, z=z.astype(float), x=x, t=tt+1, assignment=assignment,
        start=start, truth=dict(itt_y=truth_y, itt_d=truth_d, wald=truth_ratio),
        potential_y0=y0, potential_y1=y1, potential_d0=d0, potential_d1=d1,
        adoption_time0=a0, adoption_time1=a1)


def baseline_features(data: dict) -> np.ndarray:
    """Identical pretreatment information for every corrected learner."""
    from longbet._iv import baseline_features as shared_baseline_features
    return shared_baseline_features(data["y"], data["d"], data["z"], data["x"], data["t"])


class _SplineFit:
    """Arm-specific ridge GAM with covariate-by-time spline interactions."""
    def __init__(self, penalty: float):
        self.penalty = penalty

    def fit(self, x, z, responses, times):
        from sklearn.preprocessing import SplineTransformer, StandardScaler
        self.h = len(times)
        self.nonconstant = np.std(x, axis=0) > 1e-10
        self.scaler = StandardScaler().fit(x[:, self.nonconstant])
        xs = self.scaler.transform(x[:, self.nonconstant])
        self.x_spline = SplineTransformer(n_knots=3, degree=3, include_bias=False,
            knots="quantile", extrapolation="linear").fit(xs)
        self.time_spline = SplineTransformer(n_knots=min(4, self.h), degree=3,
            include_bias=False, extrapolation="linear").fit(np.asarray(times)[:, None])
        self.time_basis = self.time_spline.transform(np.asarray(times)[:, None])
        design = self._design(x)
        self.center = design.mean(axis=0)
        self.scale = design.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        design = (design-self.center)/self.scale
        self.response_center = responses.mean(axis=(0, 1))
        self.response_scale = np.maximum(responses.std(axis=(0, 1)), [1e-8, .05])
        normalized = ((responses-self.response_center)/self.response_scale).reshape(-1, 2)
        self.coefficients, self.intercepts = [], []
        for arm in (0, 1):
            keep = np.repeat(z == arm, self.h)
            a, b = design[keep], normalized[keep]
            mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
            ac, bc = a-mean_a, b-mean_b
            # Averaged-loss parameterization makes the penalty comparable across
            # panel lengths, fold sizes, and arms.
            gram = ac.T@ac/len(ac)
            gram.flat[::len(gram)+1] += self.penalty
            coef = np.linalg.solve(gram, ac.T@bc/len(ac))
            self.coefficients.append(coef)
            self.intercepts.append(mean_b-mean_a@coef)
        return self

    def _design(self, x):
        xb = self.x_spline.transform(self.scaler.transform(x[:, self.nonconstant]))
        tb = self.time_basis
        interactions = (xb[:, None, :, None]*tb[None, :, None, :]).reshape(len(x)*self.h, -1)
        return np.column_stack((np.repeat(xb, self.h, axis=0),
            np.tile(tb, (len(x), 1)), interactions))

    def predict(self, x):
        design = (self._design(x)-self.center)/self.scale
        result = np.stack([(design@coef+intercept).reshape(len(x), self.h, 2)
            for coef, intercept in zip(self.coefficients, self.intercepts)], axis=2)
        result = result*self.response_scale+self.response_center
        result[..., 1] = np.clip(result[..., 1], 0, 1)
        return result


class _ExtraTreesFit:
    """Separate-arm multivariate forests share partitions over all trajectories."""
    def __init__(self, min_samples_leaf: int, *, seed: int, trees: int = 128):
        self.min_samples_leaf, self.seed, self.trees = min_samples_leaf, seed, trees

    def fit(self, x, z, responses, times):
        from sklearn.ensemble import ExtraTreesRegressor
        self.h = len(times)
        self.center = responses.mean(axis=(0, 1))
        self.scale = np.maximum(responses.std(axis=(0, 1)), [1e-8, .05])
        normalized = ((responses-self.center)/self.scale).reshape(len(x), -1)
        self.models = []
        for arm in (0, 1):
            model = ExtraTreesRegressor(n_estimators=self.trees,
                min_samples_leaf=self.min_samples_leaf, max_features=1.,
                random_state=self.seed+arm, n_jobs=1)
            model.fit(x[z == arm], normalized[z == arm])
            self.models.append(model)
        return self

    def predict(self, x):
        result = np.stack([model.predict(x).reshape(len(x), self.h, 2)
            for model in self.models], axis=2)*self.scale+self.center
        result[..., 1] = np.clip(result[..., 1], 0, 1)
        return result


class TunedComparator:
    """Three candidates, one account holdout, followed by an outer-train refit."""
    def __init__(self, kind: str, *, seed: int = 0, trees: int = 128):
        if kind not in ("spline", "extratrees"):
            raise ValueError(kind)
        self.kind, self.seed, self.trees = kind, seed, trees

    def fit(self, x, z, responses, times):
        from longbet._iv_nuisance import inner_validation_split, nuisance_validation_score
        train, valid = inner_validation_split(z, seed=self.seed)
        candidates = (.01, .1, 1.) if self.kind == "spline" else (2, 5, 10)
        scores = []
        for candidate in candidates:
            model = self._make(candidate).fit(x[train], z[train], responses[train], times)
            scores.append(nuisance_validation_score(model.predict(x[valid]),
                z[valid], responses[valid], responses[train]))
        chosen = int(np.argmin(scores))
        self.model = self._make(candidates[chosen]).fit(x, z, responses, times)
        self.metadata_ = dict(learner=self.kind, candidate_values=candidates,
            validation_scores=scores, selected_candidate=candidates[chosen],
            inner_holdout_sha256=hashlib.sha256(np.asarray(valid, dtype="<i8").tobytes()).hexdigest(),
            inner_training_count=len(train), inner_validation_count=len(valid),
            training_count=len(x), seed=self.seed, trees=self.trees if self.kind == "extratrees" else None)
        return self

    def _make(self, candidate):
        return (_SplineFit(candidate) if self.kind == "spline" else
            _ExtraTreesFit(candidate, seed=self.seed, trees=self.trees))

    def predict(self, x):
        return self.model.predict(x)


def learner_factory(method: str, *, seed: int, config: dict):
    if method == "crossfit_longbet":
        from longbet._iv_nuisance import LongBetIVNuisance
        return lambda: LongBetIVNuisance(seed=seed, **config.get("longbet", {}))
    if method in ("crossfit_spline", "crossfit_extratrees"):
        kind = method.removeprefix("crossfit_")
        return lambda: TunedComparator(kind, seed=seed, trees=config.get("trees", 128))
    raise ValueError(method)


def reference_null_tests(estimates: np.ndarray, covariance: np.ndarray) -> list[dict]:
    """Same asymptotic average-moment nulls for the classical references."""
    tests = []
    for horizon, (effect, variance) in enumerate(zip(estimates, covariance), 1):
        for beta in NULL_MOMENT_BETAS:
            contrast = np.array([1., -beta])
            moment = float(contrast@effect)
            moment_variance = max(float(contrast@variance@contrast), 0.)
            statistic = (moment/np.sqrt(moment_variance) if moment_variance > 0
                else (0. if moment == 0 else np.copysign(np.inf, moment)))
            tests.append(dict(horizon=horizon, beta=beta, statistic=float(statistic),
                p_value=float(2*norm.sf(abs(statistic))), reject=bool(abs(statistic) > norm.ppf(.975)),
                inference="asymptotic_average_itt_moment", exact_randomization=False))
    return tests


def run_one(task: dict) -> str:
    from longbet._encourage import encouragement_effects
    from longbet._orthogonal_iv import crossfit_encouragement
    path = Path(task["output"])/"runs"/f'{task["scenario"]}_{task["replicate"]:04d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(task["output"])/"manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != task["manifest_sha256"]:
        raise ValueError("The launch manifest changed after tasks were constructed.")
    if source_hashes() != json.loads(manifest_bytes)["source_sha256"]:
        raise ValueError("Evaluated sources changed after the launch manifest was frozen.")
    data = generate_panel(task["scenario"], task["seed"], n=task["n"])
    features = baseline_features(data)
    observed = np.stack((data["y"], data["d"]), axis=-1)[:, data["start"]:]
    record = {**task, "truth": data["truth"], "methods": {},
        "observed_sha256": hashlib.sha256(np.concatenate((data["y"].ravel(),
            data["d"].ravel(), features.ravel(), data["assignment"].ravel())).astype("<f8").tobytes()).hexdigest(),
        "horizons": observed.shape[1], "baseline_periods": data["start"],
        "feature_count": features.shape[1]}
    for method in (*REFERENCE_METHODS, *CORRECTED_METHODS):
        started = time.perf_counter()
        try:
            if method == "unadjusted":
                ref = encouragement_effects(data["y"], data["d"], data["z"], data["t"])
                estimate = ref[["itt_y", "itt_d"]].to_numpy()
                cov = np.zeros((len(ref), 2, 2))
                cov[:, 0, 0], cov[:, 1, 1] = ref.itt_y_se**2, ref.itt_d_se**2
                cov[:, 0, 1] = cov[:, 1, 0] = ref.itt_y_d_cov
                result = dict(status="completed", rows=reference_rows(estimate, cov),
                    estimates=estimate, covariance_by_horizon=cov)
            elif method.endswith("ancova"):
                estimate, cov = adjusted_contrasts(data, spline=method == "spline_ancova")
                result = dict(status="completed", rows=reference_rows(estimate, cov),
                    estimates=estimate, covariance_by_horizon=cov)
            else:
                fit = crossfit_encouragement(data["y"], data["d"], data["z"], features, data["t"],
                    learner_factory=learner_factory(method, seed=task["seed"]+10000000,
                        config=task["learner_config"]), folds=task["folds"],
                    seed=task["seed"]+20000000, design="complete_randomization")
                factual_prediction = fit.predictions[np.arange(len(features)), :,
                    data["assignment"].astype(int), :]
                error = factual_prediction-observed
                # Reporting scales use all observed outcomes, only after fitting.
                # They do not affect tuning or the fitted estimator.
                scale = np.maximum(observed.std(axis=(0, 1)), [1e-8, .05])
                result = dict(status="completed", rows=fit.table, metadata=fit.metadata,
                    prediction_mse_by_response=(error**2).mean(axis=(0, 1)),
                    normalized_prediction_mse=float(np.mean((error/scale)**2)),
                    fold_ids=fit.fold_ids, estimates=fit.estimates, covariance=fit.covariance)
            if task["scenario"] == "zero":
                result["null_moment_tests"] = (
                    [fit.ar_test(beta, horizon=h) for h in range(1, observed.shape[1]+1)
                        for beta in NULL_MOMENT_BETAS] if method in CORRECTED_METHODS
                    else reference_null_tests(estimate, cov))
            record["methods"][method] = result
        except Exception as exc:
            record["methods"][method] = dict(status="failed", error=str(exc), traceback=traceback.format_exc())
        record["methods"][method]["seconds"] = time.perf_counter()-started
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(clean_json(record), indent=2)+"\n")
    temporary.replace(path)
    return f'{task["scenario"]}/{task["replicate"]} '+", ".join(
        f'{k}: {v["status"]} ({v["seconds"]:.2f}s)' for k, v in record["methods"].items())


def source_hashes() -> dict:
    files = [Path(__file__), Path(__file__).with_name("iv_comparison.py"),
        Path(__file__).with_name("orthogonal-protocol.md"),
        *sorted((SOURCE_ROOT/"src"/"longbet").glob("*.py"))]
    return {str(p.relative_to(SOURCE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def write_manifest(path: Path, manifest: dict) -> None:
    """Refuse to combine different datasets, configurations, code, or versions."""
    if path.exists():
        if json.loads(path.read_text()) != clean_json(manifest):
            raise ValueError("Output manifest differs; use a fresh output directory.")
    else:
        path.write_text(json.dumps(clean_json(manifest), indent=2)+"\n")


def main():
    from longbet._iv_nuisance import LongBetIVNuisanceConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, required=True,
        help="Use 100 for the six short guardrails; 500 for the two primary long scenarios.")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=2026091200)
    parser.add_argument("--n-short", type=int, default=160)
    parser.add_argument("--n-long", type=int, default=240)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--longbet-config", type=json.loads, default={})
    args = parser.parse_args()
    if any(n < 40 or n % 2 for n in (args.n_short, args.n_long)) or min(
            args.replications, args.folds, args.workers, args.trees) < 1:
        parser.error("Require even n >= 40 and positive replication/fold/worker/tree counts.")
    if args.folds != 2:
        parser.error("The supported conditional cross-fitting procedure requires exactly two folds.")
    if "seed" in args.longbet_config:
        parser.error("LongBet's seed is set by the common per-experiment learner seed.")
    resolved_longbet_config = asdict(LongBetIVNuisanceConfig(**args.longbet_config))
    resolved_longbet_config.pop("seed")
    if resolved_longbet_config["validation_fraction"] != .2:
        parser.error("All three learners must use the same 20% inner validation fraction.")
    if len(resolved_longbet_config["length_scales"]) != 3:
        parser.error("The matched tuning protocol requires exactly three LongBet candidates.")
    args.longbet_config = clean_json(resolved_longbet_config)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(arguments={**vars(args), "output": str(args.output)}, source_sha256=source_hashes(),
        python=platform.python_version(), numpy=np.__version__,
        packages={name: version(name) for name in ("scipy", "scikit-learn", "jax", "arviz", "bartz", "pandas")},
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip(),
        methods=[*REFERENCE_METHODS, *CORRECTED_METHODS],
        comparator_config=dict(spline_penalties=[.01, .1, 1.], spline_knots=3,
            time_spline_knots="min(4, horizons)", interaction="all covariate spline by time spline terms",
            extratrees_leaf_sizes=[2, 5, 10], extratrees_max_features=1., extratrees_trees=args.trees,
            inner_validation_fraction=.2),
        seed_offsets=dict(dataset_scenario=100000, inner_validation_and_learner=10000000,
            outer_folds=20000000), null_moment_betas=NULL_MOMENT_BETAS)
    write_manifest(args.output/"manifest.json", manifest)
    manifest_sha256 = hashlib.sha256((args.output/"manifest.json").read_bytes()).hexdigest()
    tasks = []
    for scenario in args.scenarios:
        for replicate in range(args.replications):
            task = dict(scenario=scenario, replicate=replicate,
                seed=args.seed+100000*SCENARIOS.index(scenario)+replicate,
                n=args.n_long if scenario.startswith("long_") else args.n_short,
                output=str(args.output), folds=args.folds,
                learner_config={"trees": args.trees, "longbet": args.longbet_config},
                manifest_sha256=manifest_sha256)
            path = args.output/"runs"/f"{scenario}_{replicate:04d}.json"
            if path.exists():
                existing = json.loads(path.read_text())
                if any(existing.get(key) != value for key, value in task.items()):
                    raise ValueError(f"Existing run does not match manifest: {path}")
                if set(existing.get("methods", {})) != set(manifest["methods"]):
                    raise ValueError(f"Existing run is incomplete: {path}")
            else:
                tasks.append(task)
    with ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(run_one, task) for task in tasks]):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
