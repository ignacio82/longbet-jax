"""Reproducible ordered-probit validation from ordinal.md section 9.3.

Run one dataset first, inspect its output, then the remaining seeds:
    python benchmarks/ordinal_validation.py --data-seed 20260909
    python benchmarks/ordinal_validation.py --data-seed 20260909 --random-intercept

Each process runs one dataset so compiled forest executables do not accumulate
across seeds. It saves the DGP, evaluation membership, config, retained cutpoint
and ATT draws, probability summaries, diagnostics and numerical metrics. A
failed mixing goal is recorded, never turned into a claim of sampler recovery.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import time

import jax
import numpy as np
from scipy import stats

from longbet import LongBet, LongBetConfig, att_stability, derive_exposure

SEEDS = tuple(range(20260909, 20260919))
THRESHOLDS = np.array([-np.inf, 0., 1.2, 2.5, np.inf])


def normal_categories(mean):
    """Independent SciPy generating probabilities, without sampler helpers."""
    lo = THRESHOLDS[:-1] - mean[..., None]
    hi = THRESHOLDS[1:] - mean[..., None]
    return np.where(lo >= 0, stats.norm.sf(lo)-stats.norm.sf(hi),
                    stats.norm.cdf(hi)-stats.norm.cdf(lo))


def generate_panel(seed, *, random_intercept=False, held_out=False, beta=None):
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(held_out)]))
    N, T = 300, 10
    t = np.arange(1, T+1)
    x = rng.standard_normal((N, 2))
    adoption = np.repeat([3, 5, 7, T+1], 75)
    z = (t[None, :] >= adoption[:, None]).astype(float)
    s = derive_exposure(z, t)
    if beta is None:
        beta = np.zeros(int(s.max())+1)
        for exposure in range(1, len(beta)):
            beta[exposure] = .8 + .7*(beta[exposure-1]-.8) + rng.normal(0, .05)
    else:
        beta = np.array(beta, copy=True)
    gamma = rng.normal(0, .5, N) if random_intercept else np.zeros(N)
    mu = np.sin(x[:, 0]) + x[:, 1]
    nu = 1 + .5*x[:, 0]
    eta0 = np.broadcast_to((mu + beta[0]*nu + gamma)[:, None], (N, T)).copy()
    eta1 = mu[:, None] + beta[s]*nu[:, None] + gamma[:, None]
    factual = np.where(z == 1, eta1, eta0)
    latent = factual + rng.standard_normal((N, T))
    y_complete = np.searchsorted(THRESHOLDS[1:-1], latent, side="left").astype(float)
    observed = rng.uniform(size=(N, T)) >= .1 if random_intercept else np.ones((N, T), bool)
    y = np.where(observed, y_complete, np.nan)
    return dict(x=x, z=z, t=t, s=s, y=y, y_complete=y_complete, observed=observed,
                adoption=adoption, beta=beta, gamma=gamma, eta0=eta0, eta1=eta1,
                prob0=normal_categories(eta0), prob1=normal_categories(eta1),
                prob_y=normal_categories(factual), thresholds=THRESHOLDS.copy())


def benchmark_config(seed, random_intercept=False):
    return LongBetConfig(outcome="ordinal", num_categories=4,
        num_chains=4, num_burnin=2000, num_sweeps=1000, n_skip=1,
        adaptive_coding=False, kernel_type="ar1", random_intercept=random_intercept,
        random_seed=seed+1000, inner_loop_length=100, device="cpu")


def source_digest():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "src" / "longbet").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def diagnostics(draws, chains, counts=None):
    """Last draw axis to chain-major diagnostic layout (C,D,S)."""
    arr = np.asarray(draws)
    arr = arr.reshape(arr.shape[0], chains, arr.shape[-1]//chains).transpose(1, 2, 0)
    diagnostic = att_stability(arr, min_ess=200, max_rhat=1.05, warn=False, n_treated=counts)
    return dict(summary=diagnostic.summary, by_exposure=diagnostic.by_exposure)


def evaluation_metrics(probability, labels, baseline):
    codes = labels.astype(int)
    target = np.eye(4)[codes]
    chosen = np.take_along_axis(probability, codes[..., None], axis=-1)[..., 0]
    return dict(brier=float(np.mean(np.sum((probability-target)**2, axis=-1))),
                log_loss=float(-np.mean(np.log(np.maximum(chosen, np.finfo(float).tiny)))),
                baseline_brier=float(np.mean(np.sum((baseline-target)**2, axis=-1))),
                baseline_log_loss=float(-np.mean(np.log(np.maximum(baseline[codes], np.finfo(float).tiny)))),
                predicted_frequencies=probability.mean(axis=(0, 1)),
                observed_frequencies=target.mean(axis=(0, 1)))


def run_validation(seed, *, random_intercept=False, output_root=Path("benchmarks/ordinal_results")):
    mode = "intercepts_missing" if random_intercept else "base"
    out = Path(output_root) / f"{mode}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    config = benchmark_config(seed, random_intercept)
    data = generate_panel(seed, random_intercept=random_intercept)
    heldout = None if random_intercept else generate_panel(seed, held_out=True, beta=data["beta"])
    np.savez_compressed(out / "dgp.npz", **data)
    if heldout is not None:
        np.savez_compressed(out / "heldout.npz", **heldout)
    versions = {name: importlib.metadata.version(name) for name in
                ("jax", "jaxlib", "bartz", "equinox", "numpy", "scipy", "arviz")}
    run_metadata = dict(data_seed=seed, fit_seed=seed+1000, mode=mode,
        config=dataclasses.asdict(config), versions=versions,
        devices=[str(d) for d in jax.devices()], platform=platform.platform(),
        source_digest=source_digest(), category_counts=np.bincount(data["y"][data["observed"]].astype(int), minlength=4),
        weak_category_rule="fewer than 20 observed training cells",
        heldout_unit_ids=None if heldout is None else list(range(300, 600)))
    (out / "settings.json").write_text(json.dumps(_jsonable(run_metadata), indent=2)+"\n")
    print(json.dumps(_jsonable(run_metadata)), flush=True)

    # Record synchronized batch timings using the production loop's callback.
    # First batch includes compilation; do not label it pure sampling time.
    import longbet._model as model_module
    original_run = model_module.run_longbet_mcmc
    batch_seconds = []
    def timed_run(*args, **kwargs):
        last = time.perf_counter()
        def callback(i, total, state):
            nonlocal last
            jax.block_until_ready(state.z)
            now = time.perf_counter()
            batch_seconds.append(now-last)
            last = now
            print(f"{mode} seed={seed} batch={i+1}/{total} seconds={batch_seconds[-1]:.3f}", flush=True)
        return original_run(*args, **kwargs, callback=callback)
    model_module.run_longbet_mcmc = timed_run
    start = time.perf_counter()
    try:
        model = LongBet(config).fit(**{k: data[k] for k in ("y", "x", "z", "t")})
        jax.block_until_ready(model.trace)
    finally:
        model_module.run_longbet_mcmc = original_run
    fit_seconds = time.perf_counter()-start
    print(f"{mode} seed={seed} fit completed in {fit_seconds:.1f}s; predicting training panel", flush=True)
    start = time.perf_counter()
    pred = model.predict(x=data["x"], z=data["z"], t=data["t"], summary_only=True)
    print(f"{mode} seed={seed} training prediction completed in {time.perf_counter()-start:.1f}s"
          + ("; predicting held-out panel" if heldout is not None else "; computing diagnostics"), flush=True)
    pred_heldout = (model.predict(x=heldout["x"], z=heldout["z"], t=heldout["t"], summary_only=True)
                    if heldout is not None else None)
    prediction_seconds = time.perf_counter()-start
    if heldout is not None:
        print(f"{mode} seed={seed} predictions completed in {prediction_seconds:.1f}s; computing diagnostics", flush=True)
    true_category_att = np.stack([(data["prob1"]-data["prob0"])[(data["z"] == 1) & (data["s"] == s)].mean(0)
                                  for s in range(1, len(data["beta"]))])
    category = pred.att_probabilities()
    score = pred.att_expected_score()
    true_score_att = true_category_att @ np.arange(4)
    cutpoints = pred.cutpoints_samples
    cp_bounds = np.percentile(cutpoints, [2.5, 97.5], axis=0)
    category_diag = [diagnostics(pred.att_prob_full[:, k, :], config.num_chains, pred.att_counts) for k in range(4)]
    score_diag = diagnostics(score["att_full"], config.num_chains, pred.att_counts)
    cutpoint_diag = diagnostics(cutpoints.T, config.num_chains)
    baseline = run_metadata["category_counts"] / run_metadata["category_counts"].sum()
    metrics = dict(fit_wall_seconds=fit_seconds, prediction_wall_seconds=prediction_seconds,
        first_batch_including_compilation_seconds=batch_seconds[0],
        subsequent_batch_seconds=batch_seconds[1:],
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        train_calibration=evaluation_metrics(pred.prob_y_summary.mean, data["y_complete"], baseline),
        heldout_calibration=(evaluation_metrics(pred_heldout.prob_y_summary.mean, heldout["y_complete"], baseline)
                             if heldout is not None else None),
        category_att_error=category["att"]-true_category_att,
        category_att_covered=((category["intervals"][0] <= true_category_att) &
                              (true_category_att <= category["intervals"][1])),
        score_att_error=score["att"]-true_score_att,
        score_att_covered=((score["intervals"][0] <= true_score_att) & (true_score_att <= score["intervals"][1])),
        cutpoint_mean=cutpoints.mean(0), cutpoint_intervals=cp_bounds,
        cutpoint_covered=((cp_bounds[0] <= THRESHOLDS[2:-1]) & (THRESHOLDS[2:-1] <= cp_bounds[1])),
        cutpoint_diagnostics=cutpoint_diag, category_att_diagnostics=category_diag,
        score_att_diagnostics=score_diag, weak_categories=np.flatnonzero(run_metadata["category_counts"] < 20))
    np.savez_compressed(out / "posterior.npz", cutpoints_samples=cutpoints,
        att_prob_full=pred.att_prob_full, att_counts=pred.att_counts, score_att_full=score["att_full"],
        true_category_att=true_category_att, true_score_att=true_score_att,
        prob_y_mean=pred.prob_y_summary.mean,
        heldout_prob_y_mean=pred_heldout.prob_y_summary.mean if heldout is not None else np.empty(0),
        evaluation_membership=np.stack([(data["z"] == 1) & (data["s"] == s) for s in range(1, len(data["beta"]))]))
    (out / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2)+"\n")
    print(f"Completed {out}: fit={fit_seconds:.1f}s prediction={prediction_seconds:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--random-intercept", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/ordinal_results"))
    args = parser.parse_args()
    run_validation(args.data_seed, random_intercept=args.random_intercept, output_root=args.output_root)
