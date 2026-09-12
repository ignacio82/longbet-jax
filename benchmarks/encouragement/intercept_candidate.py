"""Benchmark-only Gaussian reduced forms with independent/correlated intercepts.

This is deliberately a small linear diagnostic, not a new LongBet sampler or an
IV identification claim. Observation innovations have known diagonal covariance.
The two equations share regressors; intercept covariance is either diagonal with
proper inverse-gamma priors or full with a proper inverse-Wishart prior. Exact
Gaussian and inverse-Wishart conditionals permit a marginal/conditional (Geweke)
check before comparisons. Binary adoption is treated as an LPM working response.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.stats import invwishart

from dgp import make_population
from model_calibration import summarize_draws
from reference_matrix import environment, portable, rate


@dataclass
class GaussianState:
    beta: np.ndarray  # P,M
    gamma: np.ndarray  # N,M
    covariance: np.ndarray  # M,M


def draw_prior(rng, n, p, correlated, beta_sd=2.):
    if correlated:
        covariance = invwishart.rvs(df=7, scale=4 * np.eye(2), random_state=rng)
    else:
        # Each diagonal of IW_2(7,4I) is IG(3,2): match marginal priors
        # when comparing absence/presence of cross-outcome covariance.
        covariance = np.diag(2 / rng.gamma(3, size=2))
    return GaussianState(rng.normal(scale=beta_sd, size=(p, 2)),
                         rng.multivariate_normal(np.zeros(2), covariance, size=n), covariance)


def sweep(rng, state, x, y, noise, correlated, beta_sd=2.):
    """One complete exact conditional sweep; shared regressors, known noise."""
    n, t, p = x.shape
    flat = x.reshape(n * t, p)
    beta = np.empty_like(state.beta)
    resid = (y - state.gamma[:, None, :]).reshape(n * t, 2)
    xtx = flat.T @ flat
    for m in range(2):
        precision = xtx / noise[m] + np.eye(p) / beta_sd**2
        chol = np.linalg.cholesky(precision)
        mean = np.linalg.solve(precision, flat.T @ resid[:, m] / noise[m])
        beta[:, m] = mean + np.linalg.solve(chol.T, rng.normal(size=p))
    resid = y - np.einsum("ntp,pm->ntm", x, beta)
    precision = np.linalg.inv(state.covariance) + np.diag(t / noise)
    variance = np.linalg.inv(precision)
    mean = (resid.sum(axis=1) / noise) @ variance
    gamma = mean + rng.normal(size=(n, 2)) @ np.linalg.cholesky(variance).T
    if correlated:
        covariance = invwishart.rvs(df=7 + n, scale=4 * np.eye(2) + gamma.T @ gamma,
                                   random_state=rng)
    else:
        covariance = np.diag((2 + .5 * (gamma**2).sum(axis=0))
                             / rng.gamma(3 + n / 2, size=2))
    return GaussianState(beta, gamma, covariance)


def geweke(replications=4000, seed=6211):
    """Independent prior -> data -> Gibbs transitions preserve prior marginals.

    Starting each replication from a fresh joint-prior draw avoids serial Monte
    Carlo dependence in this verification. Several functions of the updated
    state are compared with their analytic prior expectations using measured
    Monte Carlo standard errors. This checks implementation, not DGP adequacy.
    """
    rng = np.random.default_rng(seed)
    n, t, p = 6, 3, 3
    x = rng.normal(size=(n, t, p))
    noise = np.array([.7, 1.3])
    results = {}
    for correlated in (False, True):
        values = []
        for _ in range(replications):
            prior = draw_prior(rng, n, p, correlated, beta_sd=1.)
            y = (np.einsum("ntp,pm->ntm", x, prior.beta) + prior.gamma[:, None, :]
                 + rng.normal(size=(n, t, 2)) * np.sqrt(noise))
            state = sweep(rng, prior, x, y, noise, correlated, beta_sd=1.)
            values.append([state.beta.mean(), np.mean(state.beta**2),
                           np.mean(state.gamma**2), np.trace(state.covariance) / 2,
                           state.covariance[0, 1], np.mean(state.gamma[:, 0] * state.gamma[:, 1])])
        values = np.asarray(values)
        truth = np.array([0, 1, 1, 1, 0, 0])
        mcse = values.std(axis=0, ddof=1) / np.sqrt(replications)
        discrepancy = values.mean(axis=0) - truth
        z = np.divide(discrepancy, mcse,
                      out=np.zeros_like(truth, dtype=float), where=mcse > 0)
        # A constant broken sampler is maximally inconsistent with a nonzero
        # analytic moment, not perfect agreement merely because its MCSE is zero.
        mismatch = (mcse == 0) & (discrepancy != 0)
        z[mismatch] = np.copysign(np.inf, discrepancy[mismatch])
        results["correlated" if correlated else "independent"] = {
            "functions": ["mean_beta", "mean_beta_squared", "mean_gamma_squared",
                          "mean_intercept_variance", "intercept_covariance", "gamma_crossproduct"],
            "expectation": truth, "estimate": values.mean(axis=0), "mcse": mcse, "z": z,
            "pass": bool(np.all(np.abs(z) < 5)),
        }
    return {"method": "independent marginal-conditional Geweke transitions",
            "replications": replications, "seed": seed, "results": results}


def fit_candidate(data, start, *, correlated, seed, chains=2, burnin=300, draws=300):
    n, t = data["y"].shape
    h = t - start
    # Common calendar means, baseline X, and a separate encouragement effect
    # at each horizon. Effects are regression coefficients, already all-unit.
    x = np.zeros((n, t, t + 2 + h))
    x[:, :, :t] = np.eye(t)
    x[:, :, t:t + 2] = data["x"][:, None, :]
    for j in range(h):
        x[:, start + j, t + 2 + j] = data["assignment"]
    y = np.stack([data["y"], data["d"]], axis=-1)
    # Fixed scales specified before data generation; this is a diagnostic
    # working likelihood, not a claimed correctly specified binary sampler.
    noise = np.array([1., .2])
    beta_draws = np.empty((chains, draws, h, 2))
    covariance_draws = np.empty((chains, draws, 2, 2))
    for c, child in enumerate(np.random.SeedSequence(seed).spawn(chains)):
        rng = np.random.default_rng(child)
        state = draw_prior(rng, n, x.shape[-1], correlated)
        for iteration in range(burnin + draws):
            state = sweep(rng, state, x, y, noise, correlated)
            if iteration >= burnin:
                beta_draws[c, iteration - burnin] = state.beta[-h:]
                covariance_draws[c, iteration - burnin] = state.covariance
    return {"outcome": beta_draws[:, :, :, 0].transpose(2, 0, 1)[None],
            "takeup": beta_draws[:, :, :, 1].transpose(2, 0, 1)[None]}, covariance_draws


def run_comparison(args):
    args.output.mkdir(parents=True, exist_ok=True)
    verification = geweke(args.geweke_replications)
    (args.output / "geweke.json").write_text(json.dumps(portable(verification), indent=2) + "\n")
    if not all(r["pass"] for r in verification["results"].values()):
        raise RuntimeError("Candidate failed its prior-predictive conditional verification.")
    results = []
    for scenario in args.scenarios:
        for dataset in range(args.datasets):
            seed = 21213 + 1000 * dataset + 100_000 * args.scenarios.index(scenario)
            p = make_population(args.n, scenario, seed=seed)
            data = p.assign(seed + 1)
            for correlated in (False, True):
                name = f"{scenario}_d{dataset}_{'correlated' if correlated else 'independent'}"
                values, cov_draws = fit_candidate(data, p.start, correlated=correlated, seed=seed + 2,
                                                  chains=args.chains, burnin=args.burnin, draws=args.draws)
                stats, raw = summarize_draws(values, {"conditional_mean": p.conditional_mean_truth})
                record = {"scenario": scenario, "dataset": dataset, "data_seed": seed,
                          "assignment_seed": seed + 1, "sampler_seed": seed + 2,
                          "correlated_unit_intercepts": correlated,
                          "known_innovation_variances": [1., .2], "n": args.n,
                          "intercept_variance_marginal_prior": "IG(shape=3, scale=2) for both candidates",
                          "chains": args.chains, "burnin": args.burnin, "draws": args.draws,
                          "truth": p.conditional_mean_truth, "summary": stats,
                          "mean_intercept_covariance": cov_draws.mean(axis=(0, 1)),
                          "calibration_status": "benchmark_only_working_Gaussian_model"}
                (args.output / f"{name}.json").write_text(json.dumps(portable(record), indent=2) + "\n")
                np.savez_compressed(args.output / f"{name}.npz", **raw, covariance_draws=cov_draws,
                                    **p.archive(seed + 1))
                results.append(record)
            print(f"intercepts {scenario} dataset {dataset + 1}/{args.datasets}", flush=True)
    summary = []
    for scenario in args.scenarios:
        for correlated in (False, True):
            records = [r for r in results if r["scenario"] == scenario and r["correlated_unit_intercepts"] == correlated]
            mean = np.array([np.r_[r["summary"]["itt_y"]["mean"], r["summary"]["itt_d"]["mean"]] for r in records])
            truth = np.array([np.r_[r["truth"]["itt_y"], r["truth"]["itt_d"]] for r in records])
            posterior = np.asarray([r["summary"]["posterior_itt_covariance"] for r in records])
            centered = mean - truth
            centered -= centered.mean(axis=0)
            products = centered[:, :, None] * centered[:, None, :]
            summary.append({"scenario": scenario, "correlated_unit_intercepts": correlated,
                            "datasets": len(records),
                            "empirical_error_covariance": np.cov(mean - truth, rowvar=False, ddof=1),
                            "mean_posterior_covariance": posterior.mean(axis=0),
                            "mean_posterior_covariance_mcse": posterior.std(axis=0, ddof=1) / np.sqrt(len(records)),
                            "empirical_error_covariance_mcse": products.std(axis=0, ddof=1) / np.sqrt(len(records)),
                            "coverage": {q: rate([r["summary"][q]["coverage"]["conditional_mean"] for r in records])
                                         for q in ("itt_y", "itt_d", "wald")}})
    (args.output / "summary.json").write_text(json.dumps(portable({"environment": environment(),
        "candidate": "Gaussian linear reduced forms, benchmark only", "settings": summary}), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "intercept_results")
    parser.add_argument("--scenarios", nargs="+", default=["strong", "correlated_serial_heteroskedastic"])
    parser.add_argument("--datasets", type=int, default=30)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--burnin", type=int, default=300)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--geweke-replications", type=int, default=4000)
    run_comparison(parser.parse_args())
