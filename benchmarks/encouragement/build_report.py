"""Build compact human-readable tables from retained calibration JSON records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_calibration import aggregate


def percentage(p, se):
    return "undefined" if p is None else f"{100 * p:.2f}% ({100 * se:.2f})"


def number(value):
    return "undefined" if value is None or not np.isfinite(value) else f"{value:.5g}"


def build(root):
    aggregate(root / "model_results")
    lines = ["# Recorded encouragement calibration results", "",
             "Generated from JSON by `build_report.py`. Percentages in parentheses are Monte Carlo",
             "SEs in percentage points. Coverage is nominally 95% pointwise; model rows with",
             "few experiments or failed convergence are diagnostic measurements, not validation.", "",
             "## Reference matrix", "",
             "All horizons, simultaneous coverage, refusal/undefined rates, seeds and full",
             "cross-outcome/cross-horizon covariance are in each linked record.", "",
             "| Setting | N | Outcome ITT h=1 | Adoption ITT h=1 | Wald h=1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    discrepancies = {"covariance": [], "mean_neyman": []}
    for file in sorted((root / "reference_matrix_results").glob("*_n*.json")):
        r = json.loads(file.read_text())
        c = r["pointwise_coverage"]
        rows = [percentage(c[q]["rate"][0], c[q]["mcse"][0]) for q in ("itt_y", "itt_d", "wald")]
        lines.append(f"| [{r['scenario']['name']}]({file.relative_to(root)}) | {r['n']} | " + " | ".join(rows) + " |")
        for name, estimate_key, truth_key, se_key in (
            ("covariance", "empirical_covariance", "randomization", "empirical_covariance_mcse"),
            ("mean_neyman", "mean_estimated_neyman_covariance", "expected_neyman", "mean_neyman_covariance_mcse"),
        ):
            a = np.asarray(r[estimate_key]); b = np.asarray(r["exact_covariance"][truth_key]); s = np.asarray(r[se_key])
            z = np.divide(a - b, s, out=np.zeros_like(s), where=s > 1e-12)
            discrepancies[name].append(np.abs(z).max())
    lines += ["", f"The largest absolute empirical-versus-exact covariance discrepancy is {max(discrepancies['covariance']):.2f}",
              f"Monte Carlo SEs. For the mean estimated Neyman covariance versus its expectation it is {max(discrepancies['mean_neyman']):.2f}.",
              "These maxima span many dependent entries and are descriptive checks, not a multiple-testing procedure.", "",
              "## Model fits", ""]
    records = [json.loads(p.read_text()) for p in sorted((root / "model_results").glob("*/result.json"))]
    complete = [r for r in records if r["status"] == "complete"]
    failed = [r for r in records if r["status"] == "failed"]
    pending = [r for r in records if r["status"] == "running"]
    lines += [f"Recorded runs: {len(complete)} complete, {len(failed)} failed, {len(pending)} in progress.",
              "The full [summary](model_results/summary.json) distinguishes configurations and includes all horizons.",
              "Conditional rows target conditional mean potential outcomes for study units; population rows",
              "target fresh latent units at the empirical baseline covariates. The reference finite-population",
              "target and its inclusion indicators are additionally retained in each run record.", ""]
    for target in ("conditional", "population"):
        counts = {}
        for quantity in ("itt_y", "itt_d", "wald"):
            counts[quantity] = sum(any(not d["convergence_checks_pass"]
                                      for d in r["standardizations"][target][quantity]["diagnostics"])
                                   for r in complete)
        unavailable = sum(any(d["undefined_fraction"] > 0
                              for d in r["standardizations"][target]["wald"]["diagnostics"])
                          for r in complete)
        lines += [f"{target.capitalize()}: at least one horizon fails convergence checks in "
                  f"{counts['itt_y']}/{len(complete)} outcome ITT runs, "
                  f"{counts['itt_d']}/{len(complete)} adoption ITT runs, and "
                  f"{counts['wald']}/{len(complete)} ratio runs. "
                  f"Undefined ratio draws prevent at least one interval in {unavailable} runs.", ""]
    summary = json.loads((root / "model_results" / "summary.json").read_text())
    for target in ("conditional", "population"):
        lines += ["", f"### {target.capitalize()} standardization", "",
                  "| Setting | Variant | Datasets | Y h=1 | D h=1 | Wald h=1 |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for row in summary:
            matching = [r for r in complete if r["scenario"]["name"] == row["scenario"]
                        and r["n"] == row["n"] and r["scenario"] == row["scenario_parameters"]
                        and r["variant"] == row["variant"] and r["config_id"] == row["config_id"]]
            if not matching or matching[0]["config"]["num_burnin"] != 400:
                continue
            c = row["standardizations"][target]["coverage"]
            cells = [percentage(c[q]["rate"][0], c[q]["mcse"][0]) for q in ("itt_y", "itt_d", "wald")]
            lines.append(f"| {row['scenario']} (N={row['n']}) | {row['variant']} | {row['independent_datasets']} | " + " | ".join(cells) + " |")
    lines += ["", "A measured 100% inclusion rate from eight experiments has a plug-in MCSE of zero;",
              "that boundary estimate does not imply certainty about coverage. No configuration is declared calibrated.", "",
              "For binary outcomes with a probit first stage, both equations are binary. Their SUR",
              "loading rows are fixed to zero under current marginal-scale identification, so joint_probit",
              "and separate_probit are the same uncoupled model here (shared trees are disabled).",
              "The identical draws and results are expected; no binary/binary covariance is learned.", "",
              "### Longer-chain sensitivity on the first strong dataset", "",
              "Each entry is the maximum R-hat / minimum tail ESS over all four horizons.",
              "The tails correspond to the 2.5% and 97.5% endpoints. All quantities use two chains.", "",
              "| Variant | Burn-in / saved per chain | Outcome ITT | Adoption ITT | Wald |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for r in complete:
        if r["scenario"]["name"] != "strong" or r["dataset"] != 0:
            continue
        values = r["standardizations"]["conditional"]
        cells = []
        for q in ("itt_y", "itt_d", "wald"):
            diagnostics = values[q]["diagnostics"]
            rhats = [v["rhat"] for v in diagnostics if v["rhat"] is not None]
            esses = [v["ess_tail"] for v in diagnostics if v["ess_tail"] is not None]
            cells.append(f"{max(rhats):.3f} / {min(esses):.1f}" if rhats and esses else "unavailable")
        lines.append(f"| {r['variant']} | {r['config']['num_burnin']} / {r['config']['num_sweeps']} | " + " | ".join(cells) + " |")
    lines += ["", "### Unit-level covariance diagnostic", "",
              "Shown: covariance between outcome ITT and adoption ITT at h=1. Empirical error",
              "covariance is across independently generated datasets after subtracting each dataset's",
              "conditional-mean truth. Parentheses contain Monte Carlo SEs of covariance estimates.", "",
              "| Setting | Variant | Mean posterior covariance | Empirical error covariance |",
              "| --- | --- | ---: | ---: |"]
    for r in summary:
        if r["independent_datasets"] < 2:
            continue
        a = r["standardizations"]["conditional"]
        h = len(a["mean_error"]) // 2
        cells = []
        for key, se_key in (("mean_posterior_itt_covariance", "mean_posterior_covariance_mcse"),
                            ("empirical_error_covariance", "empirical_error_covariance_mcse")):
            cells.append(f"{number(a[key][0][h])} ({number(a[se_key][0][h])})")
        lines.append(f"| {r['scenario']} (N={r['n']}) | {r['variant']} | " + " | ".join(cells) + " |")
    lines += ["", "Eight datasets give imprecise covariance estimates, and incomplete posterior exploration",
              "also affects them. Joint draw alignment alone cannot pass this calibration gate.", "",
              "## Independent versus correlated intercept candidate", "",
              "This is the benchmark-only Gaussian linear model, with 30 datasets per setting and",
              "known diagonal observation variances. It is not a production LongBet option.", "",
              "| Setting | Intercepts | Posterior covariance Y,D h=1 | Empirical error covariance | Wald h=1 coverage |",
              "| --- | --- | ---: | ---: | ---: |"]
    candidate = json.loads((root / "intercept_results" / "summary.json").read_text())
    for r in candidate["settings"]:
        post = r["mean_posterior_covariance"][0][4]
        emp = r["empirical_error_covariance"][0][4]
        coverage = r["coverage"]["wald"]
        lines.append(f"| {r['scenario']} | {'correlated' if r['correlated_unit_intercepts'] else 'independent'} | {number(post)} | {number(emp)} | {percentage(coverage['rate'][0], coverage['mcse'][0])} |")
    lines += ["", "The prior-predictive conditional checks passed for both candidates; see",
              "[Geweke moments and Monte Carlo SEs](intercept_results/geweke.json). Wide intervals,",
              "small datasets counts, and a misspecified Gaussian uptake likelihood prevent a release calibration claim.", ""]
    (root / "calibration-report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    build(Path(__file__).parent)
