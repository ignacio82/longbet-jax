"""Build a coverage/mixing report from completed ordinal validation artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from ordinal_validation import SEEDS


def wilson(successes, n):
    z = norm.ppf(.975)
    p = successes/n
    centre = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return f"{successes}/{n} [{centre-half:.2f}, {centre+half:.2f}]"


def diagnostics_extrema(metrics):
    groups = [metrics["cutpoint_diagnostics"], metrics["score_att_diagnostics"],
              *metrics["category_att_diagnostics"]]
    values = {name: [v for group in groups for v in group["by_exposure"][name]]
              for name in ("rhat", "ess_bulk", "ess_tail", "mcse")}
    missing = sum(v is None for vec in values.values() for v in vec)
    result = {name: (min if name.startswith("ess") else max)([v for v in vec if v is not None], default=np.nan)
              for name, vec in values.items()}
    result["undefined"] = missing
    return result


def plot_cutpoints(root, seed, destination):
    """Retained traces expose between-chain separation hidden by pooled bounds."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = root / f"base_{seed}"
    settings = json.loads((path/"settings.json").read_text())
    config = settings["config"]
    with np.load(path/"posterior.npz") as p:
        draws = p["cutpoints_samples"].reshape(config["num_chains"], config["num_sweeps"], 2)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout="constrained")
    for j, ax in enumerate(axes):
        for chain in range(len(draws)):
            ax.plot(np.arange(1, config["num_sweeps"]+1), draws[chain, :, j],
                    linewidth=.75, alpha=.85, label=f"Chain {chain+1}")
        ax.axhline((1.2, 2.5)[j], color="black", linestyle="--", linewidth=1, label="Truth")
        ax.set_ylabel(f"Free threshold {j+2}")
        ax.grid(alpha=.2)
    axes[0].legend(ncol=len(draws)+1, fontsize=9, loc="upper right")
    axes[1].set_xlabel("Retained draw within chain")
    fig.suptitle(f"Ordered-probit threshold mixing — data seed {seed}\n"
                 f"{config['num_burnin']:,} burn-in sweeps; {config['num_sweeps']:,} retained draws per chain",
                 fontsize=11)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def build_report(root=Path("benchmarks/ordinal_results"), destination=Path("benchmarks/ordinal_validation_report.md")):
    cases = []
    for seed in SEEDS:
        path = root / f"base_{seed}"
        if (path / "metrics.json").exists():
            cases.append((seed, json.loads((path/"settings.json").read_text()),
                          json.loads((path/"metrics.json").read_text())))
    lines = ["# Ordered-probit validation", "",
        f"Completed base datasets: **{len(cases)}/{len(SEEDS)}**. "
        + ("The initial ten-dataset run is complete." if len(cases) == len(SEEDS) else
           "Coverage validation is incomplete until all ten prescribed seeds finish."), "",
        "The implementation retains ordered probit. This experiment tests LongBet's "
        "ordered-probit sampler; it is not a comparison with cloglog. The separate "
        "prototype and weighted-treatment-tree gates in `ordinal.md` section 10 "
        "would be required before replacing this backend. No cloglog option is exposed.", "",
        "## Reproduction", "", "```sh",
        ".venv/bin/python benchmarks/ordinal_validation.py --data-seed 20260909",
        "# Inspect that dataset, then repeat for 20260910 through 20260918.",
        ".venv/bin/python benchmarks/ordinal_validation.py --data-seed 20260909 --random-intercept",
        ".venv/bin/python benchmarks/report_ordinal_validation.py",
        ".venv/bin/python -m pytest tests/test_ordinal_validation.py -m slow -q", "```", "",
        "Each case directory contains `settings.json`, `dgp.npz`, `posterior.npz`, "
        "and `metrics.json`; base cases also include `heldout.npz`. Settings record "
        "every config field, versions, device, seeds and a source digest. Posterior "
        "files retain cutpoints, category/score ATT draws, calibration probabilities, "
        "true effects and exact evaluation membership. Fits use 300 units, ten "
        "periods, 75 units in each adoption group (3, 5, 7, never), thresholds "
        "`[-inf,0,1.2,2.5,inf]`, the specified nonlinear prognostic and treatment "
        "functions, and the saved realized AR(1) exposure trajectory.", "",
        "Each fit uses four overdispersed chains, 2,000 burn-in sweeps and 1,000 "
        "retained draws per chain, thinning one, fixed `b0=b1=1`, and the AR(1) "
        "kernel. Other sampler settings are unchanged defaults, except CPU device "
        "and 100-sweep dispatches. Base validation has no unit intercepts. Its "
        "300 independent held-out units have the same rollout and exposure "
        "trajectory, new covariates/errors, and zero intercepts. This avoids using "
        "positional fitted intercepts as new-unit predictions.", "",
        "Timing synchronizes JAX at every batch. Fit time includes initialization "
        "and compilation; the first batch also contains 100 sweeps and is not "
        "reported as pure compilation time. Batch logs and metrics show delays "
        "from host memory contention. Only a CPU was available; GPU checks are "
        "not claimed.", ""]
    if cases:
        lines += ["Versions: " + ", ".join(f"{k} {v}" for k,v in cases[0][1]["versions"].items()) + ".", "",
            "## Held-out calibration and effect error", "",
            "Brier score sums squared category errors per cell. Log loss uses "
            "posterior mean category probabilities. Baselines use observed training "
            "category frequencies. ATT errors use the paired generating probabilities, "
            "including the control `beta_0*nu` term, on all treated training-panel cells.", "",
            "| Data seed | Category counts | Brier / baseline | Log loss / baseline | Category ATT RMSE | Score ATT RMSE | Fit seconds | Peak RSS MiB |",
            "|---|---|---|---|---|---|---|---|"]
        for seed, settings, m in cases:
            c = m["heldout_calibration"]
            lines.append(f"| {seed} | {settings['category_counts']} | {c['brier']:.4f} / {c['baseline_brier']:.4f} | "
                f"{c['log_loss']:.4f} / {c['baseline_log_loss']:.4f} | "
                f"{np.sqrt(np.mean(np.square(m['category_att_error']))):.4f} | "
                f"{np.sqrt(np.mean(np.square(m['score_att_error']))):.4f} | {m['fit_wall_seconds']:.1f} | {m['peak_rss_mib']:.0f} |")
        lines += ["", "## Category support and rollout overlap", "",
            "Adoption groups use the same generating covariate distribution. "
            "The table checks realized category support in treated and untreated "
            "cells and covariate balance between ever-treated and never-treated "
            "units. The largest absolute standardized mean difference uses the "
            "pooled within-group SD for x1 and x2; it is descriptive, not an "
            "acceptance threshold. All periods retain 75 never-treated units.", "",
            "| Seed | Untreated category counts | Treated category counts | Max absolute covariate SMD |",
            "|---|---|---|---|"]
        for seed, _, _ in cases:
            with np.load(root / f"base_{seed}" / "dgp.npz") as d:
                treated = d["z"] == 1
                ever = treated.any(1)
                x = d["x"]
                pooled_sd = np.sqrt((x[ever].var(0, ddof=1)+x[~ever].var(0, ddof=1))/2)
                smd = np.max(np.abs(x[ever].mean(0)-x[~ever].mean(0))/pooled_sd)
                counts = [np.bincount(d["y"][mask].astype(int), minlength=4).tolist()
                          for mask in (~treated, treated)]
                lines.append(f"| {seed} | {counts[0]} | {counts[1]} | {smd:.3f} |")
        with np.load(root / f"base_{cases[0][0]}" / "posterior.npz") as p:
            lines += ["", "Treated cells at exposures 1 through 8: "
                      + str(p["att_counts"].tolist()) + ". This membership is the same across base datasets."]
        lines += ["", "## Mixing and thresholds", "",
            "The goals are rank R-hat <1.05 and bulk/tail ESS >200 for supported "
            "cutpoints and reported category/score ATTs. The table gives extrema "
            "across all these quantities, maximum mean MCSE, and the number of "
            "undefined diagnostics. It does not diagnose unidentified forest/GP "
            "factors or use latent ATT convergence as a substitute. Categories "
            "with fewer than 20 observed cells are flagged as weakly supported.", "",
            "| Seed | Free cutpoint means (truth 1.2, 2.5) | Max R-hat | Min bulk ESS | Min tail ESS | Max MCSE | Undefined | Goals met | Weak categories |",
            "|---|---|---|---|---|---|---|---|---|"]
        for seed, _, m in cases:
            v = diagnostics_extrema(m)
            passed = v["undefined"] == 0 and v["rhat"] < 1.05 and min(v["ess_bulk"], v["ess_tail"]) > 200
            lines.append(f"| {seed} | {np.round(m['cutpoint_mean'],4).tolist()} | {v['rhat']:.3f} | "
                f"{v['ess_bulk']:.1f} | {v['ess_tail']:.1f} | {v['mcse']:.4f} | {v['undefined']} | "
                f"{'yes' if passed else 'no'} | {m['weak_categories']} |")
        figure = destination.with_name("ordinal_cutpoint_trace.png")
        plot_cutpoints(root, cases[0][0], figure)
        lines += ["", f"![Retained cutpoint draws for the first prescribed dataset]({figure.name})", "",
            "The first prescribed dataset is shown without selecting for good "
            "mixing. The retained chain traces should be read together with the "
            "cutpoint diagnostics in its metrics file."]
        lines += ["", "Sequential cutpoint Gibbs can mix slowly. Any failed goals "
            "in this table are limitations of this run, even when prediction or "
            "point estimates look accurate. They do not establish calibrated "
            "uncertainty. Longer runs or a separately derived accelerated transition "
            "would require their own evaluation.", "",
            "A separate acceleration experiment could start with a scalar partially "
            "collapsed threshold move. Hold the predictors fixed, integrate out "
            "the latents, and propose in log gaps `u_j=log(theta_j-theta_(j-1))`. "
            "The log target is the sum of observed category log probabilities "
            "minus `sum(theta_j**2)/(2*cutpoint_prior_scale**2)` plus the "
            "Jacobian `sum(u_j)`. Accept/reject using that target, then refresh "
            "all latents before any Gaussian parameter update. Keep the zero "
            "anchor and unit error variance. This is a proposed follow-up, not "
            "an implemented or validated transition. Check its invariant "
            "distribution against small numerical-integration problems and "
            "empty-category priors, then compare ESS per second including CDF "
            "cost. SUR needs its own conditional-likelihood derivation before "
            "extending the move to coupled fits.", "",
            "## Repeated-dataset interval coverage", "",
            "Entries are coverage counts for nominal 95% intervals and 95% Wilson "
            "binomial intervals across independent datasets for each fixed estimand. "
            "They are coarse with ten datasets. Categories and exposures within one "
            "dataset are correlated; they are not pooled as independent Bernoulli "
            "trials. One missed interval is not by itself evidence of a sampler bug.", "",
            "| Exposure | Category 0 | Category 1 | Category 2 | Category 3 | Rank score |",
            "|---|---|---|---|---|---|"]
        n = len(cases)
        category = np.array([m["category_att_covered"] for _,_,m in cases]).sum(0)
        score = np.array([m["score_att_covered"] for _,_,m in cases]).sum(0)
        for s in range(8):
            lines.append(f"| {s+1} | " + " | ".join(wilson(int(k),n) for k in [*category[s],score[s]]) + " |")
        cp = np.array([m["cutpoint_covered"] for _,_,m in cases]).sum(0)
        lines += ["", f"Cutpoint coverage: theta_2 {wilson(int(cp[0]),n)}; theta_3 {wilson(int(cp[1]),n)}.", ""]
    path = root / "intercepts_missing_20260909"
    lines += ["## Unit intercepts and missing outcomes", ""]
    if (path/"metrics.json").exists():
        m = json.loads((path/"metrics.json").read_text())
        v = diagnostics_extrema(m)
        c = m["train_calibration"]
        settings = json.loads((path/"settings.json").read_text())
        with np.load(path/"dgp.npz") as d:
            missing = int((~d["observed"]).sum())
            missing_treated = int(((~d["observed"]) & (d["z"] == 1)).sum())
        lines += [f"The separate panel uses Gaussian unit intercepts with SD 0.5 and "
            f"10% independent outcome missingness. Fit time was {m['fit_wall_seconds']:.1f}s; "
            f"maximum R-hat {v['rhat']:.3f}, minimum bulk/tail ESS "
            f"{v['ess_bulk']:.1f}/{v['ess_tail']:.1f}, maximum MCSE {v['mcse']:.4f}, "
            f"undefined diagnostics {v['undefined']}. Category ATT includes missing "
            "training-outcome cells under the documented prediction membership rule. "
            "Calibration here conditions on the fitted unit intercepts and is not "
            "held-out-new-unit validation. Its single dataset is not a coverage study.", ""]
        lines += [f"Observed category counts were {settings['category_counts']}; "
            f"{missing} outcomes were missing, including {missing_treated} treated "
            "cells retained in the ATT evaluation. Calibration uses the saved "
            "complete generating labels, including the masked cells.", "",
            f"Brier score / frequency baseline: {c['brier']:.4f} / {c['baseline_brier']:.4f}; "
            f"log loss / baseline: {c['log_loss']:.4f} / {c['baseline_log_loss']:.4f}. "
            f"Category ATT RMSE: {np.sqrt(np.mean(np.square(m['category_att_error']))):.4f}; "
            f"score ATT RMSE: {np.sqrt(np.mean(np.square(m['score_att_error']))):.4f}. "
            f"Nominal 95% intervals contain truth for {int(np.sum(m['category_att_covered']))}/32 "
            f"category/exposure effects and {int(np.sum(m['score_att_covered']))}/8 score effects. "
            "These within-panel counts are correlated and have no binomial interval.", "",
            f"Free cutpoint means: {np.round(m['cutpoint_mean'],4).tolist()}; "
            f"95% lower/upper bounds: {np.round(m['cutpoint_intervals'],4).tolist()}; "
            f"truth covered: {m['cutpoint_covered']}.", ""]
    else:
        lines += ["This separate panel has not yet completed.", ""]
    lines += ["## Software verification", "",
        "See `ordinal_implementation_status.md` for test commands, authoritative "
        "results, unavailable checks, and the requirement audit. Fast distributional "
        "tests use SciPy oracles and Monte Carlo tolerances; compiling a short fit "
        "is not presented as statistical validation.", ""]
    destination.write_text("\n".join(lines))


if __name__ == "__main__":
    build_report()
