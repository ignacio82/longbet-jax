"""Check that repair-study gates describe draws, not selected favorable cells."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/encouragement"))
from repair_comparison import summarize_windows
from build_repair_report import parameter_diagnostics


def test_nested_windows_preserve_bad_horizon_and_undefined_ratio():
    rng = np.random.default_rng(49)
    y = rng.normal(size=(2, 4, 1000))
    d = rng.normal(1, .1, size=y.shape)
    # Only the second horizon disagrees across chains. It must not be pooled
    # with the first or hidden by adequate tail ESS.
    y[1] += np.arange(4)[:, None] * 2
    targets = {"truth": {"itt_y": np.zeros(2), "itt_d": np.ones(2), "wald": np.zeros(2)}}
    result = summarize_windows(dict(itt_y=y, itt_d=d), targets, 10.)
    assert not result["all_effect_checks_pass"]
    assert result["windows"]["half"]["retained_per_chain"] == 500
    np.testing.assert_allclose(result["windows"]["half"]["quantities"]["itt_y"]["mean"],
                               y[..., :500].mean(axis=(1, 2)))
    d[0, 0, 0] = 0
    result = summarize_windows(dict(itt_y=y, itt_d=d), targets, 10.)
    diag = result["windows"]["full"]["quantities"]["wald"]["diagnostics"][0]
    assert diag["undefined_fraction"] == 1 / 4000
    assert not diag["convergence_checks_pass"]
    assert np.isnan(result["windows"]["full"]["quantities"]["wald"]["lower"][0])


def test_parameter_diagnostics_keep_chain_axis_and_every_unit():
    rng = np.random.default_rng(50)
    gamma = rng.normal(size=(4, 1000, 3, 2))
    gamma[:, :, 2, 1] += np.arange(4)[:, None] * 3
    result = parameter_diagnostics({"gamma": gamma})["gamma"]
    assert not result["checks_pass"]
    assert len(result["cells"]) == 6
    worst = max(result["cells"], key=lambda x: x["rhat"])
    assert worst["index"] == [2, 1]


def test_report_keeps_failures_missing_archives_and_undefined_seeds(tmp_path):
    import json
    from build_repair_report import run
    from repair_comparison import write_json

    output = tmp_path / "results"
    output.mkdir()
    base = dict(scenario={"name": "zero"}, n=8, chains=4, burnin=1000, draws=200,
                variant="parametric", config={"example": 1}, data_seed=42)
    for sampler in (0, 1):
        folder = output / str(sampler)
        folder.mkdir()
        rng = np.random.default_rng(200 + sampler)
        np.savez(folder / "effect_draws.npz", itt_y=rng.normal(size=(1, 4, 200)), itt_d=np.zeros((1, 4, 200)))
        write_json(folder / "result.json", {**base, "sampler": sampler, "status": "complete",
            "fit_seconds": 1., "targets": {"truth": {"itt_y": [0.], "itt_d": [0.], "wald": [None]}}})
    for status in ("failed", "running"):
        folder = output / status
        folder.mkdir()
        write_json(folder / "result.json", {**base, "variant": status, "status": status})
    run(output)
    summary = json.loads((output / "summary.json").read_text())
    assert {s["variant"] for s in summary} == {"parametric", "failed", "running"}
    parametric = next(s for s in summary if s["variant"] == "parametric")
    assert parametric["datasets"] == 1
    assert parametric["parameter_passes"] == 0
    assert parametric["max_rhat"] is None
    agreements = json.loads((output / "seed-agreement.json").read_text())
    ratio = next(r for r in agreements if r["quantity"] == "wald")
    assert ratio["available"] == [False]
    assert ratio["difference_in_mcse"] == [None]


def test_continuation_runner_matches_uninterrupted_sampler(tmp_path):
    from dataclasses import asdict
    import os
    from types import SimpleNamespace
    from direct_smooth_candidate import ForestConfig, fit_candidate
    from extend_repair import run
    from repair_comparison import write_json

    rng = np.random.default_rng(83)
    a = np.arange(8) % 2
    z = a[:, None] * np.array([0, 1, 1])[None]
    data = dict(assignment=a, z=z, d=z, y=rng.normal(size=(8, 3)), x=rng.normal(size=(8, 1)))
    config = ForestConfig(baseline_trees=1, effect_trees=1, cutpoints=1,
                          joint_baseline_intercept=True, joint_effect_leaves=True)
    values, prefix = fit_candidate(data, 1, seed=83, chains=4, burnin=5, draws=8, config=config)
    _, full = fit_candidate(data, 1, seed=83, chains=4, burnin=5, draws=16, config=config)
    parent = tmp_path / "example_k8_seed83"
    parent.mkdir()
    np.savez(parent / "data.npz", **data)
    np.savez(parent / "parameters.npz", **prefix)
    y, d = values["outcome"][0], values["takeup"][0]
    np.savez(parent / "effect_draws.npz", itt_y=y, itt_d=d, wald=y / d)
    write_json(parent / "result.json", dict(run_id=parent.name, status="complete", variant="direct_joint",
        scenario={"name": "clean_strong"}, n=8, chains=4, burnin=5, draws=8, sampler_seed=83,
        config=asdict(config), fit_seconds=1., total_seconds=2.,
        targets={"truth": {"itt_y": [0., 0.], "itt_d": [1., 1.], "wald": [0., 0.]}}))
    run(SimpleNamespace(output=tmp_path, cpu_ids=sorted(os.sched_getaffinity(0)),
                        variant="direct_joint", scenario="clean_strong", from_draws=8, to_draws=16))
    with np.load(tmp_path / "example_k16_seed83" / "parameters.npz", allow_pickle=False) as combined:
        for name in ("baseline_rules", "effect_rules", "baseline_leaves", "effect_leaves", "gamma", "sigma2", "gamma2", "itt"):
            np.testing.assert_allclose(combined[name], full[name], atol=2e-13)
