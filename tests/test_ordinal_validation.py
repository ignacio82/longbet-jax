# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slow validation artifacts: independent seeded DGP, coverage and diagnostics.

LONG BET statistical thresholds are reporting goals, not stochastic CI asserts.
The slow tests run the prescribed chains if artifacts do not already exist,
then verify their scope, settings, probability algebra and independent truth.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ordinal_validation", ROOT / "benchmarks" / "ordinal_validation.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_ordinal_dgp_contract():
    d = validation.generate_panel(20260909)
    assert d["y"].shape == (300,10)
    assert set(np.unique(d["y"])) == {0,1,2,3}
    np.testing.assert_array_equal(np.unique(d["adoption"],return_counts=True)[1],75)
    np.testing.assert_array_equal(d["t"],np.arange(1,11))
    np.testing.assert_array_equal(d["s"][0],[0,0,1,2,3,4,5,6,7,8])
    assert d["beta"][0] == 0 and d["observed"].all()
    np.testing.assert_allclose(d["prob_y"].sum(-1),1,atol=5e-16)
    np.testing.assert_allclose(d["eta0"],np.repeat((np.sin(d["x"][:,0])+d["x"][:,1])[:,None],10,axis=1))
    held = validation.generate_panel(20260909,held_out=True,beta=d["beta"])
    assert not np.array_equal(held["x"],d["x"])
    np.testing.assert_array_equal(held["beta"],d["beta"])


@pytest.mark.slow
@pytest.mark.parametrize("seed,intercepts", [(s,False) for s in validation.SEEDS]+[(20260909,True)])
def test_ordinal_seeded_validation(seed,intercepts):
    mode = "intercepts_missing" if intercepts else "base"
    root = ROOT / "benchmarks" / "ordinal_results"
    out = root / f"{mode}_{seed}"
    required = ("metrics.json", "settings.json", "posterior.npz", "dgp.npz")
    if not all((out / f).exists() for f in required):
        validation.run_validation(seed, random_intercept=intercepts, output_root=root)
    settings = json.loads((out / "settings.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    assert settings["config"] == validation.benchmark_config(seed,intercepts).to_dict()
    assert settings["source_digest"] == validation.source_digest(), "results predate current sampler sources"
    assert settings["fit_seed"] == seed+1000
    with np.load(out / "posterior.npz") as p, np.load(out / "dgp.npz") as d:
        assert p["cutpoints_samples"].shape == (4000,2)
        assert p["att_prob_full"].shape == (8,4,4000)
        assert np.isfinite(p["cutpoints_samples"]).all()
        assert np.isfinite(p["att_prob_full"]).all()
        np.testing.assert_allclose(p["att_prob_full"].sum(1),0,atol=1e-14)
        for s in range(1,9):
            mask = (d["z"] == 1) & (d["s"] == s)
            np.testing.assert_array_equal(p["evaluation_membership"][s-1],mask)
            np.testing.assert_allclose(p["true_category_att"][s-1],(d["prob1"]-d["prob0"])[mask].mean(0))
    assert len(metrics["category_att_diagnostics"]) == 4
    assert metrics["fit_wall_seconds"] > 0
    assert len(metrics["subsequent_batch_seconds"]) == 29
