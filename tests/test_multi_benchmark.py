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

"""The chapter benchmark must not conceal chain disagreement in its summaries."""
import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_multi_chapter.py"
spec = importlib.util.spec_from_file_location("bench_multi_chapter", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_subgroup_summary_keeps_chain_boundaries():
    rng = np.random.default_rng(942)
    chains, retained = 4, 500
    truth = np.array([1., 1., -1., -1.])
    offsets = np.array([-0.8, -0.3, 0.3, 0.8])
    draws = truth[:, None, None] + offsets[None, :, None] + rng.normal(
        0, .02, (4, chains, retained))
    draws = draws.reshape(4, -1)
    groups = {"good": truth > 0, "bad": truth < 0}
    report = benchmark.summarize(draws, truth, groups, chains, retained)
    # Pooled means look excellent, but four chains remain in separate regions.
    assert report["rmse"] < .002
    for label, mask in groups.items():
        summary = report["groups"][label]
        assert summary["rhat"] > 1.5
        assert summary["ess_bulk"] < 20
        expected = draws[mask].mean(0).reshape(chains, retained)
        np.testing.assert_allclose(summary["chain_means"], expected.mean(-1))
        np.testing.assert_allclose(summary["interval"], np.quantile(expected, [.025, .975]))


def test_subgroup_average_precedes_probability_and_interval():
    rng = np.random.default_rng(761)
    draws = rng.normal(size=(6, 2400))
    truth = np.array([.1, .2, .3, -.1, -.2, -.3])
    groups = {"good": truth > 0, "bad": truth < 0}
    report = benchmark.summarize(draws, truth, groups, 4, 600)
    for label, mask in groups.items():
        values = draws[mask].mean(0)
        expected = np.mean(values*np.sign(truth[mask].mean()) > 0)
        assert report["groups"][label]["p_correct_sign"] == expected
        assert report["groups"][label]["n"] == 3


def test_diagnostics_use_mean_ess_and_the_reported_interval_tails():
    from longbet import compute_ess

    rng = np.random.default_rng(733)
    # Skewed, correlated draws distinguish mean ESS from rank/bulk ESS.
    noise = rng.normal(size=(4, 1000))
    for i in range(1, noise.shape[1]):
        noise[:, i] += .8 * noise[:, i-1]
    values = np.exp(noise * .6)
    report = benchmark.summarize(values.reshape(1, -1), np.ones(1),
                                 {"all": np.ones(1, bool)}, 4, 1000)
    result = report["groups"]["all"]
    mean_ess = float(compute_ess(values, method="mean"))
    np.testing.assert_allclose(result["mcse_mean"], values.std(ddof=1)/np.sqrt(mean_ess))
    np.testing.assert_allclose(result["ess_tail"],
        compute_ess(values, method="tail", prob=(.025, .975)))
    assert not np.isclose(mean_ess, result["ess_bulk"])


def test_constant_event_does_not_pass_diagnostic_gate():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        report = benchmark.summarize(np.ones((1, 1000)), np.ones(1),
                                     {"all": np.ones(1, bool)}, 4, 250)
    assert not report["groups"]["all"]["diagnostics_pass"]


def test_control_dgps_keep_signed_heterogeneity_and_paired_randomness():
    path = Path(__file__).resolve().parents[1]/'benchmarks'/'make_mixing_controls.py'
    spec = importlib.util.spec_from_file_location('mixing_controls',path)
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    data = dict(controls.generate())
    baseline = data['baseline']
    assert set(data) == {'baseline','rare','correlated','unit_intercepts','complex_shape'}
    for d in data.values():
        for key in ('x','z','ze','t','col','listings','fulfilled'):
            np.testing.assert_array_equal(d[key],baseline[key])
        for outcome in ('gmv','hours','complaint'):
            truth = d['truth_'+outcome]
            good = d['fulfilled']==1
            direction = 1 if outcome=='gmv' else -1
            assert np.all(truth[good]*direction>0)
            assert np.all(truth[~good]*direction<0)
        assert set(np.unique(d['complaint'])) == {0.,1.}
    for key in ('gmv','hours','truth_gmv','truth_hours'):
        np.testing.assert_array_equal(data['rare'][key],baseline[key])
    assert data['rare']['complaint'].mean()<.15
    assert .3<baseline['complaint'].mean()<.7
