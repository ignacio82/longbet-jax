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

"""The Python side of the two-front-door contract (contract/longbet-api.yaml).

Checked in **both** directions: every contract entry must match the dataclass,
and every dataclass field must appear in the contract or be declared internal.
A one-directional check cannot catch a new option that was never written down.
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from longbet import LongBetConfig

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contract" / "longbet-api.yaml"


@pytest.fixture(scope="module")
def contract():
    assert CONTRACT_PATH.exists(), f"contract file not found: {CONTRACT_PATH}"
    with open(CONTRACT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _defaults():
    return {f.name: f.default for f in dataclasses.fields(LongBetConfig)}


def test_every_contract_argument_matches_the_config(contract):
    defaults = _defaults()
    for arg in contract["arguments"]:
        name = arg["python_name"]
        assert name in defaults, f"contract lists {name!r}, which LongBetConfig lacks"
        assert defaults[name] == arg["default"], (
            f"default mismatch for {name}: contract says {arg['default']!r}, "
            f"LongBetConfig says {defaults[name]!r}"
        )


def test_every_config_field_is_in_the_contract(contract):
    documented = {a["python_name"] for a in contract["arguments"]}
    internal = set(contract.get("internal", []))
    undocumented = set(_defaults()) - documented - internal
    assert not undocumented, (
        f"LongBetConfig fields missing from the contract: {sorted(undocumented)}. "
        "Add them to contract/longbet-api.yaml (and to the R front door), or list "
        "them under `internal`."
    )


def test_r_names_are_unique(contract):
    r_names = [a["r_name"] for a in contract["arguments"] if a.get("expose_in_r")]
    assert len(r_names) == len(set(r_names)), "duplicate r_name in the contract"


def test_yaml_files_are_byte_identical():
    c1 = (Path(__file__).resolve().parents[1] / "contract" / "longbet-api.yaml").read_bytes()
    c2 = (Path(__file__).resolve().parents[1] / "inst" / "contract" / "longbet-api.yaml").read_bytes()
    assert c1 == c2, "contract/longbet-api.yaml and inst/contract/longbet-api.yaml differ"


def test_multi_contract_structure(contract):
    assert "multi_api" in contract, "contract missing multi_api section"
    m_api = contract["multi_api"]
    from longbet._sur import SAMPLER_SEMANTICS
    from longbet._io import PRECISION_CACHE_VERSION
    assert m_api.get("sampler_semantics") == SAMPLER_SEMANTICS
    assert m_api.get("precision_cache_version") == PRECISION_CACHE_VERSION
    assert "explicitly positive sigma_prior_a and sigma_prior_b" in m_api["variance_prior_requirement"]
    assert "inputs" in m_api
    assert "fit_returns" in m_api
    assert "predict_returns" in m_api
    assert "utilities" in m_api
    for util in ("effect_draws", "joint_prob", "outcome_correlation"):
        assert util in m_api["utilities"]


def test_python_multi_exports():
    import longbet
    for symbol in (
        "LongBetMulti",
        "LongBetMultiPrediction",
        "effect_draws",
        "joint_prob",
        "outcome_correlation",
        "effect_draws_from_arrays",
        "reduce_joint_masks",
    ):
        assert hasattr(longbet, symbol), f"longbet missing export {symbol}"
        assert symbol in longbet.__all__, f"{symbol} missing from longbet.__all__"


def test_r_frontdoor_signatures_and_defaults(contract):
    """Ensure R functions longbet and longbet_multi have formals matching the contract."""
    import re
    repo_root = Path(__file__).resolve().parents[1]

    # Parse R/longbet.R
    lb_r = (repo_root / "R" / "longbet.R").read_text(encoding="utf-8")
    lb_sig_match = re.search(r"longbet\s*<-\s*function\s*\((.*?)\)\s*\{", lb_r, re.DOTALL)
    assert lb_sig_match, "Could not find longbet signature in R/longbet.R"
    lb_params = [p.strip().split("=")[0].strip() for p in lb_sig_match.group(1).split(",") if p.strip()]

    # Parse R/multi.R
    multi_r = (repo_root / "R" / "multi.R").read_text(encoding="utf-8")
    multi_sig_match = re.search(r"longbet_multi\s*<-\s*function\s*\((.*?)\)\s*\{", multi_r, re.DOTALL)
    assert multi_sig_match, "Could not find longbet_multi signature in R/multi.R"
    multi_params = [p.strip().split("=")[0].strip() for p in multi_sig_match.group(1).split(",") if p.strip()]

    # Check exposed arguments in both functions
    for arg in contract["arguments"]:
        if not arg.get("expose_in_r"):
            continue
        r_name = arg["r_name"]
        assert r_name in lb_params, f"longbet in R/longbet.R missing formal '{r_name}'"
        assert r_name in multi_params, f"longbet_multi in R/multi.R missing formal '{r_name}'"

    # Check NAMESPACE exports
    ns = (repo_root / "NAMESPACE").read_text(encoding="utf-8")
    for exp in (
        "export(longbet_multi)",
        "export(effect_draws)",
        "export(joint_prob)",
        "export(outcome_correlation)",
        "S3method(predict, longbet_multi)",
        "S3method(print, longbet_multi)",
        "S3method(summary, longbet_multi)",
        "S3method(print, longbet_multi.pred)",
        'S3method("[", longbet_multi)',
        'S3method("[", longbet_multi.pred)',
    ):
        assert exp in ns, f"NAMESPACE missing {exp}"
