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

"""LongBet: time-varying heterogeneous treatment effects in panel data.

A JAX implementation of LongBet, built on ``bartz``. See :class:`LongBet`.
"""

from longbet._config import LongBetConfig
from longbet._design import Design
from longbet._diagnostics import StabilityResult, att_stability, compute_ess, compute_rhat
from longbet._io import load_npz, save_npz
from longbet._multi_io import load_multi_npz, save_multi_npz
from longbet._multi_model import (
    LongBetMulti,
    LongBetMultiPrediction,
    effect_draws,
    effect_draws_from_arrays,
    joint_prob,
    outcome_correlation,
    reduce_joint_masks,
)
from longbet._rollout import (
    ROLLOUT_COLORS,
    adoption_cohorts,
    plot_rollout,
    rollout_summary,
)
from longbet._model import (
    LongBet,
    LongBetPrediction,
    available_devices,
    derive_exposure,
    get_att,
    get_catt,
    resolve_device,
)
from longbet._summary import (
    BlockAccumulator,
    PosteriorSummary,
    streaming_summary,
    summarize_draws,
)

__version__ = "0.1.0"

__all__ = [
    "ROLLOUT_COLORS",
    "BlockAccumulator",
    "Design",
    "LongBet",
    "LongBetConfig",
    "LongBetMulti",
    "LongBetMultiPrediction",
    "LongBetPrediction",
    "PosteriorSummary",
    "StabilityResult",
    "adoption_cohorts",
    "att_stability",
    "available_devices",
    "compute_ess",
    "compute_rhat",
    "derive_exposure",
    "effect_draws",
    "effect_draws_from_arrays",
    "get_att",
    "get_catt",
    "joint_prob",
    "load_multi_npz",
    "load_npz",
    "outcome_correlation",
    "plot_rollout",
    "resolve_device",
    "reduce_joint_masks",
    "rollout_summary",
    "save_multi_npz",
    "save_npz",
    "streaming_summary",
    "summarize_draws",
    "__version__",
]
