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
from longbet._encourage import (
    encouragement_effects,
    encouragement_summary,
    plot_encouragement,
    validate_encouragement,
)
from longbet._encourage_model import (
    ConditionalEffect,
    ConditionalEncouragementPrediction,
    EncouragementComparison,
    EncouragementDraws,
    EncouragementPrediction,
    LongBetEncourage,
    adoption_distribution,
    offer_effect_on_outcome,
)
from longbet._encourage_design import (
    EncouragementDesign,
    EncouragementDesignResult,
    design_encouragement_effects,
)
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
    "ConditionalEffect",
    "ConditionalEncouragementPrediction",
    "Design",
    "EncouragementComparison",
    "EncouragementDraws",
    "EncouragementDesign",
    "EncouragementDesignResult",
    "EncouragementPrediction",
    "LongBet",
    "LongBetConfig",
    "LongBetEncourage",
    "LongBetMulti",
    "LongBetMultiPrediction",
    "LongBetPrediction",
    "PosteriorSummary",
    "StabilityResult",
    "adoption_cohorts",
    "adoption_distribution",
    "att_stability",
    "available_devices",
    "compute_ess",
    "compute_rhat",
    "derive_exposure",
    "design_encouragement_effects",
    "effect_draws",
    "effect_draws_from_arrays",
    "encouragement_effects",
    "encouragement_summary",
    "get_att",
    "get_catt",
    "joint_prob",
    "load_multi_npz",
    "load_npz",
    "offer_effect_on_outcome",
    "outcome_correlation",
    "plot_rollout",
    "plot_encouragement",
    "resolve_device",
    "reduce_joint_masks",
    "rollout_summary",
    "save_multi_npz",
    "save_npz",
    "streaming_summary",
    "summarize_draws",
    "validate_encouragement",
    "__version__",
]
