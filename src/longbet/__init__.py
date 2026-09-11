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
    EncouragementComparison,
    EncouragementPrediction,
    LongBetEncourage,
)
from longbet._encourage_design import (
    EncouragementDesign,
    EncouragementDesignResult,
    design_encouragement_effects,
)
from longbet._direct_smooth import (
    DirectSmoothConfig,
    LongBetDirectSmooth,
)
from longbet._iv import longbet_iv
from longbet._iv_nuisance import LongBetIVNuisance, LongBetIVNuisanceConfig
from longbet._jax_nuisance import JAXLongBetIVNuisance
from longbet._duration_iv import (
    HeterogeneousDurationIVResult,
    heterogeneous_duration_iv,
)
from longbet._orthogonal_iv import CrossfitEncouragementResult, crossfit_encouragement
from longbet._randomization_ar import (
    RandomizationARResult,
    randomization_ar,
)
from longbet._encourage_bounds import (
    IdentificationBoundsResult,
    encouragement_bounds,
    identification_bounds,
)
from longbet._hazard_adoption import (
    HazardAdoptionForest,
    HazardAdoptionResult,
    HazardConfig,
    hazard_adoption_effects,
)
from longbet._coupled_hazard_iv import (
    CoupledHazardConfig,
    CoupledHazardIV,
    CoupledHazardResult,
)
from longbet._structural_deconvolution import (
    DurationDeconvolutionResult,
    duration_deconvolution_effects,
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
    "CoupledHazardConfig",
    "CoupledHazardIV",
    "CoupledHazardResult",
    "CrossfitEncouragementResult",
    "Design",
    "DirectSmoothConfig",
    "DurationDeconvolutionResult",
    "EncouragementComparison",
    "EncouragementDesign",
    "EncouragementDesignResult",
    "EncouragementPrediction",
    "HazardAdoptionForest",
    "HazardAdoptionResult",
    "HazardConfig",
    "HeterogeneousDurationIVResult",
    "IdentificationBoundsResult",
    "JAXLongBetIVNuisance",
    "LongBet",
    "LongBetConfig",
    "LongBetDirectSmooth",
    "LongBetEncourage",
    "LongBetIVNuisance",
    "LongBetIVNuisanceConfig",
    "LongBetMulti",
    "LongBetMultiPrediction",
    "LongBetPrediction",
    "PosteriorSummary",
    "RandomizationARResult",
    "StabilityResult",
    "adoption_cohorts",
    "att_stability",
    "available_devices",
    "compute_ess",
    "compute_rhat",
    "crossfit_encouragement",
    "derive_exposure",
    "design_encouragement_effects",
    "duration_deconvolution_effects",
    "effect_draws",
    "effect_draws_from_arrays",
    "encouragement_bounds",
    "encouragement_effects",
    "encouragement_summary",
    "get_att",
    "get_catt",
    "hazard_adoption_effects",
    "heterogeneous_duration_iv",
    "identification_bounds",
    "joint_prob",
    "load_multi_npz",
    "load_npz",
    "longbet_iv",
    "outcome_correlation",
    "plot_rollout",
    "plot_encouragement",
    "randomization_ar",
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
