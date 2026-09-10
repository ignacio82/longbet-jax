"""Convenience interface for experimental prediction-assisted LongBet IV."""
from __future__ import annotations

from typing import Any

import numpy as np

from longbet._encourage import _validate
from longbet._iv_nuisance import LongBetIVNuisance
from longbet._orthogonal_iv import CrossfitEncouragementResult, crossfit_encouragement


def baseline_features(y: Any, d: Any, z: Any, x: Any = None,
                      t: Any = None) -> np.ndarray:
    """Join supplied baseline covariates and all pre-assignment Y/D columns.

    No post-assignment response enters the feature matrix. A vector assignment
    treats all supplied periods as post-assignment, matching the inference API.
    The user must ensure that the supplied ``x`` was measured before assignment.
    No response-dependent standardization or feature selection is done here.
    """
    outcomes = np.asarray(y)
    if (outcomes.ndim != 2 or outcomes.dtype.kind not in "biuf"
            or not np.isfinite(outcomes).all()):
        raise ValueError("y must be a complete finite numeric unit-by-period matrix.")
    assignment = np.asarray(z)
    if assignment.ndim == 1:
        if len(assignment) != len(outcomes):
            raise ValueError("A vector z must contain one assignment per unit.")
        assignment = np.broadcast_to(assignment[:, None], outcomes.shape)
    panel = _validate(assignment, d, t)
    if outcomes.shape != panel.z.shape:
        raise ValueError("y, d and panel z must have identical shapes.")
    covariates = np.empty((len(outcomes), 0)) if x is None else np.asarray(x)
    if (covariates.ndim != 2 or covariates.shape[0] != len(outcomes)
            or covariates.dtype.kind not in "biuf" or not np.isfinite(covariates).all()):
        raise ValueError("x must be a complete finite numeric unit-by-covariate matrix.")
    features = np.column_stack([covariates, outcomes[:, :panel.start],
                                panel.d[:, :panel.start]]).astype(float)
    # A constant feature supports the no-baseline/no-covariate case without
    # inventing account-specific information for the prediction model.
    return features if features.shape[1] else np.ones((len(outcomes), 1))


def longbet_iv(y: Any, d: Any, z: Any, x: Any = None, t: Any = None, *,
               learner_config: dict[str, Any] | None = None,
               folds: int = 2, seed: int = 0, alpha: float = .05,
               design: str = "complete_randomization") -> CrossfitEncouragementResult:
    """Estimate assignment ITTs and complete IV confidence sets with LongBet.

    This experimental interface uses LongBet only for held-out prediction.
    ``learner_config`` contains :class:`LongBetIVNuisanceConfig` options. The
    sampling budget controls numerical prediction; posterior quantiles are not
    used for causal inference. All pre-assignment Y/D columns are appended to x.

    Supports complete individual-unit randomization and two conditional folds.
    The joint residual Neyman covariance and Fieller intervals are asymptotic
    and pointwise, requiring stable predictions and suitable moment conditions.
    A CACE interpretation additionally requires exclusion, monotonicity and
    relevance for the specified adoption history. Full unbounded/disconnected
    sets are retained. See :func:`crossfit_encouragement` for the generic learner
    interface and the design/inference assumptions.
    """
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer.")
    if learner_config is not None and not isinstance(learner_config, dict):
        raise ValueError("learner_config must be a dictionary or None.")
    options = dict(learner_config or {})
    options.setdefault("seed", int(seed) + 7919)
    # Validate options before beginning any fold fits.
    LongBetIVNuisance(**options)
    features = baseline_features(y, d, z, x, t)
    result = crossfit_encouragement(
        y, d, z, features, t, learner_factory=lambda: LongBetIVNuisance(**options),
        folds=folds, seed=seed, alpha=alpha, design=design,
    )
    result.metadata.update(
        engine="longbet_crossfit_iv", experimental=True,
        feature_source="supplied_baseline_x_plus_all_pre_assignment_y_d",
        feature_count=int(features.shape[1]), learner_seed=int(options["seed"]),
        posterior_causal_intervals=False,
    )
    return result
