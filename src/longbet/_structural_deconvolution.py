"""Conventional panel 2SLS for a common treatment-duration response.

The model is ``Y_it = alpha_i + lambda_t + g(sum_{s<=t} D_is) + error_it``.
Assignment-by-post-wave interactions instrument the duration regressors after
removing unit and wave effects. This changes the treatment definition when
encouragement accelerates adoption; it does not create identifying variation or
provide weak-instrument-robust inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DurationDeconvolutionResult:
    """Panel 2SLS estimates and explicit rank and inference diagnostics."""

    durations: np.ndarray
    effects: np.ndarray
    se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    first_stage_f: float
    model_type: str
    alpha: float
    summary_table: pd.DataFrame
    residuals: np.ndarray
    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    instrument_rank: int
    regressor_rank: int
    residual_df: int
    n_clusters: int
    first_stage_status: str
    first_stage_diagnostics: pd.DataFrame
    inference_method: str = "unit_cluster_CR1_t"


def _demean_two_way(mat: np.ndarray, n: int, periods: int) -> np.ndarray:
    panel = mat.reshape(n, periods, -1)
    return (
        panel
        - panel.mean(axis=1, keepdims=True)
        - panel.mean(axis=0, keepdims=True)
        + panel.mean(axis=(0, 1), keepdims=True)
    ).reshape(n * periods, -1)


def duration_deconvolution_effects(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    t: np.ndarray | None = None,
    model_type: Literal["linear", "stepwise", "quadratic", "spline"] = "linear",
    max_duration: int | None = None,
    alpha: float = 0.05,
    ridge_lambda: float = 0.0,
) -> DurationDeconvolutionResult:
    """Estimate a common duration-response curve using unregularized panel 2SLS.

    Parameters
    ----------
    y : np.ndarray
        Finite outcome panel of shape (N, T), with independent units/clusters.
    d : np.ndarray
        Binary absorbing adoption panel of shape (N, T). Exposure is counted
        from the beginning of this panel; prior exposure is not reconstructed.
    z : np.ndarray
        Binary absorbing encouragement panel of shape (N, T). Encouraged units
        must share an onset; both encouraged and never-encouraged units are needed.
    t : np.ndarray | None
        Strictly increasing, equally spaced calendar periods of length T.
        Duration counts observed periods, not the numerical units of ``t``.
        Irregular or missing periods require a different exposure construction
        and are rejected. Defaults to 1, ..., T.
    model_type : {"linear", "stepwise", "quadratic", "spline"}
        ``linear`` assumes g(s) = beta*s. ``stepwise`` uses 1(s >= k) for
        k=1,...,K. ``quadratic`` uses s and s**2; ``spline`` is its legacy alias
        (there are no spline knots). All models impose g(0)=0.
    max_duration : int | None
        Maximum evaluated duration, capped at the largest observed duration.
        In the stepwise model this also limits the basis: g(s) is constant for
        s above this value. It only changes the evaluation grid in other models.
    alpha : float
        Significance level, strictly between zero and one.
    ridge_lambda : float
        Legacy argument; only zero is accepted. Penalization cannot identify a
        rank-deficient IV model and is not part of this 2SLS estimator.

    Notes
    -----
    Identification requires the common duration-response specification, additive
    unit/time effects, exclusion through treatment history, and full rank of the
    instrumented duration regressors. Random assignment alone does not identify
    an unrestricted heterogeneous duration response.

    Covariance clusters by unit using CR1, with multiplier
    N/(N-1) * (NT-1)/(NT-K), counting K=N+T-1+p coefficients including absorbed
    fixed effects. Pointwise intervals use t(N-1) critical values. These are
    conventional large-cluster, strong-instrument approximations, not exact or
    weak-instrument-robust confidence sets. They assume units are the independent
    assignment/sampling clusters.

    ``first_stage_f`` is the unit-cluster Wald statistic divided by instrument
    rank for a single endogenous regressor. It is not the homoskedastic F used
    with Stock--Yogo tables. For multiple regressors it is NaN: individual
    first-stage diagnostics do not establish joint identification strength.
    Singular cluster covariance also gives NaN with an explicit status. A
    perfectly fitted first stage is marked ``perfect_fit`` with infinite F.
    """
    y_arr, d_arr, z_arr = (
        np.asarray(value, dtype=np.float64) for value in (y, d, z)
    )
    if any(value.ndim != 2 for value in (y_arr, d_arr, z_arr)):
        raise ValueError("y, d, and z must be 2D arrays of shape (N, T).")
    n, periods = y_arr.shape
    if d_arr.shape != y_arr.shape or z_arr.shape != y_arr.shape:
        raise ValueError("y, d, and z must have identical shapes.")
    if n < 3:
        raise ValueError("At least 3 units required for deconvolution.")
    if periods < 2:
        raise ValueError("At least 2 periods required for unit/time fixed effects.")
    if not all(np.isfinite(value).all() for value in (y_arr, d_arr, z_arr)):
        raise ValueError("y, d, and z must be finite balanced panels without missing values.")
    for name, value in (("d", d_arr), ("z", z_arr)):
        if not np.isin(value, [0.0, 1.0]).all():
            raise ValueError(f"{name} must be binary (zero or one).")
        if np.any(np.diff(value, axis=1) < 0):
            raise ValueError(f"{name} must be absorbing (no transitions from one to zero).")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one.")
    if not np.isfinite(ridge_lambda) or ridge_lambda != 0:
        raise ValueError("ridge_lambda must be zero: regularization cannot supply IV identification.")
    if model_type not in ("linear", "stepwise", "quadratic", "spline"):
        raise ValueError(f"Unknown model_type: {model_type}")

    if t is not None:
        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.shape != (periods,):
            raise ValueError(f"t must have length T={periods}.")
        if not np.isfinite(t_arr).all() or not np.all(np.diff(t_arr) > 0):
            raise ValueError("t must be finite and strictly increasing.")
        spacing = np.diff(t_arr)
        if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=0):
            raise ValueError("t must be equally spaced; duration counts complete observed periods.")

    duration_arr = np.cumsum(d_arr, axis=1)
    observed_max = int(duration_arr.max())
    if observed_max == 0:
        raise ValueError("No adoption observed; cumulative duration is 0 for all units.")
    if max_duration is not None:
        if (
            isinstance(max_duration, (bool, np.bool_))
            or not np.isscalar(max_duration)
            or not np.isfinite(max_duration)
            or max_duration < 1
            or max_duration != int(max_duration)
        ):
            raise ValueError("max_duration must be a positive integer.")
    k_dur = observed_max if max_duration is None else min(int(max_duration), observed_max)

    assigned = z_arr[:, -1]
    encouraged = assigned == 1
    if not np.any(encouraged) or np.all(encouraged):
        raise ValueError("Encouragement must include both assigned and never-encouraged units.")
    onsets = np.argmax(z_arr[encouraged], axis=1)
    if np.unique(onsets).size != 1:
        raise ValueError("Encouraged units must share a common encouragement onset.")
    onset = int(onsets[0])

    dur_flat = duration_arr.reshape(-1)
    dur_eval = np.arange(1, k_dur + 1, dtype=np.float64)
    if model_type == "linear":
        w_mat = dur_flat[:, None]
        evaluation = dur_eval[:, None]
        names = ["duration"]
    elif model_type == "stepwise":
        w_mat = (dur_flat[:, None] >= dur_eval).astype(float)
        evaluation = np.tril(np.ones((k_dur, k_dur)))
        names = [f"duration_ge_{k}" for k in range(1, k_dur + 1)]
    else:
        w_mat = np.column_stack((dur_flat, dur_flat**2))
        evaluation = np.column_stack((dur_eval, dur_eval**2))
        names = ["duration", "duration_squared"]

    # A duration trend is already a linear combination of these interactions.
    # With onset zero, omit the first wave because unit FE absorb assignment.
    waves = np.arange(max(1, onset), periods)
    v_mat = (
        assigned[:, None, None]
        * (np.arange(periods)[None, :, None] == waves[None, None, :])
    ).reshape(n * periods, -1)
    y_within = _demean_two_way(y_arr.reshape(-1, 1), n, periods)[:, 0]
    w_within = _demean_two_way(w_mat, n, periods)
    v_within = _demean_two_way(v_mat, n, periods)

    v_left, v_singular, _ = np.linalg.svd(v_within, full_matrices=False)
    tolerance = np.finfo(float).eps * max(n * periods, w_mat.shape[1])
    instrument_rank = int(np.sum(v_singular > tolerance * v_singular[0]))
    q_mat = v_left[:, :instrument_rank]
    p = w_mat.shape[1]
    if instrument_rank < p:
        raise ValueError(
            f"Underidentified duration model: {p} endogenous regressors but only "
            f"{instrument_rank} independent instruments after fixed effects."
        )

    # Scaling by each original within-column norm makes the rank check sensitive
    # to a vanishing projection, even in the one-regressor case. In particular a
    # numerical remnant of a zero first stage cannot become a fitted coefficient.
    scales = np.linalg.norm(w_within, axis=0)
    if np.any(scales <= tolerance * np.maximum(1, np.linalg.norm(w_mat, axis=0))):
        raise ValueError("Underidentified duration model: a regressor has no within-panel variation.")
    w_scaled = w_within / scales
    projected = q_mat.T @ w_scaled
    left, singular, right = np.linalg.svd(projected, full_matrices=False)
    regressor_rank = int(np.sum(singular > tolerance))
    if regressor_rank < p:
        raise ValueError(
            f"Underidentified duration model: instrumented regressor rank is "
            f"{regressor_rank}, but {p} coefficients require identification."
        )

    n_obs = n * periods
    fixed_effect_rank = n + periods - 1
    residual_df = n_obs - fixed_effect_rank - p
    first_stage_df = n_obs - fixed_effect_rank - instrument_rank
    if min(residual_df, first_stage_df) <= 0:
        raise ValueError("Insufficient residual degrees of freedom after fixed effects and instruments.")

    w_hat_scaled = q_mat @ projected
    beta_scaled = right.T @ ((left.T @ (q_mat.T @ y_within)) / singular)
    beta_hat = beta_scaled / scales
    residuals_flat = y_within - w_scaled @ beta_scaled
    bread = (right.T / singular**2) @ right
    scores = np.einsum(
        "ntp,nt->np", w_hat_scaled.reshape(n, periods, p), residuals_flat.reshape(n, periods)
    )
    correction = (n / (n - 1)) * ((n_obs - 1) / residual_df)
    cov_scaled = correction * bread @ (scores.T @ scores) @ bread
    cov_beta = cov_scaled / scales[:, None] / scales[None, :]

    diagnostics = []
    first_correction = (n / (n - 1)) * ((n_obs - 1) / first_stage_df)
    q_panel = q_mat.reshape(n, periods, instrument_rank)
    for j, name in enumerate(names):
        first_residual = w_scaled[:, j] - w_hat_scaled[:, j]
        first_scores = np.einsum("ntq,nt->nq", q_panel, first_residual.reshape(n, periods))
        first_cov = first_correction * (first_scores.T @ first_scores)
        if np.linalg.norm(first_residual) <= tolerance:
            first_f, status = float("inf"), "perfect_fit"
        elif np.linalg.matrix_rank(first_cov) < instrument_rank:
            first_f, status = float("nan"), "singular_cluster_covariance"
        else:
            first_f = float(projected[:, j] @ np.linalg.solve(first_cov, projected[:, j]) / instrument_rank)
            status = "available"
        diagnostics.append({
            "regressor": name,
            "cluster_wald_f": first_f,
            "status": status,
            "instrument_rank": instrument_rank,
            "cluster_df": n - 1,
            "residual_df": first_stage_df,
            "partial_r_squared": float(np.dot(w_hat_scaled[:, j], w_hat_scaled[:, j])),
        })
    first_stage_diagnostics = pd.DataFrame(diagnostics)
    first_stage_f = diagnostics[0]["cluster_wald_f"] if p == 1 else float("nan")
    first_stage_status = diagnostics[0]["status"] if p == 1 else "unavailable_for_multiple_endogenous_regressors"

    effects = evaluation @ beta_hat
    se_curve = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", evaluation, cov_beta, evaluation)))
    critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = effects - critical * se_curve
    ci_upper = effects + critical * se_curve
    summary_table = pd.DataFrame({
        "duration": dur_eval.astype(int),
        # Retain the historical column name: this is the effect on the current
        # outcome after cumulative exposure, not an outcome summed across waves.
        "cumulative_effect": effects,
        "std_error": se_curve,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    })
    return DurationDeconvolutionResult(
        durations=dur_eval,
        effects=effects,
        se=se_curve,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        first_stage_f=first_stage_f,
        model_type=model_type,
        alpha=alpha,
        summary_table=summary_table,
        residuals=residuals_flat.reshape(n, periods),
        coefficients=beta_hat,
        coefficient_covariance=cov_beta,
        instrument_rank=instrument_rank,
        regressor_rank=regressor_rank,
        residual_df=residual_df,
        n_clusters=n,
        first_stage_status=first_stage_status,
        first_stage_diagnostics=first_stage_diagnostics,
    )
