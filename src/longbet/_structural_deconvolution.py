"""Structural treatment-clock deconvolution for longitudinal instrumental variables.

When encouragement accelerates adoption timing, both encouraged and control units
may eventually adopt (e.g. D_{it}(1) = D_{it}(0) = 1 at later horizons). In this
setting, static adoption-stock first stages ITT_d(t) vanish to zero, causing
standard Wald ratios ITT_y(t) / ITT_d(t) to divide by zero and diverge, even though
treated units experienced strictly greater cumulative exposure duration.

This module models the structural outcome response as a function of cumulative
exposure duration:
    Y_{it} = alpha_i + lambda_t + g(duration_{it}) + epsilon_{it}
where duration_{it} = sum_{s <= t} D_{is}. The endogenous duration path is
instrumented using the history and interaction of assigned encouragement Z_i
with post-encouragement exposure time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DurationDeconvolutionResult:
    """Result of structural duration deconvolution estimation."""

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


def duration_deconvolution_effects(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    t: np.ndarray | None = None,
    model_type: Literal["linear", "stepwise", "spline"] = "linear",
    max_duration: int | None = None,
    alpha: float = 0.05,
    ridge_lambda: float = 1e-5,
) -> DurationDeconvolutionResult:
    """Estimate structural causal effects of cumulative treatment exposure duration.

    Parameters
    ----------
    y : np.ndarray
        Outcome panel of shape (N, T).
    d : np.ndarray
        Binary absorbing adoption panel of shape (N, T).
    z : np.ndarray
        Binary absorbing encouragement panel of shape (N, T).
    t : np.ndarray | None
        Calendar periods of length T. Defaults to 1, ..., T.
    model_type : {"linear", "stepwise", "spline"}
        - "linear": Y_{it} has a constant marginal effect per period of exposure:
          g(duration) = beta * duration.
        - "stepwise": non-parametric indicator for each duration step:
          g(duration) = sum_{k=1}^K beta_k * 1(duration >= k).
        - "spline": smooth quadratic spline basis over exposure duration.
    max_duration : int | None
        Maximum duration to model. Defaults to the maximum observed duration.
    alpha : float
        Significance level for confidence intervals (default 0.05).
    ridge_lambda : float
        Regularization for 2SLS projection matrix inversion.

    Returns
    -------
    DurationDeconvolutionResult
        Estimates, standard errors, confidence intervals, first-stage F-statistic,
        and summary table.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    d_arr = np.asarray(d, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)

    if y_arr.ndim != 2 or d_arr.ndim != 2 or z_arr.ndim != 2:
        raise ValueError("y, d, and z must be 2D arrays of shape (N, T).")
    n, t_len = y_arr.shape
    if d_arr.shape != (n, t_len) or z_arr.shape != (n, t_len):
        raise ValueError("y, d, and z must have identical shapes.")
    if n < 3:
        raise ValueError("At least 3 units required for deconvolution.")

    if t is None:
        t_arr = np.arange(1, t_len + 1, dtype=np.float64)
    else:
        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.shape != (t_len,):
            raise ValueError(f"t must have length T={t_len}.")

    # Compute cumulative exposure duration for each unit at each period:
    # duration_{i, t} = sum_{s <= t} d_{i, s}
    duration_arr = np.cumsum(d_arr, axis=1)  # (N, T)
    obs_max_duration = int(np.max(duration_arr))

    if obs_max_duration == 0:
        raise ValueError("No adoption observed; cumulative duration is 0 for all units.")

    if max_duration is None or max_duration > obs_max_duration:
        k_dur = obs_max_duration
    else:
        k_dur = max(1, int(max_duration))

    # Identify encouragement post-periods:
    # Z_i is 1 if unit i was assigned encouragement
    z_unit = (np.max(z_arr, axis=1) > 0).astype(np.float64)  # (N,)

    # Flatten panel to (N*T, ...)
    dur_flat = duration_arr.reshape(-1)  # (N*T,)
    y_flat = y_arr.reshape(-1)  # (N*T,)

    # Unit indices and time indices
    unit_idx = np.repeat(np.arange(n), t_len)
    time_idx = np.tile(np.arange(t_len), n)

    if model_type == "linear":
        w_mat = dur_flat[:, None]
        dur_eval = np.arange(1, k_dur + 1, dtype=np.float64)
    elif model_type == "stepwise":
        w_cols = []
        for k in range(1, k_dur + 1):
            w_cols.append((dur_flat >= k).astype(np.float64))
        w_mat = np.column_stack(w_cols)
        dur_eval = np.arange(1, k_dur + 1, dtype=np.float64)
    elif model_type == "spline":
        w_mat = np.column_stack([dur_flat, dur_flat**2])
        dur_eval = np.arange(1, k_dur + 1, dtype=np.float64)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Construct Exogenous Instruments V:
    v_cols = []
    t_enc_idx = int(np.argmax(np.any(z_arr > 0, axis=0)))
    for s in range(t_enc_idx, t_len):
        v_cols.append((z_unit[unit_idx] * (time_idx == s)).astype(np.float64))
        v_cols.append((z_unit[unit_idx] * np.maximum(0, time_idx - t_enc_idx + 1)).astype(np.float64))

    v_mat = np.column_stack(v_cols)
    _, unique_cols = np.unique(v_mat, axis=1, return_index=True)
    v_mat = v_mat[:, sorted(unique_cols)]

    # Two-way within transformation (demean by unit and demean by time period)
    def demean_two_way(mat: np.ndarray) -> np.ndarray:
        res = mat.copy()
        p = mat.shape[1]
        for j in range(p):
            col = mat[:, j].reshape(n, t_len)
            unit_mean = np.mean(col, axis=1, keepdims=True)
            time_mean = np.mean(col, axis=0, keepdims=True)
            overall_mean = np.mean(col)
            col_demeaned = col - unit_mean - time_mean + overall_mean
            res[:, j] = col_demeaned.reshape(-1)
        return res

    y_ddot = demean_two_way(y_flat[:, None])[:, 0]
    w_ddot = demean_two_way(w_mat)
    v_ddot = demean_two_way(v_mat)

    # First-stage 2SLS projection:
    vtv = v_ddot.T @ v_ddot
    reg_eye = ridge_lambda * np.eye(vtv.shape[0])
    try:
        vtv_inv = np.linalg.pinv(vtv + reg_eye)
    except np.linalg.LinAlgError:
        vtv_inv = np.linalg.pinv(vtv)

    gamma_hat = vtv_inv @ (v_ddot.T @ w_ddot)
    w_hat = v_ddot @ gamma_hat

    # First-stage F-statistic:
    r_w = w_ddot[:, 0]
    r_w_hat = w_hat[:, 0]
    ssr_1 = np.sum((r_w - r_w_hat) ** 2)
    ssm_1 = np.sum(r_w_hat ** 2)
    df_v = v_ddot.shape[1]
    df_res = max(1, n * t_len - df_v - n - t_len)
    first_stage_f = float((ssm_1 / max(1, df_v)) / max(1e-12, (ssr_1 / df_res)))

    # Second stage:
    wht_wh = w_hat.T @ w_hat
    reg_w = ridge_lambda * np.eye(wht_wh.shape[0])
    wht_wh_inv = np.linalg.pinv(wht_wh + reg_w)
    beta_hat = wht_wh_inv @ (w_hat.T @ y_ddot)

    residuals_flat = y_ddot - w_ddot @ beta_hat
    residuals = residuals_flat.reshape(n, t_len)

    # Cluster-robust Huber-White sandwich variance:
    score_i = np.zeros((n, beta_hat.shape[0]), dtype=np.float64)
    for i in range(n):
        idx_i = np.where(unit_idx == i)[0]
        score_i[i] = w_hat[idx_i].T @ residuals_flat[idx_i]

    omega = score_i.T @ score_i
    cov_beta = wht_wh_inv @ omega @ wht_wh_inv

    # Evaluate duration-response curve at durations 1, ..., k_dur:
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    if model_type == "linear":
        beta_val = beta_hat[0]
        se_val = float(np.sqrt(max(0.0, cov_beta[0, 0])))
        effects = beta_val * dur_eval
        se_curve = se_val * dur_eval
    elif model_type == "stepwise":
        effects = np.cumsum(beta_hat)
        se_curve = np.zeros(k_dur, dtype=np.float64)
        for k in range(k_dur):
            sub_cov = cov_beta[: k + 1, : k + 1]
            se_curve[k] = np.sqrt(max(0.0, float(np.sum(sub_cov))))
    elif model_type == "spline":
        x_eval = np.column_stack([dur_eval, dur_eval**2])
        effects = x_eval @ beta_hat
        se_curve = np.zeros(k_dur, dtype=np.float64)
        for k in range(k_dur):
            xk = x_eval[k : k + 1]
            var_k = float((xk @ cov_beta @ xk.T).item())
            se_curve[k] = np.sqrt(max(0.0, var_k))

    ci_lower = effects - z_crit * se_curve
    ci_upper = effects + z_crit * se_curve

    table_data = {
        "duration": dur_eval.astype(int),
        "cumulative_effect": effects,
        "std_error": se_curve,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
    summary_table = pd.DataFrame(table_data)

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
        residuals=residuals,
    )
