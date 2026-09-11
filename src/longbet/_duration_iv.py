"""Heterogeneous exposure-duration instrumental variables for panel encouragement designs.

Estimates heterogeneous causal returns to accumulated treatment exposure:
    Y_{it} = alpha_i + lambda_t + g(S_{it}, X_i) + epsilon_{it}
where S_{it} = sum_{s<=t} D_{is} is accumulated exposure duration.

Instruments exposure and its covariate interactions with randomized encouragement
history Z_{it} = Z_i * 1(t >= t_enc). Uses two-way fixed effect projection and
account-clustered CR1 sandwich covariance with t(N-1) degrees of freedom.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from longbet._encourage import _validate


@dataclass(frozen=True)
class HeterogeneousDurationIVResult:
    """Estimates, standard errors, and diagnostics for heterogeneous duration IV."""

    coefficients: np.ndarray
    covariance: np.ndarray
    regressor_names: list[str]
    unit_effects: np.ndarray
    segment_table: pd.DataFrame
    population_effect: float
    population_se: float
    population_ci: tuple[float, float]
    first_stage_f: float
    first_stage_status: str
    instrument_rank: int
    regressor_rank: int
    n_units: int
    n_periods: int
    metadata: dict[str, Any]

    def predict_curve(self, x_new: np.ndarray, durations: np.ndarray | None = None) -> np.ndarray:
        """Predict the causal exposure-duration response curve g(s, x) for new accounts."""
        x = np.atleast_2d(np.asarray(x_new, dtype=float))
        durs = np.arange(1, self.n_periods + 1) if durations is None else np.asarray(durations)
        # linear duration response: g(s, x) = s * beta(x)
        slopes = self._predict_slopes(x)
        return slopes[:, None] * durs[None, :]

    def _predict_slopes(self, x: np.ndarray) -> np.ndarray:
        x_mean = self.metadata["x_mean"]
        x_std = self.metadata["x_std"]
        xs = (x - x_mean) / x_std
        intercept = self.coefficients[0]
        interaction_coefs = self.coefficients[1:1 + xs.shape[1]]
        return intercept + xs @ interaction_coefs


def _demean_two_way(mat: np.ndarray, n: int, periods: int) -> np.ndarray:
    panel = mat.reshape(n, periods, -1)
    return (
        panel
        - panel.mean(axis=1, keepdims=True)
        - panel.mean(axis=0, keepdims=True)
        + panel.mean(axis=(0, 1), keepdims=True)
    ).reshape(n * periods, -1)


def heterogeneous_duration_iv(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x: np.ndarray | None = None,
    t: np.ndarray | None = None,
    *,
    model_type: Literal["linear", "quadratic"] = "linear",
    segments: int = 3,
    alpha: float = 0.05,
) -> HeterogeneousDurationIVResult:
    """Estimate heterogeneous exposure-duration returns using panel IV.

    Parameters
    ----------
    y : (N, T) outcome panel
    d : (N, T) binary absorbing adoption panel
    z : (N, T) binary absorbing encouragement panel (or (N,) assignment vector)
    x : (N, P) baseline covariates measured before assignment (optional)
    t : (T,) calendar periods (optional)
    model_type : "linear" (g(s, x) = s * beta(x)) or "quadratic"
    segments : number of covariate quantile segments for reporting (default 3)
    alpha : significance level (default 0.05)
    """
    outcomes = np.asarray(y, dtype=float)
    adoptions = np.asarray(d, dtype=float)
    assignment = np.asarray(z, dtype=float)
    n, periods = outcomes.shape

    if assignment.ndim == 1:
        assignment = np.broadcast_to(assignment[:, None], outcomes.shape)

    panel = _validate(assignment, adoptions, t)
    t_enc = panel.start
    times = panel.t

    # Accumulated exposure duration S_{it}
    s = np.cumsum(adoptions, axis=1)

    # Base encouragement instrument: cumulative periods of encouragement Z^{cum}_{it}
    inst_z = np.cumsum(assignment, axis=1)

    # Preprocessing baseline covariates X
    if x is None or (np.asarray(x).ndim == 2 and np.asarray(x).shape[1] == 0):
        covariates = np.zeros((n, 0))
    else:
        covariates = np.asarray(x, dtype=float)
        if covariates.shape[0] != n:
            raise ValueError("x must match the number of accounts in y.")

    p = covariates.shape[1]
    if p > 0:
        x_mean = covariates.mean(axis=0)
        x_std = np.where(covariates.std(axis=0) > 1e-8, covariates.std(axis=0), 1.0)
        xs = (covariates - x_mean) / x_std
    else:
        x_mean = np.empty(0)
        x_std = np.empty(0)
        xs = np.zeros((n, 0))

    # Construct regressors and instruments
    # Main duration: S
    reg_columns = [s.ravel()]
    reg_names = ["duration"]
    inst_columns = [inst_z.ravel()]
    inst_names = ["cumulative_encouragement"]

    # Covariate interactions: S * X_j, instrumented by Z * X_j
    for j in range(p):
        name = f"covariate_{j+1}"
        xj = xs[:, j:j+1]
        reg_columns.append((s * xj).ravel())
        reg_names.append(f"duration_x_{name}")
        inst_columns.append((inst_z * xj).ravel())
        inst_names.append(f"encouragement_x_{name}")

    if model_type == "quadratic":
        s_sq = s ** 2
        reg_columns.append(s_sq.ravel())
        reg_names.append("duration_sq")
        inst_columns.append((inst_z ** 2).ravel())
        inst_names.append("cumulative_encouragement_sq")

    regressors = np.column_stack(reg_columns)
    instruments = np.column_stack(inst_columns)

    # Two-way fixed effects demeaning
    y_dm = _demean_two_way(outcomes, n, periods)
    x_dm = _demean_two_way(regressors, n, periods)
    z_dm = _demean_two_way(instruments, n, periods)

    # Rank checks
    inst_rank = int(np.linalg.matrix_rank(z_dm))
    reg_rank = int(np.linalg.matrix_rank(x_dm))
    n_reg = x_dm.shape[1]

    if inst_rank < n_reg:
        raise ValueError(f"Instrument rank ({inst_rank}) is less than regressor count ({n_reg}).")
    if reg_rank < n_reg:
        raise ValueError(f"Regressor rank ({reg_rank}) is deficient.")

    # 2SLS estimation
    pz = z_dm @ np.linalg.pinv(z_dm.T @ z_dm) @ z_dm.T
    x_hat = pz @ x_dm
    coef = np.linalg.solve(x_hat.T @ x_hat, x_hat.T @ y_dm).ravel()

    # Residuals and account-clustered CR1 sandwich covariance
    residuals = (y_dm - x_dm @ coef[:, None]).reshape(n, periods)
    # Unit-level score: sum over periods
    unit_scores = np.zeros((n, n_reg))
    x_hat_panel = x_hat.reshape(n, periods, n_reg)
    for i in range(n):
        unit_scores[i] = np.sum(x_hat_panel[i] * residuals[i, :, None], axis=0)

    meat = unit_scores.T @ unit_scores
    bread = np.linalg.inv(x_hat.T @ x_hat)
    # CR1 small-sample degrees-of-freedom correction
    df_correction = (n / (n - 1)) * ((n * periods - 1) / (n * periods - n_reg - n - periods + 1))
    covariance = df_correction * (bread @ meat @ bread)

    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    # First-stage diagnostics: regression of duration on instrument
    fs_coef = np.linalg.solve(z_dm.T @ z_dm, z_dm.T @ x_dm[:, 0])
    fs_res = (x_dm[:, 0] - z_dm @ fs_coef).reshape(n, periods)
    fs_scores = np.array([np.sum(z_dm.reshape(n, periods, -1)[i] * fs_res[i, :, None], axis=0) for i in range(n)])
    fs_meat = fs_scores.T @ fs_scores
    fs_bread = np.linalg.inv(z_dm.T @ z_dm)
    fs_cov = df_correction * (fs_bread @ fs_meat @ fs_bread)
    fs_wald = float(fs_coef[0] ** 2 / max(fs_cov[0, 0], 1e-12))
    fs_status = "strong" if fs_wald >= 10.0 else "weak"

    # Compute unit-level duration slopes: beta(X_i)
    if p > 0:
        unit_slopes = coef[0] + xs @ coef[1:1 + p]
    else:
        unit_slopes = np.full(n, coef[0])

    # Population exposure-weighted average effect
    pop_effect = float(coef[0])
    pop_se = float(se[0])
    pop_ci = (float(pop_effect - tcrit * pop_se), float(pop_effect + tcrit * pop_se))

    # Segment table: partition accounts into segments
    segment_rows = []
    if p > 0 and segments >= 2:
        # Segment by predicted unit slope or first principal covariate
        quantiles = np.linspace(0, 1, segments + 1)
        labels = [f"Segment {k+1}" for k in range(segments)]
        cuts = np.quantile(xs[:, 0], quantiles)
        cuts[0] -= 1e-6
        for k in range(segments):
            in_seg = (xs[:, 0] > cuts[k]) & (xs[:, 0] <= cuts[k+1])
            if in_seg.sum() >= 2:
                seg_n = int(in_seg.sum())
                seg_mean_slope = float(unit_slopes[in_seg].mean())
                # Linear combination SE: w = [1, mean(xs)]
                w_seg = np.zeros(n_reg)
                w_seg[0] = 1.0
                w_seg[1:1+p] = xs[in_seg].mean(axis=0)
                seg_var = float(w_seg @ covariance @ w_seg)
                seg_se = float(np.sqrt(max(seg_var, 0.0)))
                segment_rows.append({
                    "segment": labels[k],
                    "n_accounts": seg_n,
                    "mean_duration_slope": seg_mean_slope,
                    "se": seg_se,
                    "ci_lower": seg_mean_slope - tcrit * seg_se,
                    "ci_upper": seg_mean_slope + tcrit * seg_se,
                })
    else:
        segment_rows.append({
            "segment": "Overall",
            "n_accounts": n,
            "mean_duration_slope": pop_effect,
            "se": pop_se,
            "ci_lower": pop_ci[0],
            "ci_upper": pop_ci[1],
        })

    segment_df = pd.DataFrame(segment_rows)

    metadata = {
        "model": "heterogeneous_exposure_duration_iv",
        "model_type": model_type,
        "n_units": n,
        "n_periods": periods,
        "t_enc": t_enc,
        "p_covariates": p,
        "x_mean": x_mean,
        "x_std": x_std,
        "first_stage_f": fs_wald,
        "first_stage_status": fs_status,
        "instrument_rank": inst_rank,
        "regressor_rank": reg_rank,
        "alpha": alpha,
        "critical_t": tcrit,
    }

    return HeterogeneousDurationIVResult(
        coefficients=coef,
        covariance=covariance,
        regressor_names=reg_names,
        unit_effects=unit_slopes,
        segment_table=segment_df,
        population_effect=pop_effect,
        population_se=pop_se,
        population_ci=pop_ci,
        first_stage_f=fs_wald,
        first_stage_status=fs_status,
        instrument_rank=inst_rank,
        regressor_rank=reg_rank,
        n_units=n,
        n_periods=periods,
        metadata=metadata,
    )


__all__ = [
    "HeterogeneousDurationIVResult",
    "heterogeneous_duration_iv",
]
