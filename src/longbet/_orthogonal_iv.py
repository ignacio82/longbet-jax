"""Direct Bayesian Orthogonal IV (Two-Stage BCF / R-Learner) for panel data.

Estimates heterogeneous Complier Average Causal Effects (CACE) directly via
Neyman-orthogonalized treatment trees:
    1. Stage 1: Fit dynamic first-stage compliance pi_D(X, t) and prognostic outcome mu_Y(X, t).
    2. Stage 2: Fit Bayesian Causal Forest on orthogonalized pseudo-outcomes
       Y* = (Y - mu_Y) / pi_D conditioning on pi_D in the propensity score to
       eliminate Regularization-Induced Confounding (RIC).
Directly produces posterior draws of tau_CACE without noisy denominator division.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from longbet._config import LongBetConfig
from longbet._encourage import _validate
from longbet._encourage_model import ConditionalEncouragementPrediction, _finite_matrix
from longbet._model import LongBet


class LongBetOrthogonalIV:
    """Direct Bayesian Orthogonal IV (Two-Stage BCF) for Panel Encouragement Designs.

    Parameters
    ----------
    config : LongBetConfig, optional
        Configuration for the second-stage structural treatment effect forest.
    first_stage_config : LongBetConfig, optional
        Configuration for the first-stage compliance model. Defaults to `config`.
    prognostic_config : LongBetConfig, optional
        Configuration for the prognostic outcome model. Defaults to `config`.
    min_compliance : float
        Lower bound regularization for first-stage compliance denominator (default 0.02).
    monotonic_first_stage : bool
        If True (default), enforces non-negative compliance (pi_D >= 0).
    **kwargs
        Additional keyword arguments forwarded to the default LongBetConfig.
    """

    def __init__(
        self,
        config: LongBetConfig | None = None,
        *,
        first_stage_config: LongBetConfig | None = None,
        prognostic_config: LongBetConfig | None = None,
        min_compliance: float = 0.02,
        monotonic_first_stage: bool = True,
        **kwargs: Any,
    ):
        base_cfg = LongBetConfig(**kwargs) if config is None else dataclasses.replace(config, **kwargs) if kwargs else config
        self.config = base_cfg
        self.first_stage_config = first_stage_config if first_stage_config is not None else base_cfg
        self.prognostic_config = prognostic_config if prognostic_config is not None else base_cfg
        self.min_compliance = float(min_compliance)
        if self.min_compliance <= 0:
            raise ValueError("min_compliance must be positive.")
        self.monotonic_first_stage = bool(monotonic_first_stage)

        self.first_stage_model: LongBet | None = None
        self.prognostic_model: LongBet | None = None
        self.cace_model: LongBet | None = None
        self._data: dict[str, Any] = {}
        self.panel = None

    def fit(
        self,
        y: Any,
        d: Any,
        z: Any,
        x: Any,
        t: Any = None,
        *,
        x_trt: Any = None,
    ) -> LongBetOrthogonalIV:
        """Fit the two-stage orthogonal IV causal forest on longitudinal panel data.

        Parameters
        ----------
        y : array-like of shape (N, T)
            Continuous downstream outcome panel.
        d : array-like of shape (N, T)
            Observed adoption / treatment takeup panel.
        z : array-like of shape (N, T)
            Randomized encouragement instrument panel.
        x : array-like of shape (N, P)
            Baseline covariates for prognostic and treatment forests.
        t : array-like of shape (T,), optional
            Calendar time vector.
        x_trt : array-like of shape (N, P_trt), optional
            Baseline covariates for treatment forest; defaults to `x`.

        Returns
        -------
        self : LongBetOrthogonalIV
        """
        self.panel = _validate(z, d, t)
        t_vec = self.panel.t
        start = self.panel.start
        d_np = self.panel.d.copy()
        z_np = self.panel.z.copy()
        n_units, n_periods = z_np.shape

        y_np = _finite_matrix(y, "y", n_units)
        if y_np.shape != (n_units, n_periods):
            raise ValueError(f"y must share shape ({n_units}, {n_periods}) with z and d.")
        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim != 2 or len(x_np) != n_units:
            raise ValueError(f"x must be a 2-D array of shape ({n_units}, P).")

        # Stage 1: Dynamic First-Stage Compliance Model
        self.first_stage_model = LongBet(self.first_stage_config).fit(
            d_np, x_np, z_np, t=t_vec, x_trt=x_trt
        )
        pred_d = self.first_stage_model.predict(x_np, z_np, t=t_vec, summary_only=False)
        tau_d_draws = pred_d.tauhats  # (N, T, D)
        if self.monotonic_first_stage:
            tau_d_draws = np.maximum(0.0, tau_d_draws)
        pi_d_mean = np.mean(tau_d_draws, axis=-1)  # (N, T)

        # Stage 1b: Prognostic Baseline Outcome Model
        self.prognostic_model = LongBet(self.prognostic_config).fit(
            y_np, x_np, z_np, t=t_vec, x_trt=x_trt
        )
        pred_y = self.prognostic_model.predict(x_np, z_np, t=t_vec, summary_only=False)
        mu_y_mean = np.mean(pred_y.muhats0, axis=-1)  # (N, T)

        # Stage 2: Orthogonalized Pseudo-Outcome
        y_res = y_np - mu_y_mean
        pi_star = np.maximum(pi_d_mean, self.min_compliance)
        y_star = y_res.copy()
        treated_mask = (z_np == 1)
        y_star[treated_mask] = y_res[treated_mask] / pi_star[treated_mask]

        # BCF propensity score conditioning: mean compliance across post-treatment horizons
        ps = np.mean(pi_star[:, start:], axis=1).astype(np.float32)  # (N,)

        # Fit Direct Structural Treatment Forest on y_star
        self.cace_model = LongBet(self.config).fit(
            y_star, x_np, z_np, t=t_vec, x_trt=x_trt, ps=ps
        )

        self._data = {
            "y": y_np,
            "d": d_np,
            "z": z_np,
            "x": x_np,
            "x_trt": x_trt,
            "t": t_vec,
            "start": start,
        }
        return self

    def predict_conditional(
        self,
        x: Any = None,
        z: Any = None,
        t: Any = None,
        *,
        alpha: float = 0.05,
    ) -> ConditionalEncouragementPrediction:
        """Predict unit-level conditional CACE, ITTs, and resource-constrained policies.

        Evaluates the direct structural BCF and compliance forest to produce
        calibrated posterior draws of CACE without division by noisy takeup draws.

        Parameters
        ----------
        x : array-like, optional
            Baseline covariates (N, P). If None, evaluates on fitted training units.
        z : array-like, optional
            Counterfactual encouragement schedule (N, T). Defaults to training schedule.
        t : array-like, optional
            Calendar time vector (T,). Defaults to fitted calendar.
        alpha : float
            Significance level for credible intervals (default 0.05).

        Returns
        -------
        ConditionalEncouragementPrediction
        """
        if self.cace_model is None or self.first_stage_model is None:
            raise RuntimeError("LongBetOrthogonalIV must be fitted before prediction.")

        if x is None:
            x_np = self._data["x"]
        else:
            x_np = np.asarray(x, dtype=np.float32)
            if x_np.ndim != 2:
                raise ValueError("x must be a 2-D array of shape (N, P).")

        n_units = len(x_np)
        t_vec = self._data["t"] if t is None else np.asarray(t, dtype=np.float64)
        n_periods = len(t_vec)
        start = self._data["start"]

        if z is None:
            z_mat = np.zeros((n_units, n_periods), dtype=np.float32)
            z_mat[:, start:] = 1.0
        else:
            z_mat = np.asarray(z, dtype=np.float32)
            if z_mat.shape != (n_units, n_periods):
                raise ValueError(f"z must have shape ({n_units}, {n_periods}).")

        # 1. Compliance draws from first-stage forest
        pred_d = self.first_stage_model.predict(x_np, z_mat, t=t_vec, summary_only=False)
        d_draws_full = pred_d.tauhats  # (N, T, D)
        if self.monotonic_first_stage:
            d_draws_full = np.maximum(0.0, d_draws_full)

        pi_d_mean = np.mean(d_draws_full, axis=-1)  # (N, T)
        pi_star = np.maximum(pi_d_mean, self.min_compliance)
        ps_pred = np.mean(pi_star[:, start:], axis=1).astype(np.float32)

        # 2. Structural CACE draws directly from second-stage forest using predicted ps
        pred_cace = self.cace_model.predict(x_np, z_mat, t=t_vec, ps=ps_pred, summary_only=False)
        cace_draws_full = pred_cace.tauhats  # (N, T, D)

        # Slice post-treatment horizons
        post_idx = np.arange(start, n_periods)
        cace_draws = cace_draws_full[:, post_idx, :]
        citt_d_draws = d_draws_full[:, post_idx, :]
        citt_y_draws = cace_draws * citt_d_draws

        p0_draws = (
            np.clip(pred_d.muhats0[:, post_idx, :], 0.0, 1.0)
            if pred_d.muhats0 is not None
            else None
        )

        periods = self.panel.t[start:].copy()
        horizons = self.panel.exposure[start:].copy()

        return ConditionalEncouragementPrediction(
            citt_y_draws=citt_y_draws,
            citt_d_draws=citt_d_draws,
            periods=periods,
            horizons=horizons,
            p0_draws=p0_draws,
            cace_draws=cace_draws,
            alpha=alpha,
            monotonic_first_stage=self.monotonic_first_stage,
        )

    def predict(
        self,
        *,
        new_x: Any = None,
        new_z: Any = None,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> Any:
        """Evaluate out-of-sample or in-sample conditional predictions."""
        return self.predict_conditional(x=new_x, z=new_z, alpha=alpha)
