"""Experimental modular hazard and outcome working model.

The adoption module samples a probit hazard with a spike-and-slab coefficient on
encouragement. For each hazard draw, the outcome module samples the *conditional*
working regression ``Y_it = mu_t + beta D_it + rho (D_it - Dhat_it) + error_it``.
Outcome data do not feed back into the adoption module: this is a modular (cut)
distribution, not a joint hazard/outcome posterior.

``D - Dhat`` is not generally a valid control function for endogenous binary
adoption. When relevance is zero, its column and ``D`` are collinear after time
effects are removed, so the likelihood cannot distinguish beta from rho. Proper
priors produce finite draws without identifying a causal effect. Consequently,
this module exposes experimental working coefficients and withholds causal
intervals for every fit. It is not a weak-instrument-robust IV procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from longbet._encourage import _validate


@dataclass(frozen=True)
class CoupledHazardConfig:
    """Experimental modular model configuration.

    ``prior_pi0`` is the prior probability that the relevance slab is active.
    ``beta_prior_var`` and ``rho_prior_var`` are prior variances *relative to the
    outcome error variance*, permitting exact conditional normal/inverse-gamma
    draws. Hazard intercepts have independent proper zero-mean normal priors.
    """

    num_sweeps: int = 150
    num_burnin: int = 40
    prior_pi0: float = 0.5
    slab_var: float = 1.0
    beta_prior_var: float = 4.0
    seed: int = 42
    rho_prior_var: float = 1.0
    hazard_intercept_prior_var: float = 4.0

    def __post_init__(self) -> None:
        for name in ("num_sweeps", "num_burnin", "seed"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.num_sweeps <= self.num_burnin:
            raise ValueError("num_sweeps must exceed num_burnin.")
        if not np.isfinite(self.prior_pi0) or not 0 <= self.prior_pi0 <= 1:
            raise ValueError("prior_pi0 must lie in [0, 1].")
        for name in ("slab_var", "beta_prior_var", "rho_prior_var", "hazard_intercept_prior_var"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True)
class CoupledHazardResult:
    """Experimental working-model draws, with no supported causal interval.

    The legacy ``beta_*`` fields describe a working regression coefficient, not
    a validated causal effect. ``causal_ci`` is deliberately always ``None``.
    """

    beta_post: np.ndarray
    beta_mean: float
    beta_ci: tuple[float, float]
    rho_post: np.ndarray
    rho_mean: float
    xi_inclusion_prob: float
    sigma_y_post: np.ndarray
    summary_table: pd.DataFrame
    metadata: dict[str, Any]
    causal_ci: None = None


def _outcome_posterior(
    y: np.ndarray,
    d: np.ndarray,
    d_hat: np.ndarray,
    beta_prior_var: float,
    rho_prior_var: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Exact conditional NIG parameters after integrating flat time intercepts.

    Unlike one outcome Gibbs step per changing hazard draw, sampling this
    conditional distribution targets the declared modular distribution exactly
    conditional on each adoption draw. The residual variance uses the same
    integrated model, avoiding the previous update's stale time intercepts.
    """
    n, t_len = y.shape
    y_centered = (y - y.mean(axis=0)).ravel()
    design = np.stack([d, d - d_hat], axis=-1)
    design = (design - design.mean(axis=0)).reshape(-1, 2)
    prior_precision = np.diag([1 / beta_prior_var, 1 / rho_prior_var])
    precision = design.T @ design + prior_precision
    covariance = np.linalg.solve(precision, np.eye(2))
    mean = np.linalg.solve(precision, design.T @ y_centered)
    residual = y_centered - design @ mean
    shape = 2.0 + 0.5 * (n - 1) * t_len
    scale = 1.0 + 0.5 * (residual @ residual + mean @ prior_precision @ mean)
    return mean, covariance, shape, float(scale), int(np.linalg.matrix_rank(design))


class CoupledHazardIV:
    """Experimental modular working model; causal inference is unsupported."""

    def __init__(self, config: CoupledHazardConfig | None = None) -> None:
        self.config = config or CoupledHazardConfig()

    def fit(
        self,
        y: np.ndarray,
        d: np.ndarray,
        z: np.ndarray,
        x: np.ndarray | None = None,
        t: np.ndarray | None = None,
    ) -> CoupledHazardResult:
        """Sample the modular working distribution for a complete balanced panel.

        ``x`` is unsupported and rejected instead of silently ignored. The hazard
        uses one step per observed period, requiring equally spaced observations.
        The Gaussian outcome likelihood assumes independent errors conditional on
        time effects; repeated observations need not satisfy this working model.
        """
        if x is not None:
            raise ValueError("x adjustment is not implemented in the experimental hazard model.")
        panel = _validate(z, d, t)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.shape != panel.d.shape or not np.isfinite(y_arr).all():
            raise ValueError("y must be a finite panel aligned with d and z.")
        if len(y_arr) < 2:
            raise ValueError("At least two units are required.")
        if len(panel.t) > 2 and not np.allclose(np.diff(panel.t), np.diff(panel.t)[0]):
            raise ValueError("The experimental hazard model requires equally spaced periods.")
        warnings.warn(
            "CoupledHazardIV is an experimental modular working model. Its residual "
            "regression is not validated for causal inference or weak instruments; "
            "beta_ci is a working-model interval and causal_ci is withheld.",
            UserWarning,
            stacklevel=2,
        )
        d_arr, z_arr = panel.d, panel.z
        n, t_len = y_arr.shape
        # Separate streams make the no-outcome-feedback contract verifiable.
        hazard_seed, outcome_seed = np.random.SeedSequence(self.config.seed).spawn(2)
        hazard_rng = np.random.default_rng(hazard_seed)
        outcome_rng = np.random.default_rng(outcome_seed)

        risk_set = np.column_stack([np.ones(n, dtype=bool), d_arr[:, :-1] == 0])
        risk_indices = np.where(risk_set)
        z_risk = z_arr[risk_indices]
        ztz = float(z_risk @ z_risk)
        time_mu_d = np.zeros(t_len)
        for s in range(t_len):
            if risk_set[:, s].any():
                rate = np.mean(d_arr[risk_set[:, s], s])
                time_mu_d[s] = stats.norm.ppf(np.clip(rate, 0.02, 0.98))
        xi, tau_d = 1, 0.0
        u_lat = np.zeros((n, t_len))
        retained = self.config.num_sweeps - self.config.num_burnin
        beta_trace = np.empty(retained)
        rho_trace = np.empty(retained)
        xi_trace = np.empty(retained, dtype=int)
        sigma_y_trace = np.empty(retained)
        rank_trace = np.empty(retained, dtype=int)

        for sweep in range(self.config.num_sweeps):
            eta_d = time_mu_d[None, :] + xi * tau_d * z_arr
            eta_risk = eta_d[risk_indices]
            adopted = d_arr[risk_indices] == 1
            lower = np.where(adopted, -eta_risk, -np.inf)
            upper = np.where(adopted, np.inf, -eta_risk)
            # Stable tail sampling; clipping uniform CDFs changes truncated tails.
            u_lat[risk_indices] = stats.truncnorm.rvs(
                lower, upper, loc=eta_risk, random_state=hazard_rng,
            )

            res_haz = u_lat[risk_indices] - time_mu_d[risk_indices[1]]
            zty = float(z_risk @ res_haz)
            post_precision = ztz + 1 / self.config.slab_var
            log_bayes_factor = (-0.5 * np.log(post_precision * self.config.slab_var)
                                + 0.5 * zty**2 / post_precision)
            prob_xi = expit(logit(self.config.prior_pi0) + log_bayes_factor)
            xi = int(hazard_rng.uniform() < prob_xi)
            tau_d = (float(hazard_rng.normal(zty / post_precision, 1 / np.sqrt(post_precision)))
                     if xi else float(hazard_rng.normal(0, np.sqrt(self.config.slab_var))))

            for s in range(t_len):
                indices = risk_set[:, s]
                precision = indices.sum() + 1 / self.config.hazard_intercept_prior_var
                residual_sum = np.sum(u_lat[indices, s] - xi * tau_d * z_arr[indices, s])
                # Proper priors also cover separated or completely exhausted risks.
                time_mu_d[s] = hazard_rng.normal(residual_sum / precision, 1 / np.sqrt(precision))

            if sweep < self.config.num_burnin:
                continue
            hazard = stats.norm.cdf(time_mu_d[None, :] + xi * tau_d * z_arr)
            d_hat = 1 - np.cumprod(1 - hazard, axis=1)
            mean, covariance, shape, scale, rank = _outcome_posterior(
                y_arr, d_arr, d_hat, self.config.beta_prior_var, self.config.rho_prior_var,
            )
            variance = float(stats.invgamma.rvs(shape, scale=scale, random_state=outcome_rng))
            theta = outcome_rng.multivariate_normal(mean, variance * covariance)
            index = sweep - self.config.num_burnin
            beta_trace[index], rho_trace[index] = theta
            xi_trace[index] = xi
            sigma_y_trace[index] = np.sqrt(variance)
            rank_trace[index] = rank

        beta_mean = float(beta_trace.mean())
        beta_ci = tuple(float(v) for v in np.quantile(beta_trace, [0.025, 0.975]))
        rho_mean = float(rho_trace.mean())
        xi_prob = float(xi_trace.mean())
        summary_table = pd.DataFrame({
            "parameter": ["beta (working coefficient)", "rho (working residual coefficient)",
                          "xi (hazard relevance)", "sigma_y (working error scale)"],
            "posterior_mean": [beta_mean, rho_mean, xi_prob, float(sigma_y_trace.mean())],
            "ci_lower_2.5": [beta_ci[0], float(np.quantile(rho_trace, .025)), np.nan,
                             float(np.quantile(sigma_y_trace, .025))],
            "ci_upper_97.5": [beta_ci[1], float(np.quantile(rho_trace, .975)), np.nan,
                              float(np.quantile(sigma_y_trace, .975))],
        })
        return CoupledHazardResult(
            beta_post=beta_trace, beta_mean=beta_mean, beta_ci=beta_ci,
            rho_post=rho_trace, rho_mean=rho_mean, xi_inclusion_prob=xi_prob,
            sigma_y_post=sigma_y_trace, summary_table=summary_table,
            metadata={
                **panel.metadata(),
                "experimental": True,
                "distribution": "modular_cut",
                "outcome_feedback_to_hazard": False,
                "outcome_conditional_sampler": "exact_normal_inverse_gamma_with_time_intercepts_integrated",
                "causal_inference_supported": False,
                "weak_instrument_robust": False,
                "interval_interpretation": "working-model marginal credible interval, not causal confidence interval",
                "null_relevance_fraction": float(np.mean(xi_trace == 0)),
                "rank_deficient_fraction": float(np.mean(rank_trace < 2)),
                "outcome_error_assumption": "conditionally independent Gaussian",
                "beta_prior_var_relative_to_error_variance": self.config.beta_prior_var,
                "rho_prior_var_relative_to_error_variance": self.config.rho_prior_var,
            },
        )
