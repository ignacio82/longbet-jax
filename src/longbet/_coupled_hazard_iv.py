"""Coupled joint hazard adoption and structural outcome IV model.

This module unifies the discrete-time hazard model for adoption timing on the active
risk set R_t with a structural outcome response equation:
    Stage 1 (Adoption Hazard on Risk Set R_t):
        U_{it}^* = mu_{d, t} + xi * tau_{d} * Z_i + eps_{d, it}
        D_{it} = 1 iff U_{it}^* > 0  (for i in R_t)
        xi ~ Bernoulli(pi_0) (spike-and-slab relevance indicator)
        lambda_{it} = Phi(mu_{d, t} + xi * tau_{d} * Z_i)
        D_hat_{it} = 1 - prod_{s <= t} (1 - lambda_{is})
        r_{it} = D_{it} - D_hat_{it} (generalized adoption residual)
    
    Stage 2 (Structural Outcome):
        Y_{it} = mu_{y, t} + beta * D_{it} + rho_{CF} * r_{it} + nu_{it}

The control function coefficient rho_{CF} purges unobserved time-varying confounding
between adoption and outcome shocks, while the spike-and-slab indicator xi places
exact prior mass on the null instrument hypothesis (tau_d = 0) to eliminate
weak-instrument bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CoupledHazardConfig:
    """Configuration for coupled joint hazard adoption IV model."""

    num_sweeps: int = 150
    num_burnin: int = 40
    prior_pi0: float = 0.5
    slab_var: float = 1.0
    beta_prior_var: float = 4.0
    seed: int = 42


@dataclass(frozen=True)
class CoupledHazardResult:
    """Posterior results from CoupledHazardIV."""

    beta_post: np.ndarray
    beta_mean: float
    beta_ci: tuple[float, float]
    rho_post: np.ndarray
    rho_mean: float
    xi_inclusion_prob: float
    sigma_y_post: np.ndarray
    summary_table: pd.DataFrame


class CoupledHazardIV:
    """Coupled joint hazard adoption and structural outcome IV model."""

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
        """Fit the coupled hazard-outcome IV model via MCMC."""
        y_arr = np.asarray(y, dtype=np.float64)
        d_arr = np.asarray(d, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.float64)

        n, t_len = y_arr.shape
        rng = np.random.default_rng(self.config.seed)

        # Active risk set: R_t = {i: d_{i, t-1} == 0}
        risk_set = np.zeros((n, t_len), dtype=bool)
        risk_set[:, 0] = True
        for s in range(1, t_len):
            risk_set[:, s] = d_arr[:, s - 1] == 0.0

        risk_indices = np.where(risk_set)

        # Initial parameters
        beta = 0.0
        rho_cf = 0.0
        sigma_nu = max(0.2, float(np.std(y_arr)))
        tau_d = 1.0
        xi = 1

        time_mu_y = np.zeros(t_len, dtype=np.float64)
        time_mu_d = np.zeros(t_len, dtype=np.float64)
        for s in range(t_len):
            idx_s = risk_set[:, s]
            if np.any(idx_s):
                mean_d = np.clip(np.mean(d_arr[idx_s, s]), 0.02, 0.98)
                time_mu_d[s] = float(stats.norm.ppf(mean_d))

        u_lat = np.zeros((n, t_len), dtype=np.float64)

        total_draws = self.config.num_sweeps
        burnin = self.config.num_burnin
        retained = max(1, total_draws - burnin)

        beta_trace = np.zeros(retained, dtype=np.float64)
        rho_trace = np.zeros(retained, dtype=np.float64)
        xi_trace = np.zeros(retained, dtype=int)
        sigma_y_trace = np.zeros(retained, dtype=np.float64)

        trace_idx = 0
        for sweep in range(total_draws):
            # 1. Update latent U* on risk set
            eta_d = time_mu_d[None, :] + (xi * tau_d) * z_arr
            lower_b = np.where(d_arr == 1.0, 0.0, -np.inf)
            upper_b = np.where(d_arr == 1.0, np.inf, 0.0)

            p_a = stats.norm.cdf(lower_b[risk_indices] - eta_d[risk_indices])
            p_b = stats.norm.cdf(upper_b[risk_indices] - eta_d[risk_indices])
            unif = np.clip(rng.uniform(p_a, p_b), 1e-10, 1.0 - 1e-10)
            u_lat[risk_indices] = eta_d[risk_indices] + stats.norm.ppf(unif)

            # 2. Update spike-and-slab xi on hazard
            res_haz = u_lat[risk_indices] - time_mu_d[risk_indices[1]]
            z_risk = z_arr[risk_indices]
            ztz = float(np.sum(z_risk**2))
            zty = float(np.sum(z_risk * res_haz))
            v_slab = self.config.slab_var

            post_prec = ztz + 1.0 / v_slab
            post_mean = zty / post_prec

            delta_ll = -0.5 * np.log(post_prec * v_slab) + 0.5 * (zty**2) / post_prec
            delta_ll = np.clip(delta_ll, -30.0, 30.0)
            prior_odds = self.config.prior_pi0 / (1.0 - self.config.prior_pi0 + 1e-12)
            prob_xi = 1.0 / (1.0 + (1.0 / prior_odds) * np.exp(-delta_ll))
            xi = int(rng.uniform() < prob_xi)

            if xi == 1:
                tau_d = float(rng.normal(post_mean, 1.0 / np.sqrt(post_prec)))
            else:
                tau_d = float(rng.normal(0.0, np.sqrt(v_slab)))

            # Baseline hazard time effects
            for s in range(t_len):
                idx_s = risk_set[:, s]
                if np.any(idx_s):
                    n_s = float(np.sum(idx_s))
                    mu_d_res = u_lat[idx_s, s] - (xi * tau_d) * z_arr[idx_s, s]
                    time_mu_d[s] = float(rng.normal(np.mean(mu_d_res), 1.0 / np.sqrt(n_s)))

            # 3. Cumulative adoption probabilities D_hat & control function residual r_cf
            d_hat = np.zeros((n, t_len), dtype=np.float64)
            for s in range(t_len):
                lam_s = stats.norm.cdf(time_mu_d[s] + (xi * tau_d) * z_arr[:, s])
                if s == 0:
                    d_hat[:, s] = lam_s
                else:
                    d_hat[:, s] = 1.0 - (1.0 - d_hat[:, s - 1]) * (1.0 - lam_s)
            r_cf = d_arr - d_hat

            # 4. Update (beta, rho_cf) jointly via bivariate Gaussian regression
            y_target = (y_arr - time_mu_y[None, :]).reshape(-1)
            m_mat = np.column_stack([d_arr.reshape(-1), r_cf.reshape(-1)])

            prior_prec_mat = np.diag([1.0 / self.config.beta_prior_var, 1.0])
            post_prec_theta = (m_mat.T @ m_mat) / (sigma_nu**2) + prior_prec_mat
            post_cov_theta = np.linalg.pinv(post_prec_theta)
            post_mean_theta = post_cov_theta @ (m_mat.T @ y_target / (sigma_nu**2))

            theta = rng.multivariate_normal(post_mean_theta, post_cov_theta)
            beta = float(theta[0])
            rho_cf = float(theta[1])

            # 5. Update time baseline effects mu_y
            res_no_time = y_arr - beta * d_arr - rho_cf * r_cf
            time_mu_y = rng.normal(np.mean(res_no_time, axis=0), sigma_nu / np.sqrt(n))

            # 6. Update residual variance sigma_nu
            res_nu = y_target - m_mat @ theta
            shape_ig = 2.0 + 0.5 * len(res_nu)
            scale_ig = 1.0 + 0.5 * np.sum(res_nu**2)
            sigma_nu = float(np.sqrt(stats.invgamma.rvs(shape_ig, scale=scale_ig, random_state=rng)))

            sigma_y = float(np.sqrt(sigma_nu**2 + (rho_cf**2) * float(np.var(r_cf))))

            if sweep >= burnin:
                beta_trace[trace_idx] = beta
                rho_trace[trace_idx] = rho_cf
                xi_trace[trace_idx] = xi
                sigma_y_trace[trace_idx] = sigma_y
                trace_idx += 1

        beta_mean = float(np.mean(beta_trace))
        beta_ci = (float(np.percentile(beta_trace, 2.5)), float(np.percentile(beta_trace, 97.5)))
        rho_mean = float(np.mean(rho_trace))
        xi_prob = float(np.mean(xi_trace))

        summary_data = {
            "parameter": ["beta (causal effect)", "rho_cf (endogeneity)", "xi (instrument relevance)", "sigma_y"],
            "posterior_mean": [beta_mean, rho_mean, xi_prob, float(np.mean(sigma_y_trace))],
            "ci_lower_2.5": [beta_ci[0], float(np.percentile(rho_trace, 2.5)), np.nan, float(np.percentile(sigma_y_trace, 2.5))],
            "ci_upper_97.5": [beta_ci[1], float(np.percentile(rho_trace, 97.5)), np.nan, float(np.percentile(sigma_y_trace, 97.5))],
        }
        summary_table = pd.DataFrame(summary_data)

        return CoupledHazardResult(
            beta_post=beta_trace,
            beta_mean=beta_mean,
            beta_ci=beta_ci,
            rho_post=rho_trace,
            rho_mean=rho_mean,
            xi_inclusion_prob=xi_prob,
            sigma_y_post=sigma_y_trace,
            summary_table=summary_table,
        )
