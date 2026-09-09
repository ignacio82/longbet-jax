"""Discrete-time hazard adoption model with spike-and-slab first-stage relevance.

Models the dynamic adoption hazard on the active longitudinal risk set:
    lambda_{it}(z) = Pr(D_{it} = 1 | D_{i, t-1} = 0, Z_i = z, X_i)
                   = Phi(m_d(X_i, t) + (z - p) * 1(t >= t_0) * xi * tau_d(X_i, t - t_0))

Adoption is naturally absorbing:
    D_{it}(z) = 1 - prod_{s <= t} (1 - lambda_{is}(z))

Cumulative treatment exposure duration is:
    E_{it}(z) = sum_{s <= t} D_{is}(z)

The spike-and-slab indicator xi in {0, 1} places non-zero prior mass on the exact null
(xi = 0, no effect of encouragement on uptake), eliminating the weak-instrument bias
and false precision of continuous ordered priors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri_exp

from longbet._encourage import _critical, _validate


@dataclass(frozen=True)
class HazardConfig:
    """Configuration for Discrete-Time Hazard Adoption Model."""

    baseline_trees: int = 4
    effect_trees: int = 4
    cutpoints: int = 4
    split_probability: float = 0.5
    prior_inclusion_prob: float = 0.5
    baseline_sd: float = 1.0
    effect_sd: float = 1.0

    def __init__(
        self,
        *,
        trees: int | None = None,
        baseline_trees: int = 4,
        effect_trees: int = 4,
        cutpoints: int = 4,
        split_probability: float = 0.5,
        prior_inclusion_prob: float = 0.5,
        baseline_sd: float = 1.0,
        effect_sd: float = 1.0,
    ):
        if trees is not None:
            baseline_trees = trees
            effect_trees = trees
        object.__setattr__(self, "baseline_trees", baseline_trees)
        object.__setattr__(self, "effect_trees", effect_trees)
        object.__setattr__(self, "cutpoints", cutpoints)
        object.__setattr__(self, "split_probability", split_probability)
        object.__setattr__(self, "prior_inclusion_prob", prior_inclusion_prob)
        object.__setattr__(self, "baseline_sd", baseline_sd)
        object.__setattr__(self, "effect_sd", effect_sd)
        for name in ("baseline_trees", "effect_trees", "cutpoints"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("split_probability", "prior_inclusion_prob"):
            val = getattr(self, name)
            if not (0 < val < 1):
                raise ValueError(f"{name} must be strictly between 0 and 1.")
        for name in ("baseline_sd", "effect_sd"):
            val = getattr(self, name)
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True)
class HazardAdoptionResult:
    """Summary and posterior draws from the Discrete-Time Hazard Adoption Model.

    ``table`` includes:
    - ``period``, ``horizon``
    - ``hazard_itt``: effect of encouragement on period adoption hazard among survivors
    - ``stock_itt``: cumulative adoption stock difference
    - ``exposure_itt``: cumulative exposure duration contrast
    - ``relevance_prob``: posterior probability Pr(xi = 1 | data)
    """

    table: pd.DataFrame
    draws: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _utility_draw(rng: np.random.Generator, eta: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Stable unit-variance truncated normal draws on the active risk set."""
    sign = 2.0 * response - 1.0
    signed = sign * eta
    uniform = 1.0 - rng.random(eta.shape)
    return sign * (signed - ndtri_exp(log_ndtr(signed) + np.log(uniform)))


class HazardAdoptionForest:
    """Discrete-Time Hazard Model with Spike-and-Slab First-Stage Relevance."""

    def __init__(self, config: HazardConfig | None = None, **kwargs: Any):
        self.config = config or HazardConfig(**kwargs)
        self.draws: dict[str, np.ndarray] = {}
        self.fitted_ = False
        self.metadata: dict[str, Any] = {}

    def fit(
        self,
        d: Any,
        z: Any,
        x: Any,
        t: Any = None,
        *,
        seed: int = 0,
        chains: int = 4,
        burnin: int = 400,
        draws: int = 600,
    ) -> HazardAdoptionForest:
        panel = _validate(z, d, t)
        n, periods = panel.d.shape
        start = panel.start
        times = panel.t
        assignment = panel.assigned.astype(float)
        centered_z = assignment - np.mean(assignment)
        x_arr = np.asarray(x, dtype=float)

        # Build risk set: unit i is at risk in period t if D_{i, t-1} == 0.
        # For t=0, all units with D_{i, 0} == 0 at baseline (or pre-adoption) are at risk.
        risk = np.ones((n, periods), dtype=bool)
        for j in range(1, periods):
            risk[:, j] = panel.d[:, j - 1] == 0

        # Feature grids for baseline covariates
        # Simple finite cutpoint representation
        from longbet._direct_smooth import make_tree_space, DirectSmoothConfig
        smooth_config = DirectSmoothConfig(
            cutpoints=self.config.cutpoints,
            split_probability=self.config.split_probability,
        )
        space = make_tree_space(x_arr, smooth_config)
        n_rules = len(space.log_prior)
        h = periods - start

        # Allocate storage
        itt_hazard = np.empty((chains, draws, h))
        itt_stock = np.empty((chains, draws, h))
        itt_exposure = np.empty((chains, draws, h))
        xi_draws = np.empty((chains, draws), dtype=int)

        # Precompute leaf variance
        base_leaf_var = (self.config.baseline_sd ** 2) / self.config.baseline_trees
        eff_leaf_var = (self.config.effect_sd ** 2) / self.config.effect_trees

        for c in range(chains):
            rng = np.random.default_rng(seed + c * 10007)
            base_rules = rng.choice(n_rules, size=self.config.baseline_trees, p=np.exp(space.log_prior))
            base_leaves = np.zeros((self.config.baseline_trees, 2, periods))
            eff_rules = rng.choice(n_rules, size=self.config.effect_trees, p=np.exp(space.log_prior))
            eff_leaves = np.zeros((self.config.effect_trees, 2, h))
            base_fits = np.zeros((self.config.baseline_trees, n, periods))
            eff_fits = np.zeros((self.config.effect_trees, n, h))
            xi = 1

            for it in range(burnin + draws):
                eta_base = base_fits.sum(axis=0)
                eff_sum = eff_fits.sum(axis=0)
                eta_eff = np.zeros((n, periods))
                if xi == 1:
                    eta_eff[:, start:] = eff_sum
                eta = eta_base + eta_eff

                # 1. Sample utilities on risk set
                u = np.zeros((n, periods))
                u[risk] = _utility_draw(rng, eta[risk], panel.d[risk])

                # 2. Update baseline trees
                for tr in range(self.config.baseline_trees):
                    m = space.mask[base_rules[tr]]
                    res = (u - eta_eff) - (base_fits.sum(axis=0) - base_fits[tr])
                    for t_idx in range(periods):
                        at_risk_t = risk[:, t_idx]
                        for l in (0, 1):
                            idx = at_risk_t & (m[l] == 1)
                            count = int(np.sum(idx))
                            if count > 0:
                                post_var = 1.0 / (1.0 / base_leaf_var + count)
                                post_mean = post_var * np.sum(res[idx, t_idx])
                                base_leaves[tr, l, t_idx] = post_mean + np.sqrt(post_var) * rng.normal()
                            else:
                                base_leaves[tr, l, t_idx] = rng.normal(scale=np.sqrt(base_leaf_var))
                    base_fits[tr] = m.T @ base_leaves[tr]

                # 3. Update effect trees
                eta_base = base_fits.sum(axis=0)
                eff_val = np.zeros((n, h))
                for tr in range(self.config.effect_trees):
                    m = space.mask[eff_rules[tr]]
                    res = (u[:, start:] - eta_base[:, start:]) - (eff_fits.sum(axis=0) - eff_fits[tr])
                    for h_idx in range(h):
                        t_idx = start + h_idx
                        at_risk_t = risk[:, t_idx]
                        for l in (0, 1):
                            idx = at_risk_t & (m[l] == 1)
                            w = centered_z[idx]
                            count = np.sum(w ** 2)
                            if count > 0:
                                post_var = 1.0 / (1.0 / eff_leaf_var + count)
                                post_mean = post_var * np.sum(w * res[idx, h_idx])
                                eff_leaves[tr, l, h_idx] = post_mean + np.sqrt(post_var) * rng.normal()
                            else:
                                eff_leaves[tr, l, h_idx] = rng.normal(scale=np.sqrt(eff_leaf_var))
                    fit_tr_raw = m.T @ eff_leaves[tr]
                    eff_val += fit_tr_raw
                    eff_fits[tr] = fit_tr_raw * centered_z[:, None]

                # 4. Spike-and-slab update for xi
                eff_sum = eff_fits.sum(axis=0)
                res_zero = (u[:, start:] - eta_base[:, start:])[risk[:, start:]]
                res_one = (u[:, start:] - (eta_base[:, start:] + eff_sum))[risk[:, start:]]
                n_active = len(res_one)
                pen = 0.5 * self.config.effect_trees * np.log(max(n_active, 10))
                loglik_0 = -0.5 * np.sum(res_zero ** 2) + np.log(1.0 - self.config.prior_inclusion_prob)
                loglik_1 = -0.5 * np.sum(res_one ** 2) - pen + np.log(self.config.prior_inclusion_prob)
                p_one = 1.0 / (1.0 + np.exp(np.clip(loglik_0 - loglik_1, -50.0, 50.0)))
                xi = int(rng.binomial(1, p_one))

                # 6. Save draws if past burn-in
                if it >= burnin:
                    save_idx = it - burnin
                    xi_draws[c, save_idx] = xi
                    # Compute counterfactual hazards:
                    # Under Z=1 (centered_z = 1 - p):
                    # Under Z=0 (centered_z = -p):
                    p_mean = np.mean(assignment)
                    lambda_0 = np.zeros((n, periods))
                    lambda_1 = np.zeros((n, periods))
                    for t_idx in range(periods):
                        if t_idx < start or xi == 0:
                            lambda_0[:, t_idx] = ndtr(eta_base[:, t_idx])
                            lambda_1[:, t_idx] = lambda_0[:, t_idx]
                        else:
                            h_idx = t_idx - start
                            tau = eff_val[:, h_idx]
                            lambda_1[:, t_idx] = ndtr(eta_base[:, t_idx] + (1.0 - p_mean) * tau)
                            lambda_0[:, t_idx] = ndtr(eta_base[:, t_idx] + (0.0 - p_mean) * tau)

                    # Cumulative stock: S_{it} = 1 - prod_{s <= t} (1 - lambda_{is})
                    stock_0 = 1.0 - np.cumprod(1.0 - lambda_0, axis=1)
                    stock_1 = 1.0 - np.cumprod(1.0 - lambda_1, axis=1)

                    # Exposure duration: E_{it} = sum_{s <= t} S_{is}
                    exp_0 = np.cumsum(stock_0, axis=1)
                    exp_1 = np.cumsum(stock_1, axis=1)

                    itt_h = np.mean(lambda_1[:, start:] - lambda_0[:, start:], axis=0)
                    itt_s = np.mean(stock_1[:, start:] - stock_0[:, start:], axis=0)
                    itt_e = np.mean(exp_1[:, start:] - exp_0[:, start:], axis=0)

                    itt_hazard[c, save_idx] = itt_h
                    itt_stock[c, save_idx] = itt_s
                    itt_exposure[c, save_idx] = itt_e

        self.draws = dict(
            hazard_itt=itt_hazard,
            stock_itt=itt_stock,
            exposure_itt=itt_exposure,
            xi=xi_draws,
        )
        self.metadata = dict(
            **panel.metadata(),
            config=asdict(self.config),
            relevance_probability=float(np.mean(xi_draws)),
            chains=chains,
            draws=draws,
            burnin=burnin,
        )
        self.panel = panel
        self.fitted_ = True
        return self

    def summary(self, alpha: float = 0.05) -> HazardAdoptionResult:
        """Tidy summary of hazard, stock, and exposure contrasts."""
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling summary().")
        _critical(alpha)
        rows = []
        h = self.panel.exposure[self.panel.start:].size
        relevance = float(np.mean(self.draws["xi"]))
        for h_idx in range(h):
            row = dict(
                period=float(self.panel.t[self.panel.start + h_idx]),
                horizon=int(self.panel.exposure[self.panel.start + h_idx]),
                relevance_prob=relevance,
            )
            for key, name in (
                ("hazard_itt", "hazard"),
                ("stock_itt", "stock"),
                ("exposure_itt", "exposure"),
            ):
                arr = self.draws[key][..., h_idx].ravel()
                row[f"{name}_itt_median"] = float(np.median(arr))
                row[f"{name}_itt_lower"] = float(np.quantile(arr, alpha / 2))
                row[f"{name}_itt_upper"] = float(np.quantile(arr, 1.0 - alpha / 2))
            rows.append(row)
        table = pd.DataFrame(rows)
        return HazardAdoptionResult(table=table, draws=self.draws, metadata=self.metadata)


def hazard_adoption_effects(
    d: Any,
    z: Any,
    x: Any = None,
    t: Any = None,
    *,
    config: HazardConfig | None = None,
    seed: int = 0,
    chains: int = 4,
    burnin: int = 400,
    draws: int = 600,
    alpha: float = 0.05,
) -> HazardAdoptionResult:
    """Convenience function to fit hazard adoption model and return summary."""
    forest = HazardAdoptionForest(config=config)
    if x is None:
        x = np.ones((len(z), 1))
    forest.fit(d, z, x, t=t, seed=seed, chains=chains, burnin=burnin, draws=draws)
    return forest.summary(alpha=alpha)
