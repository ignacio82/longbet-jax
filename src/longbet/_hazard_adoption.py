"""Experimental discrete-time hazard adoption forest with a relevance indicator.

Models the dynamic adoption hazard on the active longitudinal risk set:
    lambda_{it}(z) = Pr(D_{it} = 1 | D_{i, t-1} = 0, Z_i = z, X_i)
                   = Phi(m_d(X_i, t) + (z - p) * 1(t >= t_0) * xi * tau_d(X_i, t - t_0))

The modeled adoption stock probability is:
    Pr(D_{it}(z) = 1 | X_i) = 1 - prod_{s <= t} (1 - lambda_{is}(z))

Expected cumulative treatment exposure is:
    E[sum_{s <= t} D_{is}(z) | X_i] = sum_{s <= t} Pr(D_{is}(z) = 1 | X_i)

An indicator xi in {0, 1} permits exactly zero encouragement effects. This alone
does not identify an outcome effect or eliminate weak-instrument bias. One random
stump basis is fixed and shared across chains; tree partitions are not learned.
Probit utilities, baseline leaves, and a collapsed relevance/effect block target
the posterior conditional on that basis. Posterior interval calibration remains
unestablished, and convergence and sensitivity to the basis and priors matter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import expit, log_ndtr, logit, ndtr, ndtri_exp

from longbet._encourage import _critical, _validate
from longbet._encourage_model import _finite_matrix


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
    """Working-model summaries and traces conditional on a fixed random basis.

    ``table`` includes:
    - ``period``, ``horizon``
    - ``hazard_itt``: modeled hazard contrast averaged over original baseline units
    - ``stock_itt``: modeled adoption stock probability difference
    - ``exposure_itt``: modeled expected cumulative exposure contrast
    - ``relevance_prob``: estimate of posterior Pr(xi = 1 | data, fixed basis)

    The hazard contrast is not a comparison restricted to one common set of
    counterfactual survivors. Stock and exposure summaries require the working
    hazard specification, including its treatment of baseline adoption.
    Neither interval calibration nor sampler convergence follows from these
    summaries; the relevance probability is not a test of IV identification.
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


def _effect_block_parameters(
    residual: np.ndarray, risk: np.ndarray, basis: np.ndarray, leaf_variance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Gaussian effect conditionals and log slab/spike marginal likelihood ratio.

    Each period has independent N(0, leaf_variance I) coefficients. The returned
    means and Cholesky precision factors use coefficients whitened by the prior.
    All redundant and inactive basis columns remain proper augmented parameters.
    """
    horizons, columns = residual.shape[1], basis.shape[1]
    means = np.empty((horizons, columns))
    cholesky = np.empty((horizons, columns, columns))
    log_bayes_factor = 0.0
    for h in range(horizons):
        design = np.sqrt(leaf_variance) * basis[risk[:, h]]
        response = residual[risk[:, h], h]
        precision = np.eye(columns) + design.T @ design
        rhs = design.T @ response
        chol = np.linalg.cholesky(precision)
        means[h] = cho_solve((chol, True), rhs, check_finite=False)
        cholesky[h] = chol
        log_bayes_factor += 0.5 * rhs @ means[h] - np.log(np.diag(chol)).sum()
    return means, cholesky, float(log_bayes_factor)


def _sample_effect_block(
    rng: np.random.Generator, residual: np.ndarray, risk: np.ndarray,
    basis: np.ndarray, leaf_variance: float, prior_inclusion_prob: float,
) -> tuple[int, np.ndarray]:
    """Joint relevance/leaf Gibbs draw; inactive effects follow their prior."""
    means, cholesky, log_bayes_factor = _effect_block_parameters(
        residual, risk, basis, leaf_variance,
    )
    probability = expit(logit(prior_inclusion_prob) + log_bayes_factor)
    xi = int(rng.random() < probability)
    normal = rng.normal(size=means.shape)
    if xi:
        for h in range(len(means)):
            normal[h] = means[h] + solve_triangular(
                cholesky[h].T, normal[h], lower=False, check_finite=False,
            )
    return xi, np.sqrt(leaf_variance) * normal.T


class HazardAdoptionForest:
    """Experimental fixed-basis probit hazard model; calibration is unestablished."""

    def __init__(self, config: HazardConfig | None = None, **kwargs: Any):
        if config is not None and not isinstance(config, HazardConfig):
            raise TypeError("config must be a HazardConfig or None.")
        if config is not None and kwargs:
            raise ValueError("Pass either config or HazardConfig keyword options.")
        self.config = config if config is not None else HazardConfig(**kwargs)
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
        for name, value, minimum in (("seed", seed, 0), ("chains", chains, 1),
                                     ("burnin", burnin, 0), ("draws", draws, 1)):
            if (isinstance(value, (bool, np.bool_)) or
                    not isinstance(value, (int, np.integer)) or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        panel = _validate(z, d, t)
        n, periods = panel.d.shape
        start = panel.start
        times = panel.t
        assignment = panel.assigned.astype(float)
        centered_z = assignment - np.mean(assignment)
        x_arr = _finite_matrix(x, "x", n)
        if periods > 2 and not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-10, atol=0):
            raise ValueError("The hazard model requires equally spaced observed periods.")

        # Build risk set: unit i is at risk in period t if D_{i, t-1} == 0.
        # At the first observation all units are at risk: baseline adopters are
        # treated as first-period events, rather than left-truncated histories.
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

        # Condition on one random feature basis. Different fixed bases in each
        # chain would target different conditional posteriors and could not be
        # treated as replicated chains of a single posterior.
        basis_seed, sampler_seed = np.random.SeedSequence(seed).spawn(2)
        basis_rng = np.random.default_rng(basis_seed)
        base_rules = basis_rng.choice(n_rules, size=self.config.baseline_trees, p=np.exp(space.log_prior))
        eff_rules = basis_rng.choice(n_rules, size=self.config.effect_trees, p=np.exp(space.log_prior))
        effect_basis = space.mask[eff_rules].transpose(2, 0, 1).reshape(n, -1) * centered_z[:, None]
        chain_seeds = sampler_seed.spawn(chains)

        for c in range(chains):
            rng = np.random.default_rng(chain_seeds[c])
            base_leaves = np.zeros((self.config.baseline_trees, 2, periods))
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

                # 3. Integrate the effect leaves to sample relevance, then draw
                # all leaves jointly under that state. No BIC-like penalty is
                # needed: the Gaussian marginal likelihood supplies the prior's
                # actual complexity adjustment.
                eta_base = base_fits.sum(axis=0)
                xi, coefficients = _sample_effect_block(
                    rng, u[:, start:] - eta_base[:, start:], risk[:, start:],
                    effect_basis, eff_leaf_var, self.config.prior_inclusion_prob,
                )
                eff_leaves = coefficients.reshape(self.config.effect_trees, 2, h)
                raw_effects = np.einsum("jln,jlt->jnt", space.mask[eff_rules], eff_leaves)
                eff_val = raw_effects.sum(axis=0)
                eff_fits = raw_effects * centered_z[None, :, None]

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
            seed=int(seed),
            inference_version="hazard_fixed_basis_v2",
            experimental=True,
            calibration_status="not_established",
            topology="fixed_random_stumps_shared_across_chains",
            baseline_rules=base_rules.tolist(),
            effect_rules=eff_rules.tolist(),
            posterior_conditioning="observed_data_and_fixed_random_basis",
            relevance_interpretation="model_posterior_inclusion_probability_conditional_on_fixed_basis",
            sampler="probit_utilities_and_collapsed_relevance_effect_block",
            weak_instrument_robust=False,
            causal_outcome_inference_supported=False,
            duration_unit="observed_period",
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
    """Fit the experimental fixed-basis hazard model and return summaries.

    Probabilities and credible intervals condition on a fixed random stump basis.
    Calibration is unestablished; relevance does not validate IV identification.
    """
    forest = HazardAdoptionForest(config=config)
    if x is None:
        x = np.ones((len(z), 1))
    forest.fit(d, z, x, t=t, seed=seed, chains=chains, burnin=burnin, draws=draws)
    return forest.summary(alpha=alpha)
