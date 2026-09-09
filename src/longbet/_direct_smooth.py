"""Direct smooth Gaussian encouragement forests with exact joint updates.

Parameterizes the panel reduced forms directly as:
    Y_{kit} = m_k(X_i, t) + gamma_{ki} + (Z_i - p) * 1(t >= t_0) * tau_k(X_i, t - t_0) + eps_{kit}

Leaves contain smooth Gaussian Process time vectors with fixed RBF covariance.
Tree topologies are sampled using exact finite-stump collapsed Gibbs steps.
Baseline and effect leaves and unit intercepts can be drawn jointly conditional on
topologies (direct_joint), eliminating the MCMC mixing failure of multiplicative GP models.
Supports optional full Inverse-Wishart random intercept covariance across equations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import logsumexp
from scipy.stats import invwishart

from longbet._encourage import _critical, _validate, encouragement_effects
from longbet._encourage_model import EncouragementPrediction, INFERENCE_VERSION
from longbet._encourage_predict import EncouragementDraws, _target_weights


@dataclass(frozen=True)
class DirectSmoothConfig:
    """Configuration for Direct Smooth encouragement forests."""

    joint_baseline_intercept: bool = True
    joint_effect_leaves: bool = True
    baseline_trees: int = 6
    effect_trees: int = 6
    cutpoints: int = 4
    split_probability: float = 0.5
    length_scale: float = 2.0
    nugget: float = 0.05
    baseline_sd: float = 1.0
    effect_sd: float = 1.0
    sigma_shape: float = 3.0
    sigma_scale: float = 2.0
    gamma_shape: float = 3.0
    gamma_scale: float = 2.0
    correlated_intercepts: bool = False
    intercept_df: float = 7.0
    intercept_scale: float = 4.0

    def __post_init__(self):
        if not isinstance(self.joint_baseline_intercept, bool):
            raise ValueError("joint_baseline_intercept must be boolean.")
        if not isinstance(self.joint_effect_leaves, bool):
            raise ValueError("joint_effect_leaves must be boolean.")
        if self.joint_effect_leaves and not self.joint_baseline_intercept:
            raise ValueError("joint_effect_leaves requires joint_baseline_intercept.")
        if not isinstance(self.correlated_intercepts, bool):
            raise ValueError("correlated_intercepts must be boolean.")
        for name in ("baseline_trees", "effect_trees", "cutpoints"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < self.split_probability < 1:
            raise ValueError("split_probability must be strictly between 0 and 1.")
        for name in (
            "length_scale", "nugget", "baseline_sd", "effect_sd",
            "sigma_shape", "sigma_scale", "gamma_shape", "gamma_scale",
            "intercept_df", "intercept_scale",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass
class TreeSpace:
    """Finite prior support; mask[rule, leaf, unit]."""

    mask: np.ndarray
    log_prior: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray


def make_tree_space(x: np.ndarray, config: DirectSmoothConfig) -> TreeSpace:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) < 2:
        raise ValueError("x must be a finite baseline covariate matrix.")
    n = len(x)
    left = [np.ones(n, dtype=bool)]
    features, thresholds = [-1], [np.nan]
    for column in range(x.shape[1]):
        cuts = np.unique(np.quantile(
            x[:, column],
            np.arange(1, config.cutpoints + 1) / (config.cutpoints + 1),
        ))
        for cut in cuts:
            mask = x[:, column] <= cut
            if mask.any() and (~mask).any():
                left.append(mask)
                features.append(column)
                thresholds.append(cut)
    left_arr = np.asarray(left)
    right_arr = ~left_arr
    prior = np.ones(len(left_arr))
    if len(left_arr) > 1:
        prior[0] = 1 - config.split_probability
        prior[1:] = config.split_probability / (len(left_arr) - 1)
    return TreeSpace(
        np.stack([left_arr, right_arr], axis=1).astype(float),
        np.log(prior),
        np.asarray(features),
        np.asarray(thresholds),
    )


def time_covariance(times: np.ndarray, *, sd: float, length_scale: float, nugget: float) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    delta = (times[:, None] - times[None, :]) / length_scale
    correlation = (np.exp(-0.5 * delta**2) + nugget * np.eye(len(times))) / (1 + nugget)
    return sd**2 * correlation


@dataclass
class ForestDesign:
    space: TreeSpace
    weight: np.ndarray
    eigenvalues: np.ndarray
    vectors: np.ndarray
    counts: np.ndarray
    weighted_masks: np.ndarray

    @classmethod
    def build(cls, space: TreeSpace, weight: np.ndarray, covariance: np.ndarray) -> ForestDesign:
        values, vectors = np.linalg.eigh(covariance)
        if np.min(values) <= 0:
            raise ValueError("Leaf covariance must be positive definite.")
        masks = space.mask * np.asarray(weight)[None, None, :]
        return cls(space, np.asarray(weight), values, vectors,
                   np.sum(masks**2, axis=2), masks)

    def collapsed(self, residual: np.ndarray, sigma2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = self.weighted_masks @ residual
        projected = total @ self.vectors
        variance = (self.eigenvalues[None, None, :] * sigma2
                    / (sigma2 + self.counts[..., None] * self.eigenvalues))
        log_integral = 0.5 * np.sum(
            variance * (projected / sigma2)**2
            - np.log1p(self.counts[..., None] * self.eigenvalues / sigma2),
            axis=-1,
        )
        score = self.space.log_prior + log_integral.sum(axis=1)
        return score, variance, projected


def sample_tree(
    rng: np.random.Generator,
    design: ForestDesign,
    residual: np.ndarray,
    sigma2: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    score, variances, projected = design.collapsed(residual, sigma2)
    probability = np.exp(score - logsumexp(score))
    rule = int(rng.choice(len(probability), p=probability))
    eigen_mean = variances[rule] * projected[rule] / sigma2
    leaves = ((eigen_mean + np.sqrt(variances[rule]) * rng.normal(size=eigen_mean.shape))
              @ design.vectors.T)
    fitted = (design.space.mask[rule].T @ leaves) * design.weight[:, None]
    return rule, leaves, fitted


@dataclass
class SingleEquationState:
    baseline_rules: np.ndarray
    baseline_leaves: np.ndarray
    effect_rules: np.ndarray
    effect_leaves: np.ndarray
    gamma: np.ndarray
    sigma2: float
    gamma2: float
    baseline_fits: np.ndarray
    effect_fits: np.ndarray
    residual: np.ndarray


def _fitted_equation(state: SingleEquationState, start: int) -> np.ndarray:
    value = state.baseline_fits.sum(axis=0) + state.gamma[:, None]
    value[:, start:] += state.effect_fits.sum(axis=0)
    return value


def init_equation_state(
    rng: np.random.Generator,
    baseline: ForestDesign,
    effect: ForestDesign,
    config: DirectSmoothConfig,
    y: np.ndarray,
    start: int,
) -> SingleEquationState:
    def forest(design: ForestDesign, trees: int):
        rules = rng.choice(len(design.space.log_prior), size=trees,
                           p=np.exp(design.space.log_prior))
        leaves = ((rng.normal(size=(trees, 2, len(design.eigenvalues)))
                   * np.sqrt(design.eigenvalues)) @ design.vectors.T)
        fits = np.stack([(design.space.mask[rule].T @ leaf) * design.weight[:, None]
                         for rule, leaf in zip(rules, leaves)])
        return rules, leaves, fits

    br, bl, bf = forest(baseline, config.baseline_trees)
    er, el, ef = forest(effect, config.effect_trees)
    sigma2 = config.sigma_scale / rng.gamma(config.sigma_shape)
    gamma2 = config.gamma_scale / rng.gamma(config.gamma_shape)
    gamma = rng.normal(scale=np.sqrt(gamma2), size=len(y))
    state = SingleEquationState(br, bl, er, el, gamma, sigma2, gamma2, bf, ef, np.empty_like(y))
    state.residual = y - _fitted_equation(state, start)
    return state


def joint_baseline_intercept_draw(
    rng: np.random.Generator,
    state: SingleEquationState,
    baseline: ForestDesign,
    y: np.ndarray,
    start: int,
    *,
    effect: ForestDesign | None = None,
) -> None:
    n, periods = y.shape
    masks = baseline.space.mask[state.baseline_rules]
    factor = baseline.vectors * np.sqrt(baseline.eigenvalues)
    matrix = np.einsum("jln,tq->ntjlq", masks, factor).reshape(n, periods, -1)
    baseline_columns = matrix.shape[-1]
    residual_without_baseline = y.copy()
    if effect is None:
        residual_without_baseline[:, start:] -= state.effect_fits.sum(axis=0)
    else:
        effect_masks = effect.space.mask[state.effect_rules]
        effect_factor = effect.vectors * np.sqrt(effect.eigenvalues)
        effect_matrix = np.zeros((n, periods, effect_masks.shape[0] * 2 * (periods - start)))
        effect_matrix[:, start:] = (
            np.einsum("jln,tq->ntjlq", effect_masks, effect_factor)
            .reshape(n, periods - start, -1) * effect.weight[:, None, None]
        )
        matrix = np.concatenate([matrix, effect_matrix], axis=-1)
    x_mean = matrix.mean(axis=1)
    y_mean = residual_without_baseline.mean(axis=1)
    within = matrix - x_mean[:, None, :]
    within_flat = within.reshape(n * periods, -1)
    between_variance = state.sigma2 + periods * state.gamma2
    precision = (
        np.eye(matrix.shape[-1])
        + within_flat.T @ within_flat / state.sigma2
        + periods * x_mean.T @ x_mean / between_variance
    )
    rhs = (
        np.einsum("ntq,nt->q", within, residual_without_baseline - y_mean[:, None]) / state.sigma2
        + periods * x_mean.T @ y_mean / between_variance
    )
    chol = np.linalg.cholesky(precision)
    whitened = (
        cho_solve((chol, True), rhs, check_finite=False)
        + solve_triangular(chol.T, rng.normal(size=len(rhs)), lower=False, check_finite=False)
    )
    state.baseline_leaves = whitened[:baseline_columns].reshape(len(masks), 2, periods) @ factor.T
    state.baseline_fits = np.einsum("jln,jlt->jnt", masks, state.baseline_leaves)
    if effect is not None:
        state.effect_leaves = (
            whitened[baseline_columns:].reshape(len(effect_masks), 2, periods - start)
            @ effect_factor.T
        )
        state.effect_fits = (
            np.einsum("jln,jlt->jnt", effect_masks, state.effect_leaves)
            * effect.weight[None, :, None]
        )
        residual_without_baseline[:, start:] -= state.effect_fits.sum(axis=0)
    residual_without_gamma = residual_without_baseline - state.baseline_fits.sum(axis=0)
    variance = state.sigma2 * state.gamma2 / between_variance
    mean = state.gamma2 * residual_without_gamma.sum(axis=1) / between_variance
    state.gamma = mean + np.sqrt(variance) * rng.normal(size=n)
    state.residual = residual_without_gamma - state.gamma[:, None]


def sweep_equation_trees(
    rng: np.random.Generator,
    state: SingleEquationState,
    baseline: ForestDesign,
    effect: ForestDesign,
    config: DirectSmoothConfig,
    y: np.ndarray,
    start: int,
) -> None:
    for j in range(config.baseline_trees):
        state.residual += state.baseline_fits[j]
        rule, leaves, fit = sample_tree(rng, baseline, state.residual, state.sigma2)
        state.baseline_rules[j], state.baseline_leaves[j] = rule, leaves
        state.baseline_fits[j] = fit
        state.residual -= fit
    for j in range(config.effect_trees):
        state.residual[:, start:] += state.effect_fits[j]
        rule, leaves, fit = sample_tree(rng, effect, state.residual[:, start:], state.sigma2)
        state.effect_rules[j], state.effect_leaves[j] = rule, leaves
        state.effect_fits[j] = fit
        state.residual[:, start:] -= fit
    if config.joint_baseline_intercept:
        if config.joint_effect_leaves:
            joint_baseline_intercept_draw(rng, state, baseline, y, start, effect=effect)
        else:
            joint_baseline_intercept_draw(rng, state, baseline, y, start)
    else:
        state.residual += state.gamma[:, None]
        variance = 1 / (1 / state.gamma2 + y.shape[1] / state.sigma2)
        mean = variance * state.residual.sum(axis=1) / state.sigma2
        state.gamma = mean + np.sqrt(variance) * rng.normal(size=len(y))
        state.residual -= state.gamma[:, None]
    if not config.correlated_intercepts:
        state.gamma2 = ((config.gamma_scale + 0.5 * np.sum(state.gamma**2))
                        / rng.gamma(config.gamma_shape + len(y) / 2))
    state.sigma2 = ((config.sigma_scale + 0.5 * np.sum(state.residual**2))
                    / rng.gamma(config.sigma_shape + y.size / 2))


def sweep_correlated_intercepts(
    rng: np.random.Generator,
    states: list[SingleEquationState],
    config: DirectSmoothConfig,
    intercept_cov: np.ndarray,
    n_periods: int,
) -> np.ndarray:
    """Exact Gibbs draw of full 2x2 random intercept covariance and joint unit intercepts."""
    n = len(states[0].gamma)
    # Residuals without gamma for both equations: shape (n, n_periods, 2)
    resids = np.stack([s.residual + s.gamma[:, None] for s in states], axis=-1)
    sigmas = np.array([s.sigma2 for s in states])
    # Precision per unit: inv(Sigma_gamma) + diag(T / sigma_k^2)
    inv_cov = np.linalg.inv(intercept_cov)
    diag_term = np.diag(n_periods / sigmas)
    v_inv = inv_cov + diag_term
    v_gamma = np.linalg.inv(v_inv)
    chol_v = np.linalg.cholesky(v_gamma)
    # Mean: (sum_t resid_{kit} / sigma_k^2) @ v_gamma
    sum_resid = resids.sum(axis=1) / sigmas[None, :]
    mean_gamma = sum_resid @ v_gamma
    # Draw joint intercepts
    gamma_joint = mean_gamma + rng.normal(size=(n, 2)) @ chol_v.T
    for m, state in enumerate(states):
        state.gamma = gamma_joint[:, m]
        state.residual = resids[..., m] - state.gamma[:, None]
    # Update Inverse-Wishart covariance: IW(nu + N, Psi + sum gamma gamma')
    psi = config.intercept_scale * np.eye(2) + gamma_joint.T @ gamma_joint
    intercept_cov = invwishart.rvs(df=config.intercept_df + n, scale=psi, random_state=rng)
    return intercept_cov


class LongBetDirectSmooth:
    """Direct Smooth Bayesian encouragement model with exact joint Gibbs updates."""

    def __init__(self, config: DirectSmoothConfig | None = None, **kwargs: Any):
        self.config = config or DirectSmoothConfig(**kwargs)
        self._data: dict[str, np.ndarray] = {}
        self.metadata: dict[str, Any] = {}
        self.draws: dict[str, np.ndarray] = {}
        self.fitted_ = False

    def fit(
        self,
        y: Any,
        d: Any,
        z: Any,
        x: Any,
        t: Any = None,
        *,
        seed: int = 0,
        chains: int = 4,
        burnin: int = 1000,
        draws: int = 2000,
    ) -> LongBetDirectSmooth:
        panel = _validate(z, d, t)
        y_arr = np.asarray(y, dtype=float)
        d_arr = np.asarray(panel.d, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        if y_arr.shape != d_arr.shape:
            raise ValueError(f"y shape {y_arr.shape} must match d shape {d_arr.shape}.")
        n, periods = y_arr.shape
        start = panel.start
        times = panel.t
        assignment = panel.assigned.astype(float)
        raw = np.stack([y_arr, d_arr], axis=-1)  # (N, T, 2)
        center = raw.mean(axis=(0, 1))
        scale = raw.std(axis=(0, 1))
        scale = np.where(scale > 1e-10, scale, 1.0)
        standardized = (raw - center) / scale

        # Designs
        space = make_tree_space(x_arr, self.config)
        base_cov = time_covariance(
            times,
            sd=self.config.baseline_sd / np.sqrt(self.config.baseline_trees),
            length_scale=self.config.length_scale,
            nugget=self.config.nugget,
        )
        effect_cov = time_covariance(
            times[start:],
            sd=self.config.effect_sd / np.sqrt(self.config.effect_trees),
            length_scale=self.config.length_scale,
            nugget=self.config.nugget,
        )
        centered = assignment - np.mean(assignment)
        baseline_design = ForestDesign.build(space, np.ones(n), base_cov)
        effect_design = ForestDesign.build(space, centered, effect_cov)

        h = periods - start
        itt_draws = np.empty((chains, draws, h, 2))
        sigma2_draws = np.empty((chains, draws, 2))
        gamma2_draws = np.empty((chains, draws, 2))
        cov_draws = np.empty((chains, draws, 2, 2)) if self.config.correlated_intercepts else None

        for c in range(chains):
            chain_rng = np.random.default_rng(seed + c * 10007)
            states = [
                init_equation_state(chain_rng, baseline_design, effect_design, self.config, standardized[..., m], start)
                for m in range(2)
            ]
            intercept_cov = self.config.intercept_scale * np.eye(2)

            for it in range(burnin + draws):
                for m in range(2):
                    sweep_equation_trees(
                        chain_rng, states[m], baseline_design, effect_design,
                        self.config, standardized[..., m], start,
                    )
                if self.config.correlated_intercepts:
                    intercept_cov = sweep_correlated_intercepts(
                        chain_rng, states, self.config, intercept_cov, periods,
                    )
                if it >= burnin:
                    save_idx = it - burnin
                    for m in range(2):
                        # Population average contrast = sum_j (effect_fits) * scale
                        tau = states[m].effect_fits.sum(axis=0).mean(axis=0) * scale[m]
                        itt_draws[c, save_idx, :, m] = tau
                        sigma2_draws[c, save_idx, m] = states[m].sigma2
                        gamma2_draws[c, save_idx, m] = states[m].gamma2
                    if cov_draws is not None:
                        cov_draws[c, save_idx] = intercept_cov

        self._data = dict(
            y=y_arr, d=d_arr, z=panel.z, x=x_arr, t=panel.t, assignment=assignment,
        )
        self.draws = dict(
            itt_y=itt_draws[..., 0],  # (chains, draws, h)
            itt_d=itt_draws[..., 1],
            sigma2=sigma2_draws,
            gamma2=gamma2_draws,
        )
        if cov_draws is not None:
            self.draws["intercept_cov"] = cov_draws
        self.metadata = dict(
            **panel.metadata(),
            engine="direct_smooth",
            config=asdict(self.config),
            inference_version=INFERENCE_VERSION,
            target="all_original_units_equal_weight",
            correlated_intercepts=self.config.correlated_intercepts,
        )
        self.panel = panel
        self.fitted_ = True
        return self

    def predict(
        self,
        *,
        groups: Any = None,
        alpha: float = 0.05,
    ) -> EncouragementPrediction:
        """Create EncouragementPrediction object compatible with standard evaluation."""
        if not self.fitted_:
            raise RuntimeError("LongBetDirectSmooth must be fitted before predict().")
        _critical(alpha)
        dy = self.draws["itt_y"]  # (chains, draws, h)
        dd = self.draws["itt_d"]
        chains, draws, h = dy.shape
        # Format axes to (group, horizon, chain, draw)
        dy_formatted = dy.transpose(2, 0, 1)[None, ...]  # (1, h, chains, draws)
        dd_formatted = dd.transpose(2, 0, 1)[None, ...]

        # Reference
        ref_df = encouragement_effects(
            self._data["y"], self._data["d"], self._data["z"], self._data["t"], alpha=alpha,
        )
        ref_df.insert(0, "group", "all")
        ref_df["n_units"] = len(self._data["z"])

        std_draws = EncouragementDraws(
            draws={"outcome": dy_formatted, "takeup": dd_formatted},
            group_labels=("all",),
            group_counts=np.array([len(self._data["z"])]),
            weights=np.ones((1, len(self._data["z"]))) / len(self._data["z"]),
            periods=self.panel.t[self.panel.start:],
            period_indices=np.arange(self.panel.start, len(self.panel.t)),
            horizons=self.panel.exposure[self.panel.start:],
            cell_draws=None,
            cell_summaries={},
            standardization="conditional",
            provenance="direct_smooth",
        )
        return EncouragementPrediction(std_draws, ref_df, self.metadata, alpha=alpha)

    def save(self, path: str | Path) -> None:
        """Save fitted direct smooth model."""
        if not self.fitted_:
            raise RuntimeError("Cannot save unfitted model.")
        np.savez_compressed(
            path,
            y=self._data["y"], d=self._data["d"], z=self._data["z"],
            x=self._data["x"], t=self._data["t"],
            itt_y=self.draws["itt_y"], itt_d=self.draws["itt_d"],
            sigma2=self.draws["sigma2"], gamma2=self.draws["gamma2"],
            config_json=json.dumps(asdict(self.config)),
            metadata_json=json.dumps(self.metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> LongBetDirectSmooth:
        with np.load(path) as archive:
            config = DirectSmoothConfig(**json.loads(str(archive["config_json"])))
            obj = cls(config)
            obj._data = dict(
                y=archive["y"], d=archive["d"], z=archive["z"],
                x=archive["x"], t=archive["t"],
                assignment=(archive["z"][:, -1] == 1).astype(float),
            )
            obj.draws = dict(
                itt_y=archive["itt_y"], itt_d=archive["itt_d"],
                sigma2=archive["sigma2"], gamma2=archive["gamma2"],
            )
            obj.metadata = json.loads(str(archive["metadata_json"]))
            obj.panel = _validate(obj._data["z"], obj._data["d"], obj._data["t"])
            obj.fitted_ = True
            return obj
