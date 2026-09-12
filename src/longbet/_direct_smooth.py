"""Direct smooth Gaussian encouragement forests with exact joint updates.

Parameterizes the panel reduced forms directly as:
    Y_{kit} = m_k(X_i, t) + gamma_{ki} + (Z_i - p) * 1(t >= t_0) * tau_k(X_i, t - t_0) + eps_{kit}

Leaves contain smooth Gaussian Process time vectors with fixed RBF covariance.
Tree topologies are sampled using exact finite-stump collapsed Gibbs steps.
Baseline and effect leaves and unit intercepts can be drawn jointly conditional on
topologies (direct_joint), removing the multiplicative parameterization.
Convergence still requires checking for the reported quantities.
Supports optional full Inverse-Wishart random intercept covariance across equations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import logsumexp
from scipy.stats import invwishart

from longbet._encourage import _critical, _validate, encouragement_effects
from longbet._encourage_model import EncouragementPrediction, INFERENCE_VERSION, _digest, _finite_matrix
from longbet._encourage_predict import EncouragementDraws


DIRECT_ARCHIVE_VERSION = 2


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
    kernel: str = "matern32"

    def __post_init__(self):
        if not isinstance(self.joint_baseline_intercept, bool):
            raise ValueError("joint_baseline_intercept must be boolean.")
        if not isinstance(self.joint_effect_leaves, bool):
            raise ValueError("joint_effect_leaves must be boolean.")
        if self.joint_effect_leaves and not self.joint_baseline_intercept:
            raise ValueError("joint_effect_leaves requires joint_baseline_intercept.")
        if not isinstance(self.correlated_intercepts, bool):
            raise ValueError("correlated_intercepts must be boolean.")
        if self.kernel not in ("rbf", "matern32", "matern12"):
            raise ValueError(f"Unknown kernel: {self.kernel}")
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
        if self.intercept_df <= 1:
            raise ValueError("intercept_df must exceed 1 for a proper 2x2 inverse-Wishart prior.")


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


def time_covariance(
    times: np.ndarray,
    *,
    sd: float,
    length_scale: float,
    nugget: float,
    kernel: str = "rbf",
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")
    delta = (times[:, None] - times[None, :]) / length_scale
    r = np.abs(delta)
    if kernel == "rbf":
        corr = np.exp(-0.5 * delta**2)
    elif kernel == "matern32":
        corr = (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    elif kernel == "matern12":
        corr = np.exp(-r)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    correlation = (corr + nugget * np.eye(len(times))) / (1 + nugget)
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


def _effect_contrast(state: SingleEquationState, effect: ForestDesign) -> np.ndarray:
    """Average Y(Z=1)-Y(Z=0) over all study units, before response rescaling.

    ``effect_fits`` includes each unit's observed centered assignment. Its
    average is an observed fitted contribution, not an intervention contrast.
    """
    masks = effect.space.mask[state.effect_rules]
    return np.einsum("jln,jlt->nt", masks, state.effect_leaves).mean(axis=0)


def _conditional_intercept_prior(
    states: list[SingleEquationState], covariance: np.ndarray, equation: int,
) -> tuple[np.ndarray, float]:
    other = 1 - equation
    loading = covariance[equation, other] / covariance[other, other]
    return (loading * states[other].gamma,
            float(covariance[equation, equation] - loading * covariance[other, equation]))


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
    gamma_prior_mean: np.ndarray | float = 0.0,
) -> None:
    n, periods = y.shape
    masks = baseline.space.mask[state.baseline_rules]
    factor = baseline.vectors * np.sqrt(baseline.eigenvalues)
    matrix = np.einsum("jln,tq->ntjlq", masks, factor).reshape(n, periods, -1)
    baseline_columns = matrix.shape[-1]
    prior_mean = np.broadcast_to(gamma_prior_mean, (n,))
    residual_without_baseline = y - prior_mean[:, None]
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
    centered_gamma = mean + np.sqrt(variance) * rng.normal(size=n)
    state.gamma = prior_mean + centered_gamma
    state.residual = residual_without_gamma - centered_gamma[:, None]


def sweep_equation_trees(
    rng: np.random.Generator,
    state: SingleEquationState,
    baseline: ForestDesign,
    effect: ForestDesign,
    config: DirectSmoothConfig,
    y: np.ndarray,
    start: int,
    *,
    gamma_prior_mean: np.ndarray | float = 0.0,
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
            joint_baseline_intercept_draw(rng, state, baseline, y, start, effect=effect,
                                          gamma_prior_mean=gamma_prior_mean)
        else:
            joint_baseline_intercept_draw(rng, state, baseline, y, start,
                                          gamma_prior_mean=gamma_prior_mean)
    else:
        state.residual += state.gamma[:, None]
        variance = 1 / (1 / state.gamma2 + y.shape[1] / state.sigma2)
        mean = variance * (state.residual.sum(axis=1) / state.sigma2
                           + gamma_prior_mean / state.gamma2)
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
    """Gaussian reduced forms with smooth stump leaves and joint Gibbs updates.

    The take-up equation is a working LPM. The effect target averages both
    assignment counterfactuals over the original units. Posterior calibration
    is not established; smoothness and variance priors affect inference.
    """

    def __init__(self, config: DirectSmoothConfig | None = None, **kwargs: Any):
        if config is not None and not isinstance(config, DirectSmoothConfig):
            raise TypeError("config must be a DirectSmoothConfig or None.")
        if config is not None and kwargs:
            raise ValueError("Pass either config or DirectSmoothConfig keyword options.")
        self.config = config if config is not None else DirectSmoothConfig(**kwargs)
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
        n_skip: int = 1,
    ) -> LongBetDirectSmooth:
        for name, value, minimum in (("seed", seed, 0), ("chains", chains, 1),
                                     ("burnin", burnin, 0), ("draws", draws, 1),
                                     ("n_skip", n_skip, 1)):
            if (isinstance(value, (bool, np.bool_)) or
                    not isinstance(value, (int, np.integer)) or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        panel = _validate(z, d, t)
        y_arr = _finite_matrix(y, "y", len(panel.z), panel.z.shape)
        d_arr = np.asarray(panel.d, dtype=float)
        x_arr = _finite_matrix(x, "x", len(panel.z))
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
            kernel=self.config.kernel,
        )
        effect_cov = time_covariance(
            times[start:],
            sd=self.config.effect_sd / np.sqrt(self.config.effect_trees),
            length_scale=self.config.length_scale,
            nugget=self.config.nugget,
            kernel=self.config.kernel,
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

            for it in range(burnin + draws * n_skip):
                for m in range(2):
                    prior_mean = 0.0
                    if self.config.correlated_intercepts:
                        # Every equation update must condition on the current
                        # other intercept, using the same covariance as the
                        # subsequent joint intercept/IW block.
                        prior_mean, states[m].gamma2 = _conditional_intercept_prior(
                            states, intercept_cov, m,
                        )
                    sweep_equation_trees(
                        chain_rng, states[m], baseline_design, effect_design,
                        self.config, standardized[..., m], start,
                        gamma_prior_mean=prior_mean,
                    )
                if self.config.correlated_intercepts:
                    intercept_cov = sweep_correlated_intercepts(
                        chain_rng, states, self.config, intercept_cov, periods,
                    )
                if it >= burnin and (it - burnin + 1) % n_skip == 0:
                    save_idx = (it - burnin + 1) // n_skip - 1
                    for m in range(2):
                        tau = _effect_contrast(states[m], effect_design) * scale[m]
                        itt_draws[c, save_idx, :, m] = tau
                        sigma2_draws[c, save_idx, m] = states[m].sigma2
                        gamma2_draws[c, save_idx, m] = (
                            intercept_cov[m, m] if self.config.correlated_intercepts
                            else states[m].gamma2
                        )
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
            first_stage="lpm", outcome="continuous", provenance="direct_smooth_v2",
            config=asdict(self.config),
            sampler=dict(seed=int(seed), chains=int(chains), burnin=int(burnin),
                         draws=int(draws), n_skip=int(n_skip)),
            inference_version=INFERENCE_VERSION,
            target="all_original_units_equal_weight",
            correlated_intercepts=self.config.correlated_intercepts,
            unit_intercept_covariance=("inverse_wishart" if self.config.correlated_intercepts
                                       else "independent_prior"),
            innovation_coupling="independent", shared_treatment_partitions=False,
            calibration_status="not_established",
        )
        for value in self._data.values():
            value.flags.writeable = False
        self.panel = panel
        self.fitted_ = True
        return self

    def predict(
        self,
        *,
        groups: Any = None,
        alpha: float = 0.05,
        target: str = "sample",
    ) -> EncouragementPrediction:
        """Create EncouragementPrediction object compatible with standard evaluation.

        Parameters
        ----------
        groups : unsupported for direct_smooth (must be None)
        alpha : significance level (default 0.05)
        target : 'sample' for Sample Average Treatment Effect (SATE) with
                 finite-sample counterfactual imputation variance, or 'population'
                 for superpopulation conditional mean function PATE.
        """
        if not self.fitted_:
            raise RuntimeError("LongBetDirectSmooth must be fitted before predict().")
        if groups is not None:
            raise ValueError("direct_smooth stores only the all-unit target; groups are unsupported.")
        if target not in ("sample", "population"):
            raise ValueError("target must be 'sample' or 'population'.")
        _critical(alpha)
        dy = self.draws["itt_y"].copy()  # (chains, draws, h)
        dd = self.draws["itt_d"].copy()

        if target == "sample":
            # Finite-sample SATE counterfactual imputation:
            # Impute the unobserved potential outcomes for each unit and draw
            # The missing counterfactual contributes residual variance sigma^2 / N to the sample average
            n_units = len(self._data["z"])
            scale_y = float(self._data["y"].std())
            scale_d = float(self._data["d"].std())
            sig_y = np.sqrt(self.draws["sigma2"][:, :, 0]) * scale_y
            sig_d = np.sqrt(self.draws["sigma2"][:, :, 1]) * scale_d
            rng_sate = np.random.default_rng(int(self.metadata["sampler"]["seed"]) + 999)
            noise_y = rng_sate.normal(size=dy.shape) * (sig_y[..., None] / np.sqrt(n_units))
            noise_d = rng_sate.normal(size=dd.shape) * (sig_d[..., None] / np.sqrt(n_units))
            dy += noise_y
            dd += noise_d

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
        """Save a versioned, pickle-free archive of study data and all traces."""
        if not self.fitted_:
            raise RuntimeError("Cannot save unfitted model.")
        arrays = {name: value for name, value in self._data.items() if name != "assignment"}
        arrays.update(self.draws)
        metadata = dict(
            self.metadata, kind="LongBetDirectSmooth", archive_version=DIRECT_ARCHIVE_VERSION,
            input_hashes={name: _digest(value) for name, value in arrays.items()},
        )
        arrays["metadata"] = np.asarray(json.dumps(metadata, allow_nan=False))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> LongBetDirectSmooth:
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                arrays = {name: archive[name] for name in archive.files if name != "metadata"}
            if (metadata.get("kind") != "LongBetDirectSmooth" or
                    metadata.get("archive_version") != DIRECT_ARCHIVE_VERSION or
                    metadata.get("provenance") != "direct_smooth_v2" or
                    metadata.get("inference_version") != INFERENCE_VERSION):
                raise ValueError("unsupported kind or version; refit legacy direct_smooth archives")
            config = DirectSmoothConfig(**metadata["config"])
            data_names = {"y", "d", "z", "x", "t"}
            draw_names = {"itt_y", "itt_d", "sigma2", "gamma2"}
            if config.correlated_intercepts:
                draw_names.add("intercept_cov")
            if arrays.keys() != data_names | draw_names:
                raise ValueError("unexpected or missing archive arrays")
            if metadata["input_hashes"] != {name: _digest(a) for name, a in arrays.items()}:
                raise ValueError("data or draw digest mismatch")
            obj = cls(config)
            obj.panel = _validate(arrays["z"], arrays["d"], arrays["t"])
            n, periods = arrays["z"].shape
            obj._data = {name: arrays[name] for name in data_names}
            obj._data["y"] = _finite_matrix(arrays["y"], "y", n, (n, periods))
            obj._data["x"] = _finite_matrix(arrays["x"], "x", n)
            obj._data["assignment"] = obj.panel.assigned.astype(float)
            sampler = metadata["sampler"]
            shape = (sampler["chains"], sampler["draws"], periods - obj.panel.start)
            for name in draw_names:
                expected = (shape if name.startswith("itt_") else
                            (*shape[:2], 2, 2) if name == "intercept_cov" else (*shape[:2], 2))
                if arrays[name].shape != expected or not np.isfinite(arrays[name]).all():
                    raise ValueError(f"invalid {name} draws")
            if any(metadata.get(k) != v for k, v in obj.panel.metadata().items()):
                raise ValueError("model/design provenance mismatch")
            obj.draws = {name: arrays[name] for name in draw_names}
            obj.metadata = {k: v for k, v in metadata.items()
                            if k not in ("kind", "archive_version", "input_hashes")}
            for value in obj._data.values():
                value.flags.writeable = False
            obj.fitted_ = True
            return obj
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"Invalid direct_smooth archive: {exc}") from exc
