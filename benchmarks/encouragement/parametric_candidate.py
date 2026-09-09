"""Research control: direct, smooth Gaussian encouragement regression.

The model replaces the product of treatment forests, coding, and learned GP
factors with additive baseline and encouragement coefficient trajectories:

    y_it = m(x_i, t) + (A_i - p) post_t tau(x_i, h) + gamma_i + epsilon_it.

Each coefficient trajectory has a fixed, proper RBF covariance. The baseline
features are a constant and centered/scaled baseline covariates. Their common
normalization makes the average prior variance of each fitted function equal to
one (before the configurable baseline/effect SD). This matches the forest's
average marginal scale, not its nonlinear function prior. Responses are centered
and scaled using observed data, as in the paired forest experiment. Priors on
innovation and unit variances are IG(3, 2) on that working scale.

Each Gaussian equation samples *all* coefficients and unit intercepts jointly,
by analytically integrating intercepts, drawing coefficients, then intercepts.
The variance updates are conjugate. No production API or default is changed;
binary adoption is an LPM response and Gaussian coverage is not presumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from collections.abc import Mapping

import numpy as np
from scipy.linalg import cho_solve, solve_triangular


@dataclass(frozen=True)
class ParametricConfig:
    length_scale: float = 2.
    nugget: float = .05
    baseline_sd: float = 1.
    effect_sd: float = 1.
    sigma_shape: float = 3.
    sigma_scale: float = 2.
    gamma_shape: float = 3.
    gamma_scale: float = 2.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


def time_covariance(times, *, sd=1., length_scale=2., nugget=.05):
    times = np.asarray(times, dtype=float)
    delta = (times[:, None] - times[None, :]) / length_scale
    return sd**2 * (np.exp(-.5 * delta**2) + nugget * np.eye(len(times))) / (1 + nugget)


@dataclass
class GaussianState:
    coefficients: np.ndarray  # whitened coefficients, Q
    gamma: np.ndarray  # N
    sigma2: float
    gamma2: float
    residual: np.ndarray  # N,T, always y - full fitted values


@dataclass
class GaussianDesign:
    """Precomputed sufficient statistics for stable intercept marginalization."""

    matrix: np.ndarray  # N,T,Q
    mean_matrix: np.ndarray  # N,Q
    within_crossproduct: np.ndarray  # Q,Q
    between_crossproduct: np.ndarray  # Q,Q

    @classmethod
    def build(cls, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 3 or not np.isfinite(matrix).all():
            raise ValueError("matrix must be a finite N,T,Q design.")
        if min(matrix.shape) == 0:
            raise ValueError("The design cannot have an empty axis.")
        means = matrix.mean(axis=1)
        centered = (matrix - means[:, None, :]).reshape(-1, matrix.shape[-1])
        return cls(matrix, means, centered.T @ centered,
                   matrix.shape[1] * (means.T @ means))

    def response_statistics(self, y):
        y = np.asarray(y, dtype=float)
        if y.shape != self.matrix.shape[:2] or not np.isfinite(y).all():
            raise ValueError("y must be finite and match the design's unit/time axes.")
        means = y.mean(axis=1)
        # A centered representation avoids subtracting two large nearly equal
        # crossproducts when random-intercept variance is large.
        centered = self.matrix - self.mean_matrix[:, None, :]
        within = np.einsum("itq,it->q", centered, y - means[:, None])
        between = y.shape[1] * (self.mean_matrix.T @ means)
        return within, between


def coefficient_posterior(design, statistics, sigma2, gamma2):
    """Marginal beta Gaussian conditional, with all unit intercepts integrated.

    Whitened coefficients have identity prior covariance. Within-unit contrasts
    have variance sigma2, and normalized unit means sigma2 + T gamma2.
    Returns mean and a Cholesky factor of precision, never a GP precision.
    """
    if not np.isfinite([sigma2, gamma2]).all() or min(sigma2, gamma2) <= 0:
        raise ValueError("Both working variances must be finite and positive.")
    between_variance = sigma2 + design.matrix.shape[1] * gamma2
    precision = (np.eye(design.matrix.shape[-1])
                 + design.within_crossproduct / sigma2
                 + design.between_crossproduct / between_variance)
    rhs = statistics[0] / sigma2 + statistics[1] / between_variance
    chol = np.linalg.cholesky(precision)
    mean = cho_solve((chol, True), rhs, check_finite=False)
    return mean, chol


def joint_gaussian_draw(rng, design, y, statistics, sigma2, gamma2):
    """An exact joint conditional sample of every coefficient and intercept."""
    mean, chol = coefficient_posterior(design, statistics, sigma2, gamma2)
    coefficients = mean + solve_triangular(
        chol.T, rng.normal(size=len(mean)), lower=False, check_finite=False)
    residual_without_gamma = y - np.einsum("itq,q->it", design.matrix, coefficients)
    variance = sigma2 * gamma2 / (sigma2 + y.shape[1] * gamma2)
    gamma_mean = (gamma2 / (sigma2 + y.shape[1] * gamma2)
                  * residual_without_gamma.sum(axis=1))
    gamma = gamma_mean + np.sqrt(variance) * rng.normal(size=len(y))
    return coefficients, gamma, residual_without_gamma - gamma[:, None]


def prior_state(rng, design, y, config=ParametricConfig()):
    sigma2 = config.sigma_scale / rng.gamma(config.sigma_shape)
    gamma2 = config.gamma_scale / rng.gamma(config.gamma_shape)
    coefficients = rng.normal(size=design.matrix.shape[-1])
    gamma = rng.normal(scale=np.sqrt(gamma2), size=len(y))
    residual = y - np.einsum("itq,q->it", design.matrix, coefficients) - gamma[:, None]
    return GaussianState(coefficients, gamma, sigma2, gamma2, residual)


def sweep(rng, state, design, y, statistics=None, config=ParametricConfig()):
    """One exact joint-Gaussian/variance Gibbs sweep, preserving full residual."""
    if statistics is None:
        statistics = design.response_statistics(y)
    coefficients, gamma, residual = joint_gaussian_draw(
        rng, design, y, statistics, state.sigma2, state.gamma2)
    sigma2 = ((config.sigma_scale + .5 * np.sum(residual**2))
              / rng.gamma(config.sigma_shape + y.size / 2))
    gamma2 = ((config.gamma_scale + .5 * np.sum(gamma**2))
              / rng.gamma(config.gamma_shape + len(y) / 2))
    return GaussianState(coefficients, gamma, sigma2, gamma2, residual)


def make_design(x, assignment, times, start, config=ParametricConfig()):
    x = np.asarray(x, dtype=float)
    assignment = np.asarray(assignment, dtype=float)
    times = np.asarray(times, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite N,P baseline covariate matrix.")
    if (assignment.shape != (len(x),) or not np.isin(assignment, [0, 1]).all()
            or not 0 < assignment.mean() < 1):
        raise ValueError("assignment must contain both binary randomized arms.")
    if (times.ndim != 1 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("times must be a finite strictly increasing vector.")
    if (isinstance(start, bool) or not isinstance(start, (int, np.integer))
            or not 0 <= start < len(times)):
        raise ValueError("start must be a valid integer period index.")
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-10, 1., scale)
    features = np.column_stack([np.ones(len(x)), (x - center) / scale])
    # Constant columns contribute no variance after centering. Normalize by
    # actual average feature norm, not the nominal number of columns.
    normalization = np.sqrt(np.mean(np.sum(features**2, axis=1)))
    features /= normalization
    baseline_chol = np.linalg.cholesky(time_covariance(
        times, sd=config.baseline_sd, length_scale=config.length_scale, nugget=config.nugget))
    effect_chol = np.linalg.cholesky(time_covariance(
        times[start:], sd=config.effect_sd, length_scale=config.length_scale, nugget=config.nugget))
    baseline = np.einsum("ip,tq->itpq", features, baseline_chol).reshape(len(x), len(times), -1)
    effect_basis = np.zeros((len(times), len(times) - start))
    effect_basis[start:] = effect_chol
    effect = (np.einsum("ip,tq->itpq", features, effect_basis)
              * (assignment - assignment.mean())[:, None, None, None]).reshape(len(x), len(times), -1)
    matrix = np.concatenate([baseline, effect], axis=-1)
    return GaussianDesign.build(matrix), {
        "features": features, "feature_center": center, "feature_scale": scale,
        "feature_normalization": np.asarray(normalization),
        "baseline_cholesky": baseline_chol, "effect_cholesky": effect_chol,
        "assignment_probability": np.asarray(assignment.mean()),
    }


def _groups(groups, n):
    labels, memberships = ["all"], [np.ones(n, dtype=bool)]
    if groups is not None:
        if not isinstance(groups, Mapping):
            raise ValueError("groups must map labels to boolean unit-membership arrays.")
        for label, membership in groups.items():
            membership = np.asarray(membership)
            if not isinstance(label, str) or label == "all" or label in labels:
                raise ValueError("Group labels must be unique strings other than 'all'.")
            if membership.dtype.kind != "b" or membership.shape != (n,) or not membership.any():
                raise ValueError("Every group must contain at least one unit and use a boolean N-vector.")
            labels.append(label)
            memberships.append(membership)
    return np.asarray(labels), np.asarray(memberships)


def _fingerprint(data, start, config):
    digest = hashlib.sha256(json.dumps({"start": int(start), "config": asdict(config)},
                                     sort_keys=True).encode())
    for name in ("x", "assignment", "t", "y", "d"):
        value = np.ascontiguousarray(data[name], dtype=np.float64)
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit_candidate(data, start, *, seed, chains=4, burnin=300, draws=300,
                  config=ParametricConfig(), resume=None, groups=None):
    """Fit the two independent reduced forms and retain replayable raw draws.

    Returns ``({outcome,takeup}: (groups,horizons,chains,draws), archive)``.
    ``resume`` accepts a previous archive and requires burnin=0. Each resumed
    stream continues exactly from its saved state and RNG; the seed is ignored
    on continuation. Different input data/configurations or chain counts fail.
    The raw unit/parameter arrays use standardized-response units; returned
    encouragement contrasts are converted back to outcome/adoption units.
    """
    for name, value, minimum in (("chains", chains, 1), ("burnin", burnin, 0), ("draws", draws, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    config = config or ParametricConfig()
    raw_y, raw_d = np.asarray(data["y"], dtype=float), np.asarray(data["d"], dtype=float)
    if raw_y.ndim != 2 or raw_d.shape != raw_y.shape:
        raise ValueError("Require aligned complete outcome/adoption panels.")
    if not np.isin(raw_d, [0, 1]).all() or np.any(np.diff(raw_d, axis=1) < 0):
        raise ValueError("Adoption must be a complete absorbing binary panel.")
    y = np.stack([raw_y, raw_d], axis=-1)
    if y.ndim != 3 or not np.isfinite(y).all():
        raise ValueError("y and d must be finite unit-by-period response matrices.")
    n, t, equations = y.shape
    design, details = make_design(data["x"], data["assignment"], data["t"], start, config)
    if design.matrix.shape[:2] != (n, t):
        raise ValueError("Response dimensions do not match baseline covariates and times.")
    if "z" in data:
        expected_z = np.zeros_like(raw_y)
        expected_z[:, start:] = np.asarray(data["assignment"])[:, None]
        if not np.array_equal(np.asarray(data["z"]), expected_z):
            raise ValueError("Candidate supports a single randomized encouragement wave.")
    center = y.mean(axis=(0, 1))
    scale = y.std(axis=(0, 1))
    scale = np.where(scale < 1e-10, 1., scale)
    standardized = (y - center) / scale
    statistics = [design.response_statistics(standardized[..., m]) for m in range(equations)]
    group_labels, memberships = _groups(groups, n)
    feature_means = np.stack([details["features"][mask].mean(axis=0) for mask in memberships])
    h, p, q = t - start, details["features"].shape[1], design.matrix.shape[-1]
    fingerprint = _fingerprint(data, start, config)
    if resume is not None:
        if int(np.asarray(resume["format_version"]).item()) != 1:
            raise ValueError("Unsupported candidate archive version.")
        if burnin:
            raise ValueError("Resuming requires burnin=0; warmup has already been performed.")
        if str(np.asarray(resume["design_fingerprint"]).item()) != fingerprint:
            raise ValueError("Resume archive does not match input data or prior configuration.")
        if np.asarray(resume["final_coefficients"]).shape != (chains, q, equations):
            raise ValueError("Resume archive does not match the requested chain count/design.")
    coefficients = np.empty((chains, draws, q, equations))
    gamma = np.empty((chains, draws, n, equations))
    sigma2 = np.empty((chains, draws, equations))
    gamma2 = np.empty_like(sigma2)
    initial_coefficients = np.empty((chains, q, equations))
    initial_gamma = np.empty((chains, n, equations))
    initial_variances = np.empty((chains, equations, 2))
    effects = np.empty((len(group_labels), h, chains, draws, equations))
    rng_states = []
    for c, child in enumerate(np.random.SeedSequence(seed).spawn(chains)):
        rng = np.random.default_rng(child)
        if resume is None:
            states = [prior_state(rng, design, standardized[..., m], config) for m in range(equations)]
        else:
            rng.bit_generator.state = json.loads(str(resume["rng_state_json"][c]))
            states = []
            for m in range(equations):
                beta = np.asarray(resume["final_coefficients"][c, :, m]).copy()
                unit = np.asarray(resume["final_unit_intercepts"][c, :, m]).copy()
                residual = standardized[..., m] - np.einsum("itq,q->it", design.matrix, beta) - unit[:, None]
                states.append(GaussianState(beta, unit, float(resume["final_innovation_variance"][c, m]),
                                            float(resume["final_unit_variance"][c, m]), residual))
        for m, state in enumerate(states):
            initial_coefficients[c, :, m] = state.coefficients
            initial_gamma[c, :, m] = state.gamma
            initial_variances[c, m] = [state.sigma2, state.gamma2]
        for iteration in range(burnin + draws):
            for m in range(equations):
                state = sweep(rng, states[m], design, standardized[..., m], statistics[m], config)
                states[m] = state
                if iteration >= burnin:
                    k = iteration - burnin
                    coefficients[c, k, :, m] = state.coefficients
                    gamma[c, k, :, m] = state.gamma
                    sigma2[c, k, m], gamma2[c, k, m] = state.sigma2, state.gamma2
                    tau = state.coefficients[p * t:].reshape(p, h) @ details["effect_cholesky"].T
                    effects[:, :, c, k, m] = feature_means @ tau * scale[m]
        rng_states.append(json.dumps(rng.bit_generator.state, sort_keys=True))
    metadata = {
        "candidate": "direct_smooth_linear_gaussian", "config": asdict(config),
        "target": "average encouraged versus never-encouraged contrast over original units",
        "response_scaling": "observed pooled mean and population standard deviation; constant scale=1",
        "feature_prior": "independent whitened normal coefficient trajectories; average fitted marginal variance=SD squared",
        "unit_intercept_coupling": "independent", "innovation_coupling": "independent",
        "first_stage": "Gaussian LPM working response", "calibration_status": "not_established",
        "chains": chains, "burnin": burnin, "draws": draws, "seed": seed,
        "resumed": resume is not None, "start": int(start),
    }
    probe_indices = np.unique([0, n // 2, n - 1])
    baseline_curves = np.einsum(
        "ckpqm,tq->ckptm", coefficients[:, :, :p * t].reshape(chains, draws, p, t, equations),
        details["baseline_cholesky"])
    effect_curves = np.einsum(
        "ckpqm,hq->ckphm", coefficients[:, :, p * t:].reshape(chains, draws, p, h, equations),
        details["effect_cholesky"])
    baseline_probe = (np.einsum("ip,ckptm->ckitm", details["features"][probe_indices], baseline_curves)
                      * scale + center)
    effect_probe = (np.einsum("ip,ckphm->ckihm", details["features"][probe_indices], effect_curves)
                    * scale)
    archive = {
        "coefficient_draws": coefficients, "unit_intercept_draws": gamma,
        "innovation_variance_draws": sigma2, "unit_variance_draws": gamma2,
        "final_coefficients": coefficients[:, -1].copy(),
        "final_unit_intercepts": gamma[:, -1].copy(),
        "final_innovation_variance": sigma2[:, -1].copy(),
        "final_unit_variance": gamma2[:, -1].copy(),
        "initial_coefficients": initial_coefficients, "initial_unit_intercepts": initial_gamma,
        "initial_variances": initial_variances,
        "rng_state_json": np.asarray(rng_states), "design_fingerprint": np.asarray(fingerprint),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "response_center": center, "response_scale": scale,
        "group_labels": group_labels, "group_membership": memberships,
        "probe_indices": probe_indices, "baseline_probe": baseline_probe, "effect_probe": effect_probe,
        "sigma2": sigma2, "gamma2": gamma2,
        "gamma": gamma, "itt": effects[0].transpose(1, 2, 0, 3),
        "center": center, "scale": scale, "format_version": np.asarray(1),
        "times": np.asarray(data["t"], dtype=float), **details,
    }
    return {"outcome": effects[..., 0], "takeup": effects[..., 1]}, archive


def geweke(replications=3000, seed=11347, *, sweep_function=None):
    """Independent prior-data-one-Gibbs checks against analytic prior moments.

    Each replication draws a fresh joint prior then simulates the likelihood;
    one Gibbs sweep must preserve its parameter marginal. The independent
    replications make reported Monte Carlo SEs directly estimable. This checks
    exact conditional algebra, not adequacy of the RED likelihood.
    """
    if replications < 2:
        raise ValueError("At least two independent replications are required.")
    rng = np.random.default_rng(seed)
    design = GaussianDesign.build(rng.normal(size=(5, 3, 4)))
    config = ParametricConfig()
    update = sweep if sweep_function is None else sweep_function
    values = []
    for _ in range(replications):
        prior = prior_state(rng, design, np.zeros((5, 3)), config)
        y = (np.einsum("itq,q->it", design.matrix, prior.coefficients)
             + prior.gamma[:, None] + rng.normal(size=(5, 3)) * np.sqrt(prior.sigma2))
        prior.residual = y - np.einsum("itq,q->it", design.matrix, prior.coefficients) - prior.gamma[:, None]
        state = update(rng, prior, design, y, design.response_statistics(y), config)
        values.append([state.coefficients.mean(), np.mean(state.coefficients**2),
                       state.gamma.mean(), np.mean(state.gamma**2), state.sigma2,
                       state.gamma2, 1 / state.sigma2, 1 / state.gamma2,
                       state.coefficients[0] * state.gamma[0]])
    values = np.asarray(values)
    expectation = np.array([0., 1., 0., 1., 1., 1., 1.5, 1.5, 0.])
    estimate = values.mean(axis=0)
    mcse = values.std(axis=0, ddof=1) / np.sqrt(replications)
    difference = estimate - expectation
    z = np.divide(difference, mcse, out=np.zeros_like(difference), where=mcse > 0)
    z[(mcse == 0) & (difference != 0)] = np.copysign(np.inf, difference[(mcse == 0) & (difference != 0)])
    return {
        "method": "independent marginal-conditional Geweke transitions",
        "replications": replications, "seed": seed,
        "functions": ["coefficient_mean", "coefficient_second_moment", "gamma_mean",
                      "gamma_second_moment", "innovation_variance", "unit_variance",
                      "innovation_precision", "unit_precision", "coefficient_gamma_product"],
        "expectation": expectation, "estimate": estimate, "mcse": mcse, "z": z,
        "pass": bool(np.isfinite(z).all() and np.all(np.abs(z) < 5)),
    }
