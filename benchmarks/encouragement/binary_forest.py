"""Research single-horizon joint binary probit forests.

The finite stump support is the model, not an approximation to a deep BART
prior. Each component has its own proper sum-of-Gaussians prior. Probit utility
variance is fixed at one. Outcome forests estimate observational Y|D,Z,X.
Monotone uptake uses p1=Phi(f), p0=Phi(f)*Phi(g) and joint binary augmentation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri_exp

try:
    from .binary_cells import validate
    from .direct_smooth_candidate import ForestConfig, ForestDesign, TreeSpace, make_tree_space, sample_tree
except ImportError:  # Direct benchmark CLI execution.
    from binary_cells import validate
    from direct_smooth_candidate import ForestConfig, ForestDesign, TreeSpace, make_tree_space, sample_tree


@dataclass(frozen=True)
class BinaryConfig:
    trees: int = 3
    cutpoints: int = 4
    split_probability: float = .5
    prior_sd: float = 1.
    marginal_uptake_refresh: bool = False

    def __post_init__(self):
        if not isinstance(self.marginal_uptake_refresh, bool):
            raise ValueError("marginal_uptake_refresh must be boolean.")
        for value in (self.trees, self.cutpoints):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("trees and cutpoints must be positive integers.")
        if not 0 < self.split_probability < 1:
            raise ValueError("split_probability must lie strictly between zero and one.")
        if not np.isfinite(self.prior_sd) or self.prior_sd <= 0:
            raise ValueError("prior_sd must be finite and positive.")


def utility_draw(rng, eta, response):
    """Stable unit-variance truncated-normal utilities, including rare tails."""
    eta, response = np.broadcast_arrays(np.asarray(eta, float), np.asarray(response))
    if not np.isfinite(eta).all() or not np.isin(response, [0, 1]).all():
        raise ValueError("Finite predictors and binary responses are required.")
    sign = 2 * response - 1
    signed = sign * eta
    # 1-random() is in (0,1]; protecting floating-point random endpoints changes
    # no observed data, probability parameter, or small first-stage denominator.
    uniform = 1 - rng.random(eta.shape)
    return sign * (signed - ndtri_exp(log_ndtr(signed) + np.log(uniform)))


def monotone_labels(rng, f, g, d):
    """Joint draw of (Rf,Rg) given D=Rf*Rg for control-arm observations."""
    f, g, d = np.broadcast_arrays(np.asarray(f, float), np.asarray(g, float), np.asarray(d))
    if not np.isfinite(f).all() or not np.isfinite(g).all() or not np.isin(d, [0, 1]).all():
        raise ValueError("Finite predictors and binary adoption are required.")
    logp = np.stack([log_ndtr(-f) + log_ndtr(-g),
                     log_ndtr(f) + log_ndtr(-g),
                     log_ndtr(-f) + log_ndtr(g)], axis=-1)
    probability = np.exp(logp - logsumexp(logp, axis=-1, keepdims=True))
    selected = np.sum(rng.random(d.shape)[..., None] > np.cumsum(probability, axis=-1), axis=-1)
    # The final category includes any roundoff in the categorical CDF.
    rf = np.where(d == 1, 1, selected == 1).astype(int)
    rg = np.where(d == 1, 1, selected >= 2).astype(int)
    return rf, rg


def joint_leaf_draw(rng, matrix, utility, leaf_variance):
    """Exact scalar Gaussian block conditional on all forest topologies."""
    matrix = np.asarray(matrix, float)
    precision = matrix.T @ matrix + np.eye(matrix.shape[1]) / leaf_variance
    chol = np.linalg.cholesky(precision)
    return (cho_solve((chol, True), matrix.T @ utility, check_finite=False)
            + solve_triangular(chol.T, rng.normal(size=matrix.shape[1]),
                               lower=False, check_finite=False))


class ProbitForest:
    """One independent forest; likelihood rows may be empty, prior support may not."""

    def __init__(self, rng, space, grid_mask, rows, config, *, rules=None, leaves=None):
        self.space, self.grid_mask, self.rows, self.config = space, grid_mask, np.asarray(rows), config
        subset = TreeSpace(space.mask[..., self.rows], space.log_prior, space.feature, space.threshold)
        self.variance = config.prior_sd**2 / config.trees
        self.design = ForestDesign.build(subset, np.ones(len(self.rows)), np.array([[self.variance]]))
        if rules is None:
            self.rules = rng.choice(len(space.log_prior), config.trees, p=np.exp(space.log_prior))
            self.leaves = rng.normal(scale=np.sqrt(self.variance), size=(config.trees, 2))
        else:
            self.rules, self.leaves = np.array(rules, copy=True), np.array(leaves, copy=True)
        self.rebuild()
        self.residual = np.zeros(len(self.rows))

    def rebuild(self):
        self.fits = np.einsum("jln,jl->jn", self.space.mask[self.rules], self.leaves)

    def eta(self):
        return self.fits.sum(axis=0)

    def predict_grid(self):
        return np.einsum("jlg,jl->g", self.grid_mask[self.rules], self.leaves)

    def sweep(self, rng, response):
        utility = utility_draw(rng, self.eta()[self.rows], response)
        self.residual = utility - self.eta()[self.rows]
        for j in range(self.config.trees):
            self.residual += self.fits[j, self.rows]
            rule, leaves, fit = sample_tree(rng, self.design, self.residual[:, None], 1.)
            self.rules[j], self.leaves[j] = rule, leaves[:, 0]
            self.fits[j] = self.space.mask[rule].T @ leaves[:, 0]
            self.residual -= fit[:, 0]
        matrix = self.space.mask[self.rules][..., self.rows].reshape(-1, len(self.rows)).T if len(self.rows) else np.empty((0, 2 * self.config.trees))
        self.leaves = joint_leaf_draw(rng, matrix, utility, self.variance).reshape(self.config.trees, 2)
        self.rebuild()
        self.residual = utility - self.eta()[self.rows]
        return utility


def setup(x, grid, z, d, config):
    space = make_tree_space(x, ForestConfig(cutpoints=config.cutpoints, split_probability=config.split_probability))
    left = np.ones((len(space.feature), len(grid)), dtype=float)
    for k in range(1, len(left)):
        left[k] = grid[:, space.feature[k]] <= space.threshold[k]
    grid_mask = np.stack([left, 1 - left], axis=1)
    outcome_rows = [np.flatnonzero((d == dd) & (z == zz)) for dd in (0, 1) for zz in (0, 1)]
    return space, grid_mask, outcome_rows


def uptake_loglik(eta, d, z, first_stage):
    """Observed adoption likelihood, integrating all probit auxiliaries out."""
    if first_stage == "unrestricted":
        return np.sum(log_ndtr((2 * d - 1) * eta[z.astype(int), np.arange(len(d))]))
    f, g = eta
    log_one = np.where(z == 1, log_ndtr(f), log_ndtr(f) + log_ndtr(g))
    # D=0 in controls is the disjoint union Rf=0 or (Rf=1,Rg=0).
    log_zero = np.where(z == 1, log_ndtr(-f), np.logaddexp(log_ndtr(-f), log_ndtr(f) + log_ndtr(-g)))
    return np.sum(np.where(d == 1, log_one, log_zero))


def marginal_uptake_refresh(rng, forests, d, z, first_stage):
    """Elliptical slice updates with Gaussian leaf priors and marginal likelihood.

    Each uptake component is updated in turn. The auxiliary labels/utilities
    from earlier in the sweep are discarded; the next sweep regenerates them.
    Topologies, likelihood and priors are unchanged by this extra transition.
    """
    eta = np.array([f.eta() for f in forests[:2]])
    for j, f in enumerate(forests[:2]):
        direction = rng.normal(scale=np.sqrt(f.variance), size=f.leaves.shape)
        direction_eta = np.einsum("jln,jl->n", f.space.mask[f.rules], direction)
        log_threshold = uptake_loglik(eta, d, z, first_stage) + np.log(1 - rng.random())
        angle = rng.uniform(0, 2 * np.pi)
        lower, upper = angle - 2 * np.pi, angle
        for _ in range(10000):
            proposed_eta = eta.copy()
            proposed_eta[j] = eta[j] * np.cos(angle) + direction_eta * np.sin(angle)
            if uptake_loglik(proposed_eta, d, z, first_stage) >= log_threshold:
                f.leaves = f.leaves * np.cos(angle) + direction * np.sin(angle)
                f.rebuild()
                eta = proposed_eta
                break
            if angle < 0:
                lower = angle
            else:
                upper = angle
            angle = rng.uniform(lower, upper)
        else:
            raise RuntimeError("Elliptical slice bracket did not accept within 10000 evaluations.")


def model_sweep(rng, forests, d, z, y, first_stage):
    if first_stage == "monotone":
        control = np.flatnonzero(z == 0)
        rf, rg = monotone_labels(rng, forests[0].eta()[control], forests[1].eta()[control], d[control])
        labels = d.copy()
        labels[control] = rf
        forests[0].sweep(rng, labels)
        forests[1].sweep(rng, rg)
    else:
        for forest in forests[:2]:
            forest.sweep(rng, d[forest.rows])
    for forest in forests[2:]:
        forest.sweep(rng, y[forest.rows])
    if forests[0].config.marginal_uptake_refresh:
        old_eta = [f.eta()[f.rows] for f in forests[:2]]
        marginal_uptake_refresh(rng, forests, d, z, first_stage)
        # Preserve the residual against the stored utility as a numerical
        # invariant, although those utilities are discarded before the next step.
        for f, old in zip(forests[:2], old_eta):
            f.residual += old - f.eta()[f.rows]


def probabilities(eta, first_stage):
    probability = ndtr(eta)
    p = np.moveaxis(probability[..., :2, :], -2, -1).copy()
    if first_stage == "monotone":
        p = np.stack([p[..., 0] * p[..., 1], p[..., 0]], axis=-1)
    q = np.moveaxis(probability[..., 2:, :], -2, -1).reshape(*p.shape[:-1], 2, 2)
    return p, q, p * q[..., 1, :] + (1 - p) * q[..., 0, :]


def fit_binary(y, d, z, x, *, seed, chains=4, burnin=1000, draws=4000,
               first_stage="unrestricted", config=None, resume=None):
    """Sample p(C,K,G,Z), q(C,K,G,D,Z), parameter traces and resumable state."""
    y, d, z, x, grid, inverse, weights = validate(y, d, z, x)
    config = config or BinaryConfig()
    for value, minimum in ((chains, 1), (burnin, 0), (draws, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("Invalid chain, burn-in or draw count.")
    if first_stage not in ("unrestricted", "monotone"):
        raise ValueError("first_stage must be unrestricted or monotone.")
    config_json = json.dumps(dict(**asdict(config), first_stage=first_stage), sort_keys=True)
    digest = hashlib.sha256()
    for array in (y, d, z, x):
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    data_hash = digest.hexdigest()
    if resume is not None:
        saved_config = json.loads(str(resume["config_json"]))
        saved_config.setdefault("marginal_uptake_refresh", False)
        if burnin or saved_config != json.loads(config_json) or str(resume["data_hash"]) != data_hash:
            raise ValueError("Resume requires unchanged data/configuration and burnin=0.")
        if resume["rules"].shape[0] != chains:
            raise ValueError("Resume chain count differs.")
    space, grid_mask, outcome_rows = setup(x, grid, z, d, config)
    uptake_rows = ([np.arange(len(z)), np.flatnonzero(z == 0)] if first_stage == "monotone"
                   else [np.flatnonzero(z == zz) for zz in (0, 1)])
    rows = uptake_rows + outcome_rows
    rules = np.empty((chains, draws, 6, config.trees), dtype=int)
    leaves = np.empty((*rules.shape, 2))
    eta = np.empty((chains, draws, 6, len(grid)))
    rng_states = []
    for c, stream in enumerate(np.random.SeedSequence(seed).spawn(chains)):
        rng = np.random.default_rng(stream)
        if resume is not None:
            rng.bit_generator.state = json.loads(str(resume["rng_states"][c]))
        forests = [ProbitForest(rng, space, grid_mask, row, config,
                    rules=None if resume is None else resume["rules"][c, -1, j],
                    leaves=None if resume is None else resume["leaves"][c, -1, j]) for j, row in enumerate(rows)]
        for iteration in range(burnin + draws):
            model_sweep(rng, forests, d, z, y, first_stage)
            if iteration >= burnin:
                k = iteration - burnin
                rules[c, k] = [f.rules for f in forests]
                leaves[c, k] = [f.leaves for f in forests]
                eta[c, k] = [f.predict_grid() for f in forests]
        rng_states.append(json.dumps(rng.bit_generator.state, sort_keys=True))
    p, q, r = probabilities(eta, first_stage)
    return dict(p=p, q=q, r=r, weights=weights, grid=grid, rules=rules, leaves=leaves, eta=eta,
                rng_states=np.asarray(rng_states), config_json=np.asarray(config_json),
                data_hash=np.asarray(data_hash), seed=np.asarray(seed), first_stage=np.asarray(first_stage),
                burnin=np.asarray(burnin), feature=space.feature, threshold=space.threshold)
