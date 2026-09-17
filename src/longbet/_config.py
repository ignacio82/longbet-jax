# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the LongBet sampler."""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal


@dataclass(frozen=True)
class LongBetConfig:
    """Configuration of the LongBet panel model and its sampler.

    The model
    ---------
    For unit ``i`` in period ``t`` with exposure ``S`` (periods since the
    absorbing treatment started, 0 before it), the mean is

    .. code::

        alpha(mu(x, t)) + beta_S * nu(x, t) * 1{S >= 1} + gamma_i

    with ``mu`` a prognostic tree ensemble, ``nu`` a treatment tree ensemble
    for covariate (and optionally calendar-time) heterogeneity, ``beta_S`` a
    Gaussian-process trajectory over the exposure clock that gives the effect
    its time profile, and ``gamma_i`` a unit random intercept. The treatment
    term enters treated cells only, so the untreated surface is ``mu + gamma``
    and the effect at exposure ``S`` is ``beta_S * nu``. Continuous outcomes
    have Gaussian innovations; binary and ordinal outcomes use a probit latent
    response.

    The sampler
    -----------
    Every sweep updates both forests with GROW, PRUNE, CHANGE and one
    data-driven REGROW proposal per forest, then the exposure trajectory, the
    unit intercepts, the variances, a Metropolis move along the exact scale
    ridge between ``beta`` and the treatment leaves, and an inter-ensemble
    residual-transfer move between ``mu`` and ``nu``. Parallel tempering
    (``tempering_levels``) is available for posteriors whose forest modes the
    local moves connect too slowly. Chains start overdispersed from the prior
    so that R-hat measures convergence rather than seed agreement.

    Defaults
    --------
    Four chains, 2,000 burn-in sweeps and 1,000 retained sweeps per chain.
    Four thousand retained draws are what the package's own convergence gate
    (bulk and tail ESS of at least 400, rank R-hat at most 1.01 on every
    reported quantity) needs to be attainable; check ``att_stability`` or the
    encouragement model's ``stability()`` on your own panel rather than
    assuming it. ``jax`` dispatches asynchronously, so time a fit with
    ``jax.block_until_ready``.

    Parameters
    ----------
    num_sweeps
        Retained draws per chain after burn-in.
    num_burnin
        Burn-in sweeps discarded per chain.
    n_skip
        Thinning interval; ``1`` keeps every post-burn-in sweep. Total sweeps
        are ``num_burnin + n_skip * num_sweeps``.
    num_chains
        Parallel chains; at least two are needed for R-hat.
    inner_loop_length
        Sweeps per outer dispatch. Bounds compile time; ``None`` runs the whole
        chain in one dispatch.
    tempering_levels
        Parallel-tempering replicas per chain. ``1`` (default) runs the plain
        sampler. With ``K > 1`` every chain is a ladder of ``K`` replicas whose
        likelihood is raised to powers equally spaced in ``sqrt(beta)`` from 1
        down to ``tempering_beta_min``; replicas run the same sweep, adjacent
        levels exchange states after every sweep, and only the posterior
        replicas are retained. Exact, and lets chains cross between forest
        modes that single-tree moves cannot connect, at ``K`` times the cost.
        A ladder of about ``2 * sd(log-likelihood) * (1 - sqrt(beta_min))``
        levels keeps the exchange rate near a third. See ``longbet._tempering``.
    tempering_beta_min
        Inverse temperature of the hottest replica, in ``(0, 1]``.
    use_inter_ensemble_move
        Whether to run the inter-ensemble residual-transfer Metropolis-Hastings
        step between ``mu`` and ``nu`` each sweep.
    inter_ensemble_sd
        Proposal standard deviation for the level and slope shifts in the
        inter-ensemble transfer step.
    num_trees_pr, num_trees_trt
        Trees in the prognostic and treatment forests. The treatment forest
        needs more trees than a prognostic one of the same size: with 20 trees
        each tree carries a large share of a threshold surface and the chains
        freeze at their own cutpoints; with 60 the same panel mixes.
    min_points_per_leaf_pr, min_points_per_leaf_trt
        Minimum observed cells in a leaf.
    max_depth_pr, max_depth_trt
        Maximum tree depth. Trees are stored as heaps of ``2 ** depth`` nodes,
        so this bounds the archive size; posterior trees under the depth prior
        rarely exceed depth 5.
    alpha_split_pr, beta_split_pr, alpha_split_trt, beta_split_trt
        Tree-depth prior: a node at depth ``d`` splits with probability
        ``alpha / (1 + d) ** beta``. The treatment prior is shallower than the
        prognostic one, as in Bayesian causal forests.
    num_cutpoints
        Maximum quantile bins per continuous covariate.
    sig_knl, lambda_knl, kernel_type
        Marginal standard deviation, lengthscale and family (``'se'``,
        ``'matern32'`` (alias ``'matern'``), ``'matern52'``, ``'ar1'``) of the
        exposure-trajectory kernel.
    sigma_m, gp_constant_mean
        Prior standard deviation of the trajectory's constant mean, and whether
        that marginalized mean is included, so projections beyond the fitted
        horizon revert to an estimated common level rather than to zero.
    split_calendar_mu
        Whether the prognostic forest may split on calendar time.
    split_calendar_trt
        Whether the treatment forest may split on calendar time.
    split_exposure_trt
        Whether the treatment forest may also split on the exposure clock.
        Off by default: the trajectory ``beta_S`` carries the whole exposure
        profile, which keeps the product ``beta_S * nu`` identified. Turn it
        on when different kinds of units need genuinely different shapes over
        exposure, and read the diagnostics: the extra freedom is an
        exposure-shape ridge between the trajectory and the forest.
    random_intercept
        Whether to fit unit random intercepts ``gamma_i``.
    gamma_prior_a, gamma_prior_b
        Inverse-gamma prior on the unit-intercept variance.
    sigma_prior_a, sigma_prior_b
        Inverse-gamma prior on the innovation variance of a continuous outcome,
        on the standardized scale. The default IG(2, 1) is proper with prior
        mean 1; an improper reference prior can give an improper posterior in
        coupled models and is not offered.
    outcome
        ``'continuous'``, ``'binary'`` or ``'ordinal'``.
    num_categories
        Number of ordered categories for an ordinal outcome.
    cutpoint_prior_scale
        Ordered-normal prior scale of the ordinal cutpoints, latent units.
    standardize
        Whether to centre and scale a continuous response internally; priors
        are on the standardized scale and outputs are rescaled.
    random_seed
        Base PRNG seed.
    device
        ``'auto'``, ``'cpu'`` or ``'gpu'``.
    sur
        Whether ``LongBetMulti`` couples the outcomes' within-period innovations
        through a triangular seemingly-unrelated-regression likelihood. Inert
        for a scalar fit.
    sur_prior_var
        Prior variance of the SUR loadings.
    """

    # --- sampler -----------------------------------------------------------
    num_sweeps: int = 1000
    num_burnin: int = 2000
    n_skip: int = 1
    num_chains: int = 4
    inner_loop_length: int | None = None
    tempering_levels: int = 1
    tempering_beta_min: float = 0.05
    use_inter_ensemble_move: bool = True
    inter_ensemble_sd: float = 0.05

    # --- forests -----------------------------------------------------------
    num_trees_pr: int = 20
    num_trees_trt: int = 60
    min_points_per_leaf_pr: int = 10
    min_points_per_leaf_trt: int = 10
    max_depth_pr: int = 8
    max_depth_trt: int = 8
    alpha_split_pr: float = 0.95
    beta_split_pr: float = 2.0
    alpha_split_trt: float = 0.25
    beta_split_trt: float = 3.0
    num_cutpoints: int = 100

    # --- exposure trajectory -----------------------------------------------
    sig_knl: float = 1.0
    lambda_knl: float = 1.0
    kernel_type: Literal["se", "matern", "matern32", "matern52", "ar1"] = "se"
    sigma_m: float = 1.0
    gp_constant_mean: bool = True

    # --- structure ---------------------------------------------------------
    split_calendar_mu: bool = True
    split_calendar_trt: bool = True
    split_exposure_trt: bool = False

    # --- priors ------------------------------------------------------------
    random_intercept: bool = True
    gamma_prior_a: float = 1.0
    gamma_prior_b: float = 0.1
    sigma_prior_a: float = 2.0
    sigma_prior_b: float = 1.0

    # --- outcome -----------------------------------------------------------
    outcome: Literal["continuous", "binary", "ordinal"] = "continuous"
    num_categories: int | None = None
    cutpoint_prior_scale: float = 5.0
    #: Internal: hold the exposure trajectory at one (used by the BCF
    #: reduction test where there is no exposure clock).
    sample_beta: bool = True

    # --- misc --------------------------------------------------------------
    standardize: bool = True
    random_seed: int = 0
    device: Literal["auto", "cpu", "gpu"] = "auto"

    # --- multiple outcomes -------------------------------------------------
    sur: bool = True
    sur_prior_var: float = 1.0

    def __post_init__(self) -> None:
        for name in ("num_sweeps", "num_burnin", "n_skip", "num_chains", "num_trees_pr",
                     "num_trees_trt", "min_points_per_leaf_pr", "min_points_per_leaf_trt",
                     "max_depth_pr", "max_depth_trt", "num_cutpoints", "random_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        if self.inner_loop_length is not None and (
                isinstance(self.inner_loop_length, bool)
                or not isinstance(self.inner_loop_length, Integral) or self.inner_loop_length < 1):
            raise ValueError("inner_loop_length must be None or a positive integer")
        if self.num_sweeps < 1:
            raise ValueError(f"num_sweeps must be >= 1, got {self.num_sweeps}")
        if self.num_burnin < 0:
            raise ValueError(f"num_burnin must be >= 0, got {self.num_burnin}")
        if self.n_skip < 1:
            raise ValueError(f"n_skip must be >= 1, got {self.n_skip}")
        if self.num_chains < 1:
            raise ValueError(f"num_chains must be >= 1, got {self.num_chains}")
        if self.tempering_levels < 1:
            raise ValueError("tempering_levels must be an integer >= 1")
        if not (isinstance(self.tempering_beta_min, Real) and not isinstance(self.tempering_beta_min, bool)
                and 0 < self.tempering_beta_min <= 1):
            raise ValueError("tempering_beta_min must be in (0, 1]")
        if (isinstance(self.inter_ensemble_sd, bool) or not isinstance(self.inter_ensemble_sd, Real)
                or not math.isfinite(self.inter_ensemble_sd) or self.inter_ensemble_sd <= 0):
            raise ValueError("inter_ensemble_sd must be a finite positive scalar")
        object.__setattr__(self, "inter_ensemble_sd", float(self.inter_ensemble_sd))
        for name in ("num_trees_pr", "num_trees_trt", "min_points_per_leaf_pr",
                     "min_points_per_leaf_trt", "max_depth_pr", "max_depth_trt", "num_cutpoints"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.num_cutpoints > 255:
            raise ValueError(
                f"num_cutpoints must be at most 255, got {self.num_cutpoints}: the design "
                "stores split indices as unsigned 8-bit integers and the REGROW proposal "
                "scores every cutpoint of every variable at every node"
            )
        for name in ("alpha_split_pr", "alpha_split_trt"):
            value = getattr(self, name)
            if not (isinstance(value, Real) and 0 < value < 1):
                raise ValueError(f"{name} must lie strictly between 0 and 1, got {value!r}")
        for name in ("beta_split_pr", "beta_split_trt"):
            value = getattr(self, name)
            if not (isinstance(value, Real) and math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite nonnegative scalar, got {value!r}")
        for name in ("split_calendar_mu", "split_calendar_trt", "split_exposure_trt",
                     "random_intercept", "gp_constant_mean", "sample_beta",
                     "standardize", "sur", "use_inter_ensemble_move"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("gamma_prior_a", "gamma_prior_b", "sigma_prior_a", "sigma_prior_b"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive scalar, got {value!r}")
        if self.outcome not in ("continuous", "binary", "ordinal"):
            raise ValueError(
                f"outcome must be 'continuous', 'binary', or 'ordinal', got {self.outcome!r}"
            )
        if self.outcome == "ordinal":
            if (isinstance(self.num_categories, bool)
                    or not isinstance(self.num_categories, Integral)
                    or self.num_categories < 2):
                raise ValueError("num_categories must be an integer >=2 for ordinal outcomes")
            object.__setattr__(self, "num_categories", int(self.num_categories))
        elif self.num_categories is not None:
            raise ValueError("num_categories must be None for nonordinal configurations")
        if (isinstance(self.cutpoint_prior_scale, bool)
                or not isinstance(self.cutpoint_prior_scale, Real)
                or not math.isfinite(self.cutpoint_prior_scale)
                or self.cutpoint_prior_scale <= 0):
            raise ValueError("cutpoint_prior_scale must be finite and positive")
        object.__setattr__(self, "cutpoint_prior_scale", float(self.cutpoint_prior_scale))
        if self.device not in ("auto", "cpu", "gpu"):
            raise ValueError(
                f"device must be 'auto', 'cpu' or 'gpu', got {self.device!r}"
            )
        if self.kernel_type not in ("se", "matern", "matern32", "matern52", "ar1"):
            raise ValueError(f"unknown kernel_type {self.kernel_type!r}")
        for name in ("lambda_knl", "sig_knl", "sigma_m"):
            value = getattr(self, name)
            if not (isinstance(value, Real) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive, got {value!r}")
        if (isinstance(self.sur_prior_var, bool) or not isinstance(self.sur_prior_var, Real)
                or not math.isfinite(self.sur_prior_var) or self.sur_prior_var < 0.0):
            raise ValueError(
                f"sur_prior_var must be a finite nonnegative scalar, got {self.sur_prior_var!r}"
            )

    @property
    def split_time_ps(self) -> bool:
        """Deprecated alias for :attr:`split_calendar_mu`."""
        warnings.warn(
            "'split_time_ps' is deprecated and will be removed in a future release; "
            "use 'split_calendar_mu' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.split_calendar_mu

    @property
    def sur_active(self) -> bool:
        """Whether SUR coupling is active (sur is True and sur_prior_var > 0)."""
        return bool(self.sur and self.sur_prior_var > 0.0)

    def total_iterations(self) -> int:
        """Total Gibbs sweeps to run (burn-in plus saved draws with thinning)."""
        return self.num_burnin + self.num_sweeps * self.n_skip

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a plain dictionary."""
        d = dataclasses.asdict(self)
        d["split_time_ps"] = self.split_calendar_mu
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LongBetConfig:
        """Construct a configuration from a dictionary, ignoring unknown keys."""
        d = dict(d)
        if "split_time_ps" in d and "split_calendar_mu" not in d:
            d["split_calendar_mu"] = d.pop("split_time_ps")
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


_orig_longbet_config_init = LongBetConfig.__init__


def _longbet_config_init(self: LongBetConfig, *args: Any, **kwargs: Any) -> None:
    if "split_time_ps" in kwargs:
        warnings.warn(
            "'split_time_ps' is deprecated and will be removed in a future release; "
            "use 'split_calendar_mu' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        val = kwargs.pop("split_time_ps")
        if "split_calendar_mu" not in kwargs:
            kwargs["split_calendar_mu"] = val
    _orig_longbet_config_init(self, *args, **kwargs)


LongBetConfig.__init__ = _longbet_config_init  # type: ignore[method-assign]
