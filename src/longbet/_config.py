"""Configuration for the LongBet sampler."""

from __future__ import annotations

import dataclasses
import math
from numbers import Integral, Real
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class LongBetConfig:
    """Configuration parameters for the LongBet sampler.

    Sampling defaults
    -----------------
    2,000 burn-in, 250 draws per chain thinned by 2, across 4 chains: 2,500
    iterations per chain for 1,000 retained draws. Chosen by measurement on a
    2,500 x 20 panel with 20 trees per forest, three seeds per setting:

    ==========  ==========  ==========  =========
    burn-in     R-hat max   ESS median  ESS min
    ==========  ==========  ==========  =========
    500         1.120        375         40
    1000        1.070        643        101
    **2000**    **1.040**   **577**    **260**
    4000        1.028        631        256
    ==========  ==========  ==========  =========

    In that benchmark, worst-case effective sample size levels off around
    2,000 burn-in iterations. This is not a convergence guarantee on other
    panels, especially for binary outcomes or heterogeneous subgroup effects.

    Sampling is **not** free, and an earlier version of this note said it was.
    Cost is linear in ``num_burnin + n_skip * num_sweeps`` -- roughly 34 ms per
    iteration on a 1,000-unit, 20-period panel with 4 chains on a CPU. What
    misled the measurement was JAX's asynchronous dispatch: ``fit()`` returns
    before the work is done, so a naive timer reads a few seconds for a job that
    takes minutes. Anything timing this code must call
    ``jax.block_until_ready``.

    Thinning therefore trades sampling time for prediction time and memory:
    prediction cost scales with *retained* draws rather than with iterations.
    At ``num_burnin=2000`` and ``num_sweeps=250``, changing ``n_skip`` from 1
    to 2 adds 250 iterations per chain (2,250 to 2,500, about eleven percent).
    It holds the retained sample at 1,000 draws across four chains. Set
    ``n_skip=1`` if sampling time is what binds for you; the sensible way to
    decide is to measure both on your own panel, since which side dominates
    depends on how large ``N * T`` is relative to the chain length.

    Where the defaults fall short
    -----------------------------
    Chains are started **overdispersed** (see
    ``longbet._state.broadcast_to_chains``), which is what makes R-hat a
    convergence statistic rather than a measure of how far two random streams
    drifted -- and which makes it markedly less flattering. On the same panel,
    across seeds, the shipped defaults give a median R-hat on the ATT around
    1.01 and a worst-case effective sample size of a few hundred. Treat them as
    a floor rather than a guarantee: check ``att_stability`` on your own panel
    before trusting an interval endpoint.

    Read ``att_stability`` on your own panel and diagnose the subgroup and
    probability-scale effects actually used in decisions. More burn-in can
    address an initial transient; longer retained chains are needed to measure
    uncertainty and effective sample size. Neither necessarily repairs slow
    mixing or separated modes. If longer runs still disagree, investigate
    parameterization and priors rather than just discarding more iterations.

    Parameters
    ----------
    num_sweeps
        Number of MCMC draws to save per chain, after burn-in.
    num_burnin
        Number of initial MCMC iterations discarded as burn-in. This controls
        the discarded transient, not a guarantee of convergence or sufficient
        effective sample size in the retained chains.
    n_skip
        Thinning interval; 1 means every post-burn-in iteration is saved. Total
        iterations run is ``num_burnin + n_skip * num_sweeps``, matching bartz.
        At fixed retained draw count, thinning adds sampling work while keeping
        prediction cost and trace storage approximately fixed. It does not
        guarantee better effective sample size per second.
    num_chains
        Number of parallel MCMC chains. At least 2 are needed for R-hat; the
        default of 4 is the usual choice and costs no more per retained draw
        than one chain of the same total length.
    inner_loop_length
        Iterations per outer dispatch. Bounds compile time and enables a
        progress callback; ``None`` runs the whole chain in one dispatch.
    num_trees_pr
        Number of trees in the prognostic forest (mu).
    num_trees_trt
        Number of trees in the treatment forest (nu).
    min_points_per_leaf_pr, min_points_per_leaf_trt
        Minimum observations required in a leaf.
    max_depth_pr, max_depth_trt
        Maximum tree depth.
    alpha_split_pr, beta_split_pr, alpha_split_trt, beta_split_trt
        Tree-depth prior parameters.
    num_cutpoints
        Maximum number of quantile bins per continuous covariate.
    sig_knl
        Kernel standard deviation for the exposure trajectory; the marginal
        prior variance of ``beta_S`` is ``sig_knl ** 2``.
    lambda_knl
        Kernel lengthscale over the exposure index.
    kernel_type
        Kernel family: 'se' (squared exponential), 'matern32' (alias 'matern'),
        'matern52', or 'ar1'. Over an integer exposure index AR(1) has a
        tridiagonal precision and is the most numerically forgiving choice;
        'se' is the default for parity with the reference implementation.
    gp_jitter
        Relative diagonal jitter on the kernel: ``gp_jitter * sig_knl ** 2``.
    sigma_m
        Prior standard deviation of the constant GP mean. Marginalized into the
        kernel rather than sampled, so the forecast reverts to the estimated
        common level.
    gp_constant_mean
        Whether to include the marginalized constant mean.
    split_time_ps
        Whether the *prognostic* forest may split on calendar time.
    split_time_trt
        Whether the *treatment* forest may split on the exposure index S.
        Setting this False removes the ``c(S)`` component of the beta-nu
        non-identification and forces all exposure-time shape into ``beta``.
    random_intercept
        Whether to fit unit-level random intercepts gamma_i.
    gamma_prior_a, gamma_prior_b
        Inverse-gamma prior on the unit-intercept variance sigma_gamma^2.
    sigma_prior_a, sigma_prior_b
        Inverse-gamma prior on the error variance sigma^2. The default (0, 0) is
        the improper reference prior; set both positive for a proper prior (as
        the Geweke tests do). Coupled multiple-outcome fits with continuous
        outcomes require both to be positive, explicitly chosen on the working
        outcome scale. The reference prior can yield an improper joint posterior.
    sigma_b
        Prior standard deviation of the adaptive coding weights b0, b1. XBCF
        uses a prior variance of 1/2, hence ``1/sqrt(2)``.
    sigma_alpha
        Prior standard deviation of the prognostic scale alpha.
    outcome
        'continuous', 'binary', or 'ordinal' (ordered probit link).
    num_categories
        Declared ordinal category count (>=2); None for other outcomes.
    cutpoint_prior_scale
        Positive ordered-normal prior scale in latent probit units.
    sample_alpha
        Whether to sample the prognostic scale alpha. Off by default: its
        posterior concentrates tightly around 1 and it exists to help XBART
        mix, which is not this sampler's problem.
    sample_beta
        Whether to sample the exposure trajectory. Off only for the BCF
        reduction test.
    adaptive_coding
        Whether to sample the adaptive coding weights b0, b1. **On by default**:
        this is an exactly conjugate move along the global scale of the
        treatment term, and it is one of the two mechanisms by which this
        sampler traverses the beta-nu ridge. When off, b0 and b1 are held at 1,
        matching the reference R implementation.
    ridge_move
        Whether to run the Metropolis move along the beta-nu scale ridge.
        Ignored when sample_beta=False, which must keep beta fixed.
    ridge_proposal_sigma
        Proposal standard deviation for ``log c`` in the ridge move.
    standardize
        Whether to centre and scale a continuous response internally. Priors are
        expressed on the standardized scale and results are rescaled on output.
    random_seed
        Base PRNG seed.
    device
        Compute device: 'auto', 'cpu', or 'gpu'.
    sur
        Whether to use full-precision triangular SUR coupling across outcomes
        in multiple-outcome models (``LongBetMulti``). Inert for scalar
        ``LongBet``.
    sur_prior_var
        Prior variance on SUR regression loadings Gamma. 0 disables SUR.
        Inert for scalar ``LongBet``.
    num_shared_trees
        Opt-in shared treatment partitions in ``LongBetMulti``. Replaces this
        many of ``num_trees_trt``, leaving at least one private treatment tree
        per outcome. Zero (default) preserves the separate-forest sampler.
        Nonzero values are rejected by scalar ``LongBet.fit``. Shared trees
        have outcome-specific vector leaf values and common split settings.
    shared_variance_fraction
        Fixed prior variance allocation to the shared ensemble, strictly
        between zero and one. Shared leaf variance is this fraction divided
        by ``num_shared_trees``; private leaf variance is its complement
        divided by the remaining tree count. Thus total pointwise prior
        variance of nu remains one on the internal response scale. This
        does not change GP/coding priors or learn an effect correlation.
        Inert when sharing is disabled. Baseline forests remain separate.
    """

    # --- sampler -----------------------------------------------------------
    num_sweeps: int = 250
    num_burnin: int = 2000
    n_skip: int = 2
    num_chains: int = 4
    inner_loop_length: int | None = None

    # --- forests -----------------------------------------------------------
    num_trees_pr: int = 20
    num_trees_trt: int = 20
    min_points_per_leaf_pr: int = 10
    min_points_per_leaf_trt: int = 10
    max_depth_pr: int = 10
    max_depth_trt: int = 10
    alpha_split_pr: float = 0.95
    beta_split_pr: float = 2.0
    alpha_split_trt: float = 0.25
    beta_split_trt: float = 3.0
    num_cutpoints: int = 100

    # --- exposure trajectory ----------------------------------------------
    sig_knl: float = 1.0
    lambda_knl: float = 1.0
    kernel_type: Literal["se", "matern", "matern32", "matern52", "ar1"] = "se"
    gp_jitter: float = 1e-6
    sigma_m: float = 1.0
    gp_constant_mean: bool = True

    # --- structure ---------------------------------------------------------
    split_time_ps: bool = True
    split_time_trt: bool = True

    # --- priors ------------------------------------------------------------
    random_intercept: bool = True
    gamma_prior_a: float = 1.0
    gamma_prior_b: float = 0.1
    sigma_prior_a: float = 0.0
    sigma_prior_b: float = 0.0
    sigma_b: float = 0.7071067811865476
    sigma_alpha: float = 1.0

    # --- outcome and moves --------------------------------------------------
    outcome: Literal["continuous", "binary", "ordinal"] = "continuous"
    num_categories: int | None = None
    cutpoint_prior_scale: float = 5.0
    sample_alpha: bool = False
    sample_beta: bool = True
    adaptive_coding: bool = True
    ridge_move: bool = True
    ridge_proposal_sigma: float = 0.2
    standardize: bool = True
    random_seed: int = 0
    device: Literal["auto", "cpu", "gpu"] = "auto"

    # --- multi / SUR (inert for scalar LongBet) -----------------------------
    sur: bool = True
    sur_prior_var: float = 1.0
    # Shared trees replace this many of num_trees_trt, leaving >=1 private tree.
    num_shared_trees: int = 0
    shared_variance_fraction: float = 0.5

    def __post_init__(self) -> None:
        for name in ("sigma_prior_a", "sigma_prior_b"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite nonnegative scalar, got {value!r}")
        if (type(self.num_shared_trees) is not int or
                not 0 <= self.num_shared_trees < self.num_trees_trt):
            raise ValueError("num_shared_trees must be an integer in [0, num_trees_trt); at least one private treatment tree is required.")
        if (isinstance(self.shared_variance_fraction, bool) or
                not isinstance(self.shared_variance_fraction, (int, float)) or
                not math.isfinite(self.shared_variance_fraction) or
                not 0 < self.shared_variance_fraction < 1):
            raise ValueError("shared_variance_fraction must be a finite scalar strictly between 0 and 1.")
        if self.num_shared_trees:
            if not (type(self.max_depth_trt) is int and 1 <= self.max_depth_trt <= 16):
                raise ValueError("shared treatment trees require integer max_depth_trt in [1, 16].")
            if not (0 < self.alpha_split_trt < 1 and math.isfinite(self.beta_split_trt) and self.beta_split_trt >= 0):
                raise ValueError("shared treatment trees require 0 < alpha_split_trt < 1 and finite beta_split_trt >= 0.")
            if type(self.min_points_per_leaf_trt) is not int or self.min_points_per_leaf_trt < 1:
                raise ValueError("shared treatment trees require min_points_per_leaf_trt >= 1.")
        if self.num_sweeps < 1:
            raise ValueError(f"num_sweeps must be >= 1, got {self.num_sweeps}")
        if self.num_burnin < 0:
            raise ValueError(f"num_burnin must be >= 0, got {self.num_burnin}")
        if self.n_skip < 1:
            raise ValueError(f"n_skip must be >= 1, got {self.n_skip}")
        if self.num_chains < 1:
            raise ValueError(f"num_chains must be >= 1, got {self.num_chains}")
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
        if self.lambda_knl <= 0:
            raise ValueError(f"lambda_knl must be positive, got {self.lambda_knl}")
        if self.sig_knl <= 0:
            raise ValueError(f"sig_knl must be positive, got {self.sig_knl}")
        if not isinstance(self.sur, bool):
            raise ValueError(f"sur must be a boolean, got {self.sur!r}")
        if (
            isinstance(self.sur_prior_var, bool)
            or not isinstance(self.sur_prior_var, (int, float))
            or not math.isfinite(self.sur_prior_var)
            or self.sur_prior_var < 0.0
        ):
            raise ValueError(
                f"sur_prior_var must be a finite nonnegative scalar, got {self.sur_prior_var!r}"
            )

    @property
    def sur_active(self) -> bool:
        """Whether SUR coupling is active (sur is True and sur_prior_var > 0)."""
        return bool(self.sur and self.sur_prior_var > 0.0)

    def total_iterations(self) -> int:
        """Total Gibbs sweeps to run (burn-in plus saved draws with thinning)."""
        return self.num_burnin + self.num_sweeps * self.n_skip

    def validate_multi_variance_prior(self, outcomes: tuple[str, ...]) -> None:
        """Require an explicit proper target before coupled continuous inference.

        With latent binary residuals and SUR loadings, the observed likelihood
        can remain positive as a continuous innovation variance approaches zero.
        A 1/v prior then has an infinite normalizer. Positive inverse-gamma
        parameters remove that boundary defect; they do not guarantee mixing.
        Keep the scalar API and independent/binary-only reductions unchanged.
        """
        if (self.sur_active and len(outcomes) > 1 and "continuous" in outcomes
                and not (self.sigma_prior_a > 0 and self.sigma_prior_b > 0)):
            raise ValueError(
                "Coupled models with continuous outcomes require a proper "
                "innovation-variance prior: set sigma_prior_a > 0 and "
                "sigma_prior_b > 0 explicitly on the working outcome scale "
                "(for example, 2 and 1 for standardized outcomes). The default "
                "improper prior can yield an improper joint posterior. "
                "Refit from the original data; saved draws cannot be repaired "
                "by changing their prior metadata. Proper priors do not "
                "guarantee convergence.")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LongBetConfig:
        """Construct a configuration from a dictionary, ignoring unknown keys."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})
