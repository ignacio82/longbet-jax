# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smooth calendar-time trend block for the prognostic surface.

Why this exists
---------------
A sum-of-trees prognostic surface ``mu(X_i, t)`` is **piecewise constant** in
calendar time: inside a leaf whose time interval is ``t in {8, ..., 12}`` the
fitted value is one number, flat across those five periods. The treatment side
of the model has no such restriction: the Gaussian process ``beta`` carries one
free coordinate per integer exposure ``s = 1, ..., S_max``, so
``beta_{S_it} nu(X_i, t)`` represents a smooth trend over post-adoption time
with zero discretization error.

On a staggered panel the two are tied together by ``t = (E_i - 1) + S_it``, so a
baseline trend that diverges across cohorts can be carried either by ``mu`` or
by ``(gamma, beta, nu)``. The exact conjugate subspace Gibbs move of
:mod:`longbet._inter_ensemble_move` transfers the shared component between them
**given the tree partitions**, which is what it can do. What it cannot do is
make a constant leaf cancel a linear slope: where a chain's trees happen not to
have grown a time split, the compensating shift is unrepresentable, and the
residual mismatch is chain specific. Different chains discretize calendar time
at different cutpoints, park different fractions of the trend on either side of
the identity, and disagree -- which is the elevated split-Rhat reported for the
diverging-trend designs.

What this module does
---------------------
It removes the resolution mismatch by taking calendar time away from the forest
altogether and giving it to an explicit block whose time basis is *shared by
every chain*:

.. code::

    Y_it = alpha mu(X_i)                       # forest: cross-section only
           + sum_k psi_k(t) f_k                # free common-time factor
           + sum_m phi_m(t) b(X_i)' c_m        # heterogeneous trend
           + gamma_i
           + 1{S_it >= 1} beta_{S_it} nu(X_i, t) + eps_it

The two halves are a package deal, and shipping either alone is a mistake we
measured rather than guessed at:

* ``psi_1, ..., psi_{T-1}`` (:func:`build_period_basis`) are the type-II
  discrete cosine contrasts. The basis is **complete** over centred functions of
  calendar time, so replacing the forest's calendar step functions with it is
  lossless rather than a restriction. Dropping this half while still blinding
  the forest leaves nothing able to fit a common time shock.
* ``phi_1, ..., phi_d`` (:func:`build_time_basis`) are fixed orthonormal
  polynomials in calendar time, and ``b(X_i)``
  (:func:`build_unit_features`) is a fixed feature map of the unit's
  covariates -- the standardized covariates and random Fourier features,
  centred, with no constant column, all drawn from :data:`FEATURE_SEED` so the
  map is the same in every chain. This half carries trends that differ across
  units. Only the coefficients ``(f, c)`` are sampled.
* The prognostic forest no longer splits on calendar time
  (:attr:`longbet.LongBetConfig.split_calendar_mu` resolves to ``False``
  whenever this block is on). Leaving it on puts the forest and the complete
  period basis in overlapping subspaces, which is near-collinear by
  construction.

Two consequences follow, and they are the point of the module:

1. Any trend of the form ``psi_k(t)`` or ``phi_m(t) g(X_i)`` is representable
   **exactly at every period**, with no dependence on where a chain's trees
   happened to split time. Trend transfer between the prognostic surface and
   ``(gamma, beta)`` therefore has zero residual, in every chain.
2. Because the likelihood is Gaussian in ``f``, ``c``, ``gamma`` and ``beta``
   jointly, the blocks are drawn from their **exact joint conditional** in one
   step (:func:`trend_block_step`), which eliminates the remaining
   coordinate-wise Gibbs correlation between the trend, the unit intercepts and
   the exposure trajectory rather than merely reducing it.

The interaction block carries a horseshoe prior (:func:`sample_horseshoe`) so
that widening the feature map buys capacity without the null columns diluting a
single shared scale. The scheme of Makalic and Schmidt (2016) is used because
all four of its conditionals are inverse-Gamma, which keeps the model
conditionally Gaussian in ``c`` and so leaves the exact joint draw above intact.

The residual forest ``mu`` keeps every nonlinearity in the covariates; it simply
no longer competes for calendar time.
"""


from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
from jaxtyping import Array, Bool, Float32, Int32, Key

from longbet._linalg import sample_from_precision

#: Seed of the random Fourier feature draw. Fixed, not user configurable and
#: not derived from ``config.random_seed``: the feature map must be identical in
#: every chain of every fit, otherwise chains target slightly different models
#: and R-hat stops being a convergence statistic.
FEATURE_SEED = 20260918

#: Raw covariate columns fed to the feature map. Beyond this the map relies on
#: its random features; the cap keeps the design matrix small on wide panels.
MAX_RAW_COLS = 24


class TrendFeatureSpec(NamedTuple):
    """Everything needed to rebuild the unit feature map for new covariates.

    Stored on the fitted model so that :meth:`longbet.LongBet.predict` maps new
    units through exactly the same basis the sampler was fitted with.
    """

    #: Column means of the raw covariates used, shape ``(p0,)``.
    mean: np.ndarray
    #: Column standard deviations of the raw covariates used, shape ``(p0,)``.
    scale: np.ndarray
    #: Random Fourier projection matrix, shape ``(p0, F)``; empty when ``F = 0``.
    omega: np.ndarray
    #: Random Fourier phases, shape ``(F,)``; empty when ``F = 0``.
    phase: np.ndarray
    #: Column means removed after assembly, shape ``(q,)``.
    centre: np.ndarray
    #: Column scales applied after centring, shape ``(q,)``.
    col_scale: np.ndarray
    #: Number of raw covariate columns used.
    num_raw: int


def build_time_basis(T: int, degree: int) -> np.ndarray:
    """Orthonormal polynomial basis over the calendar-time index.

    Returns an array of shape ``(T, degree)`` whose columns are centred
    (``sum_t phi_k(t) = 0``) and scaled to unit mean square
    (``mean_t phi_k(t)^2 = 1``), and are mutually orthogonal.

    Centring matters: it keeps the block orthogonal to the model's intercept and
    to the unit intercepts' common level, so the smooth trend competes with
    neither. Unit scaling makes a single prior variance meaningful across
    degrees.

    Parameters
    ----------
    T
        Number of calendar periods.
    degree
        Number of basis functions; ``1`` is a linear trend, ``2`` adds
        curvature.

    Returns
    -------
    phi : np.ndarray
        Shape ``(T, degree)``, float32.
    """
    degree = max(int(degree), 0)
    if degree == 0 or T <= 1:
        return np.zeros((max(T, 1), 0), dtype=np.float32)
    # Raw monomials in normalized time, then Gram-Schmidt against the constant
    # and against each other. Done in float64: the Vandermonde system is
    # ill-conditioned even at modest degree, and the factor is computed once.
    u = np.linspace(-1.0, 1.0, T, dtype=np.float64)
    cols = []
    for k in range(1, degree + 1):
        v = u**k
        v = v - v.mean()
        for w in cols:
            v = v - (v @ w) / (w @ w) * w
        nrm = np.sqrt((v**2).mean())
        if nrm < 1e-8:
            # Degenerate at this degree (too few distinct periods): stop early
            # and return the basis that is actually identified.
            break
        cols.append(v / nrm)
    if not cols:
        return np.zeros((T, 0), dtype=np.float32)
    return np.column_stack(cols).astype(np.float32)


def build_period_basis(T: int) -> np.ndarray:
    """Free common-time effects, as an orthonormal centred contrast basis.

    Returns ``(T, T - 1)`` columns spanning *every* function of calendar time
    that sums to zero, so the block can represent an arbitrary common time
    factor -- a jump, a shock, a seasonal, anything -- exactly.

    This matters because the architecture forbids the prognostic forest from
    splitting on calendar time (:attr:`longbet.LongBetConfig.split_calendar_mu`
    is off when the block is on). Something has to carry the common time
    movement the forest used to carry with step functions, and a *complete*
    basis is what makes the substitution lossless rather than a restriction.

    The basis used is the type-II discrete cosine basis,
    ``psi_k(t) = sqrt(2) cos(pi k (t + 1/2) / T)`` for ``k = 1, ..., T - 1``.
    It is centred, orthogonal, has unit mean square, and is ordered from smooth
    to oscillatory, so a common prior scale shrinks wiggle before it shrinks
    level -- which is the right default for a time factor.
    """
    T = int(T)
    if T <= 1:
        return np.zeros((max(T, 1), 0), dtype=np.float32)
    tt = np.arange(T, dtype=np.float64) + 0.5
    k = np.arange(1, T, dtype=np.float64)
    psi = np.sqrt(2.0) * np.cos(np.pi * np.outer(tt, k) / T)
    return psi.astype(np.float32)


def extend_period_basis(period_fit: np.ndarray, T_new: int) -> np.ndarray:
    """Continue a fitted period basis over a longer prediction horizon.

    A free period effect for a period nobody observed is not identified, so
    rows past the fitted horizon are set to zero -- the prior mean. This is the
    honest choice and it is also harmless for the estimand: the common time
    factor cancels between the treated and untreated potential outcomes, so it
    does not touch ``tau``. It affects only predicted *levels* past the end of
    the panel, which are, correctly, shrunk to the common trend.
    """
    period_fit = np.asarray(period_fit, dtype=np.float32)
    T_fit, ncol = period_fit.shape
    if T_new <= T_fit:
        return period_fit[:T_new]
    out = np.zeros((T_new, ncol), dtype=np.float32)
    out[:T_fit] = period_fit
    return out


def build_unit_features(
    x_unit: np.ndarray,
    num_random_features: int,
) -> tuple[np.ndarray, TrendFeatureSpec]:
    """Fixed, chain-independent feature map of the unit-level covariates.

    The map is ``b(X_i) = [X_std, sqrt(2) cos(W' X_std + psi)]``: the
    standardized covariates, so the block can carry a trend linear in them, and
    random Fourier features, so it can carry a nonlinear one. The latter is what
    lets the block absorb loadings such as ``g(x_4) + x_1 |x_3 - 1|`` that no
    polynomial in the covariates reproduces.

    Two properties are deliberate and load-bearing.

    *No constant column, and every column centred across units.* The common-time
    block of :func:`build_period_basis` already spans every function of calendar
    time alone. A feature column with a non-zero mean would make
    ``phi_k(t) b_j(X_i)`` contain a copy of ``phi_k(t)``, which lies in that
    span -- an exactly collinear direction between the two halves of the block,
    which is precisely the kind of ridge this module exists to remove. Centring
    makes the two halves orthogonal on a balanced panel:
    ``sum_{i,t} psi_k(t) phi_m(t) b_j(X_i) = 0``. It also makes them orthogonal
    to the unit intercepts, since ``sum_t phi_m(t) = 0``. So the block splits
    cleanly into "what all units do in common" and "how units deviate from it".

    *``W`` and ``psi`` come from* :data:`FEATURE_SEED`, never from the chain key
    and never from ``config.random_seed``, so the design is identical across
    chains by construction. If it were not, chains would target slightly
    different models and R-hat would stop being a convergence statistic.

    Parameters
    ----------
    x_unit
        Baseline covariates, shape ``(N, P)``.
    num_random_features
        Number of random Fourier features ``F``. Zero gives a purely linear map.

    Returns
    -------
    features : np.ndarray
        Shape ``(N, q)`` with ``q = p0 + F``, float32, columns centred and
        scaled to unit mean square.
    spec : TrendFeatureSpec
        The map itself, for reuse on new covariates at prediction time.
    """
    x = np.asarray(x_unit, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    p0 = min(x.shape[1], MAX_RAW_COLS)
    xr = x[:, :p0]
    mean = xr.mean(axis=0)
    scale = xr.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    xs = (xr - mean) / scale

    F = max(int(num_random_features), 0)
    if F > 0:
        rng = np.random.default_rng(FEATURE_SEED)
        # Lengthscale sqrt(p0) keeps the argument O(1) whatever the covariate
        # count, so the features neither saturate nor degenerate to a constant.
        omega = rng.normal(size=(p0, F)) / np.sqrt(max(p0, 1))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=F)
        rff = np.sqrt(2.0) * np.cos(xs @ omega + phase)
        B = np.hstack([xs, rff])
    else:
        omega = np.zeros((p0, 0))
        phase = np.zeros((0,))
        B = xs

    centre = B.mean(axis=0)
    B = B - centre
    col_scale = np.sqrt((B**2).mean(axis=0))
    col_scale = np.where(col_scale < 1e-8, 1.0, col_scale)
    B = B / col_scale
    spec = TrendFeatureSpec(
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        omega=omega.astype(np.float64),
        phase=phase.astype(np.float64),
        centre=centre.astype(np.float64),
        col_scale=col_scale.astype(np.float64),
        num_raw=int(p0),
    )
    return B.astype(np.float32), spec


def apply_unit_features(x_unit: np.ndarray, spec: TrendFeatureSpec) -> np.ndarray:
    """Map new covariates through a fitted :class:`TrendFeatureSpec`.

    Uses the *fitted* centring and scaling, not the new sample's own, so that a
    prediction unit is placed on the same basis the coefficients were sampled
    against.
    """
    x = np.asarray(x_unit, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    xs = (x[:, : spec.num_raw] - spec.mean) / spec.scale
    if spec.omega.shape[1] > 0:
        rff = np.sqrt(2.0) * np.cos(xs @ spec.omega + spec.phase)
        B = np.hstack([xs, rff])
    else:
        B = xs
    return ((B - spec.centre) / spec.col_scale).astype(np.float32)


def build_trend_design(
    features: np.ndarray | Array,
    phi: np.ndarray | Array,
    unit_idx: np.ndarray | Array,
    time_idx: np.ndarray | Array,
    period: np.ndarray | Array | None = None,
) -> Float32[Array, 'n q_tot']:
    """Assemble the trend design, shape ``(n, (T - 1) + d q)``.

    Two blocks, in this order:

    ``[ psi_k(t) | phi_m(t) b_j(X_i) ]``

    The first is the free common-time factor from :func:`build_period_basis`:
    every function of calendar time, with no covariate dependence. The second is
    the heterogeneous-trend interaction: smooth functions of time times centred
    unit features, so it carries how units *deviate* from the common trend.
    Because the features are centred the two blocks are orthogonal on a balanced
    panel, which is what keeps them from competing.

    Neither depends on any sampled quantity, so the matrix is built once and
    shared by every chain -- that it does not vary by chain is what makes the
    block cure the resolution mismatch rather than relocate it.
    """
    B = jnp.asarray(features, dtype=jnp.float32)[jnp.asarray(unit_idx)]  # (n, q)
    P = jnp.asarray(phi, dtype=jnp.float32)[jnp.asarray(time_idx)]       # (n, d)
    n, q = B.shape
    d = P.shape[1]
    inter = (P[:, :, None] * B[:, None, :]).reshape(n, d * q)
    if period is None:
        return inter
    Psi = jnp.asarray(period, dtype=jnp.float32)[jnp.asarray(time_idx)]  # (n, T-1)
    return jnp.concatenate([Psi, inter], axis=1)


def trend_prior_scales(
    num_period: int,
    num_inter: int,
    prior_scale: float,
    period_scale: float,
) -> np.ndarray:
    """Per-column prior standard deviation of the trend coefficients.

    Returns a vector of length ``num_period + num_inter``. The two blocks are
    shrunk differently because they are doing different jobs.

    *Common-time block* (``num_period`` columns): ``f_t ~ N(0, period_scale^2)``.
    This is the calendar-time factor the prognostic forest used to carry with
    step functions. It is low-dimensional, always identified -- every period has
    data -- and should not be shrunk hard, or the model will fit common shocks
    worse than the forest did. The scale is *not* divided by the basis size.

    *Interaction block* (``num_inter`` columns): ``c ~ N(0, sigma_c^2)`` with
    ``sigma_c = prior_scale / sqrt(num_inter)``. This block is high-dimensional
    and is asked to discover which covariates carry diverging trends, so it is
    shrunk in proportion to its size. Because the time basis and the feature
    columns both have unit mean square, the induced deviation has standard
    deviation about ``prior_scale`` on the standardized response whatever the
    basis size -- ``prior_scale = 1`` says a unit's trend deviation is a priori
    of the same order as the outcome's own spread.

    When the horseshoe is enabled the interaction entries returned here are the
    *global* scale ``A`` of :func:`sample_horseshoe`, i.e. the median of the
    half-Cauchy the sampled ``tau`` is drawn from, so the two priors agree on
    typical magnitude and differ only in tail weight.
    """
    num_period = max(int(num_period), 0)
    num_inter = max(int(num_inter), 0)
    sigma_c = float(prior_scale) / np.sqrt(max(num_inter, 1))
    return np.concatenate(
        [
            np.full(num_period, max(float(period_scale), 1e-6), dtype=np.float32),
            np.full(num_inter, max(sigma_c, 1e-6), dtype=np.float32),
        ]
    )


def trend_prior_precision(
    num_period: int,
    num_inter: int,
    prior_scale: float,
    period_scale: float,
) -> np.ndarray:
    """Per-column prior precision, the reciprocal square of the scales.

    Retained because it is the more natural quantity to check a conditional
    against; the sampler itself works in scales, which is better conditioned
    (see :func:`trend_block_step`).
    """
    scales = trend_prior_scales(num_period, num_inter, prior_scale, period_scale)
    return (1.0 / np.square(scales)).astype(np.float32)


#: Bounds on the sampled horseshoe variances. The horseshoe's whole point is
#: that ``tau lambda_j`` is allowed to be extremely small, and the column
#: rescaling of :func:`trend_block_step` means a small scale costs no
#: conditioning; these clamps only stop float32 from reaching zero or infinity.
HS_VAR_MIN = 1e-10
HS_VAR_MAX = 1e6


def init_horseshoe(num_inter: int, global_scale: float) -> dict[str, np.ndarray]:
    """Starting values for the horseshoe scales.

    Chosen so the first sweep's effective prior is *exactly* the ridge of
    :func:`trend_prior_scales`: ``lambda_j^2 = 1`` and ``tau^2 = A^2`` give
    column scale ``tau lambda_j = A``. The horseshoe then earns any departure
    from the ridge from the data rather than from its initialization, which is
    what makes a ridge-vs-horseshoe comparison at fixed everything else
    interpretable.
    """
    num_inter = max(int(num_inter), 0)
    A2 = float(max(global_scale, 1e-6)) ** 2
    return dict(
        local2=np.ones(num_inter, dtype=np.float32),
        local_aux=np.ones(num_inter, dtype=np.float32),
        global2=np.float32(A2),
        global_aux=np.float32(1.0 / A2),
    )


def _inv_gamma(key: Key[Array, ''], shape_a: Any, rate_b: Any) -> Any:
    """Draw from ``InvGamma(shape_a, rate_b)`` as ``rate_b / Gamma(shape_a, 1)``."""
    g = jax.random.gamma(key, shape_a, shape=jnp.shape(rate_b), dtype=jnp.float32)
    return rate_b / jnp.maximum(g, 1e-30)


def sample_horseshoe(
    key: Key[Array, ''],
    coef_inter: Float32[Array, ' q_inter'],
    local2: Float32[Array, ' q_inter'],
    local_aux: Float32[Array, ' q_inter'],
    global2: Float32[Array, ''],
    global_aux: Float32[Array, ''],
    global_scale: float,
) -> tuple[
    Float32[Array, ' q_inter'],
    Float32[Array, ' q_inter'],
    Float32[Array, ''],
    Float32[Array, ''],
]:
    """One Gibbs sweep of the horseshoe scales, Makalic & Schmidt (2015).

    Why a horseshoe and not the ridge
    ---------------------------------
    The interaction block asks a wide basis -- ``d`` time functions times
    ``q`` unit features, order 10^2 columns -- which covariates carry a
    diverging baseline trend. In the designs this model is aimed at, a handful
    do and the rest do not. A single ridge scale has to be one number for all of
    them, so it either shrinks the few real loadings toward zero or lets the
    many null ones absorb noise; the compromise shows up as excess error in the
    fitted trend and, through the identity that ties the trend to the exposure
    profile, as excess error in the treatment effect.

    A global-local prior does not face that trade-off: ``tau`` pulls the whole
    block toward zero while the heavy-tailed ``lambda_j`` lets individual
    columns escape.

    Why *this* horseshoe
    --------------------
    The half-Cauchy prior is written as a scale mixture,

    .. code::

        c_j | lambda_j, tau ~ N(0, tau^2 lambda_j^2)
        lambda_j^2 | nu_j   ~ IG(1/2, 1/nu_j)      nu_j ~ IG(1/2, 1)
        tau^2 | xi          ~ IG(1/2, 1/xi)        xi   ~ IG(1/2, 1/A^2)

    so that ``lambda_j ~ C+(0, 1)`` and ``tau ~ C+(0, A)``. Every full
    conditional is then inverse-Gamma, which matters here for a specific
    reason: it leaves the model **conditionally Gaussian in** ``c`` **given the
    scales**. The exact joint ``(c, gamma, beta)`` draw of
    :func:`trend_block_step` -- the move that removes the trend/level/exposure
    ridge -- therefore survives unchanged. A horseshoe sampled by
    Metropolis-Hastings, or a non-conjugate sparsity prior, would have cost that
    joint draw, which is a far larger loss than the sparsity is a gain.

    ``A`` is the global scale, ``sigma_c`` from :func:`trend_prior_scales`, so
    the horseshoe and the ridge it replaces agree on typical magnitude.

    Parameters
    ----------
    key
        PRNG key.
    coef_inter
        Current interaction coefficients ``c``, shape ``(q_inter,)``.
    local2, local_aux
        Current ``lambda_j^2`` and ``nu_j``, shape ``(q_inter,)``.
    global2, global_aux
        Current ``tau^2`` and ``xi``, scalars.
    global_scale
        ``A``, the half-Cauchy scale of ``tau``.

    Returns
    -------
    local2, local_aux, global2, global_aux
        The updated scales.
    """
    A2 = float(max(global_scale, 1e-6)) ** 2
    c2 = jnp.square(coef_inter)
    q = c2.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # lambda_j^2 | . ~ IG(1, 1/nu_j + c_j^2 / (2 tau^2))
    local2 = _inv_gamma(
        k1, 1.0, jnp.reciprocal(local_aux) + 0.5 * c2 / global2
    )
    local2 = jnp.clip(local2, HS_VAR_MIN, HS_VAR_MAX)

    # nu_j | . ~ IG(1, 1 + 1/lambda_j^2)
    local_aux = _inv_gamma(k2, 1.0, 1.0 + jnp.reciprocal(local2))
    local_aux = jnp.clip(local_aux, HS_VAR_MIN, HS_VAR_MAX)

    # tau^2 | . ~ IG((q + 1)/2, 1/xi + (1/2) sum_j c_j^2 / lambda_j^2)
    global2 = _inv_gamma(
        k3,
        0.5 * (q + 1.0),
        jnp.reciprocal(global_aux) + 0.5 * jnp.sum(c2 / local2),
    )
    global2 = jnp.clip(global2, HS_VAR_MIN, HS_VAR_MAX)

    # xi | . ~ IG(1, 1/A^2 + 1/tau^2)
    global_aux = _inv_gamma(k4, 1.0, 1.0 / A2 + jnp.reciprocal(global2))
    global_aux = jnp.clip(global_aux, HS_VAR_MIN, HS_VAR_MAX)

    return local2, local_aux, global2, global_aux


def horseshoe_col_scales(
    base_scale: Float32[Array, ' q_tot'],
    num_period: int,
    local2: Float32[Array, ' q_inter'],
    global2: Float32[Array, ''],
) -> Float32[Array, ' q_tot']:
    """Column scales implied by the current horseshoe draw.

    The leading ``num_period`` entries -- the free common-time factor -- keep
    their fixed scale: there is no sparsity to discover there, every period has
    data, and shrinking the common trend is exactly the failure mode the block
    exists to avoid. Only the interaction columns get ``tau lambda_j``.
    """
    if num_period >= base_scale.shape[0]:
        return base_scale
    inter = jnp.sqrt(global2 * local2).astype(jnp.float32)
    return jnp.concatenate([base_scale[:num_period], inter])


def trend_block_step(
    key: Key[Array, ''],
    trend_coef: Float32[Array, ' q_tot'],
    gamma: Float32[Array, ' N_units'],
    beta: Float32[Array, ' S_max_plus_1'],
    R: Float32[Array, ' n'],
    H: Float32[Array, 'n q_tot'],
    d_vec: Float32[Array, ' n'],
    obs_mask: Bool[Array, ' n'],
    unit_idx: Int32[Array, ' n'],
    N_units: int,
    exposure_idx: Int32[Array, ' n'],
    sigma2: Float32[Array, ''],
    sigma_gamma2: Float32[Array, ''],
    K_chol: Float32[Array, 'S_max_plus_1 S_max_plus_1'],
    trend_col_scale: Float32[Array, ' q_tot'],
    random_intercept: bool,
    sample_beta: bool,
    conditional_precision: Float32[Array, ' n'] | None = None,
    temperature: Float32[Array, ''] = jnp.float32(1.0),
) -> tuple[
    Float32[Array, ' q_tot'],
    Float32[Array, ' N_units'],
    Float32[Array, ' S_max_plus_1'],
    Float32[Array, ' n'],
]:
    """Draw ``(c, gamma, beta)`` from their exact joint Gaussian conditional.

    Conditional on the two forests and on ``sigma^2``, the model is linear in
    the trend coefficients ``c``, the unit intercepts ``gamma`` and the exposure
    trajectory ``beta``, with Gaussian priors on all three. Their joint
    conditional is therefore exactly Gaussian, and this function samples from
    it: acceptance probability one, no step size, no tuning.

    Sampling the three **jointly** rather than in sequence is the point. They
    are strongly dependent -- the trend, the unit levels and the exposure
    profile all bid for the same post-adoption movement -- and a coordinate-wise
    Gibbs sweep through dependent Gaussian blocks moves along the ridge at a
    rate set by their correlation, which is exactly the pathology the
    diverging-trend designs exhibit. A joint draw has no such rate.

    Implementation
    --------------
    Three reparameterizations keep the cost and the conditioning under control.

    ``gamma`` is marginalized analytically. Its precision block is diagonal --
    a unit intercept touches only its own cells -- so the Schur complement
    ``P_ww - P_w,gamma diag(p_gamma)^-1 P_gamma,w`` costs
    ``O((q_tot + S)^2 N)`` rather than a factorization of an
    ``(q_tot + S + N)``-dimensional matrix. ``gamma`` is then drawn from its
    (diagonal, hence trivial) conditional given the sampled ``(c, beta)``.

    ``beta`` is whitened as ``beta = L u`` with ``L L' = K_tilde``, the same
    device :func:`longbet._gp.sample_beta_gp` uses. The prior precision of ``u``
    is the identity, so ``K_tilde^-1`` is never formed and the joint precision
    is well conditioned however flat the kernel is.

    ``c`` is whitened the same way, by its own prior scale: the draw is taken in
    ``w = c / s`` with design ``H diag(s)``, so the prior precision of ``w`` is
    again the identity and the sampled coefficient is recovered as ``c = s w``.
    This is not cosmetic. Under the horseshoe ``s_j = tau lambda_j`` is a
    *sampled* quantity that is supposed to become very small for the null
    columns; in the precision parameterization that puts entries of order
    ``s_j^-2`` on the diagonal of ``P_cc``, and a float32 Cholesky of a matrix
    whose diagonal spans many orders of magnitude loses the small, informative
    directions. Whitening moves the shrinkage into the design, where a small
    ``s_j`` simply makes a column contribute nothing and costs no conditioning
    at all. The two parameterizations define the same conditional; only one of
    them survives single precision.


    Parameters
    ----------
    key
        PRNG key.
    trend_coef, gamma, beta
        Current values of the three blocks.
    R
        Full-model residual in data units, zero on unobserved cells.
    H
        Trend design matrix from :func:`build_trend_design`, shape
        ``(n, q_tot)``.
    d_vec
        Treatment factor with the trajectory removed, ``b_Z nu(X_i, t)``.
    obs_mask, unit_idx, N_units, exposure_idx
        Panel bookkeeping.
    sigma2, sigma_gamma2
        Current innovation and intercept variances.
    K_chol
        Lower Cholesky factor of the marginalized GP kernel.
    trend_col_scale
        Per-column prior standard deviation of the trend coefficients, shape
        ``(q_tot,)``. Fixed under the ridge prior
        (:func:`trend_prior_scales`); under the horseshoe the interaction
        entries are the current ``tau lambda_j`` from
        :func:`horseshoe_col_scales` and therefore change every sweep.
    random_intercept, sample_beta
        Which blocks are live; a dead block is left untouched and dropped from
        the joint draw.
    conditional_precision
        Per-cell precision under the SUR likelihood. ``None`` selects the scalar
        ``temperature / sigma^2``.
    temperature
        Inverse temperature of the replica.

    Returns
    -------
    trend_coef, gamma, beta, R
        The sampled blocks and the residual they imply.
    """
    if conditional_precision is None:
        cell_prec = jnp.where(obs_mask, temperature / sigma2, 0.0).astype(jnp.float32)
    else:
        cell_prec = jnp.where(obs_mask, conditional_precision, 0.0).astype(jnp.float32)

    L = jnp.asarray(K_chol, dtype=jnp.float32)
    S1 = L.shape[-1]
    q_tot = H.shape[1]

    # Whiten the trend columns by their prior scale: the draw happens in
    # w = c / s, whose prior precision is the identity. See the docstring --
    # this is what lets the horseshoe drive s_j to ~0 without wrecking the
    # float32 Cholesky.
    s_col = jnp.asarray(trend_col_scale, dtype=jnp.float32)
    Hs = H * s_col[None, :]                          # (n, q_tot)

    # --- partial residual with all three blocks removed ---------------------
    trend_fit = H @ trend_coef
    R_tilde = R + trend_fit
    if random_intercept:
        R_tilde = R_tilde + gamma[unit_idx]
    if sample_beta:
        R_tilde = R_tilde + d_vec * beta[exposure_idx]
    R_tilde = jnp.where(obs_mask, R_tilde, 0.0)

    Hw = Hs * cell_prec[:, None]                     # (n, q_tot)

    # --- trend block ---------------------------------------------------------
    P_cc = Hs.T @ Hw + jnp.eye(q_tot, dtype=jnp.float32)
    h_c = Hw.T @ R_tilde

    # --- whitened trajectory block ------------------------------------------
    if sample_beta:
        # Row (i, t) of the whitened design is d_it * L[S_it, :].
        Dt = d_vec[:, None] * L[exposure_idx]        # (n, S1)
        Dtw = Dt * cell_prec[:, None]
        P_uu = Dt.T @ Dtw + jnp.eye(S1, dtype=jnp.float32)
        P_cu = Hw.T @ Dt                             # (q_tot, S1)
        h_u = Dtw.T @ R_tilde
        P_ww = jnp.block([[P_cc, P_cu], [P_cu.T, P_uu]])
        h_w = jnp.concatenate([h_c, h_u])
    else:
        Dt = None
        P_ww = P_cc
        h_w = h_c

    # --- marginalize the (diagonal) unit intercepts --------------------------
    if random_intercept:
        p_gg = (
            jnp.zeros(N_units, dtype=jnp.float32).at[unit_idx].add(cell_prec)
            + jnp.reciprocal(sigma_gamma2)
        )
        h_g = (
            jnp.zeros(N_units, dtype=jnp.float32)
            .at[unit_idx]
            .add(cell_prec * R_tilde)
        )
        # P_{c,gamma}: sum of the precision-weighted design rows within a unit.
        P_cg = (
            jnp.zeros((N_units, q_tot), dtype=jnp.float32).at[unit_idx].add(Hw)
        ).T                                          # (q_tot, N_units)
        if sample_beta:
            P_ug = (
                jnp.zeros((N_units, S1), dtype=jnp.float32).at[unit_idx].add(Dtw)
            ).T                                      # (S1, N_units)
            P_wg = jnp.concatenate([P_cg, P_ug], axis=0)
        else:
            P_wg = P_cg
        scaled = P_wg / p_gg[None, :]
        P_eff = P_ww - scaled @ P_wg.T
        h_eff = h_w - scaled @ h_g
    else:
        p_gg = None
        h_g = None
        P_wg = None
        P_eff = P_ww
        h_eff = h_w

    # --- joint draw of (c, u) ------------------------------------------------
    # The whitened prior contributes an identity block I, so P_eff has exact
    # eigenvalues >= 1.0. Adding 1e-5 * diag(P_ww) (~80x float32 machine
    # epsilon times the un-demeaned diagonal) prevents Schur-complement
    # cancellation at extreme precisions while avoiding any O(k^3) eigh under
    # jax.vmap.
    P_eff_sym = 0.5 * (P_eff + P_eff.T) + jnp.diag(1e-5 * jnp.diag(P_ww))
    L_eff = jnp.linalg.cholesky(P_eff_sym)
    eta_w = jax.random.normal(key, shape=h_eff.shape, dtype=jnp.float32)
    w_new = jax.scipy.linalg.cho_solve((L_eff, True), h_eff) + jax.scipy.linalg.solve_triangular(
        L_eff.T, eta_w, lower=False
    )

    # Back to the interpretable parameterization: c = s w. Everything outside
    # this function -- the trace, save/load, predict, evaluate_trend -- sees
    # coefficients against the unscaled design H, so the whitening stays local.
    trend_new = s_col * w_new[:q_tot]
    if sample_beta:
        u_new = w_new[q_tot:]
        beta_new = L @ u_new
    else:
        beta_new = beta

    # --- unit intercepts given the sampled blocks ----------------------------
    if random_intercept:
        v_gg = jnp.reciprocal(p_gg)
        mean_g = v_gg * (h_g - P_wg.T @ w_new)
        gamma_new = mean_g + jax.random.normal(
            jax.random.fold_in(key, 1), (N_units,), dtype=jnp.float32
        ) * jnp.sqrt(v_gg)
    else:
        gamma_new = gamma

    # --- residual implied by the draw ---------------------------------------
    fitted = H @ trend_new
    if random_intercept:
        fitted = fitted + gamma_new[unit_idx]
    if sample_beta:
        fitted = fitted + d_vec * beta_new[exposure_idx]
    R_new = jnp.where(obs_mask, R_tilde - fitted, 0.0)

    return trend_new, gamma_new, beta_new, R_new


def evaluate_trend(
    coef: Any,
    features: np.ndarray,
    phi: np.ndarray,
    unit_idx: np.ndarray,
    time_idx: np.ndarray,
    period: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the trend block for a draw axis, in NumPy.

    Computes ``sum_k psi_k(t) f_k + sum_m phi_m(t) b(X_i)' c_m``, matching the
    column order :func:`build_trend_design` lays down. Used by
    :meth:`longbet.LongBet.predict`, which works in NumPy and blocks over cells.

    Parameters
    ----------
    coef
        Trend coefficients, shape ``(draws, (T - 1) + d q)``.
    features
        Unit feature matrix for the prediction units, shape ``(N, q)``.
    phi
        Smooth time basis for the interaction block, shape ``(T, d)``.
    unit_idx, time_idx
        Cell indices, length ``n``.
    period
        Common-time basis, shape ``(T, T_fit - 1)``, or ``None``.

    Returns
    -------
    np.ndarray
        Shape ``(draws, n)``.
    """
    coef = np.asarray(coef, dtype=np.float32)
    if coef.ndim == 1:
        coef = coef[None, :]
    n_cells = len(unit_idx)
    out = np.zeros((coef.shape[0], n_cells), dtype=np.float32)
    if coef.shape[1] == 0:
        return out

    n_period = 0 if period is None else int(np.asarray(period).shape[1])
    if n_period > 0:
        period_f32 = np.asarray(period, dtype=np.float32)          # (T, T-1)
        period_effect = coef[:, :n_period] @ period_f32.T          # (draws, T)
        out += period_effect[:, time_idx]

    q = features.shape[1]
    d = phi.shape[1]
    if d == 0 or q == 0:
        return out
    inter = coef[:, n_period:n_period + d * q]
    C = inter.reshape(inter.shape[0], d, q)
    # (draws, d, N) then gather cells: avoids forming a (draws, n, q) array,
    # and reducing over the small degree axis ``d`` per cell makes the result
    # bit-for-bit independent of ``len(unit_idx)`` (the prediction block size).
    unit_effect = np.einsum('mdq,nq->mdn', C, features.astype(np.float32))
    phi_t = np.asarray(phi, dtype=np.float32)[time_idx].T[None, :, :]  # (1, d, n)
    out += np.sum(unit_effect[:, :, unit_idx] * phi_t, axis=1)
    return out
