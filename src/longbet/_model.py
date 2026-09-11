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

"""High-level Python API for LongBet."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
from jaxtyping import Array, Key

from bartz.mcmcloop import evaluate_trace
from bartz.mcmcloop._trace import MainTrace

from longbet._config import LongBetConfig
from longbet._design import Design, integer_grid_block, quantile_block
from longbet._diagnostics import StabilityResult, att_stability
from longbet._gp import forecast_beta_gp
from longbet._io import load_npz, save_npz
from longbet._ordinal import category_probabilities, prepare_ordinal
from longbet._summary import choose_ordinal_block_size
from longbet._loop import LongBetTrace, run_longbet_mcmc
from longbet._state import LongBetState, init_longbet
from longbet._summary import (
    BlockAccumulator,
    PosteriorSummary,
    choose_block_size,
    summarize_draws,
)


# ---------------------------------------------------------------------------
# device selection
# ---------------------------------------------------------------------------


def available_devices() -> list[str]:
    """List the JAX devices LongBet can run on, as ``'platform:id'`` strings."""
    return [f"{d.platform}:{d.id}" for d in jax.devices()]


def _gpu_devices() -> list[Any]:
    for platform in ("gpu", "cuda", "rocm"):
        try:
            found = jax.devices(platform)
        except RuntimeError:
            continue
        if found:
            return found
    return []


def resolve_device(device: str) -> Any:
    """Resolve a ``device`` setting to a concrete JAX device.

    ``'auto'`` prefers a GPU and falls back to CPU; ``'gpu'`` raises if none is
    visible, rather than silently running on the CPU.
    """
    if device == "cpu":
        return jax.devices("cpu")[0]
    gpus = _gpu_devices()
    if device == "gpu":
        if not gpus:
            raise RuntimeError(
                "device='gpu' was requested but JAX sees no GPU. Available devices: "
                f"{available_devices()}. Install a CUDA build of JAX "
                "(pip install 'longbet-jax[cuda12]') or use device='auto'."
            )
        return gpus[0]
    return gpus[0] if gpus else jax.devices("cpu")[0]


# ---------------------------------------------------------------------------
# panel helpers
# ---------------------------------------------------------------------------


def derive_exposure(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Derive the exposure index ``S`` from an absorbing treatment panel.

    ``S_it`` is the number of elapsed time units since adoption, counting the
    first treated period as 1, and 0 for every untreated cell. It coincides with
    ``sum_{s<=t} Z_is`` when ``t`` is unit-spaced.

    Parameters
    ----------
    z
        Treatment indicator, shape ``(N, T)``.
    t
        Calendar time vector, length ``T``, non-decreasing.

    Returns
    -------
    S : np.ndarray
        Integer exposure index, shape ``(N, T)``.
    """
    z = np.asarray(z)
    t = np.asarray(t, dtype=np.float64)
    treated = z == 1

    any_treated = treated.any(axis=1)
    first = np.argmax(treated, axis=1)
    # Time just before adoption; for a unit treated from the first period we
    # extrapolate one step back so that period gets S = 1.
    step = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    prev = np.where(first > 0, t[np.maximum(first - 1, 0)], t[0] - step)

    elapsed = np.rint(t[None, :] - prev[:, None]).astype(np.int64)
    S = np.where(treated & any_treated[:, None], np.maximum(elapsed, 0), 0)
    return S.astype(np.int32)


def _check_absorbing(z: np.ndarray) -> None:
    if z.ndim != 2 or any(size == 0 for size in z.shape):
        raise ValueError("z must be a nonempty (N, T) array.")
    if not np.all(np.isin(z, (0, 1))):
        raise ValueError("z must contain only finite binary treatment indicators (0 or 1).")
    if z.shape[1] > 1:
        drops = np.argwhere(np.diff(z, axis=1) < 0)
        if drops.size:
            row = int(drops[0, 0])
            raise ValueError(
                f"longbet requires absorbing treatment: unit {row} switches back from "
                "treated to untreated. Drop or reshape those units before fitting."
            )


def _check_time_vector(t: np.ndarray) -> None:
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("time vector t must be a nonempty finite 1-D array.")
    if t.size > 1:
        d = np.diff(t)
        if np.any(d <= 0):
            raise ValueError("time vector t must be strictly increasing")
        if not np.allclose(d, np.rint(d)):
            warnings.warn(
                "time vector t is not integer-spaced; the exposure index is rounded to "
                "whole time units, so unobserved exposure levels will be drawn from the "
                "GP prior.",
                UserWarning,
                stacklevel=3,
            )


def _as_cell_major(a: np.ndarray) -> np.ndarray:
    """Flatten an ``(N, T, P)`` array to ``(P, N*T)`` in unit-major cell order."""
    a = np.asarray(a, dtype=np.float32)
    n, t, p = a.shape
    return a.reshape(n * t, p).T


# ---------------------------------------------------------------------------
# prediction container
# ---------------------------------------------------------------------------


class LongBetPrediction:
    """Posterior predictions from a fitted LongBet model.

    Attributes
    ----------
    tauhats
        Treatment-effect draws, shape ``(N, T, draws)``, or ``None`` under
        ``summary_only``.
    muhats0
        Untreated-outcome draws (including the unit intercept), same shape.
    yhats
        Factual outcome draws, same shape.
    att_full
        ATT draws by exposure time, shape ``(S_max, draws)``. Always present:
        it is small, so ``summary_only`` does not suppress it, and ``get_att``
        and ``stability`` therefore work in both modes.
    tau_summary, mu0_summary, y_summary
        Per-cell posterior mean, sd and credible bounds, shape ``(N, T)``.
    beta_values
        Exposure-trajectory draws, shape ``(draws, S_total)``, extended by the
        GP projection where the prediction panel runs past the fitted horizon.
    outcome
        ``'continuous'`` or ``'binary'``. For a binary outcome every quantity
        above is on the **latent probit scale**; apply ``scipy.stats.norm.cdf``
        to ``yhats`` or ``muhats0`` for probabilities.
    num_chains
        Number of chains that produced ``att_full``'s draw axis, chain-major.
        Kept so :meth:`stability` can split it without guessing.
    att_counts
        Number of treated cells behind each exposure time in ``att_full``. Late
        exposure times are reached only by the earliest adopters, so this is
        usually the explanation for a poor diagnostic there.
    """

    def __init__(
        self,
        *,
        tauhats: np.ndarray | None,
        muhats0: np.ndarray | None,
        yhats: np.ndarray | None,
        att_full: np.ndarray,
        beta_values: np.ndarray,
        z: np.ndarray,
        s: np.ndarray,
        tau_summary: PosteriorSummary,
        mu0_summary: PosteriorSummary,
        y_summary: PosteriorSummary,
        outcome: str = "continuous",
        num_chains: int = 1,
        att_counts: np.ndarray | None = None,
        summary_only: bool = False,
        num_categories: int | None = None,
        cutpoints_samples: np.ndarray | None = None,
        prob_y: np.ndarray | None = None,
        prob_mu0: np.ndarray | None = None,
        prob_tau: np.ndarray | None = None,
        prob_y_summary: PosteriorSummary | None = None,
        prob_mu0_summary: PosteriorSummary | None = None,
        prob_tau_summary: PosteriorSummary | None = None,
        att_prob_full: np.ndarray | None = None,
    ) -> None:
        self.tauhats = tauhats
        self.muhats0 = muhats0
        self.yhats = yhats
        self.att_full = att_full
        self.beta_values = beta_values
        self.z = z
        self.s = s
        self.tau_summary = tau_summary
        self.mu0_summary = mu0_summary
        self.y_summary = y_summary
        self.outcome = outcome
        self.num_chains = int(num_chains)
        self.att_counts = att_counts
        self.summary_only = summary_only
        self.num_categories = num_categories
        self.categories = None if num_categories is None else np.arange(num_categories)
        self.cutpoints_samples = cutpoints_samples
        self.prob_y, self.prob_mu0, self.prob_tau = prob_y, prob_mu0, prob_tau
        self.prob_y_summary = prob_y_summary
        self.prob_mu0_summary = prob_mu0_summary
        self.prob_tau_summary = prob_tau_summary
        self.att_prob_full = att_prob_full

    def _require_ordinal(self) -> None:
        if self.outcome != "ordinal":
            raise ValueError("This method requires an ordinal prediction.")

    def predict_probabilities(self, arm="factual", summary=True):
        """Category summaries or draws for factual, control, or effect means.

        Public draws have shape (N,T,K,D). Effects are paired treated-minus-
        control probability differences, with category sum zero. Bounds use
        the interval chosen at predict time.
        """
        self._require_ordinal()
        fields = {"factual": "prob_y", "control": "prob_mu0", "effect": "prob_tau"}
        if arm not in fields:
            raise ValueError("arm must be 'factual', 'control', or 'effect'")
        result = getattr(self, fields[arm] + ("_summary" if summary else ""))
        if result is None:
            raise ValueError("Full probability draws were discarded; predict with summary_only=False.")
        return result

    @staticmethod
    def _summarize_ordinal_att(draws, alpha):
        if (isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha)
                or not isinstance(alpha, (int, float, np.integer, np.floating))
                or not np.isfinite(alpha) or not 0 < alpha < 1):
            raise ValueError("alpha must be finite and strictly between 0 and 1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.mean(draws, axis=-1)
            intervals = np.percentile(draws, [100*alpha/2, 100*(1-alpha/2)], axis=-1)
        return {"att": mean, "intervals": intervals, "att_full": draws,
                "exposure": np.arange(1, draws.shape[0]+1)}

    def att_probabilities(self, alpha=0.05):
        """Category ATT draws (S,K,D) and summaries over the final draw axis."""
        self._require_ordinal()
        result = self._summarize_ordinal_att(self.att_prob_full, alpha)
        result["categories"] = self.categories
        return result

    def att_expected_score(self, weights=None, alpha=0.05):
        """ATT on a user-defined score; default rank scores assume equal spacing.

        Weights need not increase: indicator weights report exceedance or
        single-category effects. Scores are formed within each posterior draw.
        """
        self._require_ordinal()
        weights = np.asarray(self.categories if weights is None else weights)
        if (weights.dtype.kind not in "biuf" or weights.shape != (self.num_categories,)
                or not np.isfinite(weights).all()):
            raise ValueError(f"weights must contain {self.num_categories} finite numeric values")
        weights = weights.astype(np.float64)
        draws = np.einsum("k,skd->sd", weights, self.att_prob_full)
        result = self._summarize_ordinal_att(draws, alpha)
        result["weights"] = weights
        return result

    def att(self, alpha: float = 0.05) -> dict[str, Any]:
        """Average treatment effect on the treated, by exposure time.

        Works with or without full draws, because ``att_full`` is always
        retained.
        """
        att_full = self.att_full
        q_low, q_high = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            att_mean = np.nanmean(att_full, axis=1)
            intervals = np.nanpercentile(att_full, [q_low, q_high], axis=1)
        return {
            "att": att_mean,
            "intervals": intervals,
            "att_full": att_full,
            "exposure": np.arange(1, att_full.shape[0] + 1),
        }

    def catt(self, alpha: float = 0.05) -> dict[str, Any]:
        """Per-cell conditional average treatment effect on the treated."""
        del alpha  # bounds were fixed at predict time
        return {
            "catt": np.asarray(self.tau_summary.mean),
            "std": np.asarray(self.tau_summary.std),
            "lower": np.asarray(self.tau_summary.lower),
            "upper": np.asarray(self.tau_summary.upper),
        }

    def stability(
        self,
        min_ess: float = 400.0,
        max_rhat: float = 1.01,
        alpha: float = 0.05,
        warn: bool = True,
    ) -> StabilityResult:
        """ESS and R-hat for the ATT series.

        The chain layout is taken from :attr:`num_chains` rather than inferred
        from the array's shape, so nothing depends on the number of draws
        happening to exceed the number of exposure times.
        """
        att = self.att_full  # (S, chains * draws), chain-major
        if self.num_chains > 1 and att.shape[1] % self.num_chains == 0:
            per = att.shape[1] // self.num_chains
            arr = att.reshape(att.shape[0], self.num_chains, per).transpose(1, 2, 0)
        else:
            arr = att.T[None, ...]
        return att_stability(
            arr, min_ess=min_ess, max_rhat=max_rhat, alpha=alpha, warn=warn,
            n_treated=self.att_counts,
        )


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------


class LongBet:
    """LongBet: Bayesian causal forests for panel data with exposure dynamics.

    Fits

    .. code::

        Y_it = alpha * mu(X_i, t, X^tv_it)
             + b_{Z_it} * beta_{S_it} * nu(X_i, S_it, t, X^trt,tv_it)
             + gamma_i + eps_it

    with ``beta`` a shared Gaussian-process trajectory over the exposure index
    and ``gamma_i`` a unit random intercept.
    """

    def __init__(self, config: LongBetConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            self.config = LongBetConfig(**kwargs)
        else:
            self.config = dataclasses.replace(config, **kwargs) if kwargs else config

        self.state: LongBetState | None = None
        self.trace: LongBetTrace | None = None
        self.design_: Design | None = None
        self.device_: Any = None

        # Outcome scaling. For a continuous outcome the sampler runs on
        # (y - meany) / sdy. For a binary one the probit intercept is handed to
        # bartz as the forest offset and therefore already sits inside the mu
        # trace, so meany stays 0 and must not be added again on output.
        self.meany: float = 0.0
        self.sdy: float = 1.0
        self.offset_: float = 0.0

        self.N_: int = 0
        self.T_: int = 0
        self.S_max_: int = 0
        self.t_fit_: np.ndarray | None = None
        self.multi_origin: dict[str, Any] | None = None

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        y: np.ndarray | Array,
        x: np.ndarray | Array,
        z: np.ndarray | Array,
        t: np.ndarray | Array | None = None,
        x_trt: np.ndarray | Array | None = None,
        x_tv: np.ndarray | Array | None = None,
        x_trt_tv: np.ndarray | Array | None = None,
        ps: np.ndarray | Array | None = None,
        key: Key[Array, ''] | None = None,
    ) -> LongBet:
        """Fit the model to a panel.

        Parameters
        ----------
        y
            Outcome, shape ``(N, T)`` or a length ``N*T`` vector in unit-major
            order. ``NaN`` marks an unobserved cell; unbalanced panels are
            handled by marginalizing those cells, not by imputing them.
        x
            Time-invariant covariates for the prognostic forest, shape ``(N, P)``.
        z
            Treatment indicator, shape ``(N, T)``. Must be absorbing.
        t
            Calendar time, length ``T``. Defaults to ``1..T``.
        x_trt
            Time-invariant covariates for the treatment forest, shape
            ``(N, P_trt)``. Defaults to ``x``, in which case the two forests
            share one block of columns rather than duplicating them.
        x_tv
            Time-varying covariates for the prognostic forest, shape
            ``(N, T, P_tv)``.
        x_trt_tv
            Time-varying covariates for the treatment forest, shape
            ``(N, T, P_trt_tv)``.
        ps
            Propensity score, shape ``(N,)`` or ``(N, T)``. Visible to the
            prognostic forest only. For staggered adoption the principled
            analogue is a timing or hazard score rather than one scalar per
            unit; per-cell values are accepted for that reason.
        key
            PRNG key. Defaults to one derived from ``config.random_seed``.

        Returns
        -------
        self
        """
        if self.config.num_shared_trees:
            raise ValueError("num_shared_trees requires LongBetMulti; a scalar fit cannot share trees across outcomes.")
        self.multi_origin = None
        self.device_ = resolve_device(self.config.device)
        with jax.default_device(self.device_):
            return self._fit_impl(y, x, z, t, x_trt, x_tv, x_trt_tv, ps, key)

    def _fit_impl(
        self,
        y: Any,
        x: Any,
        z: Any,
        t: Any,
        x_trt: Any,
        x_tv: Any,
        x_trt_tv: Any,
        ps: Any,
        key: Any,
    ) -> LongBet:
        z_np = np.asarray(z)
        if z_np.ndim != 2:
            raise ValueError(f"z must be an (N, T) array, got shape {z_np.shape}")
        N, T = z_np.shape
        self.N_, self.T_ = N, T

        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim != 2 or x_np.shape[0] != N:
            raise ValueError(f"x must be (N, P) with N={N}, got {x_np.shape}")

        if self.config.outcome == "ordinal":
            # Validate original dtype before float conversion could accept strings.
            prepare_ordinal(y, self.config.num_categories)
        y_np = np.asarray(y, dtype=np.float64)
        y_mat = y_np.reshape(N, T) if y_np.ndim == 1 else y_np
        if y_mat.shape != (N, T):
            raise ValueError(f"y has shape {y_mat.shape}, expected {(N, T)}")

        t_vec = (
            np.arange(1, T + 1, dtype=np.float32)
            if t is None
            else np.asarray(t, dtype=np.float32)
        )
        if t_vec.size != T:
            raise ValueError(f"t has length {t_vec.size}, expected {T}")
        _check_time_vector(t_vec)
        _check_absorbing(z_np)
        self.t_fit_ = t_vec

        if self.config.random_intercept:
            n_always = int(np.sum(np.all(z_np == 1, axis=1)))
            if n_always:
                warnings.warn(
                    f"{n_always} unit(s) are treated in every period, so their random "
                    "intercept is not separable from their treatment effect. Consider "
                    "dropping them or setting random_intercept=False.",
                    UserWarning,
                    stacklevel=3,
                )

        S_mat = derive_exposure(z_np, t_vec)
        self.S_max_ = int(S_mat.max())

        unit_idx = np.repeat(np.arange(N, dtype=np.int32), T)
        time_idx = np.tile(np.arange(T, dtype=np.int32), N)
        exposure_idx = S_mat.ravel().astype(np.int32)
        z_vec = z_np.ravel().astype(np.float32)
        y_vec = y_mat.ravel()
        obs_mask = np.isfinite(y_vec)
        if not obs_mask.any():
            raise ValueError("y contains no observed cells")

        # -- outcome scaling -------------------------------------------------
        # Arithmetic is done on a NaN-free copy: masked cells are marginalized,
        # and letting their NaNs through would only raise spurious warnings.
        y_clean = np.where(obs_mask, y_vec, 0.0)
        obs_y = y_vec[obs_mask]
        if self.config.outcome == "ordinal":
            prepared = prepare_ordinal(y_vec, self.config.num_categories)
            self.meany, self.sdy = 0.0, 1.0
            self.offset_ = prepared.offset
            y_proc = prepared.labels
        elif self.config.outcome == "binary":
            uniq = np.unique(obs_y)
            if not np.all(np.isin(uniq, (0.0, 1.0))):
                raise ValueError(
                    f"outcome='binary' requires y in {{0, 1}}; found values {uniq[:5]}"
                )
            rate = float(obs_y.mean())
            self.meany, self.sdy = 0.0, 1.0
            # The probit intercept becomes the forest offset, so bartz's latent
            # z starts at the right level and evaluate_trace adds it back.
            self.offset_ = float(stats.norm.ppf(np.clip(rate, 1e-4, 1.0 - 1e-4)))
            y_proc = y_clean.astype(np.float32)
        elif not self.config.standardize:
            self.meany, self.sdy, self.offset_ = 0.0, 1.0, 0.0
            y_proc = y_clean.astype(np.float32)
        else:
            self.meany = float(obs_y.mean())
            self.sdy = float(obs_y.std())
            if self.sdy == 0:
                raise ValueError("y is constant on the observed cells")
            self.offset_ = 0.0
            y_proc = np.where(
                obs_mask, (y_clean - self.meany) / self.sdy, 0.0
            ).astype(np.float32)

        # -- design ----------------------------------------------------------
        raw = self._raw_inputs(
            x=x_np,
            x_trt=x_trt,
            x_tv=x_tv,
            x_trt_tv=x_trt_tv,
            ps=ps,
            time_idx=time_idx,
            exposure_idx=exposure_idx,
            N=N,
            T=T,
        )

        prng = key if key is not None else jax.random.key(self.config.random_seed)
        k_bin, k_chains, k_mcmc = jax.random.split(prng, 3)

        self.design_ = self._build_design(
            raw, x_trt_given=x_trt is not None, key=k_bin
        )
        X_unified = self.design_.build(raw)

        num_chains = self.config.num_chains if self.config.num_chains > 1 else None
        self.state = init_longbet(
            X_unified=jnp.asarray(X_unified),
            y=jnp.asarray(y_proc),
            unit_idx=jnp.asarray(unit_idx),
            time_idx=jnp.asarray(time_idx),
            exposure_idx=jnp.asarray(exposure_idx),
            z_vec=jnp.asarray(z_vec),
            obs_mask=jnp.asarray(obs_mask),
            max_split_mu=jnp.asarray(self.design_.max_split_mu),
            max_split_nu=jnp.asarray(self.design_.max_split_nu),
            config=self.config,
            offset=self.offset_,
            num_chains=num_chains,
            chain_key=k_chains,
        )

        result = run_longbet_mcmc(
            key=k_mcmc,
            state=self.state,
            n_burn=self.config.num_burnin,
            n_save=self.config.num_sweeps,
            n_skip=self.config.n_skip,
            inner_loop_length=self.config.inner_loop_length,
        )
        self.state = result.final_state
        self.trace = result.main_trace
        if self.config.outcome == "ordinal":
            # Surface checked interval failures before returning a fitted model.
            jax.block_until_ready((self.state.z, self.trace.cutpoints))
        return self

    # -- design construction ------------------------------------------------

    def _raw_inputs(
        self,
        *,
        x: np.ndarray,
        x_trt: Any,
        x_tv: Any,
        x_trt_tv: Any,
        ps: Any,
        time_idx: np.ndarray,
        exposure_idx: np.ndarray,
        N: int,
        T: int,
    ) -> dict[str, np.ndarray]:
        """Broadcast every user input to ``(n_cols, N*T)`` in unit-major order."""
        raw: dict[str, np.ndarray] = {
            "x": np.repeat(np.asarray(x, dtype=np.float32), T, axis=0).T,
            "t": time_idx.reshape(1, -1).astype(np.float32),
            "s": exposure_idx.reshape(1, -1).astype(np.float32),
        }
        if x_trt is not None:
            a = np.asarray(x_trt, dtype=np.float32)
            if a.ndim != 2 or a.shape[0] != N:
                raise ValueError(f"x_trt must be (N, P_trt) with N={N}, got {a.shape}")
            raw["x_trt"] = np.repeat(a, T, axis=0).T
        if x_tv is not None:
            a = np.asarray(x_tv, dtype=np.float32)
            if a.ndim != 3 or a.shape[:2] != (N, T):
                raise ValueError(f"x_tv must be (N, T, P_tv) = ({N}, {T}, P), got {a.shape}")
            raw["x_tv"] = _as_cell_major(a)
        if x_trt_tv is not None:
            a = np.asarray(x_trt_tv, dtype=np.float32)
            if a.ndim != 3 or a.shape[:2] != (N, T):
                raise ValueError(
                    f"x_trt_tv must be (N, T, P) = ({N}, {T}, P), got {a.shape}"
                )
            raw["x_trt_tv"] = _as_cell_major(a)
        if ps is not None:
            a = np.asarray(ps, dtype=np.float32)
            if a.ndim == 1:
                if a.shape[0] != N:
                    raise ValueError(f"ps must have length N={N}, got {a.shape}")
                raw["ps"] = np.repeat(a, T).reshape(1, -1)
            elif a.shape == (N, T):
                raw["ps"] = a.ravel().reshape(1, -1)
            else:
                raise ValueError(f"ps must be (N,) or (N, T), got shape {a.shape}")
        return raw

    def _build_design(
        self,
        raw: dict[str, np.ndarray],
        *,
        x_trt_given: bool,
        key: Key[Array, ''],
    ) -> Design:
        """Lay out the unified matrix and decide which forest sees what."""
        cfg = self.config
        blocks = []
        keys = iter(jax.random.split(key, 5))

        # Baseline covariates. When x_trt is not supplied the two forests share
        # one block; when it is, each gets its own and neither sees the other's.
        blocks.append(
            quantile_block(
                "x",
                "x",
                raw["x"],
                cfg.num_cutpoints,
                mu_visible=True,
                nu_visible=not x_trt_given,
                key=next(keys),
            )
        )
        if x_trt_given:
            blocks.append(
                quantile_block(
                    "x_trt",
                    "x_trt",
                    raw["x_trt"],
                    cfg.num_cutpoints,
                    mu_visible=False,
                    nu_visible=True,
                    key=next(keys),
                )
            )
        if "x_tv" in raw:
            blocks.append(
                quantile_block(
                    "x_tv", "x_tv", raw["x_tv"], cfg.num_cutpoints,
                    mu_visible=True, nu_visible=False, key=next(keys),
                )
            )
        if "x_trt_tv" in raw:
            blocks.append(
                quantile_block(
                    "x_trt_tv", "x_trt_tv", raw["x_trt_tv"], cfg.num_cutpoints,
                    mu_visible=False, nu_visible=True, key=next(keys),
                )
            )

        # Calendar time: split_time_ps governs the prognostic forest.
        # split_time_trt controls exposure splits, independently of calendar time.
        blocks.append(
            integer_grid_block(
                "t", "t", self.T_, mu_visible=cfg.split_time_ps, nu_visible=True
            )
        )
        # Exposure index: never visible to mu, and visible to nu only when
        # split_time_trt is on. Turning it off removes the c(S) component of the
        # beta-nu non-identification entirely.
        blocks.append(
            integer_grid_block(
                "s", "s", self.S_max_ + 1, mu_visible=False,
                nu_visible=cfg.split_time_trt,
            )
        )
        if "ps" in raw:
            blocks.append(
                quantile_block(
                    "ps", "ps", raw["ps"], cfg.num_cutpoints,
                    mu_visible=True, nu_visible=False, key=next(keys),
                )
            )
        return Design(blocks=tuple(blocks))

    # -- prediction ---------------------------------------------------------

    def predict(
        self,
        x: np.ndarray | Array,
        z: np.ndarray | Array,
        t: np.ndarray | Array | None = None,
        x_trt: np.ndarray | Array | None = None,
        x_tv: np.ndarray | Array | None = None,
        x_trt_tv: np.ndarray | Array | None = None,
        ps: np.ndarray | Array | None = None,
        y: np.ndarray | Array | None = None,
        summary_only: bool = False,
        alpha: float = 0.05,
        key: Key[Array, ''] | None = None,
        block_size: int | None = None,
        sig_knl: float | None = None,
        lambda_knl: float | None = None,
        cache_forest_evaluations: bool = False,
    ) -> LongBetPrediction:
        """Predict treatment effects and outcomes.

        The estimand is the contrast between being ``S`` periods into treatment
        and not being treated at all,

        .. code::

            tau_t(X_i, S) = b1 * beta_S * nu(X_i, S, t)
                          - b0 * beta_0 * nu(X_i, 0,  t)

        so the treatment forest is evaluated **twice**: once on the factual
        exposure and once on a copy of the design with ``S = 0``. Under control
        the exposure index is 0, not ``S``, so both the multiplier and the
        forest's input change.

        Parameters
        ----------
        x, z, t, x_trt, x_tv, x_trt_tv, ps
            As in :meth:`fit`. Every input the model was fitted with must be
            supplied again; omitting one raises rather than silently producing a
            design whose columns no longer match the fitted trees.
        y
            Unused; accepted so a fit call's arguments can be forwarded verbatim.
        summary_only
            Compute per-cell summaries without ever materializing the
            ``(N, T, draws)`` arrays. Quantiles remain exact -- the panel is
            processed in blocks of cells, not of draws. The ATT is returned in
            full either way.
        alpha
            Tail probability for credible intervals.
        key
            PRNG key for the GP projection when the prediction panel runs past
            the fitted exposure horizon. Defaults to one derived from
            ``config.random_seed``, so projections are reproducible.
        block_size
            Cells per evaluation block. Defaults to about 32 MB per block.
        sig_knl, lambda_knl
            Optionally override the kernel used to **project** ``beta`` past the
            fitted exposure horizon. They refit nothing: the posterior draws of
            ``beta`` inside the observed window are whatever the model was
            fitted with, and only the conditional projection beyond it changes.
            A sweep over them is therefore a sensitivity analysis of the
            extrapolation prior, holding the in-sample estimates fixed -- which
            is the honest way to show how much a forecast owes to the prior
            rather than to the data. It is *not* equivalent to refitting with a
            different kernel. Defaults to the fitted values.
        cache_forest_evaluations
            Reuse the forest evaluations when consecutive calls share a design,
            as a projection-kernel sweep does. Off by default: the cache holds
            three ``(draws, cells)`` arrays -- half a gigabyte at 1,000 draws
            over 42,000 cells -- which is exactly what ``summary_only`` exists
            to avoid. Only the most recent design is retained.

        Returns
        -------
        LongBetPrediction
        """
        del y
        if self.trace is None or self.design_ is None:
            raise RuntimeError("call fit() before predict()")
        device = self.device_ if self.device_ is not None else resolve_device(
            self.config.device
        )
        with jax.default_device(device):
            return self._predict_impl(
                x, z, t, x_trt, x_tv, x_trt_tv, ps, summary_only, alpha, key, block_size,
                sig_knl=sig_knl, lambda_knl=lambda_knl,
                cache_forest_evaluations=cache_forest_evaluations,
            )

    def _predict_impl(
        self,
        x: Any,
        z: Any,
        t: Any,
        x_trt: Any,
        x_tv: Any,
        x_trt_tv: Any,
        ps: Any,
        summary_only: bool,
        alpha: float,
        key: Any,
        block_size: int | None = None,
        sig_knl: float | None = None,
        lambda_knl: float | None = None,
        cache_forest_evaluations: bool = False,
    ) -> LongBetPrediction:
        assert self.design_ is not None and self.trace is not None

        z_np = np.asarray(z)
        if z_np.ndim != 2:
            raise ValueError(f"z must be an (N, T) array, got shape {z_np.shape}")
        N, T = z_np.shape
        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim != 2 or x_np.shape[0] != N:
            raise ValueError(f"x must be (N, P) with N={N}, got {x_np.shape}")

        t_vec = (
            np.asarray(self.t_fit_, dtype=np.float32)
            if t is None
            else np.asarray(t, dtype=np.float32)
        )
        if t_vec.size != T:
            raise ValueError(f"t has length {t_vec.size}, expected {T}")
        _check_absorbing(z_np)

        _check_time_vector(t_vec)

        S_mat = derive_exposure(z_np, t_vec)
        S_max_test = int(S_mat.max())

        time_idx = np.tile(np.arange(T, dtype=np.int32), N)
        unit_idx = np.repeat(np.arange(N, dtype=np.int32), T)
        exposure_idx = S_mat.ravel().astype(np.int32)
        z_vec = z_np.ravel().astype(np.float32)
        M = N * T

        raw = self._raw_inputs(
            x=x_np, x_trt=x_trt, x_tv=x_tv, x_trt_tv=x_trt_tv, ps=ps,
            time_idx=time_idx, exposure_idx=exposure_idx, N=N, T=T,
        )
        # The treatment forest is evaluated on the fitted exposure grid, so the
        # factual index is clipped; extrapolation beyond the fitted horizon is
        # carried by the GP projection of beta, not by the forest.
        raw_zero = dict(raw)
        raw["s"] = np.clip(raw["s"], 0, self.S_max_)
        raw_zero["s"] = np.zeros_like(raw["s"])

        X_factual = np.asarray(self.design_.build(raw), dtype=np.uint8)
        X_zero_s = np.asarray(self.design_.build(raw_zero), dtype=np.uint8)

        # -- posterior draws, chains flattened into one sample axis -----------
        def flat(a: Any) -> np.ndarray:
            arr = np.asarray(a)
            return arr.reshape(-1, *arr.shape[2:]) if arr.ndim > 1 and self._chained else arr

        beta_draws = np.asarray(self.trace.beta)
        if S_max_test > self.S_max_:
            proj_key = key if key is not None else jax.random.key(
                self.config.random_seed + 1
            )
            beta_draws = np.asarray(
                forecast_beta_gp(
                    key=proj_key,
                    beta_obs=jnp.asarray(beta_draws),
                    s_obs=np.arange(self.S_max_ + 1),
                    s_fut=np.arange(self.S_max_ + 1, S_max_test + 1),
                    sig_knl=self.config.sig_knl if sig_knl is None else float(sig_knl),
                    lambda_knl=self.config.lambda_knl if lambda_knl is None else float(lambda_knl),
                    kernel_type=self.config.kernel_type,
                    sigma_m=self.config.sigma_m,
                    gp_constant_mean=self.config.gp_constant_mean,
                    jitter=self.config.gp_jitter,
                )
            )

        beta_flat = flat(beta_draws)
        b0 = flat(self.trace.b0)[:, None]
        b1 = flat(self.trace.b1)[:, None]
        alpha_d = flat(self.trace.alpha)[:, None]
        gamma_flat = flat(np.asarray(self.trace.gamma))
        D = beta_flat.shape[0]
        K = self.config.num_categories if self.config.outcome == "ordinal" else None
        cutpoints = None
        if K is not None:
            if self.trace.cutpoints is None:
                raise ValueError("Ordinal predictions require saved cutpoint draws.")
            # Explicit D avoids reshape(-1, 0) for the ordinal binary limit.
            cutpoints = np.asarray(self.trace.cutpoints).reshape(D, K - 2)

        has_gamma = gamma_flat.shape[1] == N
        if not has_gamma:
            warnings.warn(
                f"predict() received {N} units but the model was fitted on "
                f"{gamma_flat.shape[1]}; unit intercepts cannot be matched and are "
                "taken as 0 (the prior mean) for these rows.",
                UserWarning,
                stacklevel=4,
            )

        # Per-cell quantities are expanded inside the block loop, not before
        # it: a (draws, N*T) array here would defeat the whole point of blocking.
        beta_0 = beta_flat[:, 0:1]

        # -- blocked evaluation ------------------------------------------------
        blk = block_size or choose_block_size(D, M)
        if K is not None:
            blk = min(blk, choose_ordinal_block_size(D, M, K))
        blk = max(1, min(int(blk), M))
        scale, centre = self.sdy, self.meany

        S_max_out = max(S_max_test, 1)
        att_sums = np.zeros((S_max_out, D), dtype=np.float64)
        att_counts = np.zeros(S_max_out, dtype=np.int64)

        keep = not summary_only
        tau_out = np.empty((D, M), dtype=np.float32) if keep else None
        mu0_out = np.empty((D, M), dtype=np.float32) if keep else None
        y_out = np.empty((D, M), dtype=np.float32) if keep else None

        stats_tau = BlockAccumulator(M, alpha)
        stats_mu0 = BlockAccumulator(M, alpha)
        stats_y = BlockAccumulator(M, alpha)
        prob_out = prob_stats = att_prob_sums = None
        if K is not None:
            prob_out = [np.empty((M, K, D), np.float64) if keep else None for _ in range(3)]
            prob_stats = [BlockAccumulator(M * K, alpha, dtype=np.float64) for _ in range(3)]
            att_prob_sums = np.zeros((S_max_out, K, D), np.float64)

        nu_trace, mu_trace = self.trace.nu_trace, self.trace.mu_trace
        # When the treatment forest cannot split on the exposure index, the
        # S = 0 design differs from the factual one only in a column it never
        # reads, so the counterfactual evaluation is exactly the factual one.
        # Skipping it removes a third of the prediction cost.
        s_visible = any(b.name == "s" and b.nu_visible for b in self.design_.blocks)

        # Optionally reuse the forest evaluations across predict() calls whose
        # design is unchanged -- a GP-projection sweep, say, which varies only
        # beta and leaves both forests alone.
        #
        # Off by default, and deliberately so: what is cached is three
        # (draws x cells) arrays, exactly the thing summary_only exists to avoid
        # building. At 1,000 draws over 42,000 cells that is half a gigabyte.
        # Only the most recent design is kept, so the cache cannot grow without
        # bound, and the trace objects are held in the key rather than their
        # id(), which CPython reuses after collection.
        cache_key = None
        cached_blocks = None
        if cache_forest_evaluations:
            cache_key = (
                X_factual.shape,
                hash(X_factual.tobytes()),
                hash(X_zero_s.tobytes()) if s_visible else 0,
                nu_trace,
                mu_trace,
                blk,
                s_visible,
            )
            held_key, held_blocks = getattr(self, "_eval_cache", (None, None))
            if held_key is not None and _same_cache_key(held_key, cache_key):
                cached_blocks = held_blocks

        blocks_to_cache = [] if cached_blocks is None and cache_key is not None else None

        for b_idx, lo in enumerate(range(0, M, blk)):
            hi = min(lo + blk, M)
            sl = slice(lo, hi)

            # Every block is padded to the same width before being handed to
            # evaluate_trace. A ragged final block would be a second input
            # shape, and so a second XLA compilation of the tree evaluator --
            # which on a large panel costs more than the block itself.
            if cached_blocks is not None:
                nu_f, nu_0, mu_f = cached_blocks[b_idx]
            else:
                nu_f = _eval_block(X_factual, lo, hi, blk, nu_trace)
                nu_0 = _eval_block(X_zero_s, lo, hi, blk, nu_trace) if s_visible else nu_f
                mu_f = _eval_block(X_factual, lo, hi, blk, mu_trace)
                if blocks_to_cache is not None:
                    blocks_to_cache.append((nu_f, nu_0, mu_f))

            beta_S = beta_flat[:, exposure_idx[sl]]
            bz = np.where(z_vec[None, sl] == 1.0, b1, b0)
            g = gamma_flat[:, unit_idx[sl]] if has_gamma else 0.0

            tau = (b1 * beta_S * nu_f - b0 * beta_0 * nu_0) * scale
            mu0 = (alpha_d * mu_f + b0 * beta_0 * nu_0 + g) * scale + centre
            yhat = (alpha_d * mu_f + bz * beta_S * nu_f + g) * scale + centre

            # ATT sums by exposure time, accumulated rather than aligned into an
            # (N, S_max, draws) array.
            s_blk = exposure_idx[sl]
            treated = (z_vec[sl] == 1.0) & (s_blk >= 1) & (s_blk <= S_max_out)
            if treated.any():
                idx = s_blk[treated] - 1
                np.add.at(att_sums, idx, tau[:, treated].T.astype(np.float64))
                att_counts += np.bincount(idx, minlength=S_max_out)

            if K is not None:
                p0 = category_probabilities(mu0, cutpoints)
                delta = category_probabilities(mu0 + tau, cutpoints)
                delta -= p0
                py = category_probabilities(yhat, cutpoints)
                if treated.any():
                    np.add.at(att_prob_sums, idx, delta[:, treated, :].transpose(1, 2, 0))
                for acc, out, values in zip(prob_stats, prob_out, (py, p0, delta)):
                    acc.update(lo*K, hi*K, values.reshape(D, (hi-lo)*K))
                    if keep:
                        out[sl] = values.transpose(1, 2, 0)

            stats_tau.update(lo, hi, tau)
            stats_mu0.update(lo, hi, mu0)
            stats_y.update(lo, hi, yhat)
            if keep:
                tau_out[:, sl] = tau
                mu0_out[:, sl] = mu0
                y_out[:, sl] = yhat

        if blocks_to_cache is not None:
            # One entry only: a new design evicts the old one.
            self._eval_cache = (cache_key, blocks_to_cache)

        with np.errstate(invalid="ignore", divide="ignore"):
            att_full = np.where(
                att_counts[:, None] > 0, att_sums / np.maximum(att_counts, 1)[:, None], np.nan
            ).astype(np.float32)

        def to_panel(s: PosteriorSummary) -> PosteriorSummary:
            return PosteriorSummary(*(np.asarray(v).reshape(N, T) for v in s))

        ordinal_fields = {}
        if K is not None:
            ordinal_fields = dict(num_categories=K, cutpoints_samples=cutpoints,
                att_prob_full=np.where(att_counts[:, None, None] > 0,
                    att_prob_sums / np.maximum(att_counts, 1)[:, None, None], np.nan))
            for name, acc, out in zip(("prob_y", "prob_mu0", "prob_tau"), prob_stats, prob_out):
                ordinal_fields[name] = out.reshape(N, T, K, D) if keep else None
                ordinal_fields[name + "_summary"] = PosteriorSummary(
                    *(v.reshape(N, T, K) for v in acc.result()))

        return LongBetPrediction(
            tauhats=tau_out.T.reshape(N, T, D) if keep else None,
            muhats0=mu0_out.T.reshape(N, T, D) if keep else None,
            yhats=y_out.T.reshape(N, T, D) if keep else None,
            att_full=att_full,
            beta_values=beta_flat,
            z=z_np,
            s=S_mat,
            tau_summary=to_panel(stats_tau.result()),
            mu0_summary=to_panel(stats_mu0.result()),
            y_summary=to_panel(stats_y.result()),
            outcome=self.config.outcome,
            num_chains=self.config.num_chains,
            att_counts=att_counts,
            summary_only=summary_only,
            **ordinal_fields,
        )

    @property
    def _chained(self) -> bool:
        return self.config.num_chains > 1

    # -- convenience --------------------------------------------------------

    def att(self, pred: LongBetPrediction, alpha: float = 0.05) -> dict[str, Any]:
        """ATT by exposure time; see :meth:`LongBetPrediction.att`."""
        return pred.att(alpha=alpha)

    def catt(self, pred: LongBetPrediction, alpha: float = 0.05) -> dict[str, Any]:
        """Per-cell CATT; see :meth:`LongBetPrediction.catt`."""
        return pred.catt(alpha=alpha)

    def stability(
        self,
        pred: LongBetPrediction,
        min_ess: float = 400.0,
        max_rhat: float = 1.01,
        alpha: float = 0.05,
        warn: bool = True,
    ) -> StabilityResult:
        """ESS and R-hat for the ATT series.

        Diagnostics are reported on the ATT, the identified quantity, never on
        ``beta`` or a leaf.
        """
        return pred.stability(
            min_ess=min_ess, max_rhat=max_rhat, alpha=alpha, warn=warn
        )

    @staticmethod
    def rollout_summary(z: np.ndarray, t: np.ndarray | None = None, **kwargs: Any):
        """Tidy description of a staggered rollout; see :func:`rollout_summary`.

        Exposed on the estimator so it is discoverable, but it is a static
        method: inspecting the design is something you do *before* fitting.
        """
        from longbet._rollout import rollout_summary as _summary

        return _summary(z, t, **kwargs)

    @staticmethod
    def plot_rollout(z: np.ndarray, t: np.ndarray | None = None, **kwargs: Any):
        """Tile chart of a staggered rollout; see :func:`plot_rollout`."""
        from longbet._rollout import plot_rollout as _plot

        return _plot(z, t, **kwargs)

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the fitted model to a ``.npz`` archive."""
        if self.trace is None or self.design_ is None:
            raise RuntimeError("cannot save an unfitted model")
        save_npz(
            path=path,
            trace=self.trace,
            config=self.config,
            meany=self.meany,
            sdy=self.sdy,
            metadata={
                "N": self.N_,
                "T": self.T_,
                "S_max": self.S_max_,
                "offset": self.offset_,
                "t_fit": np.asarray(self.t_fit_).tolist() if self.t_fit_ is not None else [],
                "design": self.design_.to_dict(),
                "multi_origin": self.multi_origin,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> LongBet:
        """Load a model saved by :meth:`save`."""
        trace, config, meany, sdy, meta = load_npz(path)
        model = cls(config=config)
        model.trace = trace
        model.meany = meany
        model.sdy = sdy
        model.multi_origin = meta.get("multi_origin")
        model.offset_ = float(meta.get("offset", 0.0))
        model.N_ = int(meta.get("N", 0))
        model.T_ = int(meta.get("T", 0))
        model.S_max_ = int(meta.get("S_max", 0))
        if meta.get("t_fit"):
            model.t_fit_ = np.asarray(meta["t_fit"], dtype=np.float32)
        if meta.get("design"):
            model.design_ = Design.from_dict(meta["design"])
            # An extracted multi-outcome child is a scalar archive, so check
            # its grids too rather than only checking the parent multi archive.
            model.design_.validate_panel_grids(model.T_, model.S_max_)
        return model


def _same_cache_key(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    """Compare cache keys whose trace entries must match by identity."""
    if len(a) != len(b):
        return False
    return all(x is y if isinstance(x, MainTrace) else x == y for x, y in zip(a, b))


def _eval(X_block: np.ndarray, trace: Any) -> Any:
    """Evaluate a forest trace on one block of cells, chains flattened."""
    return evaluate_trace(jnp.asarray(X_block), trace, flatten_chains=True)


def _eval_block(
    X: np.ndarray, lo: int, hi: int, width: int, trace: Any
) -> np.ndarray:
    """Evaluate ``trace`` on cells ``[lo, hi)``, padded to a fixed ``width``.

    Padding keeps the input shape constant across blocks, so the tree evaluator
    is compiled once per fit rather than once per distinct block size.
    """
    block = X[:, lo:hi]
    n = hi - lo
    if n < width:
        block = np.concatenate(
            [block, np.repeat(block[:, -1:], width - n, axis=1)], axis=1
        )
    out = np.asarray(_eval(block, trace))
    return out[:, :n] if n < width else out


# ---------------------------------------------------------------------------
# module-level helpers, kept for the R bridge and for scripting
# ---------------------------------------------------------------------------


def get_att(pred: LongBetPrediction, alpha: float = 0.05) -> dict[str, Any]:
    """Average treatment effect on the treated, by exposure time.

    Works for predictions made with or without ``summary_only``.
    """
    return pred.att(alpha=alpha)


def get_catt(pred: LongBetPrediction, alpha: float = 0.05) -> dict[str, Any]:
    """Per-cell conditional average treatment effect on the treated."""
    return pred.catt(alpha=alpha)


__all__ = [
    "LongBet",
    "LongBetPrediction",
    "available_devices",
    "derive_exposure",
    "get_att",
    "get_catt",
    "resolve_device",
    "summarize_draws",
]
