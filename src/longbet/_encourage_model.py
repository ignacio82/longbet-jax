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

"""Randomized encouragement with absorbing adoption: the adoption-clock model.

Two LongBet equations are fitted and composed:

* **Outcome on the adoption clock.** ``y`` is regressed on the *observed*
  adoption ``d`` with LongBet's exposure clock counting periods since
  adoption, so the outcome equation contains the jump that adoption causes,
  its exposure profile ``beta_S`` and its covariate heterogeneity ``nu(x)``,
  plus a unit intercept. Both arms contribute adoption events.
* **Adoption as a discrete-time hazard.** A binary LongBet is fitted to the
  first-adoption indicator on the at-risk set (cells after adoption are
  masked out), with the randomized encouragement ``z`` as the treatment and
  periods since the offer as its clock. Its unit intercept is a frailty.

The offer's effects follow by composition: the hazard equation gives each
unit's adoption-time distribution with and without the offer, the outcome
equation gives the effect of having adopted ``s`` periods ago, and the offer
effect on the outcome is the difference of the expected exposure responses.
The frailty is integrated *outside* the survival product (Gauss-Hermite), as
it must be, because it makes survival positively dependent across periods.

This uses the exclusion restriction (the offer moves outcomes only through
adoption) and the assumption that, given covariates and the unit intercept,
the timing of adoption is not confounded with the outcome innovations. The
model-free design-based reference in ``reference``/``encouragement_effects``
needs neither and is the check on the population offer effects.

A ``reduced-form`` alternative (regressing ``y`` on ``z`` directly) was
retired: an adoption jump at a random period that the outcome equation does
not model is unit-specific, time-structured residual noise, and flexible
forests fit it, so the exact posterior of that model was less accurate out of
sample than early-stopped chains. See ``benchmarks/tempering_plan.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import ndtr

from longbet._config import LongBetConfig
from longbet._diagnostics import compute_ess, compute_rhat
from longbet._encourage import _critical, _validate, encouragement_effects
from longbet._model import LongBet, derive_exposure

ARCHIVE_VERSION = 2
INFERENCE_VERSION = "adoption_clock_v1"
ENGINE = "adoption_clock"
#: Gauss-Hermite nodes used to integrate a unit frailty out of a probit path.
FRAILTY_NODES = 12


def _finite_matrix(value: Any, name: str, n: int, shape: tuple | None = None) -> np.ndarray:
    a = np.asarray(value)
    if (a.dtype.kind not in "biuf" or a.ndim != 2 or a.shape[0] != n
            or (shape is not None and a.shape != shape)):
        raise ValueError(f"{name} must be a finite numeric matrix with "
                         + (f"shape {shape}." if shape else f"{n} rows."))
    a = np.array(a, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be complete and finite.")
    return a


@dataclasses.dataclass(frozen=True)
class EncouragementDraws:
    """Aligned offer-effect draws and the population over which they average.

    ``draws[name]`` has axes ``(group, horizon, chain, retained_draw)``.
    ``weights[g]`` sum to one over the study units in group ``g``.
    """

    draws: dict[str, np.ndarray]
    group_labels: tuple[str, ...]
    group_counts: np.ndarray
    weights: np.ndarray
    periods: np.ndarray
    period_indices: np.ndarray
    horizons: np.ndarray
    standardization: str
    provenance: str

    @property
    def effects(self) -> np.ndarray:
        """Equivalent array with axes ``(chain, draw, group, horizon, outcome)``."""
        return np.stack(tuple(self.draws.values()), axis=-1).transpose(2, 3, 0, 1, 4)


def _target_weights(
    n_units: int, groups: Any, weights: Any,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Normalize fixed, nonnegative unit weights within each baseline group."""
    if weights is None:
        unit_weights = np.ones(n_units, dtype=np.float64)
    else:
        unit_weights = np.asarray(weights, dtype=np.float64)
        if (unit_weights.shape != (n_units,) or
                not np.all(np.isfinite(unit_weights)) or np.any(unit_weights < 0)):
            raise ValueError("weights must be a finite nonnegative vector of length N.")
        if unit_weights.max(initial=0) > 0:
            unit_weights = unit_weights / unit_weights.max()
    if unit_weights.sum() <= 0:
        raise ValueError("weights must have a positive total.")

    masks = [np.ones(n_units, dtype=bool)]
    labels = ["all"]
    if groups is not None:
        raw_groups = np.asarray(groups)
        if raw_groups.shape != (n_units,) or np.any(pd.isna(raw_groups)):
            raise ValueError("groups must be a length-N vector of nonmissing baseline labels.")
        try:
            values = list(pd.unique(raw_groups))
        except TypeError as exc:
            raise ValueError("groups must contain scalar baseline labels.") from exc
        for value in values:
            label = str(value)
            if label in labels:
                raise ValueError("group labels must be unique as strings and cannot be 'all'.")
            labels.append(label)
            masks.append(raw_groups == value)

    counts = np.asarray([mask.sum() for mask in masks], dtype=np.int64)
    target = np.stack([unit_weights * mask for mask in masks])
    totals = target.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("every group must have positive total weight.")
    return tuple(labels), counts, target / totals[:, None]


def _reference_groups(data: dict, standardized: EncouragementDraws, alpha: float) -> pd.DataFrame:
    """Conditional-on-subgroup-arm-count reference, on the identical unit target."""
    frames = []
    for g, label in enumerate(standardized.group_labels):
        mask = standardized.weights[g] > 0
        z = data["z"][mask]
        counts = [int((z[:, -1] == a).sum()) for a in (0, 1)]
        if min(counts) > 0:
            frame = encouragement_effects(data["y"][mask], data["d"][mask], z,
                                          data["t"], alpha=alpha)
        else:
            # Retain unsupported groups. There is no observed arm mean to fill in.
            frame = encouragement_effects(data["y"], data["d"], data["z"],
                                          data["t"], alpha=alpha)
            for name in frame:
                if name.startswith(("itt_", "wald")):
                    frame[name] = np.nan
            frame["wald_set_type"] = "unavailable"
            frame["wald_reason"] = "missing_group_arm"
            frame["n_control"], frame["n_encouraged"] = counts
        frame.insert(0, "group", label)
        frame["n_units"] = int(mask.sum())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Composition of the two equations
# ---------------------------------------------------------------------------

def _flat_draws(model: LongBet, name: str) -> np.ndarray:
    """A traced parameter with chains flattened chain-major into one draw axis."""
    a = np.asarray(getattr(model.trace, name))
    return a.reshape(-1, *a.shape[2:]) if a.ndim > 1 and model._chained else a


def _frailty_rule(marginal: bool) -> tuple[np.ndarray, np.ndarray]:
    if not marginal:
        return np.zeros(1), np.ones(1)
    nodes, weights = np.polynomial.hermite_e.hermegauss(FRAILTY_NODES)
    return nodes, weights / weights.sum()


def _predict_quiet(model: LongBet, x: np.ndarray, z: np.ndarray, t: np.ndarray,
                   x_trt: np.ndarray | None = None) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"predict\(\) received .* units", category=UserWarning)
        return model.predict(x, z, t=t, x_trt=x_trt, summary_only=False)


def _without_matched_intercepts(model: LongBet, values: np.ndarray, n_units: int) -> np.ndarray:
    """Remove the fitted unit intercepts ``predict`` attaches when ``N`` matches the fit."""
    if n_units != model.N_ or not model.config.random_intercept:
        return values
    gamma = _flat_draws(model, "gamma")                       # (D, N)
    return values - gamma.T[:, None, :]


def adoption_distribution(
    adoption_model: LongBet, x: np.ndarray, z: np.ndarray, t: np.ndarray, *,
    x_trt: np.ndarray | None = None, fitted_intercepts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """First-adoption probabilities and adoption stock under an offer schedule.

    Returns ``(p_adopt, stock)``, both ``(N, T, draws)``: the probability of
    adopting first in each period and of having adopted by each period. By
    default the unit frailty is integrated out with Gauss-Hermite quadrature
    around the survival product, the estimand for a unit of that covariate
    profile drawn from the fitted population. ``fitted_intercepts`` (``(draws,
    N)``, the study units' own frailty draws) evaluates those units instead.
    """
    n_units, n_periods = z.shape
    marginal = fitted_intercepts is None
    eta = np.asarray(_predict_quiet(adoption_model, x, z, t, x_trt).yhats, dtype=np.float64)
    eta = _without_matched_intercepts(adoption_model, eta, n_units)
    if marginal:
        sd = (np.sqrt(_flat_draws(adoption_model, "sigma_gamma2"))
              if adoption_model.config.random_intercept else np.zeros(eta.shape[-1]))
    else:
        eta = eta + np.asarray(fitted_intercepts, dtype=np.float64).T[:, None, :]
        sd = np.zeros(eta.shape[-1])
    nodes, weights = _frailty_rule(marginal)
    p_adopt = np.zeros_like(eta)
    stock = np.zeros_like(eta)
    ones = np.ones((n_units, 1, eta.shape[-1]))
    for node, weight in zip(nodes, weights):
        lam = ndtr(eta + node * sd[None, None, :])
        surv = np.cumprod(1.0 - lam, axis=1)
        first = lam * np.concatenate([ones, surv[:, :-1, :]], axis=1)
        p_adopt += weight * first
        stock += weight * (1.0 - surv)
    return p_adopt, stock


def offer_effect_on_outcome(
    outcome_model: LongBet, x: np.ndarray, t: np.ndarray,
    p_adopt_offer: np.ndarray, p_adopt_control: np.ndarray, *,
    x_trt: np.ndarray | None = None, fitted_intercepts: np.ndarray | None = None,
) -> np.ndarray:
    """Offer effect on the outcome, ``(N, T, draws)``, by composition.

    For every candidate adoption period ``a`` the outcome equation is evaluated
    under adoption from ``a`` onward; the effect at calendar period ``t`` is
    then the exposure response weighted by the difference in the probability of
    adopting at ``a`` with and without the offer. For a binary outcome the
    response is on the probability scale, with the outcome's unit intercept
    integrated out, or set to the study units' own draws when
    ``fitted_intercepts`` (``(draws, N)``) is given.
    """
    n_units, n_periods = p_adopt_offer.shape[:2]
    binary = outcome_model.config.outcome == "binary"
    marginal = fitted_intercepts is None
    diff = p_adopt_offer - p_adopt_control
    effect = np.zeros_like(diff)
    nodes, weights = _frailty_rule(marginal and binary)
    sd = None
    if binary:
        sd = (np.sqrt(_flat_draws(outcome_model, "sigma_gamma2"))
              if marginal and outcome_model.config.random_intercept else np.zeros(diff.shape[-1]))
    for a in range(n_periods):
        schedule = np.zeros((n_units, n_periods), dtype=np.float32)
        schedule[:, a:] = 1.0
        pred = _predict_quiet(outcome_model, x, schedule, t, x_trt)
        tau = np.asarray(pred.tauhats, dtype=np.float64)
        weight_a = diff[:, a, None, :]
        if not binary:
            effect += weight_a * tau
            continue
        mu0 = _without_matched_intercepts(outcome_model, np.asarray(pred.muhats0, dtype=np.float64), n_units)
        if not marginal:
            mu0 = mu0 + np.asarray(fitted_intercepts, dtype=np.float64).T[:, None, :]
        for node, weight in zip(nodes, weights):
            shift = node * sd[None, None, :]
            effect += weight * weight_a * (ndtr(mu0 + tau + shift) - ndtr(mu0 + shift))
    return effect


@dataclasses.dataclass(frozen=True)
class EncouragementComparison:
    """Paired whole-unit bootstrap comparison of model and reference ITTs.

    ``replicates`` has axes (replicate, group, horizon, quantity, estimator),
    where quantities are Y,D and estimators are model,reference. Failures remain
    NaN and are listed in ``failures``; incomplete runs never issue flags.
    """

    table: pd.DataFrame
    replicates: np.ndarray
    failures: pd.DataFrame
    metadata: dict[str, Any]


class EncouragementPrediction:
    """Common-population posterior ITTs, ratios, and separate reference inference.

    ``draws`` maps ``itt_y``, ``itt_d``, and ``wald`` to arrays with axes
    (group, horizon, chain, retained draw). No draws are removed or clipped.
    Exactly zero denominators produce NaN and make that ratio's posterior
    interval unavailable. Near-zero and negative denominators remain intact.
    Quantiles of a Wald ratio are not a design-based weak-IV confidence set.
    """

    def __init__(self, standardized: EncouragementDraws, reference: pd.DataFrame,
                 metadata: dict[str, Any], *, alpha: float = 0.05):
        _critical(alpha)
        self.standardized = standardized
        self.reference = reference.copy()
        self.metadata = dict(metadata, standardization=standardized.standardization,
                             calibration_status="not_established",
                             inference_version=INFERENCE_VERSION,
                             posterior_inference="model_posterior",
                             reference_inference="normal_ar",
                             target_aligned=standardized.standardization == "conditional",
                             ratio_interpretation="Wald; CACE requires additional assumptions")
        self.alpha = float(alpha)
        dy, dd = standardized.draws["outcome"], standardized.draws["takeup"]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = np.divide(dy, dd, out=np.full_like(dy, np.nan), where=dd != 0)
        self.draws = {"itt_y": dy, "itt_d": dd, "wald": ratio}
        self.group_labels = standardized.group_labels
        self.periods = standardized.periods
        self.horizons = standardized.horizons

    def stability(self, *, min_ess: float = 400, max_rhat: float = 1.01) -> pd.DataFrame:
        """Diagnose each group/horizon ITT and ratio, including interval endpoints.

        Report bulk/tail ESS, rank R-hat, and quantile MCSE. Mean MCSE is only
        reported for the ITTs: a ratio's posterior mean need not exist. Missing,
        constant, short, or single-chain traces cannot pass the full check.
        Threshold checks never certify interval coverage or IV identification.
        """
        if not np.isfinite(min_ess) or min_ess <= 0:
            raise ValueError("min_ess must be positive and finite.")
        if not np.isfinite(max_rhat) or max_rhat < 1:
            raise ValueError("max_rhat must be finite and at least 1.")
        rows = []
        for quantity, values in self.draws.items():
            for g, group in enumerate(self.group_labels):
                for h, horizon in enumerate(self.horizons):
                    a = values[g, h]
                    row = dict(group=group, period=float(self.periods[h]), horizon=int(horizon),
                               quantity=quantity, ess_bulk=np.nan, ess_tail=np.nan,
                               rhat=np.nan, mcse_mean=np.nan, mcse_median=np.nan,
                               mcse_lower=np.nan, mcse_upper=np.nan)
                    valid = np.all(np.isfinite(a)) and a.shape[1] >= 4 and np.ptp(a) > 0
                    if valid:
                        row["ess_bulk"] = float(compute_ess(a, method="bulk"))
                        row["ess_tail"] = float(compute_ess(
                            a, method="tail", prob=(self.alpha / 2, 1 - self.alpha / 2)))
                        row["rhat"] = float(compute_rhat(a))
                        for field, p in (("median", .5), ("lower", self.alpha / 2),
                                         ("upper", 1 - self.alpha / 2)):
                            row[f"mcse_{field}"] = float(az.mcse(a, method="quantile", prob=p))
                        if quantity != "wald":
                            row["mcse_mean"] = float(az.mcse(a, method="mean"))
                    row["ess_ok"] = bool(np.isfinite(row["ess_bulk"]) and
                                         np.isfinite(row["ess_tail"]) and
                                         min(row["ess_bulk"], row["ess_tail"]) >= min_ess)
                    row["rhat_ok"] = bool(np.isfinite(row["rhat"]) and row["rhat"] <= max_rhat)
                    row["diagnostics_passed"] = row["ess_ok"] and row["rhat_ok"]
                    row["calibration_status"] = "not_established"
                    rows.append(row)
        return pd.DataFrame(rows)

    def effects(self, *, min_ess: float = 400, max_rhat: float = 1.01) -> pd.DataFrame:
        """Tidy posterior summaries with diagnostics; ratio means are absent.

        Quantiles use the empirical inverse CDF, preserving infinite values.
        Numeric summaries are retained when diagnostics fail, with an explicit
        status. Use ``wald()`` for a table that withholds unconverged ratios.
        """
        rows = []
        for quantity, values in self.draws.items():
            for g, group in enumerate(self.group_labels):
                for h, horizon in enumerate(self.horizons):
                    a = values[g, h].ravel()
                    missing = float(np.mean(np.isnan(a)))
                    q = (np.full(3, np.nan) if missing else
                         np.quantile(a, [self.alpha / 2, .5, 1 - self.alpha / 2],
                                     method="inverted_cdf"))
                    rows.append(dict(group=group, period=float(self.periods[h]), horizon=int(horizon),
                                     quantity=quantity, posterior_lower=q[0], posterior_median=q[1],
                                     posterior_upper=q[2], posterior_mean=(float(a.mean())
                                         if quantity != "wald" else np.nan),
                                     undefined_fraction=missing,
                                     infinite_fraction=float(np.mean(np.isinf(a)))))
        frame = pd.DataFrame(rows).merge(self.stability(min_ess=min_ess, max_rhat=max_rhat),
            on=["group", "period", "horizon", "quantity"], validate="one_to_one")
        frame["posterior_status"] = np.where(frame.undefined_fraction > 0, "undefined_draws",
            np.where(frame.diagnostics_passed, "diagnostics_passed_calibration_unestablished",
                     "diagnostics_failed"))
        support = self.reference[["group", "horizon", "n_units", "n_encouraged", "n_control"]]
        frame = frame.merge(support, on=["group", "horizon"], validate="many_to_one")
        frame["model_extrapolation"] = (frame.n_encouraged == 0) | (frame.n_control == 0)
        return frame

    @property
    def table(self) -> pd.DataFrame:
        return self.effects()

    def itt_covariance(self) -> pd.DataFrame:
        """Empirical joint posterior covariance across groups, horizons and ITTs.

        All entries come from the aligned joint draws. This is a posterior
        covariance under the fitted model; design-based sampling covariance is
        a different quantity. Ratio covariance is omitted because its moments
        need not exist even when its quantiles are well defined.
        """
        values = np.stack([self.draws["itt_y"], self.draws["itt_d"]], axis=2)
        values = values.reshape(-1, np.prod(values.shape[-2:]))
        index = pd.MultiIndex.from_product([self.group_labels, self.horizons, ["itt_y", "itt_d"]],
                                           names=["group", "horizon", "quantity"])
        cov = (np.cov(values, ddof=1) if values.shape[1] >= 2 else
               np.full((len(index), len(index)), np.nan))
        return pd.DataFrame(cov, index=index, columns=index)

    def wald(self, *, require_convergence: bool = True, min_ess: float = 400,
             max_rhat: float = 1.01) -> pd.DataFrame:
        """Posterior ratio medians/intervals, withheld on failed diagnostics by default.

        ``require_convergence=False`` permits inspection of unconverged empirical
        summaries. The raw draws are always available. Passing diagnostics does
        not turn these posterior intervals into calibrated weak-IV inference;
        the design-based confidence sets are in ``reference``.
        """
        all_effects = self.effects(min_ess=min_ess, max_rhat=max_rhat)
        passed = all_effects.groupby(["group", "horizon"]).diagnostics_passed.all()
        frame = all_effects.loc[all_effects.quantity == "wald"].copy().reset_index(drop=True)
        frame["joint_diagnostics_passed"] = [bool(passed.loc[(r.group, r.horizon)])
                                               for r in frame.itertuples()]
        if require_convergence:
            frame.loc[~frame.joint_diagnostics_passed,
                      ["posterior_lower", "posterior_median", "posterior_upper"]] = np.nan
        return frame

    def first_stage(self, *, practical_threshold: float | None = None,
                    min_ess: float = 400, max_rhat: float = 1.01) -> pd.DataFrame:
        """Precision/sign and optional practical relevance, without a strength cutoff.

        ``practical_threshold`` is an analyst-chosen positive adoption risk
        difference; its posterior exceedance probability is separate from
        first-stage precision and MCMC diagnostics.
        """
        if practical_threshold is not None and (
            not np.isscalar(practical_threshold) or not np.isfinite(practical_threshold)
            or practical_threshold <= 0
        ):
            raise ValueError("practical_threshold must be positive and finite.")
        frame = self.effects(min_ess=min_ess, max_rhat=max_rhat).query(
            "quantity == 'itt_d'").copy().reset_index(drop=True)
        a = self.draws["itt_d"]
        frame["posterior_probability_positive"] = (a > 0).mean(axis=(-2, -1)).ravel()
        frame["posterior_interval_includes_zero"] = (
            (frame.posterior_lower <= 0) & (frame.posterior_upper >= 0))
        frame["practical_threshold"] = practical_threshold
        frame["posterior_probability_above_threshold"] = (
            np.nan if practical_threshold is None else
            (a > practical_threshold).mean(axis=(-2, -1)).ravel())
        return frame

    def comparison(self) -> pd.DataFrame:
        """Descriptive model/reference differences on the matching finite-unit target.

        No disagreement flag is fabricated from independent standard errors of
        two estimates from the same data. ``bootstrap_comparison`` on the fitted
        wrapper obtains the required paired sampling covariance by refitting.
        Population-marginal predictions do not match the finite-unit reference.
        """
        rows = []
        aligned = self.metadata["target_aligned"]
        for g, group in enumerate(self.group_labels):
            ref = self.reference.loc[self.reference.group == group].reset_index(drop=True)
            for h, horizon in enumerate(self.horizons):
                for quantity in ("itt_y", "itt_d"):
                    model = float(self.draws[quantity][g, h].mean())
                    reference = float(ref.loc[h, quantity])
                    rows.append(dict(group=group, period=float(self.periods[h]), horizon=int(horizon),
                        quantity=quantity, model=model, reference=reference,
                        difference=model - reference if aligned else np.nan,
                        difference_se=np.nan, disagreement=pd.NA,
                        comparison_status="paired_sampling_covariance_required" if aligned
                                          else "different_standardization_targets"))
        return pd.DataFrame(rows)


@dataclasses.dataclass(frozen=True)
class ConditionalEffect:
    """Conditional treatment effect summary across units and post-treatment horizons.

    Attributes
    ----------
    draws : np.ndarray
        Array of shape (N, H, D) containing MCMC draws.
    mean : np.ndarray
        Posterior mean of shape (N, H).
    median : np.ndarray
        Posterior median of shape (N, H).
    lower : np.ndarray
        Posterior lower quantile (level alpha/2) of shape (N, H).
    upper : np.ndarray
        Posterior upper quantile (level 1 - alpha/2) of shape (N, H).
    sd : np.ndarray
        Posterior standard deviation of shape (N, H).
    """

    draws: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sd: np.ndarray


class ConditionalEncouragementPrediction:
    """Unit-level conditional ITTs, CACE, and risk-aware policy targeting.

    Attributes
    ----------
    citt_y : ConditionalEffect
        Conditional Intent-to-Treat on the outcome (N, H, D).
    citt_d : ConditionalEffect
        Conditional Intent-to-Treat on adoption / compliance (N, H, D).
    cace : ConditionalEffect
        Conditional Complier Average Causal Effect (N, H, D).
    periods : np.ndarray
        Calendar periods corresponding to post-treatment horizons (H,).
    horizons : np.ndarray
        Exposure duration indices (H,).
    alpha : float
        Significance level for credible intervals.
    """

    def __init__(
        self,
        citt_y_draws: np.ndarray,
        citt_d_draws: np.ndarray,
        periods: np.ndarray,
        horizons: np.ndarray,
        *,
        p0_draws: np.ndarray | None = None,
        alpha: float = 0.05,
        cace_stabilization: float = 0.02,
        monotonic_first_stage: bool = True,
    ):
        self.alpha = float(alpha)
        self.periods = np.asarray(periods)
        self.horizons = np.asarray(horizons)
        if not (np.isfinite(cace_stabilization) and cace_stabilization > 0):
            raise ValueError("cace_stabilization must be positive and finite.")
        self.cace_stabilization = float(cace_stabilization)
        self.monotonic_first_stage = bool(monotonic_first_stage)

        # 1. CITT_Y
        self.citt_y = self._build_effect(citt_y_draws, alpha)

        # 2. CITT_D with optional monotonicity enforcement
        if self.monotonic_first_stage:
            # Encouragement can never discourage adoption (eliminating defiers):
            citt_d_draws = np.maximum(0.0, citt_d_draws)
        self.citt_d = self._build_effect(citt_d_draws, alpha)

        # 2b. Principal Strata (Compliers, Always-Takers, Never-Takers) under Monotonicity
        self.p0_draws = np.asarray(p0_draws) if p0_draws is not None else None
        self.compliers = self.citt_d
        if self.p0_draws is not None:
            p_at = np.clip(self.p0_draws, 0.0, 1.0)
            p_c = np.clip(self.citt_d.draws, 0.0, 1.0)
            p_nt = np.clip(1.0 - (p_at + p_c), 0.0, 1.0)
            self.always_takers = self._build_effect(p_at, alpha)
            self.never_takers = self._build_effect(p_nt, alpha)
        else:
            self.always_takers = None
            self.never_takers = None

        # 3. CACE: the ratio of the two offer effects, draw by draw, with a
        # small floor on the adoption contrast so a near-zero first stage
        # cannot produce arbitrarily large ratios.
        cace_draws_eff = citt_y_draws / (self.citt_d.draws + self.cace_stabilization)
        self.cace = self._build_effect(cace_draws_eff, alpha)

    @staticmethod
    def _build_effect(draws: np.ndarray, alpha: float) -> ConditionalEffect:
        mean = np.mean(draws, axis=-1)
        median = np.median(draws, axis=-1)
        lower = np.quantile(draws, alpha / 2, axis=-1)
        upper = np.quantile(draws, 1.0 - alpha / 2, axis=-1)
        sd = np.std(draws, axis=-1, ddof=1) if draws.shape[-1] > 1 else np.zeros_like(mean)
        return ConditionalEffect(
            draws=draws, mean=mean, median=median, lower=lower, upper=upper, sd=sd
        )

    def principal_strata(
        self,
        *,
        horizon: int | None = None,
        as_counts: bool = True,
    ) -> pd.DataFrame:
        """Posterior distribution of principal strata proportions and cohort counts.

        Under the monotonicity assumption (no defiers: D(1) >= D(0)), each unit at each
        post-treatment horizon belongs to one of three latent principal strata:
        - Compliers: adopt if and only if encouraged (D(1) = 1, D(0) = 0)
        - Always-Takers: adopt even in the absence of encouragement (D(1) = 1, D(0) = 1)
        - Never-Takers: refuse adoption even when encouraged (D(1) = 0, D(0) = 0)

        Parameters
        ----------
        horizon : int, optional
            Specific post-treatment exposure index (0-based). If None, evaluates all horizons.
        as_counts : bool, default True
            If True, reports both population proportions and expected cohort counts (N * prob).

        Returns
        -------
        pd.DataFrame
            DataFrame summarizing posterior mean, median, lower/upper credible bounds,
            and expected counts for each principal stratum.
        """
        if self.always_takers is None or self.never_takers is None:
            strata_dict = {"complier": self.compliers.draws}
        else:
            strata_dict = {
                "complier": self.compliers.draws,
                "always_taker": self.always_takers.draws,
                "never_taker": self.never_takers.draws,
            }

        n_units = self.citt_d.draws.shape[0]
        h_indices = range(len(self.horizons)) if horizon is None else [horizon]

        rows = []
        for h in h_indices:
            h_idx = int(h)
            p_val = float(self.periods[h_idx])
            h_val = int(self.horizons[h_idx])

            for name, draws in strata_dict.items():
                unit_draws = draws[:, h_idx, :]  # (N, D)
                pop_draws = np.mean(unit_draws, axis=0)  # (D,)
                count_draws = np.sum(unit_draws, axis=0)  # (D,)

                row = {
                    "period": p_val,
                    "horizon": h_val,
                    "stratum": name,
                    "prob_mean": float(np.mean(pop_draws)),
                    "prob_median": float(np.median(pop_draws)),
                    "prob_lower": float(np.quantile(pop_draws, self.alpha / 2)),
                    "prob_upper": float(np.quantile(pop_draws, 1.0 - self.alpha / 2)),
                    "prob_sd": float(np.std(pop_draws, ddof=1)) if len(pop_draws) > 1 else 0.0,
                }
                if as_counts:
                    row.update({
                        "count_mean": float(np.mean(count_draws)),
                        "count_median": float(np.median(count_draws)),
                        "count_lower": float(np.quantile(count_draws, self.alpha / 2)),
                        "count_upper": float(np.quantile(count_draws, 1.0 - self.alpha / 2)),
                        "n_total": n_units,
                    })
                rows.append(row)

        return pd.DataFrame(rows)

    def strata_summary(self, horizon: int = 0) -> str:
        """Human-readable statement of principal strata estimates matching BIVA conventions."""
        df = self.principal_strata(horizon=horizon, as_counts=True)
        h_idx = int(horizon)
        p_val = self.periods[h_idx]
        h_val = self.horizons[h_idx]
        n_units = self.citt_d.draws.shape[0]

        lines = [
            f"Principal Strata Estimates at Horizon {h_val} (Period {p_val:.2f}, N = {n_units}):"
        ]
        for _, r in df.iterrows():
            st = r["stratum"].replace("_", "-").title()
            pct_mean = f"{r['prob_mean'] * 100:.1f}%"
            pct_ci = f"[{r['prob_lower'] * 100:.1f}%, {r['prob_upper'] * 100:.1f}%]"
            cnt_mean = f"{r['count_mean']:.1f}"
            cnt_ci = f"[{r['count_lower']:.1f}, {r['count_upper']:.1f}]"
            lines.append(f"  • {st:<13}: {pct_mean:>6} (95% CI: {pct_ci})  —  ~{cnt_mean} units (95% CI: {cnt_ci})")

        c_rows = df[df["stratum"] == "complier"]
        if not c_rows.empty:
            c_mean = c_rows.iloc[0]["prob_mean"]
            if c_mean < 0.10:
                lines.append("  [WARNING: Weak Instrument detected - complier share < 10%]")
            else:
                lines.append("  [Robust Instrument - complier share exceeds 10% threshold]")

        return "\n".join(lines)

    def cumulative_lift(self, start_h: int = 0, end_h: int | None = None) -> np.ndarray:
        """Draws of cumulative outcome lift: sum_{t} CITT_Y(X_i, t). Shape: (N, D)."""
        sl = slice(start_h, end_h)
        return np.sum(self.citt_y.draws[:, sl, :], axis=1)

    def cumulative_lift_summary(self, start_h: int = 0, end_h: int | None = None) -> pd.DataFrame:
        """Summary table of cumulative outcome lift across units."""
        cum = self.cumulative_lift(start_h, end_h)
        return pd.DataFrame({
            "unit": np.arange(cum.shape[0]),
            "cum_mean": np.mean(cum, axis=-1),
            "cum_median": np.median(cum, axis=-1),
            "cum_lower": np.quantile(cum, self.alpha / 2, axis=-1),
            "cum_upper": np.quantile(cum, 1.0 - self.alpha / 2, axis=-1),
            "cum_sd": np.std(cum, axis=-1, ddof=1) if cum.shape[-1] > 1 else 0.0,
        })

    def breakeven_probability(
        self, cost: float, start_h: int = 0, end_h: int | None = None
    ) -> np.ndarray:
        """Posterior probability that cumulative lift exceeds cost: Pr(sum CITT_Y > cost | Data)."""
        cum = self.cumulative_lift(start_h, end_h)
        return np.mean(cum > float(cost), axis=-1)

    def optimal_policy(
        self, cost: float, hurdle: float = 0.50, start_h: int = 0, end_h: int | None = None
    ) -> np.ndarray:
        """Risk-aware encouragement rule: 1{Pr(sum CITT_Y > cost | Data) >= hurdle}."""
        p = self.breakeven_probability(cost, start_h, end_h)
        return p >= float(hurdle)

    def knapsack_policy(
        self,
        cost: float,
        *,
        budget: float | None = None,
        capacity: int | None = None,
        hurdle: float = 0.50,
        ranking_metric: str = "expected_net_value",
        start_h: int = 0,
        end_h: int | None = None,
    ) -> np.ndarray:
        """Compute optimal targeted policy under budget and/or unit capacity constraints.

        Parameters
        ----------
        cost : float
            Cost per encouraged account.
        budget : float, optional
            Maximum allowable total expenditure (cost * sum(decision) <= budget).
        capacity : int, optional
            Maximum allowable count of encouraged accounts (sum(decision) <= capacity).
        hurdle : float
            Minimum posterior breakeven probability required for eligibility.
        ranking_metric : str
            Priority criterion when constraints bind:
            - 'expected_net_value': E[Lift_i - cost]
            - 'certainty_adjusted': E[Lift_i - cost] / (SD(Lift_i) + 1e-6)
            - 'breakeven_probability': Pr(Lift_i > cost)
        start_h, end_h : int, optional
            Horizon slice for cumulative lift.

        Returns
        -------
        np.ndarray of shape (N,)
            Boolean indicator of encouraged units satisfying all constraints.
        """
        cost = float(cost)
        if cost <= 0:
            raise ValueError("cost must be positive.")
        if ranking_metric not in ("expected_net_value", "certainty_adjusted", "breakeven_probability"):
            raise ValueError(
                f"Unknown ranking_metric '{ranking_metric}'. Available: "
                "'expected_net_value', 'certainty_adjusted', 'breakeven_probability'."
            )
        n_units = self.citt_y.draws.shape[0]

        # 1. Eligibility check
        p_breakeven = self.breakeven_probability(cost, start_h, end_h)
        cum_draws = self.cumulative_lift(start_h, end_h)
        expected_lift = np.mean(cum_draws, axis=-1)
        expected_net = expected_lift - cost
        sd_lift = np.std(cum_draws, axis=-1, ddof=1) if cum_draws.shape[-1] > 1 else np.ones(n_units)

        eligible = (p_breakeven >= float(hurdle)) & (expected_net > 0)
        eligible_indices = np.flatnonzero(eligible)

        # 2. Determine max allowed units under constraints
        max_units = n_units
        if capacity is not None:
            max_units = min(max_units, int(capacity))
        if budget is not None:
            max_units_by_budget = int(np.floor(float(budget) / cost))
            max_units = min(max_units, max_units_by_budget)

        if max_units <= 0:
            return np.zeros(n_units, dtype=bool)

        if len(eligible_indices) <= max_units:
            decision = np.zeros(n_units, dtype=bool)
            decision[eligible_indices] = True
            return decision

        # 3. Ranking metric calculation
        if ranking_metric == "expected_net_value":
            scores = expected_net
        elif ranking_metric == "certainty_adjusted":
            scores = expected_net / (sd_lift + 1e-6)
        elif ranking_metric == "breakeven_probability":
            scores = p_breakeven
        else:
            raise ValueError(
                f"Unknown ranking_metric '{ranking_metric}'. Available: "
                "'expected_net_value', 'certainty_adjusted', 'breakeven_probability'."
            )

        # 4. Rank eligible units and select top max_units
        eligible_scores = scores[eligible_indices]
        top_order = np.argsort(-eligible_scores)[:max_units]
        selected_indices = eligible_indices[top_order]

        decision = np.zeros(n_units, dtype=bool)
        decision[selected_indices] = True
        return decision

    def policy_value(
        self,
        cost: float,
        hurdle: float = 0.50,
        true_cumulative_lift: np.ndarray | None = None,
        budget: float | None = None,
        capacity: int | None = None,
        ranking_metric: str = "expected_net_value",
        start_h: int = 0,
        end_h: int | None = None,
    ) -> float:
        """Net policy value per account: mean_i 1{g(X_i)=1} * (Lift_i - cost)."""
        if budget is not None or capacity is not None:
            dec = self.knapsack_policy(
                cost,
                budget=budget,
                capacity=capacity,
                hurdle=hurdle,
                ranking_metric=ranking_metric,
                start_h=start_h,
                end_h=end_h,
            )
        else:
            dec = self.optimal_policy(cost, hurdle, start_h, end_h)
        lift = (
            np.mean(self.cumulative_lift(start_h, end_h), axis=-1)
            if true_cumulative_lift is None
            else np.asarray(true_cumulative_lift)
        )
        return float(np.mean(np.where(dec, lift - float(cost), 0.0)))


class LongBetEncourage:
    """The adoption-clock encouragement model; see the module docstring.

    ``config`` (or keyword overrides) applies to both equations; the adoption
    equation is always fitted as a binary hazard. ``outcome`` is
    ``'continuous'`` or ``'binary'``. Fit a complete single-wave individually
    randomized panel with absorbing offer and adoption indicators.
    """

    def __init__(self, config: LongBetConfig | None = None, *, outcome: str = "continuous",
                 **config_kwargs: Any):
        if outcome not in ("continuous", "binary"):
            raise ValueError("outcome must be 'continuous' or 'binary'.")
        if config is None:
            self.config = LongBetConfig(**config_kwargs)
        elif isinstance(config, LongBetConfig):
            self.config = dataclasses.replace(config, **config_kwargs)
        else:
            raise TypeError("config must be a LongBetConfig or None.")
        if self.config.outcome != "continuous" or self.config.num_categories is not None:
            raise ValueError("pass the outcome type as outcome=..., not in the config.")
        self.outcome = outcome
        self.outcome_model: LongBet | None = None
        self.adoption_model: LongBet | None = None
        self._data: dict[str, np.ndarray] = {}
        self.metadata: dict[str, Any] = {}

    # -- inputs ---------------------------------------------------------------
    def _inputs(self, y: Any, d: Any, z: Any, x: Any, t: Any, x_trt: Any) -> dict:
        panel = _validate(z, d, t)
        y = _finite_matrix(y, "y", len(panel.z), panel.z.shape)
        if self.outcome == "binary" and not np.isin(y, [0, 1]).all():
            raise ValueError("binary outcome y must contain only 0 and 1.")
        data = dict(y=y, d=panel.d.copy(), z=panel.z.copy(), t=panel.t.copy(),
                    x=_finite_matrix(x, "x", len(panel.z)))
        if x_trt is not None:
            data["x_trt"] = _finite_matrix(x_trt, "x_trt", len(panel.z))
        model_t = data["t"].astype(np.float32)
        for name in ("z", "d"):
            if not np.all(np.diff(model_t) > 0) or not np.array_equal(
                    derive_exposure(data[name], model_t), derive_exposure(data[name], data["t"])):
                raise ValueError("t loses clock precision in the float32 model; shift the "
                                 "calendar origin before fitting.")
        return data

    @staticmethod
    def _first_adoption_events(d: np.ndarray) -> np.ndarray:
        """First-adoption indicator on the at-risk set; NaN once a unit has adopted."""
        previous = np.concatenate([np.zeros((len(d), 1)), d[:, :-1]], axis=1)
        return np.where(previous == 1, np.nan, d)

    def _equation_configs(self) -> tuple[LongBetConfig, LongBetConfig]:
        outcome_cfg = dataclasses.replace(self.config, outcome=self.outcome, num_categories=None)
        adoption_cfg = dataclasses.replace(self.config, outcome="binary", num_categories=None)
        return outcome_cfg, adoption_cfg

    # -- fitting --------------------------------------------------------------
    def fit(self, y: Any, d: Any, z: Any, x: Any, t: Any = None,
            x_trt: Any = None, key: Any = None) -> LongBetEncourage:
        """Fit the outcome equation on adoption and the adoption hazard on the offer."""
        data = self._inputs(y, d, z, x, t, x_trt)
        outcome_cfg, adoption_cfg = self._equation_configs()
        if key is not None:
            import jax
            key_y, key_d = jax.random.split(key)
        else:
            key_y = key_d = None
        with warnings.catch_warnings():
            # Units adopting in the first period are 'always treated' for the
            # outcome equation; their exposure clock is right (adoption at
            # period 1), so the scalar model's advice to drop them does not apply.
            warnings.filterwarnings("ignore", message=r"\d+ unit\(s\) are treated in every period,.*",
                                    category=UserWarning)
            self.outcome_model = LongBet(outcome_cfg).fit(
                data["y"], data["x"], data["d"], t=data["t"], x_trt=data.get("x_trt"), key=key_y)
            self.adoption_model = LongBet(adoption_cfg).fit(
                self._first_adoption_events(data["d"]), data["x"], data["z"], t=data["t"],
                x_trt=data.get("x_trt"), key=key_d)
        self._data = data
        for value in self._data.values():
            value.flags.writeable = False
        self.metadata = dict(_validate(z, d, t).metadata(),
                             target="all_original_units_equal_weight", engine=ENGINE,
                             outcome=self.outcome, inference_version=INFERENCE_VERSION,
                             unit_intercept_covariance="independent_prior",
                             calibration_status="not_established")
        return self

    def _require_fit(self) -> None:
        if self.outcome_model is None or self.adoption_model is None:
            raise RuntimeError("LongBetEncourage must be fitted before prediction.")

    # -- population effects ---------------------------------------------------
    def predict(self, *, summary_only: bool = True, groups: Any = None, weights: Any = None,
                block_size: int | None = None, alpha: float = .05) -> EncouragementPrediction:
        """Offer effects standardized over the study units, with the design-based reference.

        Every study unit is evaluated under both schedules (offered from the
        common start onward, never offered) with its own fitted intercepts, so
        the target is the finite study population the reference describes.
        ``groups`` are prespecified baseline labels; ``weights`` are fixed
        nonnegative unit weights normalized within each group.
        """
        self._require_fit()
        _critical(alpha)
        del summary_only  # the standardized draws are small; kept for API symmetry
        data = self._data
        panel = _validate(data["z"], data["d"], data["t"])
        n_units, n_periods = data["z"].shape
        labels, counts, target = _target_weights(n_units, groups, weights)
        post_idx = np.arange(panel.start, n_periods)
        num_chains = self.config.num_chains
        n_draws = self.config.num_sweeps * num_chains
        offered = np.zeros((n_units, n_periods), dtype=np.float32)
        offered[:, panel.start:] = 1.0
        never = np.zeros_like(offered)
        if block_size is None:
            block_size = max(1, int(2.5e8 // (8 * n_periods * n_draws)))
        elif not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError("block_size must be a positive integer.")
        sums = {name: np.zeros((len(labels), len(post_idx), n_draws)) for name in ("outcome", "takeup")}
        t_vec = data["t"]
        gamma_d = (_flat_draws(self.adoption_model, "gamma") if self.config.random_intercept
                   else np.zeros((n_draws, n_units)))
        gamma_y = (_flat_draws(self.outcome_model, "gamma")
                   if self.config.random_intercept and self.outcome == "binary"
                   else np.zeros((n_draws, n_units)))
        x_trt_all = data.get("x_trt")
        for lo in range(0, n_units, block_size):
            sl = slice(lo, min(lo + block_size, n_units))
            x_blk = data["x"][sl]
            x_trt_blk = None if x_trt_all is None else x_trt_all[sl]
            p1, stock1 = adoption_distribution(self.adoption_model, x_blk, offered[sl], t_vec,
                                               x_trt=x_trt_blk, fitted_intercepts=gamma_d[:, sl])
            p0, stock0 = adoption_distribution(self.adoption_model, x_blk, never[sl], t_vec,
                                               x_trt=x_trt_blk, fitted_intercepts=gamma_d[:, sl])
            citt_y = offer_effect_on_outcome(self.outcome_model, x_blk, t_vec, p1, p0,
                                             x_trt=x_trt_blk, fitted_intercepts=gamma_y[:, sl])
            citt_d = stock1 - stock0
            for name, values in (("outcome", citt_y), ("takeup", citt_d)):
                sums[name] += np.einsum("gn,nhd->ghd", target[:, sl], values[:, post_idx, :])
        draws = {name: sums[name].reshape(len(labels), len(post_idx), num_chains, -1)
                 for name in sums}
        standardized = EncouragementDraws(
            draws=draws, group_labels=labels, group_counts=counts, weights=target,
            periods=panel.t[post_idx].copy(), period_indices=post_idx,
            horizons=panel.exposure[post_idx].copy(), standardization="conditional",
            provenance=INFERENCE_VERSION)
        reference = _reference_groups(data, standardized, alpha)
        return EncouragementPrediction(standardized, reference, self.metadata, alpha=alpha)

    # -- unit-level effects ---------------------------------------------------
    def predict_conditional(
        self, x: Any = None, z: Any = None, t: Any = None, *, x_trt: Any = None,
        alpha: float = 0.05, cace_stabilization: float = 0.02, monotonic_first_stage: bool = True,
    ) -> ConditionalEncouragementPrediction:
        """Unit-level offer effects, principal strata, lift and policy quantities.

        Describes a unit with covariate profile ``x`` drawn from the fitted
        population: both unit intercepts are integrated out, so the same rule
        applies to study units and to a new cohort. ``x`` defaults to the study
        units, ``z`` to an offer from the study's launch period onward, ``t``
        to the fitted calendar. The comparison schedule is never offering.
        Adoption contrasts are clipped at zero when ``monotonic_first_stage``
        (no defiers); the CACE is the draw-wise ratio of the two offer effects
        with ``cace_stabilization`` added to the adoption contrast.
        """
        self._require_fit()
        _critical(alpha)
        x_np = self._data["x"] if x is None else np.asarray(x, dtype=np.float32)
        if x_np.ndim != 2 or x_np.shape[1] != self._data["x"].shape[1]:
            raise ValueError(f"x must be a 2-D array with {self._data['x'].shape[1]} columns.")
        fitted_x_trt = self._data.get("x_trt")
        if x is None:
            x_trt_np = fitted_x_trt
        elif fitted_x_trt is None:
            if x_trt is not None:
                raise ValueError("the model was fitted without x_trt.")
            x_trt_np = None
        else:
            if x_trt is None:
                raise ValueError("the model was fitted with x_trt, so new units need x_trt too.")
            x_trt_np = _finite_matrix(x_trt, "x_trt", len(x_np), (len(x_np), fitted_x_trt.shape[1]))
        t_vec = self._data["t"] if t is None else np.asarray(t, dtype=np.float64)
        n_units, n_periods = len(x_np), len(t_vec)
        panel = _validate(self._data["z"], self._data["d"], self._data["t"])
        start = panel.start
        if z is None:
            z_mat = np.zeros((n_units, n_periods), dtype=np.float32)
            z_mat[:, start:] = 1.0
        else:
            z_mat = np.asarray(z, dtype=np.float32)
            if z_mat.shape != (n_units, n_periods):
                raise ValueError(f"z must have shape ({n_units}, {n_periods}).")
        never = np.zeros_like(z_mat)
        p1, stock1 = adoption_distribution(self.adoption_model, x_np, z_mat, t_vec, x_trt=x_trt_np)
        p0, stock0 = adoption_distribution(self.adoption_model, x_np, never, t_vec, x_trt=x_trt_np)
        citt_y = offer_effect_on_outcome(self.outcome_model, x_np, t_vec, p1, p0, x_trt=x_trt_np)
        post_idx = np.arange(start, n_periods)
        return ConditionalEncouragementPrediction(
            citt_y_draws=citt_y[:, post_idx, :],
            citt_d_draws=(stock1 - stock0)[:, post_idx, :],
            periods=np.asarray(t_vec)[post_idx].copy(),
            horizons=np.asarray(t_vec)[post_idx] - np.asarray(t_vec)[start] + 1,
            p0_draws=stock0[:, post_idx, :],
            alpha=alpha, cace_stabilization=cace_stabilization,
            monotonic_first_stage=monotonic_first_stage,
        )

    # -- paired bootstrap comparison -----------------------------------------
    def bootstrap_comparison(self, *, replicates: int = 200, groups: Any = None,
                             random_seed: int = 0, alpha: float = .05,
                             min_ess: float = 400, max_rhat: float = 1.01,
                             callback: Any = None) -> EncouragementComparison:
        """Refit paired whole-unit bootstrap samples to compare model and reference ITTs.

        Resample entire unit records within assigned arm and prespecified
        group, holding their counts fixed; refit both equations with new
        sampler seeds; recompute both estimators on the same resampled
        population, so the comparison variance includes their sampling
        covariance. Only the ITTs are compared. This requires ``replicates``
        complete fits. A disagreement flag is withheld if the original fit or
        any replicate fails diagnostics, if reference inference is unavailable,
        or if fewer than 100 replications are requested.
        """
        critical = _critical(alpha)
        if isinstance(replicates, (bool, np.bool_)) or not isinstance(replicates, (int, np.integer)) or replicates < 2:
            raise ValueError("replicates must be an integer of at least 2.")
        if callback is not None and not callable(callback):
            raise ValueError("callback must be callable or None.")
        original = self.predict(groups=groups, alpha=alpha)
        original_checks = original.stability(min_ess=min_ess, max_rhat=max_rhat)
        original_ok = original_checks.query("quantity != 'wald'").diagnostics_passed.all()
        reference_columns = ["itt_y_se", "itt_d_se", "itt_y_d_cov"]
        original_reference_ok = bool(np.isfinite(original.reference[reference_columns]).all().all())
        n = len(self._data["z"])
        group_values = np.zeros(n) if groups is None else np.asarray(groups)
        strata = []
        for label in pd.unique(group_values):
            for arm in (0, 1):
                idx = np.flatnonzero((group_values == label) & (self._data["z"][:, -1] == arm))
                if idx.size:
                    strata.append(idx)
        rng = np.random.default_rng(random_seed)
        paired = np.full((replicates, len(original.group_labels), len(original.horizons), 2, 2), np.nan)
        failures = []
        for b in range(replicates):
            indices = np.concatenate([rng.choice(idx, len(idx), replace=True) for idx in strata])
            seed = int(rng.integers(0, 2**31))
            try:
                fit = LongBetEncourage(self.config, outcome=self.outcome, random_seed=seed)
                fit.fit(self._data["y"][indices], self._data["d"][indices], self._data["z"][indices],
                        self._data["x"][indices], self._data["t"],
                        x_trt=None if "x_trt" not in self._data else self._data["x_trt"][indices])
                pred = fit.predict(groups=None if groups is None else group_values[indices], alpha=alpha)
                for g, label in enumerate(original.group_labels):
                    pg = pred.group_labels.index(label)
                    ref = pred.reference.loc[pred.reference.group == label]
                    for q, quantity in enumerate(("itt_y", "itt_d")):
                        paired[b, g, :, q, 0] = pred.draws[quantity][pg].mean(axis=(-2, -1))
                        paired[b, g, :, q, 1] = ref[quantity].to_numpy()
                checks = pred.stability(min_ess=min_ess, max_rhat=max_rhat)
                if not checks.query("quantity != 'wald'").diagnostics_passed.all():
                    failures.append(dict(replicate=b, sampler_seed=seed, reason="diagnostics_failed"))
                if not np.isfinite(paired[b]).all():
                    failures.append(dict(replicate=b, sampler_seed=seed, reason="unavailable_estimate"))
                if not np.isfinite(pred.reference[reference_columns]).all().all():
                    failures.append(dict(replicate=b, sampler_seed=seed,
                                         reason="unavailable_reference_inference"))
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                failures.append(dict(replicate=b, sampler_seed=seed,
                                     reason=f"{type(exc).__name__}: {exc}"))
            if callback is not None:
                callback(b + 1, replicates)
        frame = original.comparison()
        frame["model_sampling_variance"] = np.nan
        frame["reference_sampling_variance"] = np.nan
        frame["model_reference_covariance"] = np.nan
        complete = not failures and bool(original_ok) and original_reference_ok and replicates >= 100
        for r, row in frame.iterrows():
            g = original.group_labels.index(row.group)
            h = int(np.flatnonzero(original.horizons == row.horizon)[0])
            q = ("itt_y", "itt_d").index(row.quantity)
            pairs = paired[:, g, h, q]
            if np.isfinite(pairs).all():
                cov = np.cov(pairs, rowvar=False, ddof=1)
                se = float(np.std(pairs[:, 0] - pairs[:, 1], ddof=1))
                frame.loc[r, ["model_sampling_variance", "reference_sampling_variance",
                              "model_reference_covariance", "difference_se"]] = (
                                  cov[0, 0], cov[1, 1], cov[0, 1], se)
                if complete and np.isfinite(row.difference) and np.isfinite(se) and se > 0:
                    frame.loc[r, "disagreement"] = bool(abs(row.difference) > critical * se)
                    frame.loc[r, "comparison_status"] = "paired_unit_bootstrap_normal_pointwise"
                else:
                    frame.loc[r, "comparison_status"] = "bootstrap_checks_incomplete"
            else:
                frame.loc[r, "comparison_status"] = "bootstrap_failed_estimates"
        return EncouragementComparison(frame, paired,
            pd.DataFrame(failures, columns=["replicate", "sampler_seed", "reason"]),
            dict(method="paired_unit_bootstrap", replicates=int(replicates), random_seed=random_seed,
                 alpha=alpha, original_diagnostics_passed=bool(original_ok),
                 original_reference_inference_available=original_reference_ok,
                 resampling_unit="entire_unit_record", strata="assigned_arm_and_baseline_group",
                 target="same_resampled_units_for_both_estimators", exact_randomization=False,
                 calibration_status="not_established"))

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save a versioned, pickle-free archive with both equations and the study data."""
        self._require_fit()
        arrays = dict(self._data)
        with tempfile.TemporaryDirectory() as tmp:
            for name, model in (("outcome_model", self.outcome_model),
                                ("adoption_model", self.adoption_model)):
                nested = Path(tmp) / f"{name}.npz"
                model.save(nested)
                arrays[name] = np.frombuffer(nested.read_bytes(), dtype=np.uint8)
        meta = dict(self.metadata, kind="LongBetEncourage", archive_version=ARCHIVE_VERSION,
                    config=self.config.to_dict(),
                    input_hashes={name: _digest(value) for name, value in arrays.items()})
        arrays["metadata"] = np.asarray(json.dumps(meta, allow_nan=False))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> LongBetEncourage:
        """Validate and replay an encouragement archive."""
        try:
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"]))
                arrays = {name: archive[name] for name in archive.files if name != "metadata"}
            if (not isinstance(meta, dict) or meta.get("kind") != "LongBetEncourage" or
                    meta.get("archive_version") != ARCHIVE_VERSION or
                    meta.get("inference_version") != INFERENCE_VERSION or
                    meta.get("engine") != ENGINE):
                raise ValueError("unsupported kind or version")
            required = {"y", "d", "z", "x", "t", "outcome_model", "adoption_model"}
            if not required <= arrays.keys() or arrays.keys() - required - {"x_trt"}:
                raise ValueError("unexpected or missing archive arrays")
            if meta["input_hashes"] != {name: _digest(value) for name, value in arrays.items()}:
                raise ValueError("input or model digest mismatch")
            config = LongBetConfig.from_dict(meta["config"])
            obj = cls(config, outcome=meta["outcome"])
            models = {}
            with tempfile.TemporaryDirectory() as tmp:
                for name in ("outcome_model", "adoption_model"):
                    blob = arrays.pop(name)
                    if blob.dtype != np.uint8 or blob.ndim != 1:
                        raise ValueError("invalid nested model bytes")
                    nested = Path(tmp) / f"{name}.npz"
                    nested.write_bytes(blob.tobytes())
                    models[name] = LongBet.load(nested)
            obj._data = obj._inputs(**{**arrays, "x_trt": arrays.get("x_trt")})
            panel = _validate(arrays["z"], arrays["d"], arrays["t"])
            outcome_cfg, adoption_cfg = obj._equation_configs()
            for name, expected in (("outcome_model", outcome_cfg), ("adoption_model", adoption_cfg)):
                model = models[name]
                if (model.config != expected or model.N_ != len(arrays["y"])
                        or model.T_ != arrays["y"].shape[1]
                        or not np.array_equal(model.t_fit_, arrays["t"].astype(np.float32))):
                    raise ValueError("model/design provenance mismatch")
            if (any(meta.get(k) != v for k, v in panel.metadata().items())
                    or meta.get("target") != "all_original_units_equal_weight"
                    or meta.get("calibration_status") != "not_established"):
                raise ValueError("model/design provenance mismatch")
            for value in obj._data.values():
                value.flags.writeable = False
            obj.outcome_model, obj.adoption_model = models["outcome_model"], models["adoption_model"]
            obj.metadata = {k: v for k, v in meta.items()
                            if k not in ("kind", "archive_version", "input_hashes", "config")}
            return obj
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"Invalid encouragement archive: {exc}") from exc


def _digest(a: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(str(a.shape).encode())
    digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()
