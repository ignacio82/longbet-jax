"""Joint reduced forms for single-wave randomized encouragement experiments.

The posterior belongs to a working outcome model. Design-based confidence sets
remain separate, and neither sampler diagnostics nor a proper prior establish
calibration of the posterior Wald ratio.
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

from longbet._config import LongBetConfig
from longbet._diagnostics import compute_ess, compute_rhat
from longbet._encourage import _critical, _validate, encouragement_effects
from longbet._encourage_predict import EncouragementDraws, standardize_encouragement
from longbet._multi_model import LongBetMulti
from longbet._model import derive_exposure


ARCHIVE_VERSION = 1
INFERENCE_VERSION = "encouragement_reduced_form_v1"


def _innovation_coupling(model: LongBetMulti) -> str:
    # Binary equations have zero incoming loading rows in the current sampler.
    # A SUR config flag cannot create innovation dependence between two such rows.
    if all(kind == "binary" for kind in model.outcome):
        return "independent_binary"
    return "SUR" if model.sur_active else "independent"


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
        cace_draws: np.ndarray | None = None,
        alpha: float = 0.05,
        cace_stabilization: float = 0.02,
        monotonic_first_stage: bool = True,
        cace_shrinkage: str = "adaptive",
        shrinkage_lambda: float = 1.0,
    ):
        self.alpha = float(alpha)
        self.periods = np.asarray(periods)
        self.horizons = np.asarray(horizons)
        self.cace_stabilization = float(cace_stabilization)
        self.monotonic_first_stage = bool(monotonic_first_stage)
        self.cace_shrinkage = str(cace_shrinkage)
        self.shrinkage_lambda = float(shrinkage_lambda)

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

        # 3. CACE (direct draws or stabilized ratio)
        if cace_draws is not None:
            cace_draws_eff = np.asarray(cace_draws)
        elif self.cace_shrinkage == "adaptive":
            # Population-level Wald ratio per horizon and draw:
            # sum over units of CITT_Y / sum over units of CITT_D
            y_sum_hd = np.sum(citt_y_draws, axis=0)  # (H, D)
            d_sum_hd = np.sum(citt_d_draws, axis=0)  # (H, D)
            pop_cace_hd = y_sum_hd / (d_sum_hd + 1e-6)  # (H, D)
            pop_cace_nhd = np.broadcast_to(pop_cace_hd[None, :, :], citt_y_draws.shape)

            # Local signal-to-noise ratio per unit and horizon:
            d_mean_nh = np.mean(citt_d_draws, axis=-1, keepdims=True)  # (N, H, 1)
            d_sd_nh = (
                np.std(citt_d_draws, axis=-1, ddof=1, keepdims=True)
                if citt_d_draws.shape[-1] > 1
                else np.ones_like(d_mean_nh)
            )
            snr_nh = (d_mean_nh / (d_sd_nh + 1e-6)) ** 2
            weight_nh = snr_nh / (snr_nh + self.shrinkage_lambda)  # (N, H, 1)

            # Local raw ratio regularized by small epsilon
            eps = self.cace_stabilization
            local_cace = citt_y_draws / (citt_d_draws + eps)
            cace_draws_eff = weight_nh * local_cace + (1.0 - weight_nh) * pop_cace_nhd
        elif self.cace_shrinkage == "ridge":
            eps2 = self.cace_stabilization ** 2
            cace_draws_eff = (citt_y_draws * citt_d_draws) / (citt_d_draws ** 2 + eps2)
        elif self.cace_shrinkage == "none":
            cace_draws_eff = citt_y_draws / (citt_d_draws + 1e-8)
        else:
            raise ValueError(
                f"Unknown cace_shrinkage '{self.cace_shrinkage}'. Available: 'adaptive', 'ridge', 'none'."
            )

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
    """Bayesian joint encouragement reduced forms with a design-based reference.

    Choose ``first_stage='lpm'`` or ``'probit'`` explicitly: neither is promoted
    as a calibrated default. LPM treats binary adoption as a continuous working
    response and can predict outside [0,1]. Probit contrasts are standardized
    probability differences. ``outcome`` selects a continuous or binary Y.

    With no supplied config, proper innovation priors are IG(shape=2, scale=1)
    on standardized continuous outcomes (mean 1, infinite prior variance).
    Supplied configs must also use positive proper innovation priors; they are
    never silently rewritten. Unit-intercept variances use the config's proper
    IG prior. SUR couples within-period innovations, while unit-intercept priors
    remain independent across equations. This distinction matters for covariance
    calibration under repeated-unit dependence.
    With binary Y and probit D, innovation loadings are fixed to zero in both
    equations. Without opt-in shared treatment partitions, these two fitted
    posteriors are independent even when ``config.sur=True``.

    Fit a complete single-wave individual randomized panel. ``x`` and ``x_trt``
    are baseline covariates. Prediction reuses the original units, ordering,
    design and common equal weights; no realized-uptake adjustment, forecasting,
    generalized-design model fit, or new-unit extrapolation is performed.
    """

    def __init__(self, config: LongBetConfig | None = None, *, first_stage: str,
                 outcome: str = "continuous", **config_kwargs: Any):
        if first_stage not in ("lpm", "probit", "hazard"):
            raise ValueError("first_stage must be explicitly chosen as 'lpm', 'probit', or 'hazard'.")
        if outcome not in ("continuous", "binary"):
            raise ValueError("outcome must be 'continuous' or 'binary'.")
        self.engine = config_kwargs.pop("engine", "longbet")
        self.direct_config = config_kwargs.pop("direct_config", None)
        self._direct_model = None
        self._orthogonal_model = None
        self._hazard_forest = None
        if self.engine not in ("longbet", "direct_smooth", "orthogonal_iv"):
            raise ValueError("engine must be 'longbet', 'direct_smooth', or 'orthogonal_iv'.")
        if self.engine == "orthogonal_iv" and outcome != "continuous":
            raise ValueError("orthogonal_iv requires outcome='continuous'.")
        if self.engine == "direct_smooth":
            from longbet._direct_smooth import DirectSmoothConfig
            if first_stage != "lpm" or outcome != "continuous":
                raise ValueError("direct_smooth requires first_stage='lpm' and outcome='continuous'.")
            supported = {"random_seed", "num_chains", "num_burnin", "num_sweeps", "n_skip"}
            unsupported = config_kwargs.keys() - supported
            if config is not None:
                if not isinstance(config, LongBetConfig):
                    raise TypeError("config must be a LongBetConfig or None.")
                defaults = LongBetConfig()
                unsupported |= {field.name for field in dataclasses.fields(config)
                                if field.name not in supported and
                                getattr(config, field.name) != getattr(defaults, field.name)}
                # The wrapper's own proper-prior defaults are inert for this
                # engine; direct_config owns both Gaussian variance priors.
                for name, value in (("sigma_prior_a", 2.), ("sigma_prior_b", 1.)):
                    if getattr(config, name) == value:
                        unsupported.discard(name)
            if unsupported:
                raise ValueError("Unsupported direct_smooth LongBetConfig options: "
                                 + ", ".join(sorted(unsupported))
                                 + ". Set forest and prior options in direct_config.")
            if isinstance(self.direct_config, dict):
                self.direct_config = DirectSmoothConfig(**self.direct_config)
            elif self.direct_config is None:
                self.direct_config = DirectSmoothConfig()
            elif not isinstance(self.direct_config, DirectSmoothConfig):
                raise TypeError("direct_config must be a DirectSmoothConfig, dictionary, or None.")
        elif self.direct_config is not None:
            raise ValueError("direct_config requires engine='direct_smooth'.")
        if config is None:
            self.config = LongBetConfig(**{"sigma_prior_a": 2., "sigma_prior_b": 1.,
                                          **config_kwargs})
        elif isinstance(config, LongBetConfig):
            self.config = dataclasses.replace(config, **config_kwargs)
        else:
            raise TypeError("config must be a LongBetConfig or None.")
        if self.engine == "longbet" and (self.config.sigma_prior_a <= 0 or self.config.sigma_prior_b <= 0):
            raise ValueError("LongBetEncourage requires a proper innovation prior: "
                             "sigma_prior_a > 0 and sigma_prior_b > 0.")
        if self.engine == "longbet" and self.config.random_intercept and not all(
            np.isfinite(v) and v > 0 for v in (self.config.gamma_prior_a, self.config.gamma_prior_b)
        ):
            raise ValueError("random intercepts require proper positive gamma_prior_a and gamma_prior_b.")
        self.first_stage, self.outcome = first_stage, outcome
        self.model: LongBetMulti | None = None
        self._data: dict[str, np.ndarray] = {}
        self.metadata: dict[str, Any] = {}

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
        if self.engine == "longbet" and (not np.all(np.diff(model_t) > 0) or not np.array_equal(
            derive_exposure(data["z"], model_t), derive_exposure(data["z"], data["t"])
        )):
            raise ValueError("t loses encouragement-clock precision in the float32 model; "
                             "shift the calendar origin before fitting.")
        return data

    def fit(self, y: Any, d: Any, z: Any, x: Any, t: Any = None,
            x_trt: Any = None, key: Any = None) -> LongBetEncourage:
        """Fit Y and take-up jointly on randomized encouragement, preserving inputs."""
        if self.engine == "direct_smooth":
            from longbet._direct_smooth import LongBetDirectSmooth
            if x_trt is not None:
                raise ValueError("direct_smooth uses x for both forests; x_trt is unsupported.")
            if key is not None:
                raise ValueError("direct_smooth does not accept a JAX key; set random_seed instead.")
            data = self._inputs(y, d, z, x, t, x_trt)
            self._direct_model = LongBetDirectSmooth(self.direct_config)
            self._direct_model.fit(**data, seed=self.config.random_seed,
                                   chains=self.config.num_chains, burnin=self.config.num_burnin,
                                   draws=self.config.num_sweeps, n_skip=self.config.n_skip)
            self._data = data
            for value in self._data.values():
                value.flags.writeable = False
            self.metadata = dict(self._direct_model.metadata,
                                 wrapper_config=dataclasses.asdict(self.config))
            return self
        if self.engine == "orthogonal_iv":
            from longbet._orthogonal_iv import LongBetOrthogonalIV
            data = self._inputs(y, d, z, x, t, x_trt)
            self._orthogonal_model = LongBetOrthogonalIV(
                self.config,
                monotonic_first_stage=True,
            )
            self._orthogonal_model.fit(
                data["y"], data["d"], data["z"], data["x"], t=data["t"], x_trt=data.get("x_trt")
            )
            self._data = data
            for value in self._data.values():
                value.flags.writeable = False
            self.metadata = dict(
                _validate(z, d, t).metadata(),
                target="all_original_units_equal_weight",
                first_stage=self.first_stage,
                outcome=self.outcome,
                engine="orthogonal_iv",
                inference_version=INFERENCE_VERSION,
                calibration_status="not_established",
            )
            return self
        if self.first_stage == "hazard":
            from longbet._hazard_adoption import HazardAdoptionForest, HazardConfig
            from longbet._model import LongBet
            data = self._inputs(y, d, z, x, t, x_trt)
            hazard_cfg = HazardConfig(
                baseline_trees=min(self.config.num_trees_pr, 6),
                effect_trees=min(self.config.num_trees_trt, 6),
            )
            self._hazard_forest = HazardAdoptionForest(hazard_cfg)
            self._hazard_forest.fit(
                data["d"], data["z"], data["x"], t=data["t"],
                seed=self.config.random_seed,
                chains=self.config.num_chains,
                burnin=self.config.num_burnin,
                draws=self.config.num_sweeps,
            )
            model_y = LongBet(self.config).fit(
                data["y"], data["x"], data["z"], t=data["t"], x_trt=data.get("x_trt"), key=key
            )
            self.model, self._data = model_y, data
            for value in self._data.values():
                value.flags.writeable = False
            self.metadata = dict(
                _validate(z, d, t).metadata(),
                target="all_original_units_equal_weight",
                first_stage="hazard",
                outcome=self.outcome,
                inference_version=INFERENCE_VERSION,
                calibration_status="not_established",
            )
            return self
        data = self._inputs(y, d, z, x, t, x_trt)
        first_period = bool(np.any(data["z"][:, 0])) and self.config.random_intercept
        if first_period:
            warnings.warn("Encouragement begins in the first observed period. Randomized ITTs "
                          "remain identified; inspect prior sensitivity of the model's unit "
                          "intercepts and encouragement effects. Retain both randomized arms.",
                          UserWarning, stacklevel=2)
        with warnings.catch_warnings():
            if first_period:
                # The generic treatment model recommends dropping always-treated
                # units; here those units are the randomized encouragement arm.
                warnings.filterwarnings("ignore", message=r"\d+ unit\(s\) are treated in every period,.*",
                                        category=UserWarning)
            model = LongBetMulti(self.config).fit(
                {"outcome": data["y"], "takeup": data["d"]}, data["x"], data["z"],
                t=data["t"], x_trt=data.get("x_trt"), key=key,
                outcome={"outcome": self.outcome,
                         "takeup": "continuous" if self.first_stage == "lpm" else "binary"})
        self.model, self._data = model, data
        for value in self._data.values():
            value.flags.writeable = False
        self.metadata = dict(_validate(z, d, t).metadata(),
            target="all_original_units_equal_weight", first_stage=self.first_stage,
            outcome=self.outcome, provenance=model.provenance,
            sampler_semantics=model.sampler_semantics, inference_version=INFERENCE_VERSION,
            unit_intercept_covariance="independent_prior",
            innovation_coupling=_innovation_coupling(model),
            shared_treatment_partitions=bool(model.config.num_shared_trees),
            calibration_status="not_established")
        return self

    def predict(self, *, summary_only: bool = True, groups: Any = None,
                block_size: int | None = None, standardization: str = "conditional",
                alpha: float = .05) -> Any:
        """Standardize both ITTs on all original units and fixed baseline subgroups.

        Group labels are prespecified, nonmissing, length N, and cannot be
        ``'all'``. Missing-arm subgroups retain model predictions but their
        reference is unavailable; those predictions depend on model extrapolation.
        """
        if self.engine == "direct_smooth":
            if self._direct_model is None:
                raise RuntimeError("LongBetEncourage must be fitted before prediction.")
            if summary_only is not True or block_size is not None or standardization != "conditional":
                raise ValueError("direct_smooth supports only summary_only=True, block_size=None, "
                                 "and standardization='conditional'.")
            return self._direct_model.predict(groups=groups, alpha=alpha)
        if self.engine == "orthogonal_iv":
            if self._orthogonal_model is None:
                raise RuntimeError("LongBetEncourage must be fitted before prediction.")
            return self._orthogonal_model.predict(new_x=new_x, new_z=new_z, alpha=alpha)
        if self.model is None:
            raise RuntimeError("LongBetEncourage must be fitted before prediction.")
        result = standardize_encouragement(self.model, self._data["x"], self._data["z"],
            self._data["t"], x_trt=self._data.get("x_trt"), summary_only=summary_only,
            groups=groups, block_size=block_size, standardization=standardization, alpha=alpha)
        reference = _reference_groups(self._data, result, alpha)
        return EncouragementPrediction(result, reference, self.metadata, alpha=alpha)

    def predict_conditional(
        self,
        x: Any = None,
        z: Any = None,
        t: Any = None,
        *,
        alpha: float = 0.05,
        cace_stabilization: float = 0.02,
        monotonic_first_stage: bool = True,
        cace_shrinkage: str = "adaptive",
        shrinkage_lambda: float = 1.0,
    ) -> ConditionalEncouragementPrediction:
        """Predict unit-level conditional ITTs, CACE, and optimal encouragement policies.

        Evaluates the fitted LongBet model to produce individual-level Conditional
        Intent-to-Treat on the outcome (CITT_Y), adoption take-up (CITT_D), and
        conditional CACE across post-treatment horizons. Supports out-of-sample
        prediction on new cohorts.

        Parameters
        ----------
        x : array-like, optional
            Baseline covariates of shape (N, P). If None, evaluates on fitted units.
        z : array-like, optional
            Counterfactual encouragement assignment matrix (N, T). If None,
            assigns encouragement starting from the encouragement launch period onward.
        t : array-like, optional
            Calendar time vector (T,). Defaults to fitted calendar.
        alpha : float
            Significance level for credible intervals (default 0.05).
        cace_stabilization : float
            Ridge shrinkage parameter for regularizing small-compliance CACE (default 0.02).
        monotonic_first_stage : bool
            If True (default), eliminates defier draws by enforcing non-negative first-stage compliance.
        cace_shrinkage : {"adaptive", "ridge", "none"}
            Shrinkage method for CACE estimation. "adaptive" (default) applies Empirical Bayes /
            James-Stein shrinkage toward population CACE based on local compliance SNR.
        shrinkage_lambda : float
            Strength multiplier for adaptive shrinkage (default 1.0).

        Returns
        -------
        ConditionalEncouragementPrediction
        """
        if self.engine == "orthogonal_iv":
            if self._orthogonal_model is None:
                raise RuntimeError("LongBetEncourage must be fitted before prediction.")
            return self._orthogonal_model.predict_conditional(x=x, z=z, t=t, alpha=alpha)
        if self.engine != "longbet":
            raise ValueError(
                "predict_conditional requires engine='longbet' or 'orthogonal_iv'. The direct_smooth "
                "engine uses additive stumps and does not support out-of-sample or "
                "heterogeneous subgroup prediction."
            )
        if self.model is None:
            raise RuntimeError("LongBetEncourage must be fitted before prediction.")

        # Resolve x
        if x is None:
            x_np = self._data["x"]
        else:
            x_np = np.asarray(x, dtype=np.float32)
            if x_np.ndim != 2:
                raise ValueError("x must be a 2-D array of shape (N, P).")

        n_units = len(x_np)

        # Resolve t
        if t is None:
            t_vec = self._data["t"]
        else:
            t_vec = np.asarray(t, dtype=np.float64)

        n_periods = len(t_vec)

        # Determine encouragement start index from training panel
        panel = _validate(self._data["z"], self._data["d"], self._data["t"])
        start = panel.start

        # Resolve z (counterfactual encouragement schedule)
        if z is None:
            z_mat = np.zeros((n_units, n_periods), dtype=np.float32)
            z_mat[:, start:] = 1.0
        else:
            z_mat = np.asarray(z, dtype=np.float32)
            if z_mat.shape != (n_units, n_periods):
                raise ValueError(f"z must have shape ({n_units}, {n_periods}).")

        if self.first_stage == "hazard":
            pred_y = self.model.predict(x_np, z_mat, t=t_vec, summary_only=False)
            post_idx = np.arange(start, n_periods)
            citt_y_draws = pred_y.tauhats[:, post_idx, :]
            citt_d_draws = self._hazard_forest.predict(x if x is not None else None)
            min_d = min(citt_y_draws.shape[-1], citt_d_draws.shape[-1])
            citt_y_draws = citt_y_draws[:, :, :min_d]
            citt_d_draws = citt_d_draws[:, :, :min_d]
            periods = panel.t[start:].copy()
            horizons = panel.exposure[start:].copy()
            return ConditionalEncouragementPrediction(
                citt_y_draws=citt_y_draws,
                citt_d_draws=citt_d_draws,
                periods=periods,
                horizons=horizons,
                alpha=alpha,
                cace_stabilization=cace_stabilization,
                monotonic_first_stage=monotonic_first_stage,
                cace_shrinkage=cace_shrinkage,
                shrinkage_lambda=shrinkage_lambda,
            )

        # Run multi-outcome prediction with full draws
        pred_multi = self.model.predict(x_np, z_mat, t=t_vec, summary_only=False)

        # Extract draws
        from longbet._multi_model import effect_draws
        y_draws = effect_draws(pred_multi, "outcome")   # (N, T, D)
        d_draws = effect_draws(pred_multi, "takeup")    # (N, T, D)

        # Slice post-encouragement horizons
        post_idx = np.arange(start, n_periods)
        citt_y_draws = y_draws[:, post_idx, :]
        citt_d_draws = d_draws[:, post_idx, :]

        # Extract baseline untreated adoption draws (p0) for principal stratification
        pred_d = pred_multi["takeup"]
        if pred_d.muhats0 is not None:
            if pred_d.outcome == "binary":
                p0_draws = stats.norm.cdf(pred_d.muhats0[:, post_idx, :])
            else:
                p0_draws = np.clip(pred_d.muhats0[:, post_idx, :], 0.0, 1.0)
        else:
            p0_draws = None

        periods = panel.t[start:].copy()
        horizons = panel.exposure[start:].copy()

        return ConditionalEncouragementPrediction(
            citt_y_draws=citt_y_draws,
            citt_d_draws=citt_d_draws,
            periods=periods,
            horizons=horizons,
            p0_draws=p0_draws,
            alpha=alpha,
            cace_stabilization=cace_stabilization,
            monotonic_first_stage=monotonic_first_stage,
            cace_shrinkage=cace_shrinkage,
            shrinkage_lambda=shrinkage_lambda,
        )

    def bootstrap_comparison(self, *, replicates: int = 200, groups: Any = None,
                             random_seed: int = 0, alpha: float = .05,
                             min_ess: float = 400, max_rhat: float = 1.01,
                             callback: Any = None) -> EncouragementComparison:
        """Refit paired whole-unit bootstrap samples to compare model/reference ITTs.

        Resample entire longitudinal records within assigned arm and prespecified
        group, holding their counts fixed. Refit both equations together, using
        new sampler seeds, and recompute both estimators on each same resampled
        population. The comparison variance includes their sampling covariance.
        This is an empirical sampling approximation, not exact finite-population
        randomization inference or a test of IV assumptions. Only the ITTs are
        compared: ordinary bootstrap ratio intervals fail near a zero first stage.

        This requires ``replicates`` complete model fits and can be expensive.
        ``callback(completed, total)`` optionally reports progress. All failures
        and failed convergence checks are retained. A flag is withheld if the
        original fit or any replicate fails diagnostics, inference is unavailable,
        or fewer than 100 replications are requested. The 100-run floor is a
        computational minimum, not a calibration guarantee.
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
        # predict already validates group labels and their string representation.
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
                fit = LongBetEncourage(self.config, first_stage=self.first_stage, outcome=self.outcome,
                                       engine=self.engine, direct_config=self.direct_config,
                                       random_seed=seed)
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

    def save(self, path: str | Path) -> None:
        """Save a versioned, pickle-free archive including original study data.

        The archive contains the outcomes, assignment, adoption and baseline
        covariates needed to replay the identical target, as well as joint traces.
        """
        if self.engine == "direct_smooth" and self._direct_model is None or (
                self.engine == "longbet" and self.model is None and self.first_stage != "hazard") or (
                self.engine == "orthogonal_iv" and self._orthogonal_model is None) or (
                self.first_stage == "hazard" and (self.model is None or self._hazard_forest is None)):
            raise RuntimeError("LongBetEncourage must be fitted before saving.")
        arrays = dict(self._data)
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "model.npz"
            if self.engine == "direct_smooth":
                self._direct_model.save(nested)
            elif self.engine == "orthogonal_iv":
                f_path = Path(tmp) / "fs.npz"
                c_path = Path(tmp) / "cace.npz"
                self._orthogonal_model.first_stage_model.save(f_path)
                self._orthogonal_model.cace_model.save(c_path)
                np.savez_compressed(
                    nested,
                    first_stage_bytes=np.frombuffer(f_path.read_bytes(), dtype=np.uint8),
                    cace_bytes=np.frombuffer(c_path.read_bytes(), dtype=np.uint8),
                    min_compliance=np.asarray(self._orthogonal_model.min_compliance),
                    monotonic_first_stage=np.asarray(self._orthogonal_model.monotonic_first_stage),
                )
            elif self.first_stage == "hazard":
                y_path = Path(tmp) / "model_y.npz"
                self.model.save(y_path)
                haz = self._hazard_forest
                np.savez_compressed(
                    nested,
                    model_y_bytes=np.frombuffer(y_path.read_bytes(), dtype=np.uint8),
                    hazard_base_rules=haz.base_rules,
                    hazard_eff_rules=haz.eff_rules,
                    hazard_space_features=haz.space_features,
                    hazard_space_thresholds=haz.space_thresholds,
                    hazard_p_mean=np.asarray(haz.p_mean),
                    hazard_base_leaves=np.asarray(haz._base_leaves_list),
                    hazard_eff_leaves=np.asarray(haz._eff_leaves_list),
                    hazard_xi=np.asarray(haz._xi_list),
                    hazard_unit_stock_itt=haz.draws["unit_stock_itt"],
                    hazard_stock_itt=haz.draws["stock_itt"],
                    hazard_exposure_itt=haz.draws["exposure_itt"],
                    hazard_itt=haz.draws["hazard_itt"],
                    hazard_xi_draws=haz.draws["xi"],
                    hazard_metadata_json=np.asarray(json.dumps(haz.metadata, allow_nan=False)),
                )
            else:
                self.model.save(nested)
            arrays["model_archive"] = np.frombuffer(nested.read_bytes(), dtype=np.uint8)
        meta = dict(self.metadata, kind="LongBetEncourage", archive_version=ARCHIVE_VERSION,
                    input_hashes={name: _digest(value) for name, value in arrays.items()})
        arrays["metadata"] = np.asarray(json.dumps(meta, allow_nan=False))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> LongBetEncourage:
        """Validate and replay an encouragement archive; scalar/multi archives fail."""
        try:
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"]))
                arrays = {name: archive[name] for name in archive.files if name != "metadata"}
            if (not isinstance(meta, dict) or meta.get("kind") != "LongBetEncourage" or
                    type(meta.get("archive_version")) is not int or
                    meta.get("archive_version") != ARCHIVE_VERSION or
                    meta.get("inference_version") != INFERENCE_VERSION):
                raise ValueError("unsupported kind or version")
            required = {"y", "d", "z", "x", "t", "model_archive"}
            if not required <= arrays.keys() or arrays.keys() - required - {"x_trt"}:
                raise ValueError("unexpected or missing archive arrays")
            if meta["input_hashes"] != {name: _digest(value) for name, value in arrays.items()}:
                raise ValueError("input or model digest mismatch")
            if arrays["model_archive"].dtype != np.uint8 or arrays["model_archive"].ndim != 1:
                raise ValueError("invalid nested model bytes")
            direct = meta.get("engine", "longbet") == "direct_smooth"
            orthogonal = meta.get("engine") == "orthogonal_iv"
            hazard = meta.get("first_stage") == "hazard"

            with tempfile.TemporaryDirectory() as tmp:
                nested = Path(tmp) / "model.npz"
                nested.write_bytes(arrays.pop("model_archive").tobytes())
                if direct:
                    from longbet._direct_smooth import LongBetDirectSmooth
                    model = LongBetDirectSmooth.load(nested)
                elif orthogonal:
                    with np.load(nested, allow_pickle=False) as o_arch:
                        fs_p = Path(tmp) / "fs.npz"
                        c_p = Path(tmp) / "c.npz"
                        fs_p.write_bytes(o_arch["first_stage_bytes"].tobytes())
                        c_p.write_bytes(o_arch["cace_bytes"].tobytes())
                        from longbet._model import LongBet
                        from longbet._orthogonal_iv import LongBetOrthogonalIV
                        fs_model = LongBet.load(fs_p)
                        c_model = LongBet.load(c_p)
                        ortho = LongBetOrthogonalIV(
                            config=c_model.config,
                            first_stage_config=fs_model.config,
                            min_compliance=float(o_arch["min_compliance"]),
                            monotonic_first_stage=bool(o_arch["monotonic_first_stage"]),
                            outcome=meta.get("outcome", "continuous"),
                        )
                        ortho.first_stage_model = fs_model
                        ortho.cace_model = c_model
                        ortho.fitted_ = True
                        ortho.panel = _validate(arrays["z"], arrays["d"], arrays["t"])
                        ortho._data = dict(
                            y=arrays["y"], d=arrays["d"], z=arrays["z"],
                            x=arrays["x"], t=arrays["t"],
                            start=ortho.panel.start,
                        )
                    obj = cls(c_model.config, engine="orthogonal_iv", outcome=meta.get("outcome", "continuous"))
                    obj._data = obj._inputs(**{**arrays, "x_trt": arrays.get("x_trt")})
                    for value in obj._data.values():
                        value.flags.writeable = False
                    obj._orthogonal_model = ortho
                    obj.metadata = {k: v for k, v in meta.items()
                                    if k not in ("kind", "archive_version", "input_hashes")}
                    return obj
                elif hazard:
                    with np.load(nested, allow_pickle=False) as h_arch:
                        y_p = Path(tmp) / "y.npz"
                        y_p.write_bytes(h_arch["model_y_bytes"].tobytes())
                        from longbet._model import LongBet
                        from longbet._hazard_adoption import HazardAdoptionForest, HazardConfig
                        model_y = LongBet.load(y_p)
                        haz_meta = json.loads(str(h_arch["hazard_metadata_json"]))
                        haz_cfg = HazardConfig(**haz_meta.get("config", {}))
                        haz = HazardAdoptionForest(haz_cfg)
                        haz.base_rules = h_arch["hazard_base_rules"]
                        haz.eff_rules = h_arch["hazard_eff_rules"]
                        haz.space_features = h_arch["hazard_space_features"]
                        haz.space_thresholds = h_arch["hazard_space_thresholds"]
                        haz.p_mean = float(h_arch["hazard_p_mean"])
                        haz._base_leaves_list = list(h_arch["hazard_base_leaves"])
                        haz._eff_leaves_list = list(h_arch["hazard_eff_leaves"])
                        haz._xi_list = list(h_arch["hazard_xi"])
                        haz.draws = dict(
                            unit_stock_itt=h_arch["hazard_unit_stock_itt"],
                            stock_itt=h_arch["hazard_stock_itt"],
                            exposure_itt=h_arch["hazard_exposure_itt"],
                            hazard_itt=h_arch["hazard_itt"],
                            xi=h_arch["hazard_xi_draws"],
                        )
                        haz.metadata = haz_meta
                        haz.fitted_ = True
                        haz.panel = _validate(arrays["z"], arrays["d"], arrays["t"])
                    obj = cls(model_y.config, first_stage="hazard", outcome=meta["outcome"])
                    obj._data = obj._inputs(**{**arrays, "x_trt": arrays.get("x_trt")})
                    for value in obj._data.values():
                        value.flags.writeable = False
                    obj.model = model_y
                    obj._hazard_forest = haz
                    obj.metadata = {k: v for k, v in meta.items()
                                    if k not in ("kind", "archive_version", "input_hashes")}
                    return obj
                else:
                    model = LongBetMulti.load(nested)
            if direct:
                obj = cls(LongBetConfig(**meta["wrapper_config"]),
                          engine="direct_smooth", direct_config=model.config,
                          first_stage=meta["first_stage"], outcome=meta["outcome"])
                obj._data = obj._inputs(**{**arrays, "x_trt": arrays.get("x_trt")})
                if (any(not np.array_equal(value, model._data.get(name))
                        for name, value in obj._data.items()) or
                        any(meta.get(k) != v for k, v in model.metadata.items()) or
                        model.metadata["sampler"] != dict(
                            seed=obj.config.random_seed, chains=obj.config.num_chains,
                            burnin=obj.config.num_burnin, draws=obj.config.num_sweeps,
                            n_skip=obj.config.n_skip)):
                    raise ValueError("direct_smooth model/design provenance mismatch")
                for value in obj._data.values():
                    value.flags.writeable = False
                obj._direct_model = model
                obj.metadata = {k: v for k, v in meta.items()
                                if k not in ("kind", "archive_version", "input_hashes")}
                return obj
            obj = cls(model.config, first_stage=meta["first_stage"], outcome=meta["outcome"])
            obj._data = obj._inputs(**{**arrays, "x_trt": arrays.get("x_trt")})
            panel = _validate(arrays["z"], arrays["d"], arrays["t"])
            expected_types = (obj.outcome, "continuous" if obj.first_stage == "lpm" else "binary")
            if (model.outcome_names != ("outcome", "takeup") or model.outcome != expected_types or
                    model.N_ != len(arrays["y"]) or model.T_ != arrays["y"].shape[1] or
                    not np.array_equal(model.fitted_t_, arrays["t"].astype(np.float32)) or
                    not np.array_equal(derive_exposure(arrays["z"], model.fitted_t_),
                                       derive_exposure(arrays["z"], arrays["t"])) or
                    model.provenance != meta["provenance"] or
                    model.sampler_semantics != meta["sampler_semantics"] or
                    meta["target"] != "all_original_units_equal_weight" or
                    meta.get("unit_intercept_covariance") != "independent_prior" or
                    meta.get("calibration_status") != "not_established" or
                    meta.get("innovation_coupling") != _innovation_coupling(model) or
                    meta.get("shared_treatment_partitions") != bool(model.config.num_shared_trees) or
                    any(meta.get(k) != v for k, v in panel.metadata().items())):
                raise ValueError("model/design provenance mismatch")
            for value in obj._data.values():
                value.flags.writeable = False
            obj.model = model
            obj.metadata = {k: v for k, v in meta.items()
                            if k not in ("kind", "archive_version", "input_hashes")}
            return obj
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"Invalid encouragement archive: {exc}") from exc


def _digest(a: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(str(a.shape).encode())
    digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()
