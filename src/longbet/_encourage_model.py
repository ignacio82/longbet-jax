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
        if first_stage not in ("lpm", "probit"):
            raise ValueError("first_stage must be explicitly chosen as 'lpm' or 'probit'.")
        if outcome not in ("continuous", "binary"):
            raise ValueError("outcome must be 'continuous' or 'binary'.")
        self.engine = config_kwargs.pop("engine", "longbet")
        self.direct_config = config_kwargs.pop("direct_config", None)
        self._direct_model = None
        if config is None:
            self.config = LongBetConfig(**{"sigma_prior_a": 2., "sigma_prior_b": 1.,
                                          **config_kwargs})
        elif isinstance(config, LongBetConfig):
            self.config = dataclasses.replace(config, **config_kwargs)
        else:
            raise TypeError("config must be a LongBetConfig or None.")
        if self.config.sigma_prior_a <= 0 or self.config.sigma_prior_b <= 0:
            raise ValueError("LongBetEncourage requires a proper innovation prior: "
                             "sigma_prior_a > 0 and sigma_prior_b > 0.")
        if self.config.random_intercept and not all(
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
        if not np.all(np.diff(model_t) > 0) or not np.array_equal(
            derive_exposure(data["z"], model_t), derive_exposure(data["z"], data["t"])
        ):
            raise ValueError("t loses encouragement-clock precision in the float32 model; "
                             "shift the calendar origin before fitting.")
        return data

    def fit(self, y: Any, d: Any, z: Any, x: Any, t: Any = None,
            x_trt: Any = None, key: Any = None) -> LongBetEncourage:
        """Fit Y and take-up jointly on randomized encouragement, preserving inputs."""
        if self.engine == "direct_smooth":
            from longbet._direct_smooth import LongBetDirectSmooth
            self._direct_model = LongBetDirectSmooth(self.direct_config)
            self._direct_model.fit(y, d, z, x, t=t)
            self._data = self._direct_model._data
            self.metadata = self._direct_model.metadata
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
                alpha: float = .05) -> EncouragementPrediction:
        """Standardize both ITTs on all original units and fixed baseline subgroups.

        Group labels are prespecified, nonmissing, length N, and cannot be
        ``'all'``. Missing-arm subgroups retain model predictions but their
        reference is unavailable; those predictions depend on model extrapolation.
        """
        if self.engine == "direct_smooth":
            if self._direct_model is None:
                raise RuntimeError("LongBetEncourage must be fitted before prediction.")
            return self._direct_model.predict(groups=groups, alpha=alpha)
        if self.model is None:
            raise RuntimeError("LongBetEncourage must be fitted before prediction.")
        result = standardize_encouragement(self.model, self._data["x"], self._data["z"],
            self._data["t"], x_trt=self._data.get("x_trt"), summary_only=summary_only,
            groups=groups, block_size=block_size, standardization=standardization, alpha=alpha)
        reference = _reference_groups(self._data, result, alpha)
        return EncouragementPrediction(result, reference, self.metadata, alpha=alpha)

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
        if self.model is None:
            raise RuntimeError("LongBetEncourage must be fitted before saving.")
        arrays = dict(self._data)
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "model.npz"
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
            with tempfile.TemporaryDirectory() as tmp:
                nested = Path(tmp) / "model.npz"
                nested.write_bytes(arrays.pop("model_archive").tobytes())
                model = LongBetMulti.load(nested)
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
