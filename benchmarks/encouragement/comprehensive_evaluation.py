"""Comprehensive evaluation of LongBet IV upgrades vs existing methods.

Evaluates:
Part A: Pointwise CACE / Reduced Form Nuisance Benchmark
  1. Linear ANCOVA IV
  2. Spline ANCOVA IV
  3. ExtraTrees IV
  4. Baseline LongBet IV (Direct Smooth with RBF kernel, LPM Gaussian adoption)
  5. Upgraded LongBet IV (Phase 1+2: Matérn-3/2, ANCOVA backbone, hazard adoption, JAX)
  Across scenarios: 'linear', 'smooth', 'abrupt'.

Part B: Dynamic Exposure-Duration IV Benchmark (Phase 3)
  1. Static Wald IV (collapses under adoption catch-up)
  2. Homogeneous Duration 2SLS (misses heterogeneity across accounts)
  3. Heterogeneous Duration IV (Phase 3: g(S_{it}, X_i) with clustered CR1 covariance)
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor

from longbet import (
    DirectSmoothConfig,
    LongBetDirectSmooth,
    LongBetIVNuisance,
    LongBetIVNuisanceConfig,
    crossfit_encouragement,
    heterogeneous_duration_iv,
)
from longbet._encourage import _wald_set
from benchmarks.encouragement.iv_comparison import (
    generate_panel,
    adjusted_contrasts,
    reference_rows,
)


class ExtraTreesNuisance:
    """ExtraTrees nuisance learner for cross-fitting."""
    def __init__(self, n_estimators: int = 50, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, x_train, assignment_train, responses_train, times_post):
        self.n_horizons = responses_train.shape[1]
        self.models_ = {}
        for arm in (0, 1):
            mask = (assignment_train == arm)
            x_arm = x_train[mask]
            # Flatten responses for multi-output regression (horizons x 2)
            y_d = responses_train[mask].reshape(len(x_arm), -1)
            reg = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=5,
                random_state=self.seed + arm,
                n_jobs=1
            )
            reg.fit(x_arm, y_d)
            self.models_[arm] = reg
        return self

    def predict(self, x_test):
        n = len(x_test)
        preds = np.zeros((n, self.n_horizons, 2, 2))
        for arm in (0, 1):
            pred_flat = self.models_[arm].predict(x_test)
            preds[:, :, arm, :] = pred_flat.reshape(n, self.n_horizons, 2)
        # Clip adoption probabilities to [0, 1]
        preds[:, :, :, 1] = np.clip(preds[:, :, :, 1], 0.0, 1.0)
        return preds


def run_part_a_replication(scenario: str, seed: int, n: int = 160):
    data = generate_panel(scenario, seed=seed, n=n)
    truth_wald = data["truth"]["wald"]
    truth_d = data["truth"]["itt_d"]
    truth_y = data["truth"]["itt_y"]
    n_horizons = len(truth_wald)

    results = {}

    # Construct full pre-treatment feature set including baseline outcome
    y_pre = data["y"][:, :data["start"]].mean(axis=1, keepdims=True)
    y_pre_std = float(np.std(y_pre))
    y_pre_norm = (y_pre - float(np.mean(y_pre))) / (y_pre_std if y_pre_std > 1e-8 else 1.0)
    x_full = np.column_stack([data["x"], y_pre_norm])

    # 1. Linear ANCOVA IV
    t0 = time.perf_counter()
    est_lin, cov_lin = adjusted_contrasts(data, spline=False)
    t_lin = time.perf_counter() - t0
    rows_lin = reference_rows(est_lin, cov_lin)
    results["Linear ANCOVA IV"] = {"rows": rows_lin, "time": t_lin}

    # 2. Spline ANCOVA IV
    t0 = time.perf_counter()
    est_spl, cov_spl = adjusted_contrasts(data, spline=True)
    t_spl = time.perf_counter() - t0
    rows_spl = reference_rows(est_spl, cov_spl)
    results["Spline ANCOVA IV"] = {"rows": rows_spl, "time": t_spl}

    # 3. ExtraTrees IV
    t0 = time.perf_counter()
    res_et = crossfit_encouragement(
        y=data["y"], d=data["d"], z=data["z"], x=x_full, t=data["t"],
        learner_factory=lambda: ExtraTreesNuisance(n_estimators=40, seed=seed),
        seed=seed
    )
    t_et = time.perf_counter() - t0
    results["ExtraTrees IV"] = {"rows": res_et.table, "time": t_et}

    # 4. Baseline LongBet IV (Direct Smooth: RBF kernel, LPM Gaussian adoption)
    t0 = time.perf_counter()
    cfg_base = DirectSmoothConfig(
        baseline_trees=4, effect_trees=4, correlated_intercepts=False
    )
    fit_base = LongBetDirectSmooth(cfg_base).fit(
        data["y"], data["d"], data["z"], x_full, data["t"],
        chains=2, burnin=200, draws=400, seed=seed + 50000
    )
    pred_base = fit_base.predict().effects()
    t_base = time.perf_counter() - t0
    rows_base = []
    for r in pred_base.itertuples():
        rows_base.append({
            "horizon": r.horizon, "quantity": r.quantity,
            "estimate": r.posterior_median, "lower": r.posterior_lower,
            "upper": r.posterior_upper, "set_type": "posterior"
        })
    results["Baseline LongBet IV"] = {"rows": rows_base, "time": t_base}

    # 5. Upgraded LongBet IV (Phase 1+2: Matérn-3/2, ANCOVA backbone, hazard adoption, JAX)
    t0 = time.perf_counter()
    cfg_upgraded = LongBetIVNuisanceConfig(
        kernel="matern32",
        linear_ancova_backbone=True,
        adoption_model="hazard",
        engine="jax",
        baseline_trees=4,
        effect_trees=4,
        burnin=100,
        draws=200,
        length_scales=(2.0,)
    )
    res_upgraded = crossfit_encouragement(
        y=data["y"], d=data["d"], z=data["z"], x=x_full, t=data["t"],
        learner_factory=lambda: LongBetIVNuisance(cfg_upgraded),
        seed=seed
    )
    t_upgraded = time.perf_counter() - t0
    results["Upgraded LongBet IV"] = {"rows": res_upgraded.table, "time": t_upgraded}

    # Extract metrics per method
    metrics = {}
    for mname, mdata in results.items():
        rows = mdata["rows"]
        wald_estimates = []
        wald_covered = []
        d_estimates = []
        d_covered = []

        for h in range(1, n_horizons + 1):
            h_rows = {r["quantity"]: r for r in rows if r["horizon"] == h}
            
            # Wald / CACE
            if "wald" in h_rows:
                rw = h_rows["wald"]
                w_est = rw.get("estimate", np.nan)
                w_lo = rw.get("lower", np.nan)
                w_hi = rw.get("upper", np.nan)
                wald_estimates.append(w_est)
                if np.isfinite(w_lo) and np.isfinite(w_hi) and np.isfinite(truth_wald[h - 1]):
                    wald_covered.append(w_lo <= truth_wald[h - 1] <= w_hi)
                else:
                    wald_covered.append(False)

            # Adoption ITT_D
            if "itt_d" in h_rows:
                rd = h_rows["itt_d"]
                d_est = rd.get("estimate", np.nan)
                d_lo = rd.get("lower", np.nan)
                d_hi = rd.get("upper", np.nan)
                d_estimates.append(d_est)
                if np.isfinite(d_lo) and np.isfinite(d_hi):
                    d_covered.append(d_lo <= truth_d[h - 1] <= d_hi)
                else:
                    d_covered.append(False)

        wald_err = np.array(wald_estimates) - truth_wald
        d_err = np.array(d_estimates) - truth_d

        metrics[mname] = {
            "cace_sq_err": np.mean(wald_err ** 2),
            "cace_covered": np.mean(wald_covered),
            "d_sq_err": np.mean(d_err ** 2),
            "d_covered": np.mean(d_covered),
            "time": mdata["time"]
        }

    return metrics


def run_part_b_replication(seed: int, n_units: int = 300, n_periods: int = 12):
    """Simulate panel encouragement with adoption catch-up and heterogeneous returns."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_units, 2))
    z = rng.binomial(1, 0.5, size=(n_units, 1))

    # Heterogeneous returns: beta_i = beta_0 + beta_1 * X_1 + beta_2 * X_2
    beta_0 = 5.0
    beta_1 = 2.5
    beta_2 = -1.5
    true_slopes = beta_0 + beta_1 * x[:, 0] + beta_2 * x[:, 1]  # (n_units,)

    alpha_i = rng.standard_normal((n_units, 1)) * 1.5
    lambda_t = (0.2 * np.arange(n_periods))[None, :]

    # Adoption dynamics: Encouraged adopt early (weeks 1-4); Control catch up (weeks 5-10)
    # By week 10, both arms have near 100% adoption (ITT_D -> 0)
    d = np.zeros((n_units, n_periods))
    for i in range(n_units):
        p_adopt = 0.45 if z[i, 0] == 1 else 0.12
        adopted = False
        for t in range(n_periods):
            # Acceleration phase
            if not adopted and rng.random() < p_adopt:
                adopted = True
            # Catch-up phase: control adoption hazard increases after t=5
            if not adopted and z[i, 0] == 0 and t >= 5 and rng.random() < 0.35:
                adopted = True
            if adopted:
                d[i, t] = 1.0

    # Accumulated exposure duration
    s = np.cumsum(d, axis=1)

    # Outcomes
    y_true = alpha_i + lambda_t + s * true_slopes[:, None]
    eps = rng.standard_normal((n_units, n_periods)) * 1.0
    y = y_true + eps

    z_panel = np.broadcast_to(z, (n_units, n_periods))

    # 1. Heterogeneous Duration IV (Phase 3)
    t0 = time.perf_counter()
    res_hdiv = heterogeneous_duration_iv(y, d, z_panel, x=x, model_type="linear", segments=3)
    t_hdiv = time.perf_counter() - t0

    pred_slopes = res_hdiv.unit_effects
    hdiv_rmse = np.sqrt(np.mean((pred_slopes - true_slopes) ** 2))
    pop_slope_covered = res_hdiv.population_ci[0] <= np.mean(true_slopes) <= res_hdiv.population_ci[1]

    # 2. Homogeneous Duration 2SLS (assumes uniform duration effect)
    t0 = time.perf_counter()
    res_homo = heterogeneous_duration_iv(y, d, z_panel, x=None, model_type="linear")
    t_homo = time.perf_counter() - t0
    homo_slope = res_homo.population_effect
    homo_rmse = np.sqrt(np.mean((homo_slope - true_slopes) ** 2))
    homo_pop_covered = res_homo.population_ci[0] <= np.mean(true_slopes) <= res_homo.population_ci[1]

    # 3. Static Wald IV at final period (where catch-up occurred)
    # Late horizon Wald ratio ITT_Y / ITT_D
    itt_y_late = np.mean(y[z[:, 0] == 1, -1]) - np.mean(y[z[:, 0] == 0, -1])
    itt_d_late = np.mean(d[z[:, 0] == 1, -1]) - np.mean(d[z[:, 0] == 0, -1])
    static_wald_late = itt_y_late / itt_d_late if abs(itt_d_late) > 0.001 else np.nan
    static_first_stage = abs(itt_d_late)

    return {
        "hdiv_rmse": hdiv_rmse,
        "hdiv_covered": pop_slope_covered,
        "hdiv_f": res_hdiv.first_stage_f,
        "hdiv_time": t_hdiv,
        "homo_rmse": homo_rmse,
        "homo_covered": homo_pop_covered,
        "homo_time": t_homo,
        "late_itt_d": static_first_stage,
        "static_wald_late": static_wald_late,
    }


def main():
    print("=" * 80)
    print("STARTING COMPREHENSIVE EVALUATION: LONGBET VS EXISTING ALTERNATIVES")
    print("=" * 80)

    # --- PART A: POINTWISE REDUCED FORM / CACE BENCHMARK ---
    n_reps = 30
    scenarios = ["linear", "smooth", "abrupt"]
    methods = [
        "Linear ANCOVA IV",
        "Spline ANCOVA IV",
        "ExtraTrees IV",
        "Baseline LongBet IV",
        "Upgraded LongBet IV",
    ]

    print(f"\nPART A: Pointwise Nuisance Benchmark ({n_reps} replications per scenario)")
    part_a_records = []

    for sc in scenarios:
        print(f"\n--> Running Scenario: {sc.upper()}")
        sc_metrics = {m: {"cace_sq": [], "cace_cov": [], "d_sq": [], "d_cov": [], "time": []} for m in methods}
        
        for r in range(n_reps):
            seed = 20260900 + 1000 * scenarios.index(sc) + r
            rep_res = run_part_a_replication(sc, seed=seed, n=160)
            for m in methods:
                sc_metrics[m]["cace_sq"].append(rep_res[m]["cace_sq_err"])
                sc_metrics[m]["cace_cov"].append(rep_res[m]["cace_covered"])
                sc_metrics[m]["d_sq"].append(rep_res[m]["d_sq_err"])
                sc_metrics[m]["d_cov"].append(rep_res[m]["d_covered"])
                sc_metrics[m]["time"].append(rep_res[m]["time"])
            if (r + 1) % 10 == 0:
                print(f"    Completed {r + 1}/{n_reps} reps...")

        for m in methods:
            part_a_records.append({
                "Scenario": sc,
                "Method": m,
                "CACE_RMSE": np.sqrt(np.mean(sc_metrics[m]["cace_sq"])),
                "CACE_Coverage": np.mean(sc_metrics[m]["cace_cov"]) * 100,
                "ITT_D_RMSE": np.sqrt(np.mean(sc_metrics[m]["d_sq"])),
                "ITT_D_Coverage": np.mean(sc_metrics[m]["d_cov"]) * 100,
                "Avg_Time_Sec": np.mean(sc_metrics[m]["time"]),
            })

    df_part_a = pd.DataFrame(part_a_records)
    print("\n" + "=" * 80)
    print("PART A RESULTS TABLE:")
    print("=" * 80)
    print(df_part_a.to_string(index=False))

    # --- PART B: DYNAMIC EXPOSURE-DURATION IV BENCHMARK ---
    n_reps_b = 40
    print(f"\n\nPART B: Dynamic Heterogeneous Duration IV Benchmark ({n_reps_b} replications)")
    b_records = []
    for r in range(n_reps_b):
        seed = 20261900 + r
        res_b = run_part_b_replication(seed=seed, n_units=300, n_periods=12)
        b_records.append(res_b)
        if (r + 1) % 10 == 0:
            print(f"    Completed {r + 1}/{n_reps_b} duration replications...")

    df_b = pd.DataFrame(b_records)
    print("\n" + "=" * 80)
    print("PART B RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Average Final-Period Adoption ITT (Catch-up): {df_b['late_itt_d'].mean():.4f}")
    print(f"  -> Static Wald IV failure rate (ITT_D < 0.05): {(df_b['late_itt_d'] < 0.05).mean() * 100:.1f}%")
    print(f"\nHeterogeneous Duration IV (Phase 3):")
    print(f"  Account-Level Slope RMSE: {df_b['hdiv_rmse'].mean():.4f}")
    print(f"  Population Slope 95% Coverage: {df_b['hdiv_covered'].mean() * 100:.1f}%")
    print(f"  Average First-Stage F: {df_b['hdiv_f'].mean():.1f}")
    print(f"  Average Runtime: {df_b['hdiv_time'].mean():.4f}s")
    print(f"\nHomogeneous Duration 2SLS:")
    print(f"  Account-Level Slope RMSE: {df_b['homo_rmse'].mean():.4f}")
    print(f"  Population Slope 95% Coverage: {df_b['homo_covered'].mean() * 100:.1f}%")
    print(f"  Average Runtime: {df_b['homo_time'].mean():.4f}s")

    # Save results to CSV
    df_part_a.to_csv("/home/ignacio/longbet-iv-review/benchmarks/encouragement/evaluation_part_a.csv", index=False)
    df_b.to_csv("/home/ignacio/longbet-iv-review/benchmarks/encouragement/evaluation_part_b.csv", index=False)
    print("\nBenchmark results saved successfully!")


if __name__ == "__main__":
    main()
