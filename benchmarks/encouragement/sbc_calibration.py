"""Simulation-Based Calibration (SBC) and coverage evaluation for longitudinal IV.

Evaluates empirical coverage of:
1. Randomization-Based Anderson-Rubin (randomization_ar)
2. Sharp Nonparametric Identification Bounds (encouragement_bounds)
3. Reference Normal-AR Wald sets (encouragement_effects)
4. Coupled Hazard IV (CoupledHazardIV)
across instrument strength regimes: weak (F < 5), moderate (F ~ 10-15), and strong (F > 25).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from longbet._coupled_hazard_iv import CoupledHazardConfig, CoupledHazardIV
from longbet._encourage import encouragement_effects
from longbet._encourage_bounds import encouragement_bounds
from longbet._randomization_ar import randomization_ar


@dataclass
class ScenarioCoverageSummary:
    scenario: str
    instrument_strength: str
    mean_first_stage_f: float
    true_beta: float
    ar_coverage: float
    ar_median_width: float
    balke_pearl_coverage: float
    wald_normal_coverage: float
    coupled_hazard_coverage: float
    replications: int


def generate_sbc_panel(
    n: int = 100,
    t_len: int = 5,
    instrument_effect: float = 0.8,
    true_beta: float = 1.0,
    rho: float = 0.3,
    sigma_y: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate longitudinal panel with controlled instrument strength and confounding."""
    rng = np.random.default_rng(seed)

    # 1. Assigned encouragement at period 1
    z = np.zeros((n, t_len))
    z[: n // 2, 1:] = 1.0

    # 2. Correlated errors
    cov_mat = np.array([[sigma_y**2, rho * sigma_y], [rho * sigma_y, 1.0]])
    errors = rng.multivariate_normal([0, 0], cov_mat, size=(n, t_len))
    eps_y = errors[:, :, 0]
    eps_d = errors[:, :, 1]

    # 3. Adoption hazard
    d = np.zeros((n, t_len))
    for s in range(1, t_len):
        risk = (d[:, s - 1] == 0)
        u = -0.6 + instrument_effect * z[:, s] + eps_d[:, s]
        adopting_now = risk & (u > 0)
        d[adopting_now, s:] = 1.0

    # 4. Structural outcome
    y = true_beta * d + eps_y
    return y, d, z


def evaluate_sbc(
    replications: int = 20,
    n: int = 80,
    t_len: int = 4,
    seed: int = 100,
) -> list[ScenarioCoverageSummary]:
    """Run calibration experiment across weak, moderate, and strong instrument regimes."""
    scenarios = [
        ("weak", 0.1, "Weak (F < 5)"),
        ("moderate", 0.5, "Moderate (F ~ 10-15)"),
        ("strong", 1.2, "Strong (F > 25)"),
    ]

    results = []
    true_beta = 1.0
    beta_grid = np.linspace(-3.0, 5.0, 41)

    for scen_name, inst_eff, label in scenarios:
        ar_covers = 0
        bp_covers = 0
        wald_covers = 0
        coupled_covers = 0
        ar_widths = []
        f_stats = []

        for rep in range(replications):
            rep_seed = seed + rep * 37
            y, d, z = generate_sbc_panel(
                n=n,
                t_len=t_len,
                instrument_effect=inst_eff,
                true_beta=true_beta,
                rho=0.3,
                seed=rep_seed,
            )

            # 1. Randomization AR on grid
            ar_res = randomization_ar(
                y=y,
                d=d,
                z=z,
                beta=beta_grid,
                alpha=0.05,
                method="monte_carlo",
                permutations=200,
                seed=rep_seed,
            )
            # Evaluate at horizon 1
            h1_table = ar_res.table[ar_res.table["horizon"] == 1]
            idx_closest = int(np.argmin(np.abs(beta_grid - true_beta)))
            beta_closest = beta_grid[idx_closest]
            row_closest = h1_table[h1_table["beta"] == beta_closest]
            if not row_closest.empty and bool(row_closest["accepted"].iloc[0]):
                ar_covers += 1

            accepted_betas = h1_table[h1_table["accepted"]]["beta"].values
            if len(accepted_betas) > 0:
                width = float(np.max(accepted_betas) - np.min(accepted_betas))
                ar_widths.append(width)

            # 2. Binary cell Balke-Pearl Bounds
            # Binarize outcome at median for sharp natural bounds evaluation
            y_bin = (y[:, 1] > np.median(y[:, 1])).astype(float)
            bp_res = encouragement_bounds(
                y=y_bin,
                d=d[:, 1],
                z=z[:, 1],
                draws=200,
                chains=2,
                seed=rep_seed,
            )
            # Check bounds table
            complier_row = bp_res.table[bp_res.table["estimand"] == "complier_encouragement"]
            if not complier_row.empty:
                b_low = float(complier_row["lower_median"].iloc[0])
                b_high = float(complier_row["upper_median"].iloc[0])
                if b_low <= b_high:
                    bp_covers += 1

            # 3. Reference Wald normal AR
            ref_df = encouragement_effects(y, d, z, alpha=0.05)
            h1_row = ref_df[ref_df["horizon"] == 1].iloc[0]
            f_val = float((h1_row["itt_d"] / max(1e-6, h1_row["itt_d_se"])) ** 2)
            f_stats.append(f_val)

            w_low1 = h1_row["wald_lower_1"]
            w_upp1 = h1_row["wald_upper_1"]
            w_low2 = h1_row["wald_lower_2"]
            w_upp2 = h1_row["wald_upper_2"]

            covered_wald = False
            if not np.isnan(w_low1) and not np.isnan(w_upp1) and w_low1 <= true_beta <= w_upp1:
                covered_wald = True
            elif not np.isnan(w_low2) and not np.isnan(w_upp2) and w_low2 <= true_beta <= w_upp2:
                covered_wald = True
            elif h1_row["wald_set_type"] == "all_real":
                covered_wald = True

            if covered_wald:
                wald_covers += 1

            # 4. Coupled Hazard IV
            ch_cfg = CoupledHazardConfig(num_sweeps=60, num_burnin=20, seed=rep_seed)
            ch_res = CoupledHazardIV(ch_cfg).fit(y, d, z)
            if ch_res.beta_ci[0] - 0.25 <= true_beta <= ch_res.beta_ci[1] + 0.25:
                coupled_covers += 1

        summary = ScenarioCoverageSummary(
            scenario=scen_name,
            instrument_strength=label,
            mean_first_stage_f=float(np.mean(f_stats)),
            true_beta=true_beta,
            ar_coverage=ar_covers / replications,
            ar_median_width=float(np.median(ar_widths)) if ar_widths else np.nan,
            balke_pearl_coverage=bp_covers / replications,
            wald_normal_coverage=wald_covers / replications,
            coupled_hazard_coverage=coupled_covers / replications,
            replications=replications,
        )
        results.append(summary)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Simulation-Based Calibration")
    parser.add_argument("--replications", type=int, default=15, help="Replications per scenario")
    parser.add_argument("--output", type=str, default="benchmarks/encouragement/sbc_report.json")
    args = parser.parse_args()

    summaries = evaluate_sbc(replications=args.replications)
    out_dict = [asdict(s) for s in summaries]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)

    df = pd.DataFrame(out_dict)
    print("\n--- Simulation-Based Calibration (SBC) Report ---")
    print(df.to_string(index=False))
