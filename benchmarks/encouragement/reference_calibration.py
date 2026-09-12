"""Repeated-randomization smoke study for the single-wave reference estimator.

This exercises finite-population coverage of the two ITTs and normal-test-inversion
Wald sets; it is not validation of the future Bayesian wrapper or all IV designs.
Run: python benchmarks/encouragement/reference_calibration.py --replications 500
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from longbet import encouragement_effects


def run(n: int = 400, replications: int = 500, seed: int = 2026) -> dict:
    if n < 40 or replications < 2:
        raise ValueError("Use at least 40 units and two replications.")
    results = []
    for k, (label, share) in enumerate((("strong", 0.4), ("weak", 0.05), ("zero", 0.0))):
        rng = np.random.default_rng(seed + k)
        # Hold potential outcomes and types fixed across assignments: coverage
        # is over complete randomization, not repeated draws from a model prior.
        x = rng.normal(size=n)
        n_always, n_compliers = int(0.15 * n), int(share * n)
        always = np.arange(n) < n_always
        complier = (np.arange(n) >= n_always) & (np.arange(n) < n_always + n_compliers)
        theta = 1 + 0.4 * x
        baseline = 0.7 * always - 0.4 * (~always & ~complier) + x + rng.normal(size=n)
        base_panel = baseline[:, None] + np.array([0.0, 0.1, 0.2])
        d0 = np.tile(always[:, None], (1, 3)).astype(float)
        d1 = d0.copy()
        d1[complier, 1:] = 1
        # Always-takers have identical exposure under both assignments.
        always_effect = always[:, None] * theta[:, None] * np.sqrt([1, 2, 3])
        y0 = base_panel + always_effect
        y1 = y0.copy()
        y1[complier, 1:] += theta[complier, None] * np.sqrt([1, 2])
        truth_y = (y1 - y0).mean(axis=0)[1:]
        truth_d = (d1 - d0).mean(axis=0)[1:]
        truth_wald = truth_y / truth_d if n_compliers else None
        cover_y = np.zeros(2)
        cover_d = np.zeros(2)
        cover_wald = np.zeros(2)
        negative = np.zeros(2)
        set_types: dict[str, int] = {}
        for _ in range(replications):
            assigned = np.zeros(n, dtype=bool)
            assigned[rng.permutation(n)[:n // 2]] = True
            z = np.zeros((n, 3))
            z[assigned, 1:] = 1
            d = np.where(assigned[:, None], d1, d0)
            y = np.where(assigned[:, None], y1, y0)
            result = encouragement_effects(y, d, z)
            cover_y += (result.itt_y_lower <= truth_y) & (truth_y <= result.itt_y_upper)
            cover_d += (result.itt_d_lower <= truth_d) & (truth_d <= result.itt_d_upper)
            negative += result.itt_d < 0
            for j, row in enumerate(result.to_dict("records")):
                kind = row["wald_set_type"]
                set_types[kind] = set_types.get(kind, 0) + 1
                if truth_wald is not None:
                    value = truth_wald[j]
                    cover_wald[j] += any(row[f"wald_lower_{r}"] <= value <= row[f"wald_upper_{r}"]
                                         for r in (1, 2))
        def coverage(counts):
            p = np.asarray(counts) / replications
            return {"coverage": p.tolist(), "mcse": np.sqrt(p * (1 - p) / replications).tolist()}
        results.append({
            "regime": label, "n_compliers": n_compliers,
            "truth_itt_y": truth_y.tolist(), "truth_itt_d": truth_d.tolist(),
            "truth_wald": None if truth_wald is None else truth_wald.tolist(),
            "itt_y": coverage(cover_y), "itt_d": coverage(cover_d),
            "wald": None if truth_wald is None else coverage(cover_wald),
            "negative_first_stage_rate": (np.asarray(negative) / replications).tolist(),
            "confidence_set_counts_across_horizons": set_types,
        })
    return {"n": n, "replications": replications, "seed": seed,
            "method": "normal_ar", "nominal_pointwise_coverage": 0.95,
            "scope": "fixed populations; individual complete randomization; immediate uptake",
            "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run(args.n, args.replications, args.seed), indent=2, allow_nan=False))
