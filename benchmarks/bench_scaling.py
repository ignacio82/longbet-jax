"""Scaling benchmark for the LongBet JAX engine.

**Every timed section blocks on its result.** JAX dispatches asynchronously, so
a timer that stops when ``fit()`` returns measures queue latency, not sampling:
on one panel that reported 2.8 seconds for work that took 154. The cost then
reappears in whichever call blocks first, which makes prediction look like the
bottleneck when it is not. ``jax.block_until_ready`` is not optional here.

Reports **seconds per effective draw of the ATT**, not seconds per sweep.
Sweeps-per-second rewards a sampler for producing correlated output faster,
which is exactly the trap this engine has to avoid claiming its way out of.

Compile time is reported separately from sampling time because it is fixed
rather than proportional: on a small panel XLA compilation dominates the entire
run, and the total is what a user actually waits for.

Run with ``python benchmarks/bench_scaling.py --help`` for options.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from longbet import LongBet, LongBetConfig, available_devices, get_att


def generate_panel(n: int, T: int = 20, p: int = 5, seed: int = 42):
    """Staggered-adoption panel with an exposure-dependent effect."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p)).astype(np.float32)
    t = np.arange(1, T + 1, dtype=np.float32)

    adopt = rng.integers(3, T - 1, size=n)
    adopt = np.where(rng.binomial(1, 0.5, size=n).astype(bool), adopt, 10**6)
    z = np.zeros((n, T), dtype=np.float32)
    for i in range(n):
        if adopt[i] <= T:
            z[i, adopt[i] - 1:] = 1.0
    s = np.where(z == 1, np.maximum(t[None, :] - (adopt[:, None] - 1), 0), 0)

    gamma = rng.normal(0.0, 0.5, size=n).astype(np.float32)
    y = (
        (0.5 * x[:, 0] + gamma)[:, None]
        + 0.05 * t[None, :]
        + 1.2 * np.sqrt(s)
        + rng.normal(0.0, 0.3, size=(n, T))
    ).astype(np.float32)
    return x, y, z, t


def benchmark(n, T, num_sweeps, num_burnin, num_trees, num_chains, device):
    """Fit once and report compile, sample and predict time plus ATT ESS."""
    x, y, z, t = generate_panel(n, T)
    config = LongBetConfig(
        num_sweeps=num_sweeps, num_burnin=num_burnin, num_chains=num_chains,
        num_trees_pr=num_trees, num_trees_trt=num_trees,
        device=device, random_seed=42,
    )

    # Compile time is fixed rather than proportional, so on a small panel it is
    # most of what a user waits for and must be reported separately. A one-sweep
    # fit pays for compilation; the full fit then reuses the cache and measures
    # sampling alone.
    probe = LongBetConfig(**{**config.to_dict(), "num_sweeps": 1, "num_burnin": 0})
    t0 = time.perf_counter()
    warm = LongBet(probe)
    warm.fit(y=y, x=x, z=z, t=t)
    jax.block_until_ready(warm.trace.beta)
    compile_time = time.perf_counter() - t0

    model = LongBet(config)
    t0 = time.perf_counter()
    model.fit(y=y, x=x, z=z, t=t)
    jax.block_until_ready(model.trace.beta)   # see the module docstring
    sample_time = time.perf_counter() - t0
    cold_fit_time = compile_time + sample_time

    t0 = time.perf_counter()
    pred = model.predict(x=x, z=z, t=t, summary_only=True)
    att = get_att(pred)
    predict_time = time.perf_counter() - t0

    stab = pred.stability(warn=False)
    ess = float(stab.summary["ess_median"])
    total = cold_fit_time + predict_time

    return {
        "N": n, "T": T, "M": n * T,
        "num_chains": num_chains, "num_sweeps": num_sweeps,
        "device": device,
        "compile_time_s": round(compile_time, 3),
        "sample_time_s": round(sample_time, 3),
        "cold_fit_time_s": round(cold_fit_time, 3),
        "predict_time_s": round(predict_time, 3),
        "total_time_s": round(total, 3),
        "iterations": config.total_iterations(),
        "ms_per_iteration": round(1000.0 * sample_time / config.total_iterations(), 2),
        "att_ess_median": round(ess, 1),
        "att_rhat_max": round(float(stab.summary["rhat_max"]), 4),
        "seconds_per_effective_att_draw": round(total / ess, 4) if ess > 0 else None,
        "att": [round(float(v), 4) for v in att["att"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 2500, 10000],
                        help="panel sizes N to sweep")
    parser.add_argument("--periods", type=int, default=20)
    parser.add_argument("--sweeps", type=int, default=200)
    parser.add_argument("--burnin", type=int, default=100)
    parser.add_argument("--trees", type=int, default=20)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "benchmark_results.json")
    args = parser.parse_args()

    print(f"Devices visible to JAX: {', '.join(available_devices())}")
    print(f"{'N':>7} {'M':>9} {'compile':>8} {'sample':>8} {'ms/iter':>8} "
          f"{'predict':>8} {'total':>8} {'ATT ESS':>8} {'s/eff draw':>11}")
    print("-" * 84)

    results = []
    for n in args.sizes:
        r = benchmark(n, args.periods, args.sweeps, args.burnin,
                      args.trees, args.chains, args.device)
        results.append(r)
        print(f"{r['N']:>7} {r['M']:>9,} {r['compile_time_s']:>8.2f} "
              f"{r['sample_time_s']:>8.2f} {r['ms_per_iteration']:>8.1f} "
              f"{r['predict_time_s']:>8.2f} "
              f"{r['total_time_s']:>8.2f} {r['att_ess_median']:>8.1f} "
              f"{r['seconds_per_effective_att_draw']:>11.4f}")

    payload = {
        "config": vars(args) | {"out": str(args.out)},
        "devices": available_devices(),
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
