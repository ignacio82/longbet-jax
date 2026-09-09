"""Replay half/full parameter ESS before deciding whether more sampling helps."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from build_repair_report import parameter_diagnostics
from repair_comparison import write_json


def run(output, variant, draws):
    records = []
    for path in sorted(output.glob("*/parameters.npz")):
        record = json.loads((path.parent / "result.json").read_text())
        if record["status"] != "complete" or record["variant"] != variant or record["draws"] != draws:
            continue
        with np.load(path, allow_pickle=False) as archive:
            full = parameter_diagnostics(archive)
            half = {k: (v[:, :draws // 2] if v.ndim >= 2 and v.shape[:2] == (record["chains"], draws) else v)
                    for k, v in archive.items()}
        half = parameter_diagnostics(half)
        result = {"run": record["run_id"], "parameters": {}}
        for k, d in full.items():
            growth = {kind: np.array([c[kind] for c in d["cells"]])
                      / np.array([c[kind] for c in half[k]["cells"]]) for kind in ("ess_bulk", "ess_tail")}
            failed = [{**c, "bulk_growth": growth["ess_bulk"][j], "tail_growth": growth["ess_tail"][j]}
                      for j, c in enumerate(d["cells"]) if c["rhat"] > 1.01 or c["ess_bulk"] < 400 or c["ess_tail"] < 400]
            result["parameters"][k] = {"max_rhat": d["max_rhat"], "min_bulk_ess": d["min_bulk_ess"],
                "min_tail_ess": d["min_tail_ess"], "failed_cells": failed,
                "min_growth": {kind: np.min(v) for kind, v in growth.items()},
                "median_growth": {kind: np.median(v) for kind, v in growth.items()}}
        records.append(result)
    write_json(output / f"parameter-growth-{variant}-k{draws}.json", records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "repair_results")
    p.add_argument("--variant", default="direct_joint")
    p.add_argument("--draws", type=int, default=2000)
    args = p.parse_args()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, [4, 5])
    run(args.output, args.variant, args.draws)
