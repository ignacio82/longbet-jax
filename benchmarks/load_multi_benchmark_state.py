"""Load an experimental final state without rerunning MCMC.

This requires matching LongBet state layout/initialization, the exact benchmark
data, and its report settings. The Equinox file is an internal diagnostic
checkpoint, not a public fit/prediction archive. No chains are fitted here.

Example:
  python benchmarks/load_multi_benchmark_state.py RUN_DIRECTORY --data INPUT.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from longbet import LongBetConfig, LongBetMulti, _multi_model
from longbet._shared_forest import enable_x64


def load_state(directory, data_path):
    directory, data_path = Path(directory), Path(data_path)
    report = json.loads((directory / 'report.json').read_text())
    if report['mode'] != 'multi':
        raise ValueError('this loader requires a multi-outcome benchmark')
    if any(report.get(k) for k in ('experimental_no_calendar_trt', 'experimental_treatment_only')):
        raise ValueError('model-restriction checkpoints require their original initialization hooks')
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != report['data_sha256']:
        raise ValueError('data hash does not match the checkpoint report')
    with np.load(data_path, allow_pickle=False) as archive:
        data = dict(archive)
    captured = []
    original = _multi_model.run_multi_longbet_mcmc

    class Loaded(Exception):
        pass

    def driver(*args, **kwargs):
        captured.append(eqx.tree_deserialise_leaves(directory / 'final_state.eqx', kwargs['state']))
        raise Loaded

    _multi_model.run_multi_longbet_mcmc = driver
    try:
        settings = dict(report['settings'], random_seed=report['base_seed'])
        types = {'gmv': 'continuous', 'hours': 'continuous', 'complaint': 'binary'}
        with enable_x64(True):
            LongBetMulti(LongBetConfig(**settings)).fit({k: data[k] for k in types},
                data['x'], data['z'], t=data['t'], outcome=types)
    except Loaded:
        return captured[0], report
    finally:
        _multi_model.run_multi_longbet_mcmc = original
    raise RuntimeError('benchmark initialization did not reach the checkpoint hook')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    state, report = load_state(args.directory, args.data)
    jax.block_until_ready(state)
    print(json.dumps({'internal_order': report['parameter_outcome_order'],
        'chains': state.num_chains, 'units': state.states[0].N_units,
        'coding_by_outcome': [{'b0': np.asarray(s.b0).tolist(), 'b1': np.asarray(s.b1).tolist()}
                              for s in state.states]}, indent=2))


if __name__ == '__main__':
    main()
