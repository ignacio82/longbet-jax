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

"""Plot complete fifth-pass subgroup traces as sampler diagnostics.

Each row is one prespecified business-scale subgroup effect; columns are the
two model seeds. This only reads completed reports/draw archives. It does not
fit a model or certify inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path('/home/ignacio/longbet-bug-audit/refresh-pass')
DATA = Path('/home/ignacio/longbet-bug-audit/topology-pass/chapter-input.npz')
OUTCOMES = ('gmv', 'hours', 'complaint')
ROWS = [(outcome, group) for outcome in OUTCOMES for group in ('good', 'bad')]
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_run(directory, data, data_hash):
    """Validate completion, input identity and existing chain-major ordering."""
    report_path, draws_path = directory / 'report.json', directory / 'draws.npz'
    if not report_path.exists() or not draws_path.exists():
        raise ValueError(f'{directory.name}: report/draw archive is incomplete')
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'{directory.name}: report.json is incomplete') from error
    if set(report.get('outcomes', {})) != set(OUTCOMES):
        raise ValueError(f'{directory.name}: all three outcomes must be complete before plotting')
    if report['data_sha256'] != data_hash:
        raise ValueError(f'{directory.name}: report data hash does not match the frozen input')
    if (report['mode'] != 'multi' or report['parameter_outcome_order']
            != ['complaint', 'gmv', 'hours']):
        raise ValueError(f'{directory.name}: unexpected mode or internal outcome order')
    if not report.get('experimental_refresh_trees') or report.get('fixed_topology'):
        raise ValueError(f'{directory.name}: expected an unrestricted whole-tree refresh run')
    expected_seed = {'refresh-314159': 314159, 'refresh-271828': 271828}.get(directory.name)
    if expected_seed is not None and report['base_seed'] != expected_seed:
        raise ValueError(f'{directory.name}: report seed does not match its prespecified run')
    settings = report['settings']
    prior = (settings.get('sigma_prior_a', 0.), settings.get('sigma_prior_b', 0.))
    explicit = bool(report.get('explicit_proper_variance_prior'))
    if explicit:
        if not all(np.isfinite(v) and v > 0 for v in prior):
            raise ValueError(f'{directory.name}: invalid explicit variance prior')
        prior_label = f'IG({prior[0]:g},{prior[1]:g}); changed model'
    else:
        if prior != (0., 0.):
            raise ValueError(f'{directory.name}: unlabelled variance-prior change')
        prior_label = 'original prior; improper chapter target'
    chains, retained, skip = (settings[name] for name in ('num_chains', 'num_sweeps', 'n_skip'))
    if chains != 4 or retained < 4 or skip < 1:
        raise ValueError(f'{directory.name}: expected four chains and at least four retained draws')
    groups = dict(good=(data['listings'] >= 80) & (data['fulfilled'] == 1),
                  bad=(data['listings'] <= 25) & (data['fulfilled'] == 0))
    series, truth = {}, {}
    with np.load(draws_path, allow_pickle=False) as archive:
        if set(archive.files) != set(OUTCOMES):
            raise ValueError(f'{directory.name}: draw archive has incomplete outcomes')
        for outcome in OUTCOMES:
            draws = archive[outcome]
            if draws.shape != (data['x'].shape[0], chains * retained) or not np.isfinite(draws).all():
                raise ValueError(f'{directory.name}/{outcome}: invalid draw shape or values')
            targets = data[f'truth_{outcome}']
            if outcome != 'complaint':
                targets = np.expm1(targets)
            for group, mask in groups.items():
                if not mask.any():
                    raise ValueError(f'Prespecified subgroup {group} is empty')
                values = draws[mask].mean(0).reshape(chains, retained)
                target = float(targets[mask].mean())
                summary = report['outcomes'][outcome]['groups'][group]
                if summary['n'] != int(mask.sum()):
                    raise ValueError(f'{directory.name}/{outcome}/{group}: subgroup membership changed')
                np.testing.assert_allclose(values.mean(1), summary['chain_means'], rtol=3e-5, atol=1e-7,
                    err_msg=f'{directory.name}/{outcome}/{group}: draw/report chain ordering mismatch')
                np.testing.assert_allclose(target, summary['truth'], rtol=3e-5, atol=1e-7)
                # Saved continuous effects are already expm1(log effects).
                # Multiplication converts them to %, and risk differences to pp.
                series[outcome, group] = values * 100
                truth[outcome, group] = target * 100
    return dict(name=directory.name, report=report, series=series, truth=truth,
        variance_prior=prior, variance_prior_label=prior_label,
        sweeps=np.arange(1, retained + 1) * skip,
        report_sha256=sha256(report_path), draws_sha256=sha256(draws_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--runs', nargs=2, default=['refresh-314159', 'refresh-271828'],
                        metavar=('FIRST_RUN', 'SECOND_RUN'))
    parser.add_argument('--output', type=Path,
                        help='Output directory; defaults to --root. Files are refresh-traces.png and .pdf.')
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as archive:
        data = dict(archive)
    data_hash = sha256(args.data)
    # Finish validation for BOTH archives before creating any figure or output.
    runs = [read_run(args.root / name, data, data_hash) for name in args.runs]
    output = args.output if args.output is not None else args.root
    output.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(6, 2, figsize=(13, 16), sharey='row',
                                 sharex='col')
        fig.subplots_adjust(top=.89, bottom=.05, left=.095, right=.99,
                            hspace=.36, wspace=.07)
        for col, run in enumerate(runs):
            for row, (outcome, group) in enumerate(ROWS):
                ax = axes[row, col]
                for chain, values in enumerate(run['series'][outcome, group]):
                    ax.plot(run['sweeps'], values, color=COLORS[chain],
                            linewidth=.75, alpha=.75)
                ax.axhline(run['truth'][outcome, group], color='#202020',
                           linestyle='--', linewidth=1.15)
                summary = run['report']['outcomes'][outcome]['groups'][group]
                statistics = (f'R-hat {summary["rhat"]:.3f}  |  '
                              f'bulk ESS {summary["ess_bulk"]:.1f}  |  '
                              f'tail ESS {summary["ess_tail"]:.1f}')
                title = statistics
                if row == 0:
                    title = (f'Seed {run["report"]["base_seed"]} · {run["variance_prior_label"]}'
                             f'\n{statistics}')
                ax.set_title(title, fontsize=9, loc='left')
                ax.set_xlim(0, run['sweeps'][-1])
                ax.grid(axis='y', color='#dddddd', linewidth=.5)
                if col == 0:
                    label = {'gmv': 'GMV', 'hours': 'Hours', 'complaint': 'Complaint'}[outcome]
                    subgroup = 'Benefit subgroup' if group == 'good' else 'Harm subgroup'
                    unit = 'Relative effect (%)' if outcome != 'complaint' else 'Risk difference (pp)'
                    ax.set_ylabel(f'{label} · {subgroup}\n{unit}')
                if row == len(ROWS) - 1:
                    ax.set_xlabel('Retained sweeps after burn-in')
        legend = [Line2D([0], [0], color=color, linewidth=1.4, label=f'Chain {chain + 1}')
                  for chain, color in enumerate(COLORS)]
        legend.append(Line2D([0], [0], color='#202020', linestyle='--', label='Simulated truth'))
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.53, .95),
                   ncol=5, frameon=False)
        fig.suptitle('Whole-tree refresh: sampler diagnostics\n'
                     'Prespecified subgroup draw means; posterior inference is not certified',
                     fontsize=14, y=.987)
        description = ('Diagnostic traces only; no inference claim. Input SHA256: '
                       + data_hash + '. Runs: ' + ', '.join(args.runs))
        fig.savefig(output / 'refresh-traces.png', dpi=160,
                    metadata={'Title': 'Whole-tree refresh sampler diagnostics', 'Description': description})
        fig.savefig(output / 'refresh-traces.pdf',
                    metadata={'Title': 'Whole-tree refresh sampler diagnostics', 'Subject': description})
        plt.close(fig)
    provenance = dict(data=str(args.data), data_sha256=data_hash,
        script_sha256=sha256(__file__), rows=[list(pair) for pair in ROWS],
        chain_colors=list(COLORS),
        x_axis='(1,...,retained_draw_count) * thinning_interval; completed sweeps after burn-in',
        qualification='Sampler diagnostics only. No inference or calibration claim.',
        runs=[dict(name=run['name'], seed=run['report']['base_seed'],
                   variance_prior=run['variance_prior'],
                   variance_prior_label=run['variance_prior_label'],
                   report_sha256=run['report_sha256'], draws_sha256=run['draws_sha256']) for run in runs])
    (output / 'refresh-traces.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(output / 'refresh-traces.png')
    print(output / 'refresh-traces.pdf')


if __name__ == '__main__':
    main()
