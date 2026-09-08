"""Validate fifth-pass archives and compare the two prespecified model seeds.

No model is fitted. Partial benchmark reports are skipped until all three
outcomes and their archives exist; --require-complete instead fails. Optional
checkpoint validation loads saved states and reconstructs fits from actual
rules using an independent NumPy traversal, without cached memberships/fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import arviz as az
import numpy as np

from longbet import compute_ess, compute_rhat


DEFAULT_ROOT = Path('/home/ignacio/longbet-bug-audit/refresh-pass')
DEFAULT_DATA = Path('/home/ignacio/longbet-bug-audit/topology-pass/chapter-input.npz')
OUTCOMES = ('gmv', 'hours', 'complaint')
INTERNAL_ORDER = ['complaint', 'gmv', 'hours']
SEEDS = (314159, 271828)
REFRESH_FIELDS = ['attempted', 'accepted', 'log_ratio', 'current_leaves',
                  'candidate_leaves', 'supported', 'shape_changed', 'root_changed']


class IncompleteRun(RuntimeError):
    """An archive still lacks a complete benchmark result."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_report(directory):
    path = directory / 'report.json'
    if not path.exists():
        raise IncompleteRun(f'{directory.name}: no report.json')
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise IncompleteRun(f'{directory.name}: report.json is being written') from error
    if set(report.get('outcomes', {})) != set(OUTCOMES):
        raise IncompleteRun(f'{directory.name}: fewer than three complete outcomes')
    needed = ['draws.npz', 'parameters.npz', 'topology.npz']
    if report.get('experimental_refresh_trees'):
        needed.append('refresh.npz')
    if report.get('experimental_collapsed_exposure'):
        needed.append('collapsed.npz')
    for filename in needed:
        if not (directory / filename).exists():
            raise IncompleteRun(f'{directory.name}: missing {filename}')
    return report, needed


def effect_diagnostics(values, truth=None):
    """Diagnose draw-wise averages with their existing chain boundaries."""
    bulk = float(compute_ess(values, method='bulk'))
    tail = float(compute_ess(values, method='tail', prob=.05))
    rhat = float(compute_rhat(values))
    mean_ess = float(az.ess(values, method='mean'))
    mcse = float(az.mcse(values, method='mean'))
    mean = float(values.mean())
    interval = np.quantile(values, [.025, .975]).tolist()
    finite = all(np.isfinite(v) for v in (rhat, bulk, tail))
    result = dict(mean=mean, interval=interval, chain_means=values.mean(1).tolist(),
        rhat=rhat, ess_bulk=bulk, ess_tail=tail, ess_mean=mean_ess,
        mcse_mean=mcse, mcse_method="arviz.mcse(method='mean')",
        old_bulk_ess_mcse_approximation=float(values.std(ddof=1) / np.sqrt(bulk)),
        convergence_gate=bool(finite and rhat <= 1.01 and bulk >= 400 and tail >= 400),
        decision_threshold=0.,
        mcse_relative_to_abs_mean=mcse / abs(mean) if mean else None,
        mcse_qualification='Monte Carlo precision estimates are not certified while convergence fails.')
    if truth is not None:
        result.update(truth=float(truth),
            covers_truth=bool(interval[0] <= truth <= interval[1]),
            p_correct_sign=float(np.mean(values * np.sign(truth) > 0)))
    return result


def _validate_parameters(directory, chains, retained):
    with np.load(directory / 'parameters.npz', allow_pickle=False) as archive:
        require(archive['sur_loadings'].shape == (chains, retained, 3, 3),
                'SUR parameter chain/draw axes do not match report')
        for name in archive.files:
            array = archive[name]
            require(array.shape[:2] == (chains, retained) and np.isfinite(array).all(),
                    f'invalid parameter array {name}')
        for m, name in enumerate(INTERNAL_ORDER):
            for parameter in ('beta', 'b0', 'b1', 'alpha', 'sigma2', 'sigma_gamma2'):
                key = f'internal_{m}_{parameter}'
                require(key in archive.files, f'missing parameter {key}')
                if parameter == 'beta':
                    require(archive[key].ndim == 3 and archive[key].shape[2] >= 1,
                            f'invalid {key} shape')
                else:
                    require(archive[key].shape == (chains, retained), f'invalid {key} shape')
            require(np.all(archive[f'internal_{m}_sigma2'] > 0), 'nonpositive innovation variance')
            if name == 'complaint':
                require(np.all(archive[f'internal_{m}_sigma2'] == 1),
                        'binary identifying variance is not exactly one')
        require(np.all(np.triu(archive['sur_loadings']) == 0),
                'SUR loadings are not strictly lower triangular')


def _root_summary(var, split):
    rules = np.where(split[..., 1] != 0,
                     (var[..., 1].astype(np.int64) + 1) * 65536 + split[..., 1], 0)
    changes = rules[:, 1:] != rules[:, :-1]
    unchanged = ~changes.any(axis=1)
    return dict(trees=int(rules.shape[-1]),
        changes_between_saves_by_chain=changes.sum((1, 2)).tolist(),
        never_changed_roots_by_chain=unchanged.sum(1).tolist(),
        never_changed_root_indices_by_chain=[np.flatnonzero(row).tolist() for row in unchanged])


def topology_diagnostics(directory, report):
    settings = report['settings']
    chains, retained = settings['num_chains'], settings['num_sweeps']
    shared_count = settings['num_shared_trees']
    output, shared_reference = {}, None
    with np.load(directory / 'topology.npz', allow_pickle=False) as archive:
        for outcome in OUTCOMES:
            output[outcome] = {}
            for label, tree_count, depth in (
                    ('mu', settings['num_trees_pr'], settings['max_depth_pr']),
                    ('nu', settings['num_trees_trt'], settings['max_depth_trt'])):
                var, split = (archive[f'{outcome}_{label}_{field}'] for field in ('var', 'split'))
                expected = (chains, retained, tree_count, 2**(depth - 1))
                require(var.shape == split.shape == expected, f'{outcome}/{label}: topology shape mismatch')
                item = _root_summary(var, split)
                item['mean_active_leaves_by_chain'] = ((split != 0).sum((-1, -2)) + tree_count).mean(1).tolist()
                if label == 'nu' and shared_count:
                    private_count = tree_count - shared_count
                    item['private'] = _root_summary(var[..., :private_count, :], split[..., :private_count, :])
                    item['shared'] = _root_summary(var[..., private_count:, :], split[..., private_count:, :])
                    sv, ss = var[..., private_count:, :], split[..., private_count:, :]
                    if shared_reference is None:
                        shared_reference = sv.copy(), ss.copy()
                    else:
                        require(np.array_equal(ss, shared_reference[1]), 'shared split rules differ across outcomes')
                        require(np.all((ss == 0) | (sv == shared_reference[0])),
                                'shared variable rules differ across outcomes')
                output[outcome][label] = item
    return dict(outcomes=output,
        qualification='Actual retained tree rules; changes between saves miss reversals and changes during burn-in.')


def _refresh_summary(info, capacity):
    attempted, accepted = info[..., 0] == 1, info[..., 1] == 1
    supported = info[..., 5] == 1
    count_rejection = attempted & ~supported
    overflow_proposal = attempted & (info[..., 4] > capacity)
    capacity_rejection = attempted & supported & (info[..., 4] > capacity)
    mh_rejection = attempted & supported & ~overflow_proposal & ~accepted
    result = dict(records=int(info.shape[0]),
        scheduled_slots=int(attempted.size), attempted=int(attempted.sum()),
        accepted=int(accepted.sum()), count_rejections=int(count_rejection.sum()),
        current_capacity_skips=int((~attempted).sum()),
        candidate_capacity_rejections_after_count_check=int(capacity_rejection.sum()),
        candidate_capacity_overflow_including_count_rejections=int(overflow_proposal.sum()),
        supported_metropolis_rejections=int(mh_rejection.sum()),
        accepted_shape_changes=int(info[..., 6].sum()), accepted_root_changes=int(info[..., 7].sum()))
    require(result['attempted'] == result['accepted'] + result['count_rejections']
            + result['candidate_capacity_rejections_after_count_check']
            + result['supported_metropolis_rejections'], 'refresh rejection categories do not partition attempts')
    result['by_chain_outcome_forest'] = {name: mask.sum(0).tolist() for name, mask in (
        ('attempted', attempted), ('accepted', accepted), ('count_rejections', count_rejection),
        ('current_capacity_skips', ~attempted), ('candidate_capacity_rejections', capacity_rejection),
        ('accepted_shape_changes', info[..., 6]), ('accepted_root_changes', info[..., 7]))}
    result['maximum_current_leaves'] = float(info[..., 3].max()) if info.size else None
    result['maximum_candidate_leaves'] = float(info[..., 4].max()) if info.size else None
    return result


def refresh_diagnostics(directory, report):
    with np.load(directory / 'refresh.npz', allow_pickle=False) as archive:
        require(set(archive.files) == {'info'}, 'unexpected refresh archive fields')
        info = archive['info']
    settings = report['settings']
    total_sweeps = settings['num_burnin'] + settings['num_sweeps'] * settings['n_skip']
    period, capacity = report['collapsed_every'], report['collapsed_capacity']
    expected = (total_sweeps // period, settings['num_chains'], 3, 2, 8)
    require(info.shape == expected, f'refresh shape {info.shape} != {expected}')
    require(report['refresh_diagnostics']['fields'] == REFRESH_FIELDS, 'refresh field semantics changed')
    require(report['refresh_diagnostics']['forest_order'] == ['prognostic', 'private_treatment'],
            'refresh forest order changed')
    require(np.isfinite(info[..., [0, 1, 3, 4, 5, 6, 7]]).all(), 'nonfinite refresh counters')
    require(not np.isnan(info[..., 2]).any() and not np.isposinf(info[..., 2]).any(),
            'invalid refresh likelihood ratios')
    require(np.isin(info[..., [0, 1, 5, 6, 7]], [0, 1]).all(), 'nonbinary refresh flags')
    accepted = info[..., 1] == 1
    require(np.all((info[..., 0] == 1) | (info[..., 3] > capacity)),
            'refresh skipped a current state that fits capacity')
    require(np.all(~accepted | ((info[..., 0] == 1) & (info[..., 5] == 1)
                               & (info[..., 4] <= capacity) & np.isfinite(info[..., 2]))),
            'refresh accepted an unsupported candidate')
    require(np.all(info[..., 6:8] <= info[..., 1, None]), 'refresh changed an unaccepted tree')
    np.testing.assert_array_equal(info[..., 1].sum(0),
                                  report['refresh_diagnostics']['accepted_by_chain_outcome_forest'])
    for field, index in [('shape_changes_by_chain_outcome_forest', 6),
                         ('root_changes_by_chain_outcome_forest', 7)]:
        np.testing.assert_array_equal(info[..., index].sum(0), report['refresh_diagnostics'][field])
    start = settings['num_burnin'] // period
    return dict(fields=REFRESH_FIELDS, internal_outcome_order=INTERNAL_ORDER,
        forest_order=['prognostic', 'private_treatment'], period=period, capacity=capacity,
        retained_phase_start_record=start,
        phase_definition='Records occur at completed sweep numbers period, 2*period, ...; retained phase uses sweep > burn-in.',
        total=_refresh_summary(info, capacity), retained=_refresh_summary(info[start:], capacity))


def _numpy_ids(X, var, split):
    """Traverse heap rules directly, without any saved membership cache."""
    ids = np.ones(X.shape[1], np.int64)
    for _ in range(split.size.bit_length()):
        active = ids < split.size
        active &= split[np.minimum(ids, split.size - 1)] != 0
        if not active.any():
            return ids
        rows = np.flatnonzero(active)
        nodes = ids[rows]
        ids[rows] = 2 * nodes + (X[var[nodes].astype(int), rows] >= split[nodes])
    raise ValueError('tree traversal did not terminate within its heap depth')


def validate_checkpoint(directory, data_path, report):
    import equinox as eqx
    import jax
    from load_multi_benchmark_state import load_state
    from longbet._multi_state import split_multi_chain_fields

    checkpoint = directory / 'final_state.eqx'
    if not checkpoint.exists():
        raise IncompleteRun(f'{directory.name}: no final_state.eqx')
    state, _ = load_state(directory, data_path)
    per, constants = split_multi_chain_fields(state)
    errors, binary_variances = [], []
    for chain in range(state.num_chains):
        single = eqx.combine(jax.tree.map(lambda a: a[chain], per), constants) if state.has_chain_axis else state
        shared = single.shared_forest
        chain_errors, chain_binary = [], {}
        for m, child in enumerate(single.states):
            X = np.asarray(child.X)
            def fit(forest):
                values = np.asarray(forest.leaf_tree, np.float64)
                ids = np.stack([_numpy_ids(X, v, s) for v, s in
                                zip(np.asarray(forest.var_tree), np.asarray(forest.split_tree))])
                return (float(forest.offset) + float(forest.leaf_unit)
                        * np.take_along_axis(values, ids, axis=1).sum(0))
            mu, nu = fit(child.forest), fit(child.forest_nu)
            common, common_error = np.zeros_like(nu), 0.
            if shared is not None:
                ids = np.stack([_numpy_ids(X, v, s) for v, s in
                                zip(np.asarray(shared.var_tree), np.asarray(shared.split_tree))])
                common = np.take_along_axis(np.asarray(shared.leaf_tree[:, m], np.float64), ids, axis=1).sum(0)
                common_error = float(np.max(np.abs(common - np.asarray(shared.fit[m]))))
                nu += common
            weight = np.where(np.asarray(child.z_vec) == 1, float(child.b1), float(child.b0))
            weight *= np.asarray(child.beta, np.float64)[np.asarray(child.exposure_idx)]
            fitted = float(child.alpha) * mu + weight * nu + np.asarray(child.gamma, np.float64)[np.asarray(child.unit_idx)]
            response = np.asarray(child.z if child.outcome_type_str == 'binary' else child.y, np.float64)
            residual = np.where(np.asarray(child.obs_mask), response - fitted, 0.)
            chain_errors.append([float(np.max(np.abs(mu - np.asarray(child.mu_fit)))),
                float(np.max(np.abs(nu - np.asarray(child.nu_fit)))), common_error,
                float(np.max(np.abs(residual - np.asarray(child.resid))))])
            if child.outcome_type_str == 'binary':
                chain_binary[INTERNAL_ORDER[m]] = float(child.sigma2)
                require(float(child.sigma2) == 1., 'checkpoint binary variance is not exactly one')
        errors.append(chain_errors)
        binary_variances.append(chain_binary)
    maximum = float(np.max(errors))
    require(np.isfinite(errors).all() and maximum < 3e-4,
            f'{directory.name}: checkpoint reconstruction error {maximum} exceeds 3e-4')
    result = dict(fields=['mu_fit', 'nu_fit', 'shared_fit', 'raw_residual'],
        internal_order=report['parameter_outcome_order'], errors_by_chain=errors,
        maximum_error=maximum, tolerance=3e-4, binary_variances_by_chain=binary_variances,
        checkpoint_sha256=sha256(checkpoint), reconstruction='Independent NumPy traversal of actual rules; stored memberships and fitted caches are not used to reconstruct means.')
    (directory / 'checkpoint-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def analyze_run(directory, data, data_path, data_hash, *, checkpoints=False):
    report, files = load_report(directory)
    require(report['mode'] == 'multi' and report['parameter_outcome_order'] == INTERNAL_ORDER,
            f'{directory.name}: incompatible outcome order or mode')
    require(report['data_sha256'] == data_hash, f'{directory.name}: wrong input hash')
    require(re.fullmatch('[0-9a-f]{64}', report['python_source_sha256']) is not None,
            'invalid recorded source fingerprint')
    require(not report.get('fixed_topology') and not report.get('experimental_treatment_only')
            and not report.get('experimental_no_calendar_trt'), 'comparison requires original model posterior')
    settings = report['settings']
    chains, retained = settings['num_chains'], settings['num_sweeps']
    require(chains >= 2 and retained >= 4, 'comparison requires multiple chains and at least four draws')
    _validate_parameters(directory, chains, retained)
    groups = dict(good=(data['listings'] >= 80) & (data['fulfilled'] == 1),
                  bad=(data['listings'] <= 25) & (data['fulfilled'] == 0))
    summaries, prefixes, coverage_failures, individual_recovery = {}, {}, [], {}
    with np.load(directory / 'draws.npz', allow_pickle=False) as archive:
        require(set(archive.files) == set(OUTCOMES), 'draw archive is missing an outcome')
        for outcome in OUTCOMES:
            draws = archive[outcome]
            require(draws.shape == (data['x'].shape[0], chains * retained)
                    and np.isfinite(draws).all(), f'{outcome}: invalid effect draw shape or values')
            truth = data[f'truth_{outcome}']
            if outcome != 'complaint':
                truth = np.expm1(truth)
            means = draws.mean(-1)
            intervals = np.quantile(draws, [.025, .975], axis=-1)
            covered = (intervals[0] <= truth) & (truth <= intervals[1])
            wrong_sign = np.sign(means) != np.sign(truth)
            recovery = dict(rmse=float(np.sqrt(np.mean((means - truth)**2))),
                sign_accuracy=float(np.mean(~wrong_sign)),
                pointwise_95pct_coverage=float(covered.mean()),
                coverage_failure_indices=np.flatnonzero(~covered).tolist(),
                wrong_posterior_mean_sign_indices=np.flatnonzero(wrong_sign).tolist())
            for field in ('rmse', 'sign_accuracy'):
                np.testing.assert_allclose(recovery[field], report['outcomes'][outcome][field],
                                            rtol=3e-5, atol=1e-7)
            individual_recovery[outcome] = recovery
            summaries[outcome] = {}
            for label, mask in groups.items():
                values = draws[mask].mean(0).reshape(chains, retained)
                target = float(truth[mask].mean())
                saved = report['outcomes'][outcome]['groups'][label]
                actual = effect_diagnostics(values, target)
                require(saved['n'] == int(mask.sum()), 'group membership differs from frozen input')
                for field in ('truth', 'mean', 'chain_means', 'interval', 'rhat', 'ess_bulk', 'ess_tail'):
                    np.testing.assert_allclose(actual[field], saved[field], rtol=3e-5, atol=1e-7,
                        err_msg=f'{directory.name}/{outcome}/{label}/{field}: report/archive or chain ordering mismatch')
                actual['n'] = int(mask.sum())
                summaries[outcome][label] = actual
                if not actual['covers_truth']:
                    coverage_failures.append(f'{outcome}/{label}')
                lengths = sorted({n for n in (125, 250, 500, retained) if n <= retained})
                prefix = {str(n): effect_diagnostics(values[:, :n], target) for n in lengths}
                if retained >= 8:
                    prefix['last_half'] = effect_diagnostics(values[:, retained // 2:], target)
                if len(lengths) >= 2:
                    prefix['window_growth'] = lengths[-1] / lengths[0]
                    prefix['bulk_ess_growth'] = prefix[str(lengths[-1])]['ess_bulk'] / prefix[str(lengths[0])]['ess_bulk']
                prefixes[f'{outcome}/{label}'] = prefix
    flat = [g for outcome in summaries.values() for g in outcome.values()]
    target = ('proper inverse-gamma innovation-variance model'
              if report.get('explicit_proper_variance_prior') else 'original variance-prior settings')
    result = dict(directory=str(directory), target=target, report=report,
        seed=report['base_seed'], chains=chains, retained=retained,
        burnin=settings['num_burnin'], skip=settings['n_skip'],
        fit_seconds=report['fit_seconds']['multi'], data_sha256=data_hash,
        source_sha256=report['python_source_sha256'],
        archive_sha256={filename: sha256(directory / filename) for filename in ['report.json'] + files},
        chain_order_validation='Chain-major effect draws: reshaped subgroup per-chain means agree with archived report; parameters and topology retain matching explicit chain/draw axes and internal outcome order.',
        summaries=summaries, prefixes=prefixes, coverage_failures=coverage_failures,
        individual_recovery=individual_recovery,
        gates_passed=sum(g['convergence_gate'] for g in flat), gates_total=6,
        worst_rhat=max(g['rhat'] for g in flat), minimum_bulk_ess=min(g['ess_bulk'] for g in flat),
        minimum_tail_ess=min(g['ess_tail'] for g in flat), topology=topology_diagnostics(directory, report))
    if report.get('experimental_refresh_trees'):
        result['refresh'] = refresh_diagnostics(directory, report)
    if checkpoints:
        result['checkpoint'] = validate_checkpoint(directory, data_path, report)
    return result


def validate_paired_metadata(baseline, candidate, *, proper_prior):
    """Require the prespecified single change, allowing omitted historical false flags."""
    old, new = baseline['settings'], candidate['settings']
    defaults = {'sigma_prior_a': 0., 'sigma_prior_b': 0.}
    differences = {key: [old.get(key, defaults.get(key)), new.get(key, defaults.get(key))]
                   for key in set(old) | set(new)
                   if old.get(key, defaults.get(key)) != new.get(key, defaults.get(key))}
    allowed = {'num_burnin', 'num_sweeps', 'n_skip', 'device'}
    if proper_prior:
        allowed |= {'sigma_prior_a', 'sigma_prior_b'}
    require(set(differences) <= allowed, f'paired model settings differ: {differences}')
    flags = {key for key in set(baseline) | set(candidate) if key.startswith('experimental_')}
    flag_differences = {key: [baseline.get(key, False), candidate.get(key, False)]
                        for key in flags if baseline.get(key, False) != candidate.get(key, False)}
    expected_flag_differences = {} if proper_prior else {'experimental_refresh_trees': [False, True]}
    require(flag_differences == expected_flag_differences,
            f'paired experimental flags differ beyond the prespecified change: {flag_differences}')
    collapsed_options = ('collapsed_every', 'collapsed_capacity', 'collapsed_scale', 'collapsed_proposal')
    for option in collapsed_options:
        require(option in baseline and option in candidate and baseline[option] == candidate[option],
                f'paired collapsed option {option} differs: {baseline.get(option)} vs {candidate.get(option)}')
    same_source = baseline['python_source_sha256'] == candidate['python_source_sha256']
    if proper_prior:
        require(same_source, 'proper-prior pair must use identical Python sampler source')
    return dict(settings_differences=differences, experimental_flag_differences=flag_differences,
        matched_collapsed_options={option: baseline[option] for option in collapsed_options},
        identical_python_source=same_source,
        source_qualification=('The prior comparison uses identical sampler source.' if proper_prior else
                              'The refresh transition adds source; both recorded source fingerprints are preserved.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--validate-checkpoints', action='store_true')
    parser.add_argument('--proper-prior', action='store_true',
                        help='Compare proper-SEED under --root to refresh-SEED under sibling refresh-pass; explicitly changes the model')
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as archive:
        data = dict(archive)
    data_hash = sha256(args.data)
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(), data_sha256=data_hash,
        arviz_version=az.__version__, script_sha256=sha256(__file__),
        qualifications=[
            'Fit runtimes are contended and are not fair efficiency comparisons.',
            'The six R-hat/bulk-ESS/tail-ESS gates do not establish calibration or complete chapter validation.',
            'MCSE uses the actual mean statistic through ArviZ; failed convergence invalidates precision assurances.',
            'Nested prefixes and last-half summaries reuse draws and are not independent replicated fits.',
            'The zero sign threshold is reported with MCSE/absolute mean; no substantive decision margin is imposed.'],
        runs={}, incomplete={}, paired_comparisons={})
    kinds = (('refresh', args.root.parent / 'refresh-pass'), ('proper', args.root)) if args.proper_prior else (
        ('coding', args.root.parent / 'coding-pass'), ('refresh', args.root))
    for seed in SEEDS:
        for kind, parent in kinds:
            name = f'{kind}-{seed}'
            try:
                analyzed = analyze_run(parent / name, data, args.data, data_hash,
                                       checkpoints=args.validate_checkpoints)
                require(analyzed['seed'] == seed, f'{name}: report seed does not match directory')
                require(bool(analyzed['report'].get('experimental_refresh_trees')) == (kind != 'coding'),
                        f'{name}: unexpected refresh transition setting')
                if kind == 'proper':
                    require(analyzed['report'].get('explicit_proper_variance_prior')
                            and analyzed['report']['settings'].get('sigma_prior_a') == 2.
                            and analyzed['report']['settings'].get('sigma_prior_b') == 1.,
                            f'{name}: expected prespecified IG(2,1) model')
                else:
                    prior_settings = analyzed['report']['settings']
                    require(not analyzed['report'].get('explicit_proper_variance_prior')
                            and prior_settings.get('sigma_prior_a', 0.) == 0.
                            and prior_settings.get('sigma_prior_b', 0.) == 0.,
                            f'{name}: expected original zero-hyperparameter variance prior')
                result['runs'][name] = analyzed
                print(f'{name}: R-hat={analyzed["worst_rhat"]:.4f}, bulk ESS={analyzed["minimum_bulk_ess"]:.1f}, '
                      f'tail ESS={analyzed["minimum_tail_ess"]:.1f}, gates={analyzed["gates_passed"]}/6', flush=True)
            except IncompleteRun as error:
                result['incomplete'][name] = str(error)
                print(f'Skipping incomplete {name}: {error}', flush=True)
        baseline, candidate = (result['runs'].get(f'{kind}-{seed}') for kind, _ in kinds)
        if baseline is not None and candidate is not None:
            metadata = validate_paired_metadata(baseline['report'], candidate['report'],
                                                proper_prior=args.proper_prior)
            result['paired_comparisons'][str(seed)] = dict(**metadata,
                model_changed=bool(args.proper_prior),
                baseline_gates=baseline['gates_passed'], candidate_gates=candidate['gates_passed'],
                groups={f'{outcome}/{group}': {
                    'mean_difference': candidate['summaries'][outcome][group]['mean'] - baseline['summaries'][outcome][group]['mean'],
                    'baseline': baseline['summaries'][outcome][group],
                    'candidate': candidate['summaries'][outcome][group]}
                    for outcome in OUTCOMES for group in ('good', 'bad')})
    args.root.mkdir(parents=True, exist_ok=True)
    destination = args.root / 'comparison.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=True) + '\n')
    temporary.replace(destination)
    if args.require_complete and result['incomplete']:
        raise SystemExit(f'{len(result["incomplete"])} prespecified runs are incomplete; see {destination}')


if __name__ == '__main__':
    main()
