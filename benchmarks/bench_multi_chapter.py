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

"""Reproduce the archived chapter comparison without changing its simulation.

Input: the chapter-input.npz exported by the chapter review, with x, z, ze, t,
col (ZERO-based), gmv, hours, complaint, truth_<outcome>, listings, fulfilled.
Truth for continuous outcomes is on the LOG scale; binary truth is a risk
difference. All outcomes use the SAME intervention scenario and target column.

Example:
  python benchmarks/bench_multi_chapter.py --data chapter-input.npz \
      --output audit/joint --mode multi --coding fixed --burnin 8000 \
      --draws 500 --skip 4

This is a recovery/convergence benchmark, not a repeated-simulation calibration
study. Do not use a low RMSE to override failed convergence diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np

from longbet import (LongBet, LongBetConfig, LongBetMulti, compute_ess,
                     compute_rhat, effect_draws)


def summarize(draws, truth, groups, chains, retained):
    """Diagnose draw-wise subgroup averages, preserving the chain boundaries."""
    result = {"rmse": float(np.sqrt(np.mean((draws.mean(-1)-truth)**2))),
              "sign_accuracy": float(np.mean(np.sign(draws.mean(-1)) == np.sign(truth))),
              "groups": {}}
    for label, mask in groups.items():
        values = draws[mask].mean(0).reshape(chains, retained)
        target = float(truth[mask].mean())
        bulk = float(compute_ess(values, method="bulk"))
        tail = float(compute_ess(values, method="tail", prob=(.025, .975)))
        mean_ess = float(compute_ess(values, method="mean"))
        rhat = float(compute_rhat(values))
        result["groups"][label] = {
            "n": int(mask.sum()), "truth": target,
            "mean": float(values.mean()),
            "interval": np.quantile(values, [.025, .975]).tolist(),
            "p_correct_sign": float(np.mean(values*np.sign(target)>0)),
            "rhat": rhat, "ess_bulk": bulk, "ess_tail": tail,
            "ess_mean": mean_ess,
            "mcse_mean": float(values.std(ddof=1)/np.sqrt(mean_ess)),
            "diagnostics_pass": bool(np.isfinite([rhat, bulk, tail]).all()
                                     and rhat <= 1.01 and min(bulk, tail) >= 400),
            "chain_means": values.mean(-1).tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("multi", "scalar"), default="multi")
    parser.add_argument("--coding", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--burnin", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--shared-trees", type=int, default=0)
    parser.add_argument("--shared-variance-fraction", type=float, default=.5)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--joint-gaussian", action="store_true",
                        help="Experimental exact joint GP/unit-intercept Gibbs block")
    parser.add_argument("--joint-location", action="store_true",
                        help="Also block the global prognostic-forest shift; requires --joint-gaussian")
    parser.add_argument("--interweave-unit-variance", action="store_true")
    parser.add_argument("--interweave-coding", action="store_true",
                        help="Experimental coding/GP reparameterization preserving effective weights")
    parser.add_argument("--change-prognostic", action="store_true",
                        help="Experimental root-rule changes and swaps every 10 sweeps")
    parser.add_argument("--joint-prognostic", action="store_true",
                        help="Experimental exact block: prognostic leaves with b0/b1")
    parser.add_argument("--joint-forest-pair", action="store_true",
                        help="Experimental exact block: paired prognostic/treatment tree leaves")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--save-topology", action="store_true",
                        help="Save forest rules, proposal diagnostics, and final topology")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the multi posterior archive for independent prediction checks")
    parser.add_argument("--fixed-topology", type=Path,
                        help="Diagnostic conditional posterior at reference chain 0's topology")
    parser.add_argument("--collapsed-exposure", action="store_true",
                        help="Experimental GP proposal integrating all forest leaves and units")
    parser.add_argument("--collapsed-every", type=int, default=10)
    parser.add_argument("--collapsed-capacity", type=int, default=128)
    parser.add_argument("--collapsed-scale", type=float, default=.3,
                        help="pCN scale in [0,1]; zero is a joint-leaves/units-only control")
    parser.add_argument("--collapsed-proposal", choices=("pcn", "coding", "joint"), default="pcn",
                        help="GP pCN, coding elliptical slice, or joint coding/GP elliptical slice")
    parser.add_argument("--collapsed-tree", action="store_true",
                        help="Also propose a prognostic root with ALL forests and units integrated")
    parser.add_argument("--refresh-trees", action="store_true",
                        help="Whole private prognostic/treatment tree prior proposals with ALL leaves and units integrated")
    parser.add_argument("--sigma-prior-a", type=float,
                        help="Explicit proper inverse-gamma innovation-variance shape; changes the model, requires --sigma-prior-b")
    parser.add_argument("--sigma-prior-b", type=float,
                        help="Explicit proper inverse-gamma innovation-variance scale; changes the model, requires --sigma-prior-a")
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--no-sur", action="store_true")
    parser.add_argument("--no-exposure-splits", action="store_true")
    parser.add_argument("--no-calendar-trt", action="store_true",
                        help="Experimental model restriction: no calendar-time treatment splits (exposure still allowed)")
    parser.add_argument("--fixed-beta", action="store_true")
    parser.add_argument("--treatment-only", action="store_true",
                        help="Experimental model change: fixed b0=0, b1=1 (multi only)")
    args = parser.parse_args()
    if args.chains < 2:
        parser.error("at least two chains are required for this diagnostic benchmark")
    if args.shared_trees and args.mode != "multi":
        parser.error("--shared-trees requires --mode multi")
    if args.joint_gaussian and args.mode != "multi":
        parser.error("--joint-gaussian requires --mode multi")
    if args.interweave_unit_variance and args.mode != "multi":
        parser.error("--interweave-unit-variance requires --mode multi")
    if args.interweave_coding and args.mode != "multi":
        parser.error("--interweave-coding requires --mode multi")
    if args.change_prognostic and (args.mode != "multi" or args.fixed_topology):
        parser.error("--change-prognostic requires unrestricted multi topology")
    if args.joint_prognostic and args.mode != "multi":
        parser.error("--joint-prognostic requires --mode multi")
    if args.joint_prognostic and args.coding != "adaptive":
        parser.error("--joint-prognostic blocks in b0/b1, so it needs adaptive coding")
    if args.joint_forest_pair and args.mode != "multi":
        parser.error("--joint-forest-pair requires --mode multi")
    if args.joint_location and not args.joint_gaussian:
        parser.error("--joint-location requires --joint-gaussian")
    if args.treatment_only and args.mode != "multi":
        parser.error("--treatment-only requires --mode multi")
    if args.no_calendar_trt and args.mode != "multi":
        parser.error("--no-calendar-trt requires --mode multi")
    if (args.save_topology or args.fixed_topology) and args.mode != "multi":
        parser.error("topology diagnostics require --mode multi")
    if args.save_model and args.mode != "multi":
        parser.error("--save-model requires --mode multi")
    if args.collapsed_exposure and args.mode != "multi":
        parser.error("--collapsed-exposure requires --mode multi")
    if args.collapsed_tree and (not args.collapsed_exposure or args.fixed_topology):
        parser.error("--collapsed-tree requires --collapsed-exposure and unrestricted topology")
    if args.refresh_trees and (args.mode != "multi" or args.fixed_topology):
        parser.error("--refresh-trees requires unrestricted multi topology")
    if (args.sigma_prior_a is None) != (args.sigma_prior_b is None):
        parser.error("set both --sigma-prior-a and --sigma-prior-b for the changed prior")
    if args.sigma_prior_a is not None and not all(
            np.isfinite(v) and v > 0 for v in (args.sigma_prior_a, args.sigma_prior_b)):
        parser.error("explicit innovation-variance prior parameters must be finite and positive")
    if args.collapsed_every < 1 or args.collapsed_capacity < 1 or not 0 <= args.collapsed_scale <= 1:
        parser.error("collapsed period/capacity must be positive and scale must be in [0,1]")
    if any((args.output / name).exists() for name in ("draws.npz", "report.json", "model.npz")):
        parser.error("output already contains a benchmark; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    types = {"gmv": "continuous", "hours": "continuous", "complaint": "binary"}
    groups = {"good": (d["listings"] >= 80) & (d["fulfilled"] == 1),
              "bad": (d["listings"] <= 25) & (d["fulfilled"] == 0)}
    if not all(mask.any() for mask in groups.values()):
        parser.error("the input must contain both prespecified subgroups")
    column = int(d["col"])
    if not 0 <= column < d["ze"].shape[1]:
        parser.error("col must be a zero-based evaluation-panel column")
    settings = dict(num_burnin=args.burnin, num_sweeps=args.draws,
        n_skip=args.skip, num_chains=args.chains, num_trees_pr=20,
        num_trees_trt=20, lambda_knl=2, device="cpu",
        max_depth_pr=args.max_depth, max_depth_trt=args.max_depth,
        adaptive_coding=args.coding == "adaptive", num_shared_trees=args.shared_trees,
        shared_variance_fraction=args.shared_variance_fraction)
    settings.update(random_intercept=not args.no_intercept,
                    sur=not args.no_sur, split_time_trt=not args.no_exposure_splits,
                    sample_beta=not args.fixed_beta)
    if args.sigma_prior_a is not None:
        settings.update(sigma_prior_a=args.sigma_prior_a, sigma_prior_b=args.sigma_prior_b)
    if args.mode == "multi":
        try:
            LongBetConfig(**settings).validate_multi_variance_prior(tuple(types.values()))
        except ValueError as error:
            parser.error(str(error))
    if args.treatment_only:
        settings['adaptive_coding'] = False
    source = Path(__file__).resolve().parents[1] / "src" / "longbet"
    fingerprint = hashlib.sha256()
    for path in sorted(source.glob("*.py")):
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())
    report = {"mode": args.mode, "settings": settings, "base_seed": args.seed,
              "diagnostic_version": 2, "tail_probabilities": [.025, .975],
              "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "python_source_sha256": fingerprint.hexdigest(),
              "experimental_joint_gaussian": args.joint_gaussian,
              "experimental_joint_location": args.joint_location,
              "experimental_unit_interweave": args.interweave_unit_variance,
              "experimental_coding_interweave": args.interweave_coding,
              "experimental_change_prognostic": args.change_prognostic,
              "experimental_joint_prognostic": args.joint_prognostic,
              "experimental_joint_forest_pair": args.joint_forest_pair,
              "experimental_treatment_only": args.treatment_only,
              "experimental_no_calendar_trt": args.no_calendar_trt,
              "fixed_topology": str(args.fixed_topology) if args.fixed_topology else None,
              "fixed_topology_sha256": (hashlib.sha256(args.fixed_topology.read_bytes()).hexdigest()
                                        if args.fixed_topology else None),
              "experimental_collapsed_exposure": args.collapsed_exposure,
              "collapsed_every": args.collapsed_every,
              "collapsed_capacity": args.collapsed_capacity,
              "collapsed_scale": args.collapsed_scale,
              "collapsed_proposal": args.collapsed_proposal,
              "experimental_collapsed_tree": args.collapsed_tree,
              "experimental_refresh_trees": args.refresh_trees,
              "explicit_proper_variance_prior": args.sigma_prior_a is not None,
              "outcomes": {}, "fit_seconds": {}}
    multi = None
    collapsed_records = []
    change_records = []
    collapsed_tree_records = []
    refresh_records = []
    if args.mode == "multi":
        # Benchmark-local opt-in; do not silently change the public sampler.
        from longbet import _multi_loop, _multi_model
        from longbet._shared_forest import enable_x64
        original_driver = _multi_model.run_multi_longbet_mcmc
        if args.fixed_topology:
            from topology_diagnostic import install_fixed_topology_hooks
            install_fixed_topology_hooks()
        if args.no_calendar_trt:
            original_init = _multi_model.init_multi_longbet
            def restricted_init(**kw):
                # This benchmark supplies x only: the unified columns are x,t,s.
                # Restrict BEFORE initialization, so proposal/cutpoint caches and
                # the shared forest all use the same variable support.
                assert kw['max_split_nu'].shape == (d['x'].shape[1]+2,)
                kw['max_split_nu'] = kw['max_split_nu'].at[d['x'].shape[1]].set(0)
                return original_init(**kw)
            _multi_model.init_multi_longbet = restricted_init
        if (args.joint_gaussian or args.interweave_unit_variance
                or args.joint_prognostic or args.joint_forest_pair or args.collapsed_exposure
                or args.interweave_coding or args.change_prognostic or args.refresh_trees):
            from longbet._joint_gaussian import joint_gaussian_step
            from longbet._joint_prognostic import joint_prognostic_step
            from longbet._joint_forest_pair import joint_forest_pair_step
            from longbet._unit_interweave import unit_interweave_step
            if args.interweave_coding:
                from longbet._coding_interweave import coding_interweave_step
            if args.change_prognostic:
                from longbet._change_prognostic import change_prognostic_step
                def record_change(info):
                    change_records.append(np.asarray(info).copy())
            if args.refresh_trees:
                from longbet._refresh_trees import refresh_trees_step
                def record_refresh(info):
                    refresh_records.append(np.asarray(info).copy())
            if args.collapsed_exposure:
                from longbet._collapsed_exposure import collapsed_exposure_step
                def record_collapsed(info):
                    collapsed_records.append(np.asarray(info).copy())
                def record_collapsed_tree(info):
                    collapsed_tree_records.append(np.asarray(info).copy())
            original_step = _multi_loop.multi_step
            def step(key, state, sweep_index):
                state = original_step(key,state,sweep_index)
                if args.joint_gaussian:
                    state = joint_gaussian_step(jax.random.fold_in(key,733),
                        state,location=args.joint_location)
                if args.joint_prognostic:
                    state = joint_prognostic_step(jax.random.fold_in(key,735),state)
                if args.joint_forest_pair:
                    state = joint_forest_pair_step(jax.random.fold_in(key,736),state)
                if args.interweave_unit_variance:
                    state = unit_interweave_step(jax.random.fold_in(key,734),state)
                if args.interweave_coding:
                    state = coding_interweave_step(jax.random.fold_in(key,738),state)
                if args.change_prognostic:
                    def change(s):
                        updated, info = change_prognostic_step(jax.random.fold_in(key,739), s)
                        jax.debug.callback(record_change, info, ordered=True)
                        return updated
                    state = jax.lax.cond((sweep_index+1) % 10 == 0, change, lambda s: s, state)
                if args.collapsed_exposure:
                    def collapse(s):
                        updated, info = collapsed_exposure_step(jax.random.fold_in(key,737), s,
                            capacity=args.collapsed_capacity, proposal_scale=args.collapsed_scale,
                            proposal=args.collapsed_proposal)
                        jax.debug.callback(record_collapsed, info, ordered=True)
                        if args.collapsed_tree:
                            updated, tree_info = collapsed_exposure_step(jax.random.fold_in(key,740),
                                updated, capacity=args.collapsed_capacity, proposal='tree')
                            jax.debug.callback(record_collapsed_tree, tree_info, ordered=True)
                        return updated
                    state = jax.lax.cond((sweep_index+1) % args.collapsed_every == 0,
                                         collapse, lambda s: s, state)
                if args.refresh_trees:
                    def refresh(s):
                        updated, info = refresh_trees_step(jax.random.fold_in(key,741), s,
                            capacity=args.collapsed_capacity)
                        jax.debug.callback(record_refresh, info, ordered=True)
                        return updated
                    state = jax.lax.cond((sweep_index+1) % args.collapsed_every == 0,
                                         refresh, lambda s: s, state)
                return state
            _multi_loop.multi_step = step
        def driver(*pos, **kw):
            if args.fixed_topology:
                from topology_diagnostic import initialize_common_topology
                kw['state'] = initialize_common_topology(kw['state'], args.fixed_topology)
            if args.treatment_only:
                import equinox as eqx
                import jax.numpy as jnp
                state = kw['state']
                # Forests are empty at initialization, so no residual correction
                # is needed. This is a distinct model, NOT the same-posterior block.
                children = tuple(eqx.tree_at(lambda s:(s.b0,s.b1),s,
                    (jnp.zeros_like(s.b0),jnp.ones_like(s.b1))) for s in state.states)
                kw['state'] = eqx.tree_at(lambda s:s.states,state,children)
            def progress(i,n,state):
                jax.block_until_ready(state.gamma_loadings)
                print(f"batch {i+1}/{n}; elapsed {time.monotonic()-start:.1f}s",flush=True)
            kw.update(inner_loop_length=args.progress_every, callback=progress)
            with enable_x64(True if (args.joint_gaussian or args.interweave_unit_variance
                                     or args.joint_prognostic or args.joint_forest_pair
                                     or args.collapsed_exposure or args.interweave_coding
                                     or args.change_prognostic or args.refresh_trees)
                            else jax.config.x64_enabled):
                result = original_driver(*pos,**kw)
            if args.save_topology:
                from topology_diagnostic import save_final_topology
                save_final_topology(result.final_state, args.output / 'final_topology.npz')
            return result
        _multi_model.run_multi_longbet_mcmc = driver
        start = time.monotonic()
        multi = LongBetMulti(LongBetConfig(**settings, random_seed=args.seed)).fit(
            {k: d[k] for k in types}, d["x"], d["z"], t=d["t"], outcome=types)
        jax.block_until_ready(multi["complaint"].trace)
        report["fit_seconds"]["multi"] = time.monotonic()-start
        if args.save_model:
            multi.save(args.output / "model.npz")
        report['parameter_outcome_order'] = [multi.outcome_names[i] for i in multi.order]
        parameters = {'sur_loadings':np.asarray(multi.trace.gamma_loadings)}
        for i,tr in enumerate(multi.trace.traces):
            for name in ('beta','b0','b1','alpha','sigma2','sigma_gamma2'):
                parameters[f'internal_{i}_{name}'] = np.asarray(getattr(tr,name))
        np.savez_compressed(args.output / 'parameters.npz',**parameters)
        if args.collapsed_exposure:
            jax.effects_barrier()
            records = np.asarray(collapsed_records)
            np.savez_compressed(args.output / 'collapsed.npz', info=records)
            report['collapsed_diagnostics'] = {
                'fields': ['attempted', 'accepted', 'log_ratio', 'leaf_count', 'likelihood_evaluations'],
                'shape': list(records.shape),
                'attempted_by_chain_outcome': records[..., 0].sum(0).tolist(),
                'accepted_by_chain_outcome': records[..., 1].sum(0).tolist(),
                'max_leaves_by_chain_outcome': records[..., 3].max(0).tolist()}
            if args.collapsed_tree:
                records = np.asarray(collapsed_tree_records)
                np.savez_compressed(args.output / 'collapsed_tree.npz', info=records)
                report['collapsed_tree_diagnostics'] = {
                    'fields': ['attempted', 'root_changed', 'log_likelihood_ratio', 'leaf_count', 'likelihood_evaluations'],
                    'shape': list(records.shape),
                    'root_changes_by_chain_outcome': records[..., 1].sum(0).tolist()}
        if args.save_topology:
            from topology_diagnostic import summarize_topology
            report['topology'] = summarize_topology(multi, args.output)
        if args.refresh_trees:
            jax.effects_barrier()
            records = np.asarray(refresh_records)
            np.savez_compressed(args.output / 'refresh.npz', info=records)
            report['refresh_diagnostics'] = {
                'fields': ['attempted', 'accepted', 'log_ratio', 'current_leaves',
                           'candidate_leaves', 'supported', 'shape_changed', 'root_changed'],
                'forest_order': ['prognostic', 'private_treatment'],
                'shape': list(records.shape),
                'accepted_by_chain_outcome_forest': records[..., 1].sum(0).tolist(),
                'shape_changes_by_chain_outcome_forest': records[..., 6].sum(0).tolist(),
                'root_changes_by_chain_outcome_forest': records[..., 7].sum(0).tolist()}
        if args.change_prognostic:
            jax.effects_barrier()
            records = np.asarray(change_records)
            np.savez_compressed(args.output / 'change.npz', info=records)
            report['change_diagnostics'] = {
                'fields': ['allowed', 'accepted', 'log_ratio', 'root_changed', 'kind'],
                'shape': list(records.shape),
                'root_changes_by_chain_outcome': records[..., 3].sum((0, 3)).tolist()}
    kept = {}
    for m, (name, outcome) in enumerate(types.items()):
        if multi is None:
            start = time.monotonic()
            fit = LongBet(LongBetConfig(**settings, outcome=outcome,
                random_seed=args.seed+m+1)).fit(d[name], d["x"], d["z"], t=d["t"])
            jax.block_until_ready(fit.trace)
            report["fit_seconds"][name] = time.monotonic()-start
        else:
            fit = multi[name]
        pred = fit.predict(d["x"], d["ze"], t=d["t"])
        raw = effect_draws(pred)[:, column, :]
        truth = d["truth_"+name]
        draws = raw if outcome == "binary" else np.expm1(raw)
        truth = truth if outcome == "binary" else np.expm1(truth)
        kept[name] = draws
        report["outcomes"][name] = summarize(draws, truth, groups, args.chains, args.draws)
        np.savez_compressed(args.output / "draws.npz", **kept)
        (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(name, json.dumps(report["outcomes"][name]), flush=True)
        del pred, fit


if __name__ == "__main__":
    main()
