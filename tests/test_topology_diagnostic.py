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

"""Ensure the topology control really conditions on trees, with valid residuals."""
import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from longbet import LongBetMulti
from longbet._shared_forest import enable_x64
from test_shared_forest import _panel, _cfg

path = Path(__file__).resolve().parents[1] / 'benchmarks' / 'topology_diagnostic.py'
spec = importlib.util.spec_from_file_location('topology_diagnostic', path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_common_topology_preserves_dispersed_parameters_and_fixed_rules(tmp_path, monkeypatch):
    from bartz.mcmcstep import _step as bartz
    from longbet import _shared_forest, _multi_model
    from longbet._multi_step import multi_step
    from longbet._multi_state import split_multi_chain_fields
    from longbet._forest_cache import current_forest_fit

    x, z, y, types = _panel()
    cfg = _cfg(num_chains=2, num_burnin=10, num_sweeps=4)
    learned = LongBetMulti(cfg).fit(y, x, z, outcome=types)
    target = tmp_path / 'topology.npz'
    diagnostic.save_final_topology(learned.state, target)

    original_driver = _multi_model.run_multi_longbet_mcmc
    starts = []
    def driver(*args, **kwargs):
        fresh = kwargs['state']
        initialized = diagnostic.initialize_common_topology(fresh, target)
        starts.append(initialized)
        for before, after in zip(fresh.states, initialized.states):
            for field in ('b0', 'b1', 'beta', 'gamma', 'resid', 'mu_fit', 'nu_fit'):
                np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
            assert not np.array_equal(np.asarray(after.beta[0]), np.asarray(after.beta[1]))
        kwargs['state'] = initialized
        return original_driver(*args, **kwargs)
    monkeypatch.setattr(_multi_model, 'run_multi_longbet_mcmc', driver)
    # Register original values with pytest before installing both hooks.
    monkeypatch.setattr(bartz, 'propose_moves', bartz.propose_moves)
    monkeypatch.setattr(_shared_forest, 'shared_tree_step', _shared_forest.shared_tree_step)
    diagnostic.install_fixed_topology_hooks()
    jax.clear_caches()  # the preceding pilot was compiled without the hooks
    try:
        fit = LongBetMulti(cfg).fit(y, x, z, outcome=types)
        initial = starts[0]
        for m, child in enumerate(fit.state.states):
            for field in ('forest', 'forest_nu'):
                before = getattr(initial.states[m], field)
                after = getattr(child, field)
                np.testing.assert_array_equal(after.split_tree, before.split_tree)
                mask = np.asarray(before.split_tree) != 0
                np.testing.assert_array_equal(np.asarray(after.var_tree)[mask], np.asarray(before.var_tree)[mask])
                np.testing.assert_array_equal(after.grow_acc_count, 0)
                np.testing.assert_array_equal(after.prune_acc_count, 0)
                assert np.any(np.asarray(after.leaf_tree) != 0)
            np.testing.assert_array_equal(child.forest.split_tree[0], child.forest.split_tree[1])
        np.testing.assert_array_equal(fit.state.shared_forest.split_tree, initial.shared_forest.split_tree)

        def errors(state):
            values = []
            for m, child in enumerate(state.states):
                mu = current_forest_fit(child.forest)
                nu = current_forest_fit(child.forest_nu) + state.shared_forest.fit[m]
                fitted = (child.alpha*mu + jnp.where(child.z_vec == 1, child.b1, child.b0)
                          * child.beta[child.exposure_idx]*nu + child.gamma[child.unit_idx])
                response = child.z if child.outcome_type_str == 'binary' else child.y
                values.append(jnp.max(jnp.abs(child.resid - jnp.where(child.obs_mask, response-fitted, 0))))
                values.append(jnp.max(jnp.abs(child.mu_fit-mu)))
                values.append(jnp.max(jnp.abs(child.nu_fit-nu)))
            return jnp.stack(values)
        per, shared = split_multi_chain_fields(fit.state)
        with enable_x64(True):
            got = jax.vmap(lambda p: errors(eqx.combine(p, shared)))(per)
        assert np.max(np.asarray(got)) < 3e-4
        np.testing.assert_array_equal(fit.state.states[0].sigma2, 1.)
        report = diagnostic.summarize_topology(fit, tmp_path)
        for outcome in report.values():
            for forest in outcome.values():
                assert forest['root_changes_per_chain'] == [0, 0]
                assert forest['changed_rules_between_saves_chain_means'] == [0., 0.]
    finally:
        # Clear compiled closures before pytest restores the hooks.
        jax.clear_caches()
