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

"""Inspect coding angles conditional on saved states; this is not inference."""
import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from load_multi_benchmark_state import load_state
from longbet._collapsed_exposure import (_pack, coding_sufficient_statistics,
    coding_statistics, collapsed_statistics)
from longbet._multi_state import split_multi_chain_fields
from longbet._shared_forest import enable_x64, observed_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state, report = load_state(args.directory, args.data)
    per, constants = split_multi_chain_fields(state)
    records, arrays = [], {}
    with enable_x64(True):
        for c in range(state.num_chains):
            chain = eqx.combine(jax.tree.map(lambda x: x[c], per), constants)
            observed = jnp.stack([s.obs_mask for s in chain.states])
            omega = observed_precision(observed, chain.gamma_loadings,
                                       jnp.stack([s.sigma2 for s in chain.states]))
            raw = jnp.stack([s.resid for s in chain.states]).astype(jnp.float64)
            for m, s in enumerate(chain.states):
                base, is_trt, _, _, _, count, _ = _pack(s, chain.shared_forest, m, 128)
                if int(count) > 128:
                    raise ValueError('profile capacity exceeded')
                precision = omega[:, m, m]
                score = jnp.einsum('ik,ki->i', omega[:, m, :], raw)
                beta = s.beta[s.exposure_idx].astype(jnp.float64)
                w = jnp.where(s.z_vec == 1, s.b1, s.b0)*beta
                old_mean = s.alpha*s.mu_fit+w*s.nu_fit+s.gamma[s.unit_idx]
                response = score/jnp.where(precision > 0, precision, 1.)+old_mean-s.alpha*s.forest.offset
                sd = jnp.sqrt(s.sigma_gamma2) if s.random_intercept else 0.
                if not s.random_intercept:
                    response -= s.gamma[s.unit_idx]
                response = jnp.where(s.obs_mask, response, 0.)
                sufficient = jax.jit(coding_sufficient_statistics, static_argnums=(10,))(
                    base, is_trt, s.alpha, beta, s.z_vec, response,
                    s.forest_nu.offset, precision, s.unit_idx, sd, s.N_units)
                coding = jnp.array([s.b0, s.b1], jnp.float64)
                actual = coding_statistics(coding, sufficient)
                direct = collapsed_statistics(base*jnp.where(is_trt, w[:, None], s.alpha),
                    response-w*s.forest_nu.offset, precision, s.unit_idx, sd, s.N_units)
                np.testing.assert_allclose(actual[-1], direct[-1], atol=1e-6)
                angle = jnp.arctan2(coding[1], coding[0])
                theta = angle+jnp.linspace(-jnp.pi/2, jnp.pi/2, 257)
                points = jnp.linalg.norm(coding)*jnp.column_stack((jnp.cos(theta), jnp.sin(theta)))
                def evaluate(b):
                    stats = coding_statistics(b, sufficient)
                    return stats[-1]
                likelihood = np.asarray(jax.jit(lambda pts: jax.lax.map(evaluate, pts))(points))
                current = float(actual[-1])
                name = f'chain_{c}_{report["parameter_outcome_order"][m]}'
                arrays[name+'_theta'] = np.asarray(theta)
                arrays[name+'_log_likelihood'] = likelihood
                best = int(np.argmax(likelihood))
                row = dict(chain=c, outcome=report['parameter_outcome_order'][m],
                    coding=np.asarray(coding).tolist(), leaves=int(count),
                    current_angle=float(angle), best_angle=float(theta[best]),
                    improvement=float(likelihood.max()-current),
                    range=float(np.ptp(likelihood)),
                    grid_points_within_2_log_units=int(np.sum(likelihood > likelihood.max()-2)),
                    grid_points=257,
                    direct_integral_error=float(actual[-1]-direct[-1]))
                records.append(row)
                print(json.dumps(row), flush=True)
    np.savez_compressed(args.output/'profiles.npz', **arrays)
    (args.output/'report.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
