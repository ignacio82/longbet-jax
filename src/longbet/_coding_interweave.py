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

"""Experimental exact coding/GP reparameterization; benchmark opt-in only.

For the supported exposure convention, z=0 iff S=0. Define effective weights
a[0]=b0*beta[0], a[1:]=b1*beta[1:]. Hold a fixed and update coding under its
conditional PRIOR in this parameterization: the entire likelihood is invariant.
The beta-to-a Jacobian contributes |b0|^-1 |b1|^-(d-1). For u=log(abs(b)),
the additional coding Jacobian gives (1-n[j])*u[j], n=(1,d-1).

Enumerate all four coding sign choices exactly, then slice each log magnitude
and their common shift. The common shift preserves the shape of beta and helps
when the GP prior tightly couples the two magnitudes.
The slice reference is N(0,1); its log density is removed from the target before
calling the generic elliptical slice routine. The GP prior remains exactly the
one encoded by K_chol. This move explores a different coordinate system from
the collapsed forest block; it leaves current within-grid effects invariant.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular

from longbet._collapsed_exposure import elliptical_slice
from longbet._multi_state import split_multi_chain_fields


def interweave_coding_gp(key, beta, coding, K_chol, coding_sd):
    """Return beta/coding with the same effective weights and exact prior law."""
    beta, coding, K_chol = [jnp.asarray(v, jnp.float64) for v in (beta, coding, K_chol)]
    groups = (jnp.arange(beta.size) > 0).astype(jnp.int32)
    weights = beta*coding[groups]
    W = weights[:, None]*(groups[:, None] == jnp.arange(2)[None, :])
    V = solve_triangular(K_chol, W, lower=True)
    counts = jnp.array([1, beta.size-1], jnp.float64)
    u = jnp.log(jnp.abs(coding))
    ks, km = random.split(key)
    signs = jnp.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], jnp.float64)
    inverse = signs*jnp.exp(-u)
    energy = jnp.sum((inverse@V.T)**2, axis=1)
    sign = signs[random.categorical(ks, -.5*energy)]
    evaluations = jnp.int32(0)
    for j in range(3):
        kd, ka = random.split(random.fold_in(km, j))
        current = u[j] if j < 2 else u.mean()
        def position(value):
            return u.at[j].set(value) if j < 2 else u+value-current
        def stats(value):
            v = position(value)
            b = sign*jnp.exp(v)
            inverse_b = sign*jnp.exp(-v)
            target = -.5*(jnp.sum(b*b)/coding_sd**2+jnp.sum((V@inverse_b)**2))
            target += jnp.sum((1-counts)*v)
            return (target+.5*value**2,)
        updated, _, _, used = elliptical_slice(ka, current,
            random.normal(kd, dtype=jnp.float64), stats)
        u = position(updated)
        evaluations += used
    new_coding = sign*jnp.exp(u)
    return weights/new_coding[groups], new_coding, evaluations


@jax.jit
def coding_interweave_single_step(key, state):
    children = list(state.states)
    for m, s in enumerate(children):
        if not (s.sample_beta and s.adaptive_coding):
            continue
        # This restriction is data-dependent, not a state-dependent choice of
        # different kernels. Unusual custom exposure conventions are skipped.
        supported = jnp.all((s.z_vec == 1) == (s.exposure_idx > 0))
        coding = jnp.array([s.b0, s.b1], jnp.float64)
        def move(child):
            beta, bc, _ = interweave_coding_gp(random.fold_in(key, m), child.beta,
                coding, child.K_chol, jnp.float64(child.sigma_b))
            beta, bc = beta.astype(jnp.float32), bc.astype(jnp.float32)
            old_w = jnp.where(child.z_vec == 1, child.b1, child.b0)*child.beta[child.exposure_idx]
            new_w = jnp.where(child.z_vec == 1, bc[1], bc[0])*beta[child.exposure_idx]
            residual = jnp.where(child.obs_mask, child.resid+(old_w-new_w)*child.nu_fit, 0.)
            return eqx.tree_at(lambda t: (t.beta, t.b0, t.b1, t.resid), child,
                                (beta, bc[0], bc[1], residual))
        children[m] = jax.lax.cond(supported & jnp.all(coding != 0), move, lambda c: c, s)
    return eqx.tree_at(lambda t: t.states, state, tuple(children))


def coding_interweave_step(key, state):
    if not state.has_chain_axis:
        return coding_interweave_single_step(key, state)
    per, constants = split_multi_chain_fields(state)
    def one(k, p):
        result = coding_interweave_single_step(k, eqx.combine(p, constants))
        return split_multi_chain_fields(result)[0]
    return eqx.combine(jax.vmap(one)(random.split(key, state.num_chains), per), constants)
