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

"""Benchmark-only noncentered unit-effect variance interweaving transition."""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from longbet._multi_state import split_multi_chain_fields
from longbet._shared_forest import observed_precision


def unit_scale_move(key, residual, omega, observed, unit, gamma, variance,
                    prior_a, prior_b, outcome, proposal_sd=.2):
    """MH in log SD, holding gamma/sqrt(variance) fixed.

    The gamma-prior normalization cancels its N-coordinate Jacobian. Including
    the variance coordinate and log-scale proposal leaves -2*a*log(c), not
    -(N+2*a)*log(c). Full SUR likelihood feedback is retained.
    """
    kp,ka = random.split(key)
    log_c = proposal_sd*random.normal(kp,(),jnp.float64)
    c = jnp.exp(log_c)
    delta = (c-1)*gamma[unit]
    score = jnp.einsum('ik,ki->i',omega[:,outcome,:],residual)
    log_lik = jnp.sum(delta*score-.5*delta**2*omega[:,outcome,outcome])
    log_prior = -2*prior_a*log_c+prior_b/variance*(-jnp.expm1(-2*log_c))
    accept = jnp.log(random.uniform(ka,(),dtype=jnp.float64))<log_lik+log_prior
    scale = jnp.where(accept,c,1.)
    new_gamma = (gamma*scale).astype(gamma.dtype)
    new_variance = (variance*scale**2).astype(variance.dtype)
    new_residual = residual.at[outcome].add(jnp.where(observed,
        -(new_gamma-gamma)[unit],0.)).astype(residual.dtype)
    return new_residual,new_gamma,new_variance,accept,log_lik+log_prior


@jax.jit
def unit_interweave_single_step(key,state):
    children = list(state.states)
    observed = jnp.stack([s.obs_mask for s in children])
    G = state.gamma_loadings if state.sur_active else jnp.zeros_like(state.gamma_loadings)
    omega = observed_precision(observed,G,jnp.stack([s.sigma2 for s in children]))
    raw = jnp.stack([s.resid for s in children])
    for m,s in enumerate(children):
        if not s.random_intercept:
            continue
        raw,gamma,variance,_,_ = unit_scale_move(random.fold_in(key,m),raw,omega,
            s.obs_mask,s.unit_idx,s.gamma,s.sigma_gamma2,s.gamma_prior_a,s.gamma_prior_b,m)
        children[m] = eqx.tree_at(lambda s:(s.gamma,s.sigma_gamma2),s,(gamma,variance))
    children = tuple(eqx.tree_at(lambda s:s.resid,s,raw[m]) for m,s in enumerate(children))
    return eqx.tree_at(lambda s:s.states,state,children)


def unit_interweave_step(key,state):
    if not state.has_chain_axis:
        return unit_interweave_single_step(key,state)
    per,shared = split_multi_chain_fields(state)
    def one(k,p):
        return split_multi_chain_fields(unit_interweave_single_step(k,eqx.combine(p,shared)))[0]
    return eqx.combine(jax.vmap(one)(random.split(key,state.num_chains),per),shared)
