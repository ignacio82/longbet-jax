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

"""Experimental exact joint GP/unit-intercept Gibbs block.

Not enabled by the public sampler. The benchmark can compose this transition
with an ordinary sweep to test mixing without changing the posterior. Callers
must keep JAX X64 enabled through tracing AND execution. Stored state stays f32.
"""
from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular, block_diag

from longbet._multi_state import split_multi_chain_fields
from longbet._shared_forest import observed_precision
from longbet._ridge import compute_active_leaf_stats


def _solve(chol, rhs):
    return solve_triangular(chol.T, solve_triangular(chol, rhs, lower=True), lower=False)


def joint_statistics(y, omega, weights, exposure, unit, gp_chol, unit_sd, n_units,
                     location_weights=None, location_sd=None):
    """Schur statistics in prior-whitened coordinates; y/weights shape (M,n).

    Exploit equal exposure indices across outcomes and block-diagonal unit
    precision. Storage is O(N M^2 d + M^2 d^2), not O((N M + M d)^2).
    Zero weights or SDs represent inactive, independent dummy coordinates.
    """
    y, omega, weights, gp_chol, unit_sd = [jnp.asarray(a, jnp.float64)
        for a in (y, omega, weights, gp_chol, unit_sd)]
    M, d = gp_chol.shape[:2]
    D = M*d
    score = jnp.einsum('imk,ki->im', omega, y)
    gram = omega * weights.T[:, :, None] * weights.T[:, None, :]
    by_s = jnp.zeros((d, M, M), jnp.float64).at[exposure].add(gram)
    q_raw = jnp.einsum('smk,st->mskt', by_s, jnp.eye(d)).reshape(D,D)
    h_raw = jnp.zeros((d,M), jnp.float64).at[exposure].add(weights.T*score).T.reshape(D)
    cross_raw = jnp.zeros((n_units,d,M,M), jnp.float64).at[unit,exposure].add(
        weights.T[:,:,None]*omega*unit_sd[None,None,:])
    cross_raw = cross_raw.transpose(0,2,1,3).reshape(n_units,D,M)
    unit_q = jnp.eye(M)[None,:,:] + jnp.zeros((n_units,M,M),jnp.float64).at[unit].add(
        unit_sd[None,:,None]*omega*unit_sd[None,None,:])
    unit_h = jnp.zeros((n_units,M),jnp.float64).at[unit].add(score*unit_sd)
    unit_chol = jnp.linalg.cholesky(unit_q)
    L = block_diag(*[gp_chol[m] for m in range(M)])
    if location_weights is not None:
        a = jnp.asarray(location_weights,jnp.float64)
        beta_location = jnp.zeros((d,M,M),jnp.float64).at[exposure].add(
            weights.T[:,:,None]*omega*a.T[:,None,:]).transpose(1,0,2).reshape(D,M)
        location_q = jnp.einsum('imk,mi,ki->mk',omega,a,a)
        q_raw = jnp.block([[q_raw,beta_location],[beta_location.T,location_q]])
        h_raw = jnp.concatenate((h_raw,jnp.sum(a.T*score,axis=0)))
        location_cross = jnp.zeros((n_units,M,M),jnp.float64).at[unit].add(
            a.T[:,:,None]*omega*unit_sd[None,None,:])
        cross_raw = jnp.concatenate((cross_raw,location_cross),axis=1)
        L = block_diag(L,jnp.diag(jnp.asarray(location_sd,jnp.float64)))
        D += M
    cross = jnp.einsum('ab,ubk->uak', L.T, cross_raw)
    inv_cross = jax.vmap(_solve)(unit_chol, cross.transpose(0,2,1))
    inv_h = jax.vmap(_solve)(unit_chol, unit_h)
    q = jnp.eye(D) + L.T@q_raw@L - cross.transpose(1,0,2).reshape(D,-1)@inv_cross.reshape(-1,D)
    h = L.T@h_raw - jnp.einsum('udm,um->d', cross, inv_h)
    q = (q+q.T)*.5
    chol = jnp.linalg.cholesky(q)
    return L, chol, _solve(chol,h), unit_chol, unit_h, cross


def joint_draw(key, stats):
    """Draw whitened GP and unit coordinates from their joint conditional."""
    _, chol, mean, unit_chol, unit_h, cross = stats
    k_gp,k_unit = random.split(key)
    u = mean + solve_triangular(chol.T, random.normal(k_gp, mean.shape, jnp.float64), lower=False)
    h = unit_h-jnp.einsum('udm,d->um',cross,u)
    v_mean = jax.vmap(_solve)(unit_chol,h)
    noise = random.normal(k_unit,unit_h.shape,jnp.float64)
    v = v_mean+jax.vmap(lambda c,z: solve_triangular(c.T,z,lower=False))(unit_chol,noise)
    return u,v


@partial(jax.jit,static_argnames=('location',))
def joint_gaussian_single_step(key, state, *, location=False):
    states = state.states
    first = states[0]
    raw = jnp.stack([s.resid for s in states]).astype(jnp.float64)
    old_beta = jnp.stack([s.beta for s in states])
    old_gamma = jnp.stack([s.gamma for s in states])
    w_actual = jnp.stack([jnp.where(s.z_vec == 1, s.b1, s.b0)*s.nu_fit for s in states])
    active_beta = jnp.array([s.sample_beta for s in states])
    active_gamma = jnp.array([s.random_intercept for s in states])
    w = jnp.where(active_beta[:,None],w_actual,0.).astype(jnp.float64)
    sd = jnp.where(active_gamma, jnp.sqrt(jnp.stack([s.sigma_gamma2 for s in states])),0.)
    obs = jnp.stack([s.obs_mask for s in states])
    G = state.gamma_loadings if state.sur_active else jnp.zeros_like(state.gamma_loadings)
    omega = observed_precision(obs,G,jnp.stack([s.sigma2 for s in states]))
    partial = raw+w*old_beta[:,first.exposure_idx]+jnp.where(active_gamma[:,None],old_gamma[:,first.unit_idx],0.)
    partial = jnp.where(obs,partial,0.)
    old_location,location_sd,location_weights = None,None,None
    leaf_masks = []
    if location:
        centers,sds = [],[]
        for s in states:
            count,_,mask = compute_active_leaf_stats(s.forest.split_tree,s.forest.leaf_tree)
            trees = s.forest.leaf_tree.shape[0]
            # Coordinate along a common physical shift / J in every active leaf.
            # Orthogonal leaf contrasts are held fixed. Its Gaussian prior is
            # independent of those contrasts (conditional on the topology).
            centers.append(trees*jnp.sum(jnp.where(mask,s.forest.leaf_tree*s.forest.leaf_unit,0.))/count)
            sds.append(trees/jnp.sqrt(count*s.forest.leaf_prior_cov_inv))
            leaf_masks.append(mask)
        old_location = jnp.stack(centers)
        location_sd = jnp.stack(sds)
        location_weights = jnp.broadcast_to(jnp.stack([s.alpha for s in states])[:,None],raw.shape)
        partial = jnp.where(obs,partial+location_weights*old_location[:,None],0.)
    stats = joint_statistics(partial,omega,w,first.exposure_idx,first.unit_idx,
        jnp.stack([s.K_chol for s in states]),sd,first.N_units,location_weights,location_sd)
    u,v = joint_draw(key,stats)
    parameters = stats[0]@u
    new_beta = jnp.where(active_beta[:,None],parameters[:old_beta.size].reshape(old_beta.shape),old_beta).astype(jnp.float32)
    new_gamma = jnp.where(active_gamma[:,None],(v*sd).T,old_gamma).astype(jnp.float32)
    resid = jnp.where(obs,raw-w_actual*(new_beta-old_beta)[:,first.exposure_idx]
        -(new_gamma-old_gamma)[:,first.unit_idx],0.).astype(jnp.float32)
    updated = []
    for m,s in enumerate(states):
        new = eqx.tree_at(lambda s:(s.beta,s.gamma,s.resid),s,
            (new_beta[m],new_gamma[m],resid[m]))
        if location:
            delta = (parameters[old_beta.size+m]-old_location[m]).astype(jnp.float32)
            leaves = jnp.where(leaf_masks[m],s.forest.leaf_tree+
                delta/(s.forest.leaf_tree.shape[0]*s.forest.leaf_unit),s.forest.leaf_tree)
            new = eqx.tree_at(lambda s:(s.forest.leaf_tree,s.mu_fit,s.resid),new,
                (leaves,s.mu_fit+delta,jnp.where(s.obs_mask,new.resid-s.alpha*delta,0.)))
        updated.append(new)
    return eqx.tree_at(lambda s:s.states,state,tuple(updated))


def joint_gaussian_step(key,state,*,location=False):
    if not state.has_chain_axis:
        return joint_gaussian_single_step(key,state,location=location)
    per,shared = split_multi_chain_fields(state)
    def one(k,p):
        updated = joint_gaussian_single_step(k,eqx.combine(p,shared),location=location)
        return split_multi_chain_fields(updated)[0]
    updated = jax.vmap(one)(random.split(key,state.num_chains),per)
    return eqx.combine(updated,shared)
