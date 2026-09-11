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

"""Dense independent Gaussian oracle for the experimental blocked transition."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from longbet._joint_gaussian import joint_statistics, joint_draw
from longbet._shared_forest import observed_precision, enable_x64


@pytest.mark.parametrize('coupled,inactive', [(True,False),(False,False),(True,True)])
@pytest.mark.parametrize('location',[False,True])
def test_joint_block_against_dense_gaussian(coupled,inactive,location):
    rng = np.random.default_rng(921)
    N,T,M,d = 4,5,3,4
    n = N*T
    unit = np.repeat(np.arange(N),T)
    exposure = np.tile(np.arange(T)%d,N)
    G = np.array([[0,0,0],[.7,0,0],[-.4,.6,0]])*coupled
    variance = np.array([1.,.6,1.4])
    B = np.eye(M)-G
    Sigma = np.linalg.solve(B,np.diag(variance))@np.linalg.inv(B.T)
    obs = np.arange(M)[:,None] < (np.arange(n)%(M+1))[None,:]
    if not coupled:
        obs = rng.random((M,n))>.3  # arbitrary masks legal without SUR
    obs[:,unit == N-1] = False  # prior-only unit, including its cross covariance
    w,y = rng.normal(size=(2,M,n))
    w[1,2] = 0
    sd = np.array([.4,.8,1.2])
    if inactive:
        w[0] = 0
        sd[1] = 0
    K = np.exp(-.5*(np.arange(d)[:,None]-np.arange(d)[None,:])**2/2)+np.eye(d)*.02
    L = np.stack([np.linalg.cholesky(K*(1+m*.3)) for m in range(M)])
    location_weights = rng.normal(size=(M,n)) if location else None
    location_sd = np.array([.4,.7,.2]) if location else None
    D = M*d+(M if location else 0)
    P = np.eye(D+N*M)
    h = np.zeros(D+N*M)
    for i in range(n):
        ix = np.flatnonzero(obs[:,i])
        if not len(ix):
            continue
        A = np.zeros((M,D+N*M))
        for m in range(M):
            A[m,m*d:(m+1)*d] = w[m,i]*L[m,exposure[i]]
            if location:
                A[m,M*d+m] = location_weights[m,i]*location_sd[m]
            A[m,D+unit[i]*M+m] = sd[m]
        A = A[ix]
        cov = Sigma[np.ix_(ix,ix)]
        P += A.T@np.linalg.solve(cov,A)
        h += A.T@np.linalg.solve(cov,y[ix,i])
    expected_cov = np.linalg.inv(P)
    expected_mean = np.linalg.solve(P,h)
    with enable_x64(True):
        omega = observed_precision(jnp.array(obs),jnp.array(G),jnp.array(variance))
        stats = jax.jit(joint_statistics,static_argnums=7)(jnp.array(y),omega,
            jnp.array(w),jnp.array(exposure),jnp.array(unit),jnp.array(L),jnp.array(sd),N,
            location_weights,location_sd)
        np.testing.assert_allclose(stats[2],expected_mean[:D],atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(stats[1]@stats[1].T),expected_cov[:D,:D],atol=1e-12)
        u,v = jax.jit(jax.vmap(lambda k:joint_draw(k,stats)))(jax.random.split(jax.random.key(54),16000))
        draws = np.concatenate((u,np.asarray(v).reshape(16000,-1)),axis=1)
    np.testing.assert_allclose(draws.mean(0),expected_mean,atol=.027)
    np.testing.assert_allclose(np.cov(draws,rowvar=False),expected_cov,atol=.035)


@pytest.mark.parametrize('chains,shared,fixed', [(1,0,False),(2,2,False),(2,0,True)])
@pytest.mark.parametrize('location',[False,True])
def test_joint_state_residual_invariant(chains,shared,fixed,location):
    from longbet import LongBetConfig,LongBetMulti
    from longbet._joint_gaussian import joint_gaussian_step
    rng = np.random.default_rng(84)
    N,T = 20,5
    x = rng.normal(size=(N,2))
    z = np.zeros((N,T)); z[:12,2:] = 1
    y = dict(g=rng.normal(size=(N,T)),h=rng.normal(size=(N,T)),
             b=(rng.normal(size=(N,T))>.4).astype(float))
    y['h'][0,0] = np.nan
    cfg = LongBetConfig(num_burnin=3,num_sweeps=3,num_chains=chains,n_skip=1,
        sigma_prior_a=2,sigma_prior_b=1,
        num_trees_pr=2,num_trees_trt=4,num_shared_trees=shared,
        max_depth_pr=3,max_depth_trt=3,min_points_per_leaf_pr=2,
        min_points_per_leaf_trt=2,sample_beta=not fixed,random_intercept=not fixed,
        sample_alpha=bool(shared),
        random_seed=88)
    with enable_x64(True):
        old = LongBetMulti(cfg).fit(y,x,z,outcome=dict(g='continuous',h='continuous',b='binary')).state
        new = jax.jit(lambda k,s:joint_gaussian_step(k,s,location=location))(jax.random.key(921),old)
        for a,b in zip(old.states,new.states):
            if fixed:
                np.testing.assert_array_equal(a.beta,b.beta)
                np.testing.assert_array_equal(a.gamma,b.gamma)
                if not location:
                    np.testing.assert_array_equal(a.resid,b.resid)
            response = b.z if b.outcome_type_str == 'binary' else b.y
            expand = lambda p:p[...,None]
            coding = jnp.where(b.z_vec==1,expand(b.b1),expand(b.b0))
            fit = expand(b.alpha)*b.mu_fit+coding*b.beta[...,b.exposure_idx]*b.nu_fit+b.gamma[...,b.unit_idx]
            np.testing.assert_allclose(b.resid,jnp.where(b.obs_mask,response-fit,0),atol=3e-5)
            assert b.resid.dtype == b.beta.dtype == b.gamma.dtype == jnp.float32
            np.testing.assert_array_equal(a.sigma2,b.sigma2)
            if location:
                import equinox as eqx
                from bartz.grove import evaluate_forest
                from longbet._state import chain_filter_spec
                def actual(s):
                    return evaluate_forest(s.X,s.forest,sum_batch_axis=-1)+s.forest.offset
                if chains>1:
                    per,constants = eqx.partition(b,chain_filter_spec(b))
                    mu = jax.vmap(lambda p:actual(eqx.combine(p,constants)))(per)
                else:
                    mu = actual(b)
                np.testing.assert_allclose(jnp.where(b.obs_mask,b.mu_fit,0),
                    jnp.where(b.obs_mask,mu,0),atol=3e-5)
        np.testing.assert_array_equal(new.states[0].sigma2,1)
        np.testing.assert_array_equal(old.gamma_loadings,new.gamma_loadings)
