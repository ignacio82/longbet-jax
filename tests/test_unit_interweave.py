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

"""Check the experimental variance move against the full density and Jacobian."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from longbet._shared_forest import enable_x64,observed_precision
from longbet._unit_interweave import unit_scale_move


@pytest.mark.parametrize('m',[0,1,2])
def test_scale_ratio_against_full_joint_density(m):
    rng = np.random.default_rng(53)
    M,N,T = 3,5,4
    unit = np.repeat(np.arange(N),T)
    raw = rng.normal(size=(M,N*T))
    gamma = rng.normal(size=N)
    obs = np.arange(M)[:,None] < (np.arange(N*T)%(M+1))[None,:]
    raw = np.where(obs,raw,0)
    G = np.array([[0,0,0],[.5,0,0],[-.3,.7,0]])
    variance,a,b = .6,2.3,.4
    with enable_x64(True):
        omega = np.asarray(observed_precision(jnp.array(obs),jnp.array(G),jnp.array([1,.8,.6])))
        def log_target(r,g,v):
            return -.5*np.einsum('mi,imk,ki->',r,omega,r)-.5*np.sum(g*g)/v-(N/2+a+1)*np.log(v)-b/v
        for i in range(24):
            key = jax.random.key(i)
            kp,ka = jax.random.split(key)
            log_c = float(.2*jax.random.normal(kp,(),jnp.float64))
            c = np.exp(log_c)
            candidate = raw.copy()
            candidate[m] -= np.where(obs[m],(c-1)*gamma[unit],0)
            expected = log_target(candidate,c*gamma,c*c*variance)-log_target(raw,gamma,variance)+(N+2)*log_c
            r,g,v,accepted,ratio = jax.jit(unit_scale_move,static_argnums=9)(key,
                jnp.array(raw),jnp.array(omega),jnp.array(obs[m]),jnp.array(unit),
                jnp.array(gamma),jnp.array(variance),a,b,m)
            np.testing.assert_allclose(ratio,expected,atol=2e-13)
            expected_accept = float(jnp.log(jax.random.uniform(ka,(),dtype=jnp.float64)))<expected
            assert bool(accepted)==expected_accept
            np.testing.assert_allclose(g,gamma*(c if expected_accept else 1),atol=1e-13)
            np.testing.assert_allclose(v,variance*(c*c if expected_accept else 1),atol=1e-13)
            np.testing.assert_allclose(r,candidate if expected_accept else raw,atol=1e-13)


@pytest.mark.parametrize('random_intercept',[True,False])
def test_interweaving_state_consistency(random_intercept):
    from longbet import LongBetConfig,LongBetMulti
    from longbet._unit_interweave import unit_interweave_step
    rng = np.random.default_rng(641)
    x = rng.normal(size=(20,2)); z = np.zeros((20,5)); z[:12,2:] = 1
    y = dict(g=rng.normal(size=z.shape),h=rng.normal(size=z.shape),
             q=(rng.normal(size=z.shape)>.5).astype(float))
    cfg = LongBetConfig(num_burnin=3,num_sweeps=3,n_skip=1,num_chains=2,
        sigma_prior_a=2,sigma_prior_b=1,
        num_trees_pr=2,num_trees_trt=4,num_shared_trees=2,max_depth_pr=3,max_depth_trt=3,
        min_points_per_leaf_pr=2,min_points_per_leaf_trt=2,
        random_intercept=random_intercept,random_seed=81)
    with enable_x64(True):
        old = LongBetMulti(cfg).fit(y,x,z,outcome=dict(g='continuous',h='continuous',q='binary')).state
        new = jax.jit(unit_interweave_step)(jax.random.key(419),old)
        for a,b in zip(old.states,new.states):
            np.testing.assert_array_equal(a.beta,b.beta)
            np.testing.assert_array_equal(a.sigma2,b.sigma2)
            np.testing.assert_allclose(b.resid,a.resid-(b.gamma-a.gamma)[:,a.unit_idx],atol=2e-7)
            np.testing.assert_allclose(b.gamma/jnp.sqrt(b.sigma_gamma2)[:,None],
                a.gamma/jnp.sqrt(a.sigma_gamma2)[:,None],atol=2e-7)
            assert b.sigma_gamma2.dtype == b.gamma.dtype == b.resid.dtype == jnp.float32
            if not random_intercept:
                np.testing.assert_array_equal(a.sigma_gamma2,b.sigma_gamma2)
                np.testing.assert_array_equal(a.gamma,b.gamma)
        np.testing.assert_array_equal(new.states[0].sigma2,1)
