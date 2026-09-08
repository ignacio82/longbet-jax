"""Independent posterior, topology and full-state checks for shared treatment trees."""
import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
import scipy.special

from bartz.grove import evaluate_forest, traverse_forest
from longbet import LongBet, LongBetConfig, LongBetMulti, effect_draws, joint_prob
from longbet._shared_forest import (init_shared_forest, observed_precision,
    vector_leaf_posterior, shared_tree_step, shared_forest_step, shared_ridge_step,
    SHARED_SAMPLER_SEMANTICS)
from longbet._shared_forest import enable_x64


@pytest.fixture(autouse=True)
def matrix_precision_context():
    # Low-level kernel oracles must keep X64 enabled during outer lowering.
    # The production multi-MCMC driver establishes this context automatically.
    with enable_x64(True):
        yield


@pytest.mark.parametrize("option", [dict(num_shared_trees=-1), dict(num_shared_trees=True),
    dict(num_shared_trees=1.2), dict(num_shared_trees=20),
    dict(shared_variance_fraction=0), dict(shared_variance_fraction=1),
    dict(shared_variance_fraction=True), dict(shared_variance_fraction=np.nan)])
def test_invalid_shared_config(option):
    with pytest.raises(ValueError):
        LongBetConfig(**option)


def _cfg(**kwargs):
    return LongBetConfig(**(dict(num_trees_pr=2, num_trees_trt=4, num_shared_trees=2,
        sigma_prior_a=2, sigma_prior_b=1,
        max_depth_pr=3, max_depth_trt=3, min_points_per_leaf_pr=2,
        min_points_per_leaf_trt=2, num_burnin=5, num_sweeps=8, n_skip=1,
        num_chains=1, random_seed=51) | kwargs))


def test_vector_leaf_statistics_against_dense_observed_likelihood():
    rng = np.random.default_rng(88)
    n, M = 13, 3
    G = np.array([[0,0,0],[.8,0,0],[-.6,.5,0]])
    variance = np.array([1., .7, 1.3])
    B = np.eye(M)-G
    Sigma = np.linalg.solve(B, np.diag(variance)) @ np.linalg.inv(B.T)
    obs = np.arange(M)[:, None] < (np.arange(n) % (M+1))[None, :]
    omega = np.asarray(observed_precision(jnp.array(obs), jnp.array(G), jnp.array(variance)))
    w = rng.normal(size=(M,n)); w[0,3] = 0
    y = rng.normal(size=(M,n))
    prior = np.array([[2.,.1,-.3],[.1,1.5,.2],[-.3,.2,1.]])
    ids = np.where(np.arange(n)<6, 1, 2)
    gram = omega*w.T[:,:,None]*w.T[:,None,:]
    score = w.T*np.einsum("imk,ki->im", omega, y)
    P, mean, chol, integrated = vector_leaf_posterior(jnp.array(ids), jnp.array(gram),
        jnp.array(score), jnp.array(prior), 4)
    for leaf in (1,2):
        expected_P = prior.copy(); expected_h = np.zeros(M)
        blocks, designs, ys = [], [], []
        for i in np.flatnonzero(ids == leaf):
            ix = np.flatnonzero(obs[:,i])
            if not len(ix):
                continue
            cov = Sigma[np.ix_(ix, ix)]
            A = np.diag(w[:,i])[ix]
            expected_P += A.T @ np.linalg.solve(cov, A)
            expected_h += A.T @ np.linalg.solve(cov, y[ix,i])
            blocks.append(cov); designs.append(A); ys.append(y[ix,i])
        noise = scipy.linalg.block_diag(*blocks)
        design = np.concatenate(designs); response = np.concatenate(ys)
        marginal = noise + design @ np.linalg.solve(prior, design.T)
        log_bf = .5*(np.linalg.slogdet(noise)[1]-np.linalg.slogdet(marginal)[1]
            + response @ np.linalg.solve(noise,response)-response @ np.linalg.solve(marginal,response))
        np.testing.assert_allclose(P[leaf], expected_P, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(mean[leaf], np.linalg.solve(expected_P,expected_h), atol=2e-6)
        np.testing.assert_allclose(chol[leaf]@chol[leaf].T, expected_P, atol=3e-6)
        np.testing.assert_allclose(integrated[leaf], log_bf, atol=3e-6)


def test_actual_vector_leaf_draws_have_full_conditional_covariance():
    """Sample the actual topology/leaf kernel with all splits disabled."""
    rng = np.random.default_rng(63)
    n, M = 12, 3
    cfg = _cfg(num_trees_trt=2, num_shared_trees=1, max_depth_trt=1)
    f = init_shared_forest(cfg, M, n, jnp.zeros(1,jnp.uint8))
    omega = jnp.broadcast_to(jnp.array([[2.,.4,-.2],[.4,1.,.3],[-.2,.3,1.5]]),(n,M,M))
    w = jnp.array(rng.normal(size=(M,n))); y = jnp.array(rng.normal(size=(M,n)))
    gram = omega*w.T[:,:,None]*w.T[:,None,:]
    score = w.T*jnp.einsum("imk,ki->im",omega,y)
    P = np.asarray(gram).sum(0)+np.asarray(f.prior_precision)
    expected_cov = np.linalg.inv(P); expected_mean = expected_cov@np.asarray(score).sum(0)
    def draw(k):
        return shared_tree_step(k,f.leaf_tree[0],f.var_tree[0],f.split_tree[0],
            f.leaf_indices[0],y,gram,omega,w,jnp.ones(n,bool),f,jnp.zeros((1,n),jnp.uint8))[0][:,1]
    draws = np.asarray(jax.jit(jax.vmap(draw))(jax.random.split(jax.random.key(94),8000)))
    np.testing.assert_allclose(draws.mean(0),expected_mean,atol=.015)
    np.testing.assert_allclose(np.cov(draws.T),expected_cov,atol=.003,rtol=.1)


@pytest.mark.slow
def test_topology_frequencies_match_enumerated_posterior():
    """Nine depth-two trees; independently integrate leaves and enumerate priors.

    Includes differing growable/prunable counts, null boundary proposals and
    infeasible repeated-variable rules. Omitting proposal counts fails this.
    """
    X = np.tile(np.array([[0,0,1,1],[0,1,0,1]],np.uint8), (1,2))
    n, M = X.shape[1], 2
    y = np.array([.3*(2*X[0].astype(float)-1), -.25*(2*X[1].astype(float)-1)])
    w = np.array([np.linspace(.6,1.4,n),np.linspace(1.2,-.5,n)])
    O = np.array([[1.,.4],[.4,1.5]])
    omega = jnp.broadcast_to(jnp.array(O),(n,M,M))
    gram = omega*jnp.array(w.T[:,:,None]*w.T[:,None,:])
    cfg = _cfg(num_trees_trt=2,num_shared_trees=1,alpha_split_trt=.65,beta_split_trt=1.)
    f = init_shared_forest(cfg,M,n,jnp.ones(2,jnp.uint8))
    topologies, log_weights = [], []
    for root in (-1,0,1):
        for children in ([0] if root == -1 else range(4)):
            var, split = np.zeros(4,np.uint8),np.zeros(4,np.uint8)
            if root == -1:
                ids = np.ones(n,int); log_prior = np.log(1-.65)
            else:
                var[1],split[1] = root,1
                ids = 2+(X[root]>=1).astype(int)
                log_prior = np.log(.65/2)
                for node, bit in ((2,1),(3,2)):
                    if children & bit:
                        var[node],split[node] = 1-root,1
                        ids = np.where(ids==node,2*node+(X[1-root]>=1),ids)
                        log_prior += np.log((.65/2)/2)
                    else:
                        log_prior += np.log(1-.65/2)
            log_lik = 0.
            for leaf in np.unique(ids):
                ix = np.flatnonzero(ids==leaf)
                A = np.concatenate([np.diag(w[:,i]) for i in ix])
                response = y[:,ix].T.reshape(-1)
                noise = scipy.linalg.block_diag(*[np.linalg.inv(O) for _ in ix])
                marginal = noise+A@np.linalg.inv(np.asarray(f.prior_precision))@A.T
                log_lik += -.5*(np.linalg.slogdet(marginal)[1]+response@np.linalg.solve(marginal,response))
            topologies.append((var,split)); log_weights.append(log_prior+log_lik)
    target = scipy.special.softmax(log_weights)
    def step(carry,k):
        leaves,var,split,ids = carry
        result = shared_tree_step(k,leaves,var,split,ids,jnp.array(y),gram,
            omega,jnp.array(w),jnp.ones(n,bool),f,jnp.array(X))
        return result[:4], (result[1],result[2])
    _, (vs,ss) = jax.jit(lambda: jax.lax.scan(step,
        (f.leaf_tree[0],f.var_tree[0],f.split_tree[0],f.leaf_indices[0]),
        jax.random.split(jax.random.key(722),18000)))()
    vs,ss = np.asarray(vs)[1000:],np.asarray(ss)[1000:]
    frequencies = [np.mean(np.all(ss==s,axis=1) & np.all((vs==v)|(s==0),axis=1)) for v,s in topologies]
    np.testing.assert_allclose(sum(frequencies),1.,atol=1e-8)
    np.testing.assert_allclose(frequencies,target,atol=.015)


def _panel():
    rng = np.random.default_rng(336)
    N,T = 26,5
    x = rng.normal(size=(N,3)); z = np.zeros((N,T)); z[:18,2:]=1
    y = {"g":rng.normal(size=(N,T)),"h":rng.normal(size=(N,T)),
         "b":(rng.normal(size=(N,T))-.4>0).astype(float)}
    return x,z,y, {"g":"continuous","h":"continuous","b":"binary"}


@pytest.mark.parametrize("sur", [True,False])
def test_shared_private_residuals_priors_trace_and_roundtrip(sur,tmp_path):
    x,z,y,types = _panel()
    # Non-nested masks supported without SUR; predecessor-closed with it.
    y["h"][0,0] = np.nan
    if not sur: y["b"][1,0] = np.nan
    cfg = _cfg(sur=sur,num_chains=2,sample_alpha=True)
    model = LongBetMulti(cfg).fit(y,x,z,outcome=types)
    assert model.sampler_semantics == SHARED_SAMPLER_SEMANTICS
    st = model.state; shared = st.shared_forest
    assert shared.fit.shape == (2,3,z.size)
    private_variance = ((cfg.num_trees_trt-cfg.num_shared_trees)/
                        float(st.states[0].leaf_prior_cov_inv_nu))
    shared_variance = cfg.num_shared_trees/float(shared.prior_precision[0,0])
    np.testing.assert_allclose(private_variance+shared_variance,1.)
    for m,child in enumerate(st.states):
        def fits(mu,nu):
            return (evaluate_forest(child.X,mu,sum_batch_axis=-1)+mu.offset,
                    evaluate_forest(child.X,nu,sum_batch_axis=-1)+nu.offset)
        # Forest constants are shared, so use the package's chain partition.
        from longbet._state import chain_filter_spec
        per,constants = eqx.partition(child,chain_filter_spec(child))
        mu,private = jax.vmap(lambda p: fits(eqx.combine(p,constants).forest,
                                           eqx.combine(p,constants).forest_nu))(per)
        total = private+shared.fit[:,m]
        fitted = child.alpha[:,None]*mu+jnp.where(child.z_vec==1,child.b1[:,None],child.b0[:,None])*child.beta[:,child.exposure_idx]*total+child.gamma[:,child.unit_idx]
        response = child.z if child.outcome_type_str=="binary" else child.y
        np.testing.assert_allclose(child.resid,jnp.where(child.obs_mask,response-fitted,0),atol=2e-4)
        np.testing.assert_allclose(jnp.where(child.obs_mask,child.nu_fit,0),jnp.where(child.obs_mask,total,0),atol=2e-4)
        tr = model.trace.traces[m].nu_trace
        assert tr.leaf_tree.shape == (2,cfg.num_sweeps,4,8)
        np.testing.assert_array_equal(tr.split_tree[..., -2:, :],model.trace.traces[0].nu_trace.split_tree[..., -2:, :])
    np.testing.assert_array_equal(st.states[0].sigma2,1.)
    if not sur: np.testing.assert_array_equal(st.gamma_loadings,0.)
    pred = model.predict(x,z)
    before = effect_draws(pred["b"])
    assert np.all(np.isfinite(joint_prob(pred,{"g":lambda d:d>0,"h":lambda d:d<0,"b":lambda d:d<0})))
    path = tmp_path/"joint.npz"; model.save(path)
    restored = LongBetMulti.load(path)
    assert restored.trace.num_shared_trees == 2
    np.testing.assert_array_equal(effect_draws(restored.predict(x,z)["b"]),before)
    assert restored["b"].multi_origin["provenance"] == model.provenance
    child_path = tmp_path/"child.npz"; restored["b"].save(child_path)
    scalar = LongBet.load(child_path)
    assert scalar.multi_origin == restored["b"].multi_origin
    np.testing.assert_array_equal(effect_draws(scalar.predict(x,z)),before)
    with np.load(path) as archive: arrays = dict(archive)
    meta = json.loads(str(arrays["_meta_json"]))
    assert meta["format_version"] == 2
    arrays["outcome_1_nu_split_tree"] = arrays["outcome_1_nu_split_tree"].copy()
    arrays["outcome_1_nu_split_tree"][0,0,-1,1] ^= 1
    corrupt = tmp_path/"corrupt.npz"; np.savez(corrupt,**arrays)
    with pytest.raises(ValueError,match="shared topology"):
        LongBetMulti.load(corrupt)


def test_scalar_refuses_shared_tree_request():
    x,z,y,_ = _panel()
    with pytest.raises(ValueError,match="requires LongBetMulti"):
        LongBet(_cfg()).fit(y["g"],x,z)


def test_sharing_fit_restores_the_callers_dtype_policy():
    x,z,y,types = _panel()
    with enable_x64(False):
        model = LongBetMulti(_cfg(num_burnin=0,num_sweeps=2)).fit(y,x,z,outcome=types)
        assert not jax.config.x64_enabled
        assert model.state.shared_forest.leaf_tree.dtype == jnp.float32


def test_nearly_collinear_leaf_precision_retains_the_prior():
    G = jnp.array([[0.,0.],[-1.,0.]],jnp.float32)
    omega = observed_precision(jnp.ones((2,8),bool),G,jnp.array([1.,1e-10],jnp.float32))
    prior = jnp.eye(2)*2
    P,mean,chol,_ = jax.jit(lambda: vector_leaf_posterior(jnp.ones(8,jnp.int32),
        omega,jnp.zeros((8,2)),prior,2))()
    assert P.dtype == jnp.float64
    assert np.all(np.isfinite(chol))
    eig = np.linalg.eigvalsh(np.asarray(P[1]))
    # The poorly identified direction is regularized, not rounded away.
    assert 5.9 < eig[0] < 6.1


def test_shared_private_ridge_ratio_and_likelihood_invariance():
    """Independent active-node enumeration and prior-density/Jacobian oracle."""
    x,z,y,types = _panel()
    model = LongBetMulti(_cfg()).fit(y,x,z,outcome=types)
    forest, states = model.state.shared_forest, model.state.states

    def active(split):
        masks = np.zeros((len(split),2*split.shape[-1]),bool)
        for j,tree in enumerate(np.asarray(split)):
            pending = [1]
            while pending:
                node = pending.pop()
                if node >= len(tree) or tree[node] == 0:
                    masks[j,node] = True
                else:
                    pending.extend((2*node,2*node+1))
        return masks

    smask = active(forest.split_tree)
    accepted = 0
    for seed in range(6):
        key = jax.random.key(seed)
        updated, children = shared_ridge_step(key,forest,states)
        for m,(old,new) in enumerate(zip(states,children)):
            pmask = active(old.forest_nu.split_tree)
            kp,ka = jax.random.split(jax.random.fold_in(key,m))
            log_c = float(old.ridge_proposal_sigma * jax.random.normal(kp,dtype=jnp.float32))
            c = float(jnp.exp(jnp.float32(log_c)))
            beta = np.asarray(old.beta,dtype=float)
            L = np.asarray(old.K_chol,dtype=float)
            q_beta = np.sum(scipy.linalg.solve_triangular(L,beta,lower=True)**2)
            pv = np.asarray(old.forest_nu.leaf_tree)[pmask]*float(old.forest_nu.leaf_unit)
            sv = np.asarray(forest.leaf_tree[:,m,:])[smask]
            q_leaf = float(old.leaf_prior_cov_inv_nu)*np.sum(pv**2)
            q_leaf += float(forest.prior_precision[m,m])*np.sum(sv**2)
            log_ratio = -.5*(c*c-1)*q_beta-.5*(c**-2-1)*q_leaf
            log_ratio += (len(beta)-int(pmask.sum())-int(smask.sum()))*log_c
            accept = np.log(float(jax.random.uniform(ka,dtype=jnp.float32))) < log_ratio
            scale = c if accept else 1.
            accepted += accept
            np.testing.assert_allclose(new.beta,old.beta*scale,rtol=2e-6)
            np.testing.assert_allclose(updated.leaf_tree[:,m,:],forest.leaf_tree[:,m,:]/scale,rtol=2e-6)
            np.testing.assert_allclose(new.forest_nu.leaf_tree,
                np.where(pmask,np.asarray(old.forest_nu.leaf_tree)/scale,old.forest_nu.leaf_tree),rtol=2e-6)
            np.testing.assert_allclose(new.beta[new.exposure_idx]*new.nu_fit,
                                       old.beta[old.exposure_idx]*old.nu_fit,atol=2e-6)
            np.testing.assert_array_equal(new.resid,old.resid)
    assert accepted > 0  # The invariance checks must exercise actual moves.


@pytest.mark.slow
def test_shared_private_mixed_posterior_against_observed_data_quadrature():
    """Actual full mean/latent/shared/private updates, known Gamma and variances.

    Constant forests imply total mean prior N(offset, (2.25+1)*I).
    Holding covariance fixed makes the observed probit/Gaussian likelihood
    directly integrable, independently of any latent-response stand-in.
    """
    import scipy.stats as stats
    from longbet._multi_input import normalize_multi_inputs
    from longbet._multi_state import init_multi_longbet
    from longbet._multi_step import multi_single_step
    yb = np.array([[0.,1.,0.,0.,1.]])
    yc = np.array([[-1.2,.3,.1,.5,1.3]])
    cfg = _cfg(num_trees_pr=1,num_trees_trt=2,num_shared_trees=1,
        max_depth_pr=1,max_depth_trt=1,min_points_per_leaf_pr=1,min_points_per_leaf_trt=1,
        sample_beta=False,adaptive_coding=False,random_intercept=False,standardize=False)
    norm = normalize_multi_inputs([yb,yc],outcome=["binary","continuous"],outcome_names=None,config=cfg)
    st = init_multi_longbet(X_unified=jnp.zeros((1,5),jnp.uint8),unit_idx=jnp.zeros(5,jnp.int32),
        time_idx=jnp.arange(5,dtype=jnp.int32),exposure_idx=jnp.zeros(5,jnp.int32),z_vec=jnp.zeros(5,jnp.float32),
        max_split_mu=jnp.zeros(1,jnp.uint8),max_split_nu=jnp.zeros(1,jnp.uint8),norm_input=norm,config=cfg)
    G = jnp.array([[0.,0.],[.6,0.]],jnp.float32); v = jnp.array([1.,.7],jnp.float32)
    def fix(s):
        children = tuple(eqx.tree_at(lambda c:(c.sigma2,c.error_cov_inv.value),c,(v[m],1/v[m])) for m,c in enumerate(s.states))
        return eqx.tree_at(lambda a:(a.states,a.gamma_loadings),s,(children,G))
    st = fix(st)
    grid = np.linspace(-5,5,701)
    a,b = np.meshgrid(grid,grid,indexing="ij")
    logp = -.5*((a-norm.offset_[0])**2+b**2)/3.25
    var_c = .7+.6**2; cond_sd = np.sqrt(.7/var_c)
    for bit,value in zip(yb[0],yc[0]):
        logp += stats.norm.logpdf(value,loc=b,scale=np.sqrt(var_c))
        logp += scipy.special.log_ndtr((2*bit-1)*(a+.6/var_c*(value-b))/cond_sd)
    mass = np.exp(logp-logp.max()); mass /= mass.sum()
    expected_mean = np.array([np.sum(mass*a),np.sum(mass*b)])
    da,db = a-expected_mean[0],b-expected_mean[1]
    expected_cov = np.array([[np.sum(mass*da*da),np.sum(mass*da*db)],
                             [np.sum(mass*da*db),np.sum(mass*db*db)]])
    event = np.sum(mass[(a<0)&(b>0)])
    def step(s,k):
        ko,kl = jax.random.split(k)
        updated = fix(multi_single_step(jax.random.split(ko,2),jax.random.split(kl,2),s,jnp.int32(1)))
        theta = jnp.stack([c.mu_fit[0]+c.nu_fit[0] for c in updated.states])
        return updated,theta
    _,draws = jax.jit(lambda:jax.lax.scan(step,st,jax.random.split(jax.random.key(757),16000)))()
    draws = np.asarray(draws)[2000:]
    np.testing.assert_allclose(draws.mean(0),expected_mean,atol=.035)
    np.testing.assert_allclose(np.cov(draws.T),expected_cov,rtol=.12,atol=.008)
    np.testing.assert_allclose(np.mean((draws[:,0]<0)&(draws[:,1]>0)),event,atol=.03)
