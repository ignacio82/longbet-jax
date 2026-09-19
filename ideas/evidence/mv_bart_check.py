"""Can bartz run a MULTIVARIATE forest with a fixed dense error precision and dense leaf prior?
Section 8.4 of the plan claims this nests the vector-leaf engine at q=3 with an identity design."""
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
from bartz.mcmcstep import init, step, Wishart, make_p_nonterminal

rng = np.random.default_rng(0)
n, k, p = 200, 3, 2
X = np.vstack([rng.integers(0, 4, n) for _ in range(p)]).astype(np.uint8)
M = rng.standard_normal((k, k)); V = M @ M.T / k + np.eye(k); P = np.linalg.inv(V)
y = rng.multivariate_normal(np.zeros(k), V, size=n).T.astype(np.float32)   # (k, n)

st = init(X=jnp.asarray(X), y=jnp.asarray(y), offset=jnp.zeros(k),
          max_split=jnp.full(p, 3, jnp.uint8), num_trees=5,
          p_nonterminal=make_p_nonterminal(4, 0.95, 2.0),
          leaf_prior_cov_inv=jnp.asarray(2.0 * P, jnp.float32),   # dense leaf prior precision
          error_cov_inv=Wishart(nu=jnp.array(3.0), rate=jnp.eye(k), value=jnp.asarray(P, jnp.float32)))
print("init ok; leaf_tree", st.forest.leaf_tree.shape, "resid", st.resid.shape,
      "prec_tree", None if st.forest.prec_tree is None else st.forest.prec_tree.shape)
frozen = eqx.tree_at(lambda w: (w.nu, w.rate), st.error_cov_inv, (None, None), is_leaf=lambda x: x is None)
st = eqx.tree_at(lambda s: s.error_cov_inv, st, frozen)
key = jax.random.key(1)
for i in range(30):
    key, sub = jax.random.split(key)
    st = step(sub, st)
print("30 steps ok; error precision fixed:", np.allclose(np.asarray(st.error_cov_inv.value), P, atol=1e-5),
      "| leaves/tree", int(np.count_nonzero(np.asarray(st.forest.split_tree))) / 5,
      "| resid finite", bool(np.isfinite(np.asarray(st.resid)).all()))
