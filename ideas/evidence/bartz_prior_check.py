"""Does the oracle's tree prior match bartz's GROW/PRUNE stationary law under a flat likelihood?"""
import sys, collections
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
from jax import lax
from bartz.mcmcstep import init, step, Wishart, make_p_nonterminal
import dose_reference as ref

rng = np.random.default_rng(3)
n, max_split, max_depth, alpha, beta = 14, np.array([2, 1], np.uint8), 3, 0.5, 1.0
X = np.vstack([rng.integers(0, 3, n), rng.integers(0, 2, n)]).astype(np.uint8)
y = np.zeros(n, np.float32)

def fresh():
    st = init(X=jnp.asarray(X), y=jnp.asarray(y), offset=0.0, max_split=jnp.asarray(max_split),
              num_trees=1, p_nonterminal=make_p_nonterminal(max_depth, alpha, beta),
              leaf_prior_cov_inv=1.0,
              error_cov_inv=Wishart(nu=jnp.array(3.0), rate=jnp.array(3.0), value=jnp.array(1e-4)))
    frozen = eqx.tree_at(lambda w: (w.nu, w.rate), st.error_cov_inv, (None, None), is_leaf=lambda x: x is None)
    return eqx.tree_at(lambda s: s.error_cov_inv, st, frozen)

@jax.jit
def run(key, state, n_steps=200_000):
    def body(carry, _):
        st, k = carry
        k, sub = jax.random.split(k)
        st = step(sub, st)
        return (st, k), (st.forest.var_tree[0], st.forest.split_tree[0])
    (_, _), out = lax.scan(body, (state, key), None, length=n_steps)
    return out

var, split = map(np.asarray, run(jax.random.key(0), fresh()))
counts = collections.Counter()
for v, s in zip(var[2000:], split[2000:]):
    counts[tuple((node, int(v[node]), int(s[node])) for node in range(1, len(s)) if s[node] > 0)] += 1
total = sum(counts.values())
trees = ref.enumerate_trees(max_split, max_depth)
prior = {t.rules: float(np.exp(ref.tree_log_prior(t, max_split, max_depth, alpha, beta))) for t in trees}
assert set(counts) <= set(prior), set(counts) - set(prior)
print(f"{len(prior)} trees in the oracle, {len(counts)} visited by bartz")
worst = 0.0
for rules, p in sorted(prior.items(), key=lambda kv: -kv[1]):
    f = counts.get(rules, 0) / total
    worst = max(worst, abs(f - p))
    print(f"prior {p:.4f}  bartz {f:.4f}  diff {f - p:+.4f}  {rules}")
print("max abs diff", round(worst, 4), " TV", round(0.5 * sum(abs(counts.get(r, 0) / total - p) for r, p in prior.items()), 4))
