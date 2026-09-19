import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from bartz.mcmcstep._step import _sample_wishart_bartlett
rng = np.random.default_rng(0)
q, df = 3, 9.0
M = rng.standard_normal((q, q)); S = M @ M.T + np.eye(q)          # inverse-Wishart scale
keys = jax.random.split(jax.random.key(1), 40000)
P = np.asarray(jax.vmap(lambda k: _sample_wishart_bartlett(k, df, jnp.asarray(S)))(keys))
print("dtype", P.dtype)
print("E[P] vs df*inv(S): max rel err", np.abs(P.mean(0) - df * np.linalg.inv(S)).max() / np.abs(df * np.linalg.inv(S)).max())
V = np.linalg.inv(P)
print("E[V] vs S/(df-q-1): max rel err", np.abs(V.mean(0) - S / (df - q - 1)).max() / np.abs(S / (df - q - 1)).max())
