"""Does the level + random-intercept likelihood absorb selection on unit levels?

DGP: Y_it = alpha_i + 0.1 t + tau * Z_it + e_it, alpha_i = a * G_i + u_i.
Parallel trends holds exactly; only the LEVEL differs between arms.
A DiD estimator is unbiased for every a. We compare LongBet's ATT.
"""
import sys, numpy as np
from longbet import LongBet, LongBetConfig, get_att

def run(a, T, seed, N=200, tau=0.5, sd_u=1.0, sd_e=0.5):
    rng = np.random.default_rng(seed)
    G = np.zeros(N, int); G[: N // 2] = 1
    g0 = T // 2  # first treated column index
    z = np.zeros((N, T)); z[G == 1, g0:] = 1
    x = rng.normal(size=(N, 3))                     # pure noise covariates
    alpha = a * G + sd_u * rng.normal(size=N)
    t = np.arange(1, T + 1)
    y = alpha[:, None] + 0.1 * t[None, :] + tau * z + sd_e * rng.normal(size=(N, T))
    did = (y[G == 1, g0:].mean() - y[G == 1, :g0].mean()) - (y[G == 0, g0:].mean() - y[G == 0, :g0].mean())
    cfg = LongBetConfig(num_chains=2, num_burnin=400, num_sweeps=300, num_trees_pr=20,
                        num_trees_trt=20, random_seed=seed, device="cpu")
    m = LongBet(cfg).fit(y=y, x=x, z=z, t=t)
    pred = m.predict(x=x, z=z, t=t, summary_only=True)
    att = get_att(pred)
    return did, float(np.mean(att["att"])), att["att"], att["intervals"]

if __name__ == "__main__":
    T = int(sys.argv[1])
    for a in (0.0, 2.0):
        for seed in (1, 2):
            did, lb, by_s, iv = run(a, T, seed)
            print(f"T={T} a={a} seed={seed}  truth=0.500  DiD={did:.3f}  LongBet mean ATT={lb:.3f}  by exposure={np.round(by_s,3)}  lo={np.round(iv[0],3)} hi={np.round(iv[1],3)}", flush=True)
