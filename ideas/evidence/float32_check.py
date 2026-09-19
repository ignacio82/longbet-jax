"""How much does float32 perturb the GROW log-likelihood ratio of a realistic treatment node?"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
import dose_reference as ref

def log_ml(A, b, kappa, dtype):
    A, b = A.astype(dtype), b.astype(dtype)
    Lam = (np.asarray(kappa, dtype) * np.eye(b.size, dtype=dtype) + A)
    chol = np.linalg.cholesky(Lam)
    half = np.linalg.solve(chol, b).astype(dtype)      # triangular system, generic solve is fine here
    return dtype(0.5) * b.size * np.log(dtype(kappa)) - np.sum(np.log(np.diag(chol))) + dtype(0.5) * half @ half

def ref_stats(h_w, cohort, E_w, P, R, dt):
    K, L = h_w.shape[1], next(iter(E_w.values())).shape[1]
    A, b = np.zeros((K * L, K * L), dt), np.zeros(K * L, dt)
    for g, E in E_w.items():
        rows = np.flatnonzero(cohort == g)
        if rows.size:
            A += np.kron(h_w[rows].T @ h_w[rows], E.T @ P @ E).astype(dt)
            b += np.einsum("ik,il->kl", h_w[rows], R[rows] @ P @ E).reshape(-1).astype(dt)
    return A, b

rng = np.random.default_rng(7)
T, K, cohorts, N, M = 12, 6, (4, 7, 10), 600, 50
q, L = T - 1, T - min(cohorts) + 1
dose = rng.uniform(0.1, 1.0, N); cohort = rng.choice(cohorts, N)
knots = ref.clamped_knots(dose.min(), dose.max(), np.quantile(dose, [1/3, 2/3]))
h = ref.bspline_basis(dose, knots)
L_d = np.linalg.cholesky(ref.dose_prior_cov(h, 4.0)); L_s = np.linalg.cholesky(ref.exposure_kernel(L))
h_w = h @ L_d
E_w = {g: ref.exposure_design(T, g, L) @ L_s for g in cohorts}
A_diff = ref.difference_matrix(T)
V = 0.5 * A_diff @ A_diff.T; P = np.linalg.inv(V)
x = rng.uniform(size=N)
errs, conds, sizes = [], [], []
for rep in range(20):
    # residuals with a real subgroup effect, so the node statistics are large
    F = np.stack([ref.unit_design(h_w[i], E_w[cohort[i]]) @ (rng.standard_normal(K * L) * 0 + 1.0) * (x[i] > 0.5)
                  for i in range(N)])
    R = F + rng.multivariate_normal(np.zeros(q), V, size=N)
    left, right = x <= 0.5, x > 0.5
    kappa = M / 1.0
    out = {}
    for dt in (np.float64, np.float32):
        # accumulate the statistics themselves in the target precision
        E_dt = {g: e.astype(dt) for g, e in E_w.items()}
        stats = {name: ref_stats(h_w[m].astype(dt), cohort[m], E_dt, P.astype(dt), R[m].astype(dt), dt)
                 for name, m in (("l", left), ("r", right), ("p", np.ones(N, bool)))}
        out[dt] = sum(s * log_ml(*stats[k], kappa, dt) for k, s in (("l", 1), ("r", 1), ("p", -1)))
    errs.append(abs(float(out[np.float32]) - float(out[np.float64])))
    sizes.append(float(out[np.float64]))
    conds.append(np.linalg.cond(kappa * np.eye(K * L) + stats["p"][0].astype(np.float64)))
print(f"p = {K*L}, N = {N}; condition number of Lambda: median {np.median(conds):.1e}")
print(f"GROW log-likelihood ratio: median {np.median(sizes):.1f}")
print(f"|float32 - float64| of that ratio: median {np.median(errs):.3f}, max {np.max(errs):.3f} nats")
