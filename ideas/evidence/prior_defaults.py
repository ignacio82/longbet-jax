"""Root-only (no covariate splits) two-period dose model: closed-form posterior.
Checks that the proposed defaults (K, lambda_d, tau) neither oversmooth nor undercover."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
import dose_reference as ref

def one(seed, N, K, lam, tau, curve, noise_sd=1.0):
    rng = np.random.default_rng(seed)
    treated = rng.uniform(size=N) > 0.4
    dose = np.where(treated, rng.uniform(0.1, 1.0, N), 0.0)
    dy = 0.3 + np.where(treated, curve(dose), 0.0) + noise_sd * rng.standard_normal(N)
    # standardize as the plan prescribes: centre and scale by untreated differences
    c, s = dy[~treated].mean(), dy[~treated].std()
    v = (dy - c) / s
    d_t = dose[treated]
    interior = np.quantile(d_t, np.linspace(0, 1, K - 2)[1:-1])
    knots = ref.clamped_knots(d_t.min(), d_t.max(), interior)
    h = ref.bspline_basis(d_t, knots)
    L_d = np.linalg.cholesky(ref.dose_prior_cov(h, lam))
    hw = h @ L_d
    # untreated mean: a single coefficient with prior N(0, 1.5^2); sigma2 fixed at its truth (1 after scaling)
    Z = np.zeros((N, 1 + K)); Z[:, 0] = 1.0; Z[treated, 1:] = hw
    prior_prec = np.diag(np.r_[1 / 1.5**2, np.full(K, 1 / tau**2)])
    cov = np.linalg.inv(prior_prec + Z.T @ Z)
    mean = cov @ Z.T @ v
    grid = np.linspace(0.15, 0.95, 41)
    Hg = np.c_[np.zeros(41), ref.bspline_basis(grid, knots) @ L_d]
    est, sd = s * (Hg @ mean), s * np.sqrt(np.einsum("ij,jk,ik->i", Hg, cov, Hg))
    truth = curve(grid)
    return np.mean((est - truth) ** 2), np.mean(np.abs(est - truth) <= 1.96 * sd), np.mean(2 * 1.96 * sd)

curves = {"1-exp(-3d)": lambda d: 1 - np.exp(-3 * d), "cubic": lambda d: 2 * (d - 0.5) ** 3 + 0.5 * d,
          "null": lambda d: 0 * d, "sin(2 pi d)": lambda d: 0.5 * np.sin(2 * np.pi * d)}
print(f"{'curve':14s} {'K':>2s} {'lam':>5s} {'tau':>4s} {'N':>5s}  {'RMSE':>6s} {'cover':>6s} {'width':>6s}")
for name, curve in curves.items():
    for K, lam in ((6, 1.0), (6, 4.0), (6, 16.0), (8, 4.0)):
        for N in (500,):
            res = np.array([one(s, N, K, lam, 1.0, curve) for s in range(200)])
            print(f"{name:14s} {K:2d} {lam:5.1f} {1.0:4.1f} {N:5d}  {np.sqrt(res[:,0].mean()):6.3f} {res[:,1].mean():6.3f} {res[:,2].mean():6.3f}")
