"""Dense float64 reference oracle for the LongBetDose specification.

This file is *test ground truth*, not the sampler. It implements, with NumPy and
SciPy only and no regard for speed, every piece of closed-form algebra that
``ideas/IMPLEMENTATION_PLAN.md`` asks the JAX engine to reproduce:

* the first-difference operator and the cohort exposure designs ``E_g`` (plan 4.2);
* the clamped B-spline dose basis and its analytic derivative (plan 4.4);
* the normalized prior covariances and their whitening (plan 4.5);
* a node's sufficient statistics, dense and Kronecker-factored (plan 5.1);
* the leaf posterior and the leaf-integrated log likelihood (plan 5.1);
* the exact posterior over a fully enumerated tree space, for one tree and for a
  small forest with all leaves integrated jointly (plan 8.3);
* the joint Gaussian posterior of all leaf coefficients for fixed trees (plan 8.2);
* the covariance conditionals (plan 4.3);
* the identity "difference likelihood == level model with a flat unit intercept".

Run ``python ideas/dose_reference.py`` to execute the self-checks: each formula
is verified against an independent brute-force calculation, so a passing run
means the *specification* is internally consistent. At implementation time move
this file to ``tests/_dose_reference.py`` (next to ``tests/_panels.py``) and
import it from the ``tests/test_dose_*.py`` files. Do not import it from
``src/``: an oracle that shares code with the implementation checks nothing.

Conventions (identical to the plan)
-----------------------------------
* Periods ``t = 1..T``; differences ``j = 1..q`` with ``q = T - 1`` and
  ``v_ij = Y_{i,j+1} - Y_{i,j}``.
* Cohort ``g`` in ``2..T`` is the first treated period; never treated units have
  no cohort design (they contribute nothing to a treatment leaf).
* Exposure ``s = t - g + 1`` for ``t >= g``; the exposure grid is ``1..L``.
* A treatment leaf holds a whitened coefficient matrix ``U`` of shape ``(K, L)``
  (dose index first). Its row-major flattening ``theta = U.reshape(-1)`` has the
  prior ``N(0, I / kappa)`` with ``kappa = num_trees / tau**2``.
* The level effect of a leaf is ``f(s, d) = h_w(d)' U r_w(s)`` with whitened
  bases ``h_w(d) = L_d' h(d)`` and ``r_w(s) = L_s' e_s``.
* Tree rules follow bartz: binned predictors take values ``0..max_split[v]``; a
  rule ``(v, c)`` with ``c`` in ``1..max_split[v]`` sends ``x_v >= c`` right; heap
  node 1 is the root and node ``n`` has children ``2n`` and ``2n + 1``.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------------------
# panel geometry
# ---------------------------------------------------------------------------


def difference_matrix(T: int) -> np.ndarray:
    """``A`` of shape ``(T - 1, T)`` with ``v = A @ y``."""
    A = np.zeros((T - 1, T))
    for j in range(T - 1):
        A[j, j], A[j, j + 1] = -1.0, 1.0
    return A


def exposure_design(T: int, g: int, L: int) -> np.ndarray:
    """Unwhitened ``E_g`` of shape ``(q, L)``: differences of exposure indicators.

    ``E_g @ f`` is the vector of first differences of the level effect when
    ``f[s - 1]`` is the level effect at exposure ``s``.
    """
    if not 2 <= g <= T:
        raise ValueError("cohort g must lie in 2..T (no treatment in period 1)")
    E = np.zeros((T - 1, L))
    for j in range(T - 1):
        t = j + 2                       # the later period of difference j (1-based)
        s_now = t - g + 1 if t >= g else 0
        s_prev = (t - 1) - g + 1 if (t - 1) >= g else 0
        if s_now >= 1:
            E[j, s_now - 1] += 1.0
        if s_prev >= 1:
            E[j, s_prev - 1] -= 1.0
    return E


# ---------------------------------------------------------------------------
# dose basis
# ---------------------------------------------------------------------------


def clamped_knots(lo: float, hi: float, interior: np.ndarray, degree: int = 3) -> np.ndarray:
    return np.concatenate([np.repeat(lo, degree + 1), np.sort(interior), np.repeat(hi, degree + 1)])


def bspline_basis(d: np.ndarray, knots: np.ndarray, degree: int = 3, deriv: int = 0) -> np.ndarray:
    """``(n, K)`` design of the clamped B-spline basis, or of its ``deriv``-th derivative."""
    d = np.atleast_1d(np.asarray(d, dtype=np.float64))
    K = len(knots) - degree - 1
    out = np.empty((d.size, K))
    for k in range(K):
        coef = np.zeros(K)
        coef[k] = 1.0
        spline = BSpline(knots, coef, degree, extrapolate=False)
        if deriv:
            spline = spline.derivative(deriv)
        out[:, k] = spline(d)
    if np.isnan(out).any():
        raise ValueError("dose outside the fitted knot range: extrapolation is rejected")
    return out


def second_difference(K: int) -> np.ndarray:
    D = np.zeros((K - 2, K))
    for r in range(K - 2):
        D[r, r:r + 3] = (1.0, -2.0, 1.0)
    return D


def dose_prior_cov(h_train: np.ndarray, lam: float) -> np.ndarray:
    """``C_d = c (I + lam D2'D2)^-1`` with ``mean_i h_i' C_d h_i = 1`` over ``h_train``."""
    K = h_train.shape[1]
    D2 = second_difference(K)
    C = np.linalg.inv(np.eye(K) + lam * D2.T @ D2)
    scale = np.mean(np.einsum("ik,kl,il->i", h_train, C, h_train))
    return C / scale


def exposure_kernel(L: int, lengthscale: float = 1.0, const_share: float = 0.5,
                    jitter: float = 1e-6) -> np.ndarray:
    """Unit-diagonal squared-exponential kernel plus a marginalized constant mean.

    Equals ``longbet._gp.build_kernel_matrix(sig_knl=1, sigma_m=1)`` divided by its
    mean diagonal when ``const_share = 0.5``.
    """
    s = np.arange(1, L + 1, dtype=np.float64)
    dist = np.abs(s[:, None] - s[None, :])
    se = np.exp(-0.5 * (dist / lengthscale) ** 2) + jitter * np.eye(L)
    Kmat = const_share * np.ones((L, L)) + (1.0 - const_share) * se
    return Kmat / np.mean(np.diag(Kmat))


# ---------------------------------------------------------------------------
# leaf algebra
# ---------------------------------------------------------------------------


def unit_design(h_w: np.ndarray, E_w: np.ndarray) -> np.ndarray:
    """Whitened ``H_i = h_w' (x) E_w`` of shape ``(q, K L)``, dose index major."""
    return np.kron(h_w[None, :], E_w)


def node_stats_dense(h_w, cohort, E_w_by_cohort, P, R):
    """``A = sum_i H_i' P H_i`` and ``b = sum_i H_i' P r_i`` from explicit designs."""
    p = h_w.shape[1] * next(iter(E_w_by_cohort.values())).shape[1]
    A, b = np.zeros((p, p)), np.zeros(p)
    for i in range(h_w.shape[0]):
        H = unit_design(h_w[i], E_w_by_cohort[cohort[i]])
        A += H.T @ P @ H
        b += H.T @ P @ R[i]
    return A, b


def node_stats_factored(h_w, cohort, E_w_by_cohort, P, R):
    """The same statistics without ever forming ``H_i``.

    ``A = sum_g G_g (x) Q_g`` with ``G_g = sum_{i in g} h_i h_i'`` and
    ``Q_g = E_g' P E_g``;  ``b = sum_i h_i (x) (E_g' P r_i)``.
    This is the form the JAX engine must use.
    """
    K = h_w.shape[1]
    L = next(iter(E_w_by_cohort.values())).shape[1]
    A, b = np.zeros((K * L, K * L)), np.zeros(K * L)
    for g, E in E_w_by_cohort.items():
        rows = np.flatnonzero(cohort == g)
        if rows.size == 0:
            continue
        G = h_w[rows].T @ h_w[rows]
        A += np.kron(G, E.T @ P @ E)
        c = R[rows] @ P @ E                       # (n_g, L): row i is E' P r_i
        b += np.einsum("ik,il->kl", h_w[rows], c).reshape(-1)
    return A, b


def leaf_posterior(A, b, kappa):
    """Mean, covariance and integrated log likelihood of one leaf.

    ``log_ml`` is the log of the ratio between the leaf-integrated likelihood and
    the likelihood with the leaf coefficients set to zero, so it is 0 for an empty
    node and comparable across tree structures:

        log_ml = p/2 log(kappa) - 1/2 log|kappa I + A| + 1/2 b' (kappa I + A)^-1 b
    """
    p = b.size
    Lam = kappa * np.eye(p) + A
    chol = np.linalg.cholesky(Lam)
    sol = np.linalg.solve(Lam, b)
    log_ml = 0.5 * p * np.log(kappa) - np.sum(np.log(np.diag(chol))) + 0.5 * b @ sol
    return sol, np.linalg.inv(Lam), log_ml


def leaf_log_ml_bruteforce(h_w, cohort, E_w_by_cohort, V, R, kappa):
    """``log N(r; 0, blockdiag(V) + H H'/kappa) - log N(r; 0, blockdiag(V))``."""
    n, q = R.shape
    H = np.vstack([unit_design(h_w[i], E_w_by_cohort[cohort[i]]) for i in range(n)])
    big_V = np.kron(np.eye(n), V)
    r = R.reshape(-1)
    marginal = multivariate_normal(np.zeros(n * q), big_V + H @ H.T / kappa, allow_singular=False)
    null = multivariate_normal(np.zeros(n * q), big_V)
    return marginal.logpdf(r) - null.logpdf(r)


def draw_leaf(rng, A, b, kappa):
    """One exact draw: ``theta = Lam^-1 b + Lc'^-1 z`` with ``Lam = Lc Lc'``.

    ``solve(Lc, z)`` instead of ``solve(Lc.T, z)`` has the wrong covariance; the
    self-check below demonstrates the difference.
    """
    p = b.size
    Lam = kappa * np.eye(p) + A
    Lc = np.linalg.cholesky(Lam)
    return np.linalg.solve(Lam, b) + np.linalg.solve(Lc.T, rng.standard_normal(p))


# ---------------------------------------------------------------------------
# covariance conditionals
# ---------------------------------------------------------------------------


def inverse_wishart_conditional(nu0, S0, resid, beta=1.0):
    """``V | . ~ IW(nu0 + beta N, S0 + beta sum_i e_i e_i')`` (beta = inverse temperature).

    Convention: ``IW(nu, S)`` has density proportional to
    ``|V|^{-(nu + q + 1)/2} exp(-tr(S V^-1)/2)`` and mean ``S / (nu - q - 1)``; it is
    ``scipy.stats.invwishart(df=nu, scale=S)``.
    """
    return nu0 + beta * resid.shape[0], S0 + beta * resid.T @ resid


def difference_iid_conditional(a0, b0, resid, Omega0_inv, beta=1.0):
    """``sigma2 | . ~ IG(a0 + beta N q / 2, b0 + beta/2 sum_i e_i' Omega0^-1 e_i)`` for ``V = sigma2 Omega0``."""
    n, q = resid.shape
    quad = np.einsum("ij,jk,ik->", resid, Omega0_inv, resid)
    return a0 + beta * n * q / 2.0, b0 + beta * quad / 2.0


# ---------------------------------------------------------------------------
# exact tree space (bartz conventions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tree:
    """A tree as a sorted tuple of ``(node, var, cut)`` for its decision nodes."""

    rules: tuple

    def as_dict(self):
        return {node: (var, cut) for node, var, cut in self.rules}


def _ranges_after(ranges, var, cut, go_right):
    out = list(ranges)
    lo, hi = out[var]
    out[var] = (cut + 1, hi) if go_right else (lo, cut)
    return tuple(out)


def _enumerate_subtrees(node, depth, ranges, max_depth):
    """All subtrees rooted at ``node``; ``ranges[v] = (lo, hi)`` means cuts ``lo..hi-1`` are available."""
    yield ()
    if depth >= max_depth - 1:                     # bartz: nodes at the last level cannot split
        return
    for var, (lo, hi) in enumerate(ranges):
        for cut in range(lo, hi):
            lefts = list(_enumerate_subtrees(2 * node, depth + 1, _ranges_after(ranges, var, cut, False), max_depth))
            rights = list(_enumerate_subtrees(2 * node + 1, depth + 1, _ranges_after(ranges, var, cut, True), max_depth))
            for left, right in itertools.product(lefts, rights):
                yield ((node, var, cut),) + left + right


def enumerate_trees(max_split, max_depth):
    """Every legal tree. ``max_depth = 1`` is the root only, as in ``bartz.make_p_nonterminal``."""
    ranges = tuple((1, int(m) + 1) for m in max_split)
    return [Tree(tuple(sorted(rules))) for rules in _enumerate_subtrees(1, 0, ranges, max_depth)]


def tree_log_prior(tree, max_split, max_depth, alpha, beta):
    """bartz's tree prior: ``p_nt(depth) = alpha / (1 + depth)^beta`` where a rule is available.

    The rule prior is uniform over the variables with an available cut, then uniform
    over that variable's available cuts. Count thresholds are *not* part of it.
    """
    rules = tree.as_dict()

    def visit(node, depth, ranges):
        available = [v for v, (lo, hi) in enumerate(ranges) if hi > lo]
        can_split = bool(available) and depth < max_depth - 1
        p_nt = alpha / (1.0 + depth) ** beta if can_split else 0.0
        if node not in rules:
            return np.log1p(-p_nt)
        var, cut = rules[node]
        lo, hi = ranges[var]
        logp = np.log(p_nt) - np.log(len(available)) - np.log(hi - lo)
        logp += visit(2 * node, depth + 1, _ranges_after(ranges, var, cut, False))
        logp += visit(2 * node + 1, depth + 1, _ranges_after(ranges, var, cut, True))
        return logp

    return visit(1, 0, tuple((1, int(m) + 1) for m in max_split))


def sample_tree_prior(rng, max_split, max_depth, alpha, beta):
    """One draw from bartz's tree prior, for prior simulators (Geweke tree sizes, SBC).

    Count thresholds are not part of the prior: simulators that use this function
    must fit with the minimum-units options switched off.
    """
    rules = []

    def grow(node, depth, ranges):
        available = [v for v, (lo, hi) in enumerate(ranges) if hi > lo]
        if not available or depth >= max_depth - 1:
            return
        if rng.uniform() >= alpha / (1.0 + depth) ** beta:
            return
        var = available[rng.integers(len(available))]
        lo, hi = ranges[var]
        cut = int(rng.integers(lo, hi))
        rules.append((node, var, cut))
        grow(2 * node, depth + 1, _ranges_after(ranges, var, cut, False))
        grow(2 * node + 1, depth + 1, _ranges_after(ranges, var, cut, True))

    grow(1, 0, tuple((1, int(m) + 1) for m in max_split))
    return Tree(tuple(sorted(rules)))


def leaf_membership(tree, X):
    """Heap index of the leaf each column of the binned ``X`` (shape ``(p, n)``) falls into."""
    rules = tree.as_dict()
    node = np.ones(X.shape[1], dtype=np.int64)
    moved = True
    while moved:
        moved = False
        for n_id, (var, cut) in rules.items():
            here = node == n_id
            if here.any():
                node[here] = 2 * n_id + (X[var, here] >= cut)
                moved = True
    return node


def exact_tree_posterior(trees, X, h_w, cohort, contributes, E_w_by_cohort, P, R, kappa,
                         max_split, max_depth, alpha, beta, min_units_per_leaf=None):
    """Normalized posterior over ``trees`` with the leaf coefficients integrated out.

    ``contributes`` marks the units that carry information about this forest
    (treated units for a treatment forest). ``min_units_per_leaf`` applies bartz's
    veto semantics: trees with a leaf holding fewer contributing units are outside
    the support, and the posterior is renormalized over what remains.
    """
    log_post = np.full(len(trees), -np.inf)
    for k, tree in enumerate(trees):
        leaves = leaf_membership(tree, X)
        ok, total = True, 0.0
        for leaf in np.unique(leaves):
            rows = np.flatnonzero((leaves == leaf) & contributes)
            if min_units_per_leaf is not None and rows.size < min_units_per_leaf:
                ok = False
                break
            if rows.size:
                A, b = node_stats_factored(h_w[rows], cohort[rows], E_w_by_cohort, P, R[rows])
                total += leaf_posterior(A, b, kappa)[2]
        if ok:
            log_post[k] = total + tree_log_prior(tree, max_split, max_depth, alpha, beta)
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    return post / post.sum()


def exact_forest_posterior(trees, num_trees, X, H_w, contributes, P, R, kappa,
                           max_split, max_depth, alpha, beta):
    """Exact posterior over every ``num_trees``-tuple of trees, all leaves integrated jointly.

    The leaves of different trees are *not* independent a posteriori, so this is
    not a product of single-tree posteriors: each tuple's integrated likelihood
    comes from the joint Gaussian linear model whose design places unit ``i``'s
    block ``H_w[i]`` (shape ``(q, p)``) in the columns of the leaf it occupies in
    every tree. A backfitting sampler that updates one tree at a time, conditional
    on the others' current leaf values, must have this as the stationary law of its
    tree structures. ``kappa`` is ``num_trees / tau**2``. Feasible only for tiny
    spaces: the cost is ``len(trees) ** num_trees`` dense factorizations.
    """
    n, q, p = H_w.shape
    memberships, log_priors = [], []
    for tree in trees:
        leaves = leaf_membership(tree, X)
        ids = {leaf: k for k, leaf in enumerate(np.unique(leaves))}
        memberships.append((np.array([ids[leaf] for leaf in leaves]), len(ids)))
        log_priors.append(tree_log_prior(tree, max_split, max_depth, alpha, beta))
    Hc = H_w * contributes[:, None, None]
    tuples = list(itertools.product(range(len(trees)), repeat=num_trees))
    log_post = np.empty(len(tuples))
    for k, combo in enumerate(tuples):
        widths = [memberships[t][1] * p for t in combo]
        offsets = np.concatenate([[0], np.cumsum(widths)])
        Z = np.zeros((n, q, offsets[-1]))
        for slot, t in enumerate(combo):
            member = memberships[t][0]
            for i in range(n):
                start = offsets[slot] + member[i] * p
                Z[i, :, start:start + p] = Hc[i]
        A = np.einsum("nqa,qr,nrb->ab", Z, P, Z)
        b = np.einsum("nqa,qr,nr->a", Z, P, R)
        log_post[k] = leaf_posterior(A, b, kappa)[2] + sum(log_priors[t] for t in combo)
    post = np.exp(log_post - log_post.max())
    return tuples, post / post.sum()


# ---------------------------------------------------------------------------
# fixed-tree joint posterior
# ---------------------------------------------------------------------------


def joint_gaussian_posterior(Z_by_unit, P, v, prior_prec_diag):
    """Posterior of ``theta`` in ``v_i = Z_i theta + e_i``, ``e_i ~ N(0, P^-1)``, ``theta ~ N(0, diag^-1)``.

    Stack every leaf coefficient of every tree of both forests into ``theta`` and
    build ``Z_i`` by placing unit ``i``'s design block in the columns of the leaf it
    falls into. A Gibbs sampler that backfits tree by tree with the trees held
    fixed must reproduce this mean and covariance.
    """
    Lam = np.diag(np.asarray(prior_prec_diag, dtype=np.float64))
    rhs = np.zeros(Lam.shape[0])
    for Z, vi in zip(Z_by_unit, v):
        Lam = Lam + Z.T @ P @ Z
        rhs = rhs + Z.T @ P @ vi
    cov = np.linalg.inv(Lam)
    return cov @ rhs, cov


# ---------------------------------------------------------------------------
# level-model bias (why the likelihood is written on differences)
# ---------------------------------------------------------------------------


def random_intercept_level_bias(a, T, sigma2, sigma_gamma2, post_share):
    """Large-sample bias of a random-intercept level model under selection on levels.

    Treated units' intercepts are shifted by ``a``; parallel trends holds exactly.
    ``theta = 1 - sqrt(sigma2 / (sigma2 + T sigma_gamma2))`` is the quasi-demeaning
    weight and ``post_share`` the fraction of treated periods. The difference
    likelihood has zero bias for every ``a``.
    """
    theta = 1.0 - np.sqrt(sigma2 / (sigma2 + T * sigma_gamma2))
    return a * (1.0 - theta) ** 2 / (1.0 - theta * post_share * (2.0 - theta))


# ---------------------------------------------------------------------------
# self-checks
# ---------------------------------------------------------------------------


def _fixture(rng, n=14, T=4, K=4, lam=4.0):
    q, cohorts = T - 1, (2, 3)
    L = T - min(cohorts) + 1
    cohort = np.array([2, 3] * (n // 2))
    dose = rng.uniform(0.1, 1.0, n)
    knots = clamped_knots(0.1, 1.0, np.quantile(dose, np.linspace(0, 1, K - 2)[1:-1]))
    h = bspline_basis(dose, knots)
    C_d = dose_prior_cov(h, lam)
    K_s = exposure_kernel(L)
    L_d, L_s = np.linalg.cholesky(C_d), np.linalg.cholesky(K_s)
    h_w = h @ L_d
    E_w = {g: exposure_design(T, g, L) @ L_s for g in cohorts}
    M = rng.standard_normal((q, q))
    V = M @ M.T / q + 0.5 * np.eye(q)
    R = rng.standard_normal((n, q))
    return dict(n=n, T=T, q=q, K=K, L=L, cohort=cohort, dose=dose, knots=knots, h=h, h_w=h_w,
                C_d=C_d, K_s=K_s, L_d=L_d, L_s=L_s, E_w=E_w, V=V, P=np.linalg.inv(V), R=R)


def run_self_checks(seed: int = 20260917) -> None:
    rng = np.random.default_rng(seed)
    fx = _fixture(rng)
    T, q, L, K = fx["T"], fx["q"], fx["L"], fx["K"]

    # 1. increments sum to the level effect, for every cohort
    for g in (2, 3):
        f = rng.standard_normal(L)
        level = np.array([f[t - g] if t >= g else 0.0 for t in range(1, T + 1)])
        assert np.allclose(np.cumsum(exposure_design(T, g, L) @ f), level[1:])
        assert np.allclose(difference_matrix(T) @ level, exposure_design(T, g, L) @ f)

    # 2. basis: partition of unity, full rank, derivative against finite differences
    grid = np.linspace(0.12, 0.98, 41)
    B = bspline_basis(grid, fx["knots"])
    assert np.allclose(B.sum(axis=1), 1.0) and np.linalg.matrix_rank(B) == K
    step = 1e-6
    fd = (bspline_basis(grid + step, fx["knots"]) - bspline_basis(grid - step, fx["knots"])) / (2 * step)
    assert np.allclose(bspline_basis(grid, fx["knots"], deriv=1), fd, atol=1e-5)

    # 3. prior normalization, and the Kronecker order of row-major flattening
    assert np.isclose(np.mean(np.einsum("ik,kl,il->i", fx["h"], fx["C_d"], fx["h"])), 1.0)
    assert np.allclose(np.diag(fx["K_s"]).mean(), 1.0)
    W = rng.standard_normal((K, L))
    h0, s0 = fx["h"][0], 1
    assert np.isclose(h0 @ W[:, s0], np.kron(h0, np.eye(L)[s0]) @ W.reshape(-1))
    big = np.kron(fx["C_d"], fx["K_s"])                       # covariance of W.reshape(-1)
    assert np.allclose(np.kron(fx["L_d"], fx["L_s"]) @ np.kron(fx["L_d"], fx["L_s"]).T, big)
    # whitening: f = h' W e_s with vec(W) = kron(L_d, L_s) theta  ==  (L_d'h)' U (L_s'e_s)
    U = rng.standard_normal((K, L))
    W_phys = (np.kron(fx["L_d"], fx["L_s"]) @ U.reshape(-1)).reshape(K, L)
    assert np.isclose(h0 @ W_phys[:, s0], (fx["L_d"].T @ h0) @ U @ (fx["L_s"].T @ np.eye(L)[s0]))

    # 4. factored statistics == dense statistics
    A1, b1 = node_stats_dense(fx["h_w"], fx["cohort"], fx["E_w"], fx["P"], fx["R"])
    A2, b2 = node_stats_factored(fx["h_w"], fx["cohort"], fx["E_w"], fx["P"], fx["R"])
    assert np.allclose(A1, A2) and np.allclose(b1, b2)

    # 5. integrated likelihood == brute-force marginal density ratio
    kappa = 30.0 / 1.0 ** 2
    _, cov, log_ml = leaf_posterior(A2, b2, kappa)
    brute = leaf_log_ml_bruteforce(fx["h_w"], fx["cohort"], fx["E_w"], fx["V"], fx["R"], kappa)
    assert np.isclose(log_ml, brute, rtol=0, atol=1e-8), (log_ml, brute)
    assert np.isclose(leaf_posterior(np.zeros_like(A2), np.zeros_like(b2), kappa)[2], 0.0)

    # 6. Cholesky orientation of the leaf draw. The two orientations coincide when
    #    the precision is nearly isotropic, so this uses an informative node
    #    (weak prior, data-dominated precision) where they differ visibly.
    k6 = 0.05
    cov6 = leaf_posterior(A2, b2, k6)[1]
    draws = np.array([draw_leaf(rng, A2, b2, k6) for _ in range(40000)])
    err_right = np.abs(np.cov(draws.T) - cov6).max() / np.abs(cov6).max()
    Lc = np.linalg.cholesky(k6 * np.eye(b2.size) + A2)
    wrong = np.array([np.linalg.solve(Lc, rng.standard_normal(b2.size)) for _ in range(40000)])
    err_wrong = np.abs(np.cov(wrong.T) - cov6).max() / np.abs(cov6).max()
    assert err_right < 0.05 < 0.25 < err_wrong, (err_right, err_wrong)

    # 7. the tree prior is a probability distribution over the enumerated space
    max_split, max_depth, alpha, beta = np.array([2, 1]), 3, 0.5, 1.0
    trees = enumerate_trees(max_split, max_depth)
    total = np.sum(np.exp([tree_log_prior(t, max_split, max_depth, alpha, beta) for t in trees]))
    assert np.isclose(total, 1.0), total
    assert len(set(trees)) == len(trees)

    #    ... and the forward prior sampler draws from that same distribution
    draws = collections.Counter(sample_tree_prior(rng, max_split, max_depth, alpha, beta).rules
                                for _ in range(40000))
    prior_by_rules = {t.rules: np.exp(tree_log_prior(t, max_split, max_depth, alpha, beta)) for t in trees}
    assert set(draws) <= set(prior_by_rules)
    assert 0.5 * sum(abs(draws.get(r, 0) / 40000 - pr) for r, pr in prior_by_rules.items()) < 0.01

    # 8. exact posterior: sums to one, reduces to the prior without information,
    #    and the count veto removes exactly the trees with an under-filled leaf
    X = np.vstack([rng.integers(0, 3, fx["n"]), rng.integers(0, 2, fx["n"])])
    everyone = np.ones(fx["n"], bool)
    args = (trees, X, fx["h_w"], fx["cohort"], everyone, fx["E_w"])
    tail = (kappa, max_split, max_depth, alpha, beta)
    post = exact_tree_posterior(*args, fx["P"], fx["R"], *tail)
    assert np.isclose(post.sum(), 1.0)
    flat = exact_tree_posterior(*args, 1e-12 * fx["P"], fx["R"], *tail)
    prior = np.exp([tree_log_prior(t, max_split, max_depth, alpha, beta) for t in trees])
    assert np.allclose(flat, prior, atol=1e-8)
    vetoed = exact_tree_posterior(*args, fx["P"], fx["R"], *tail, min_units_per_leaf=3)
    for tree, pr in zip(trees, vetoed):
        counts = np.unique(leaf_membership(tree, X), return_counts=True)[1]
        assert (pr > 0) == bool(counts.min() >= 3)

    #    ... and the forest-level oracle reduces to it for a single tree
    H_w = np.stack([unit_design(fx["h_w"][i], fx["E_w"][fx["cohort"][i]]) for i in range(fx["n"])])
    _, forest_post = exact_forest_posterior(trees, 1, X, H_w, everyone, fx["P"], fx["R"], *tail)
    assert np.allclose(forest_post, post)

    # 9. difference likelihood == level likelihood with a flat unit intercept, constants included
    sigma2, y, mu = 0.7, rng.standard_normal(T), rng.standard_normal(T)
    A = difference_matrix(T)
    lhs = multivariate_normal(A @ mu, sigma2 * A @ A.T).logpdf(A @ y)
    Mproj = np.eye(T) - np.ones((T, T)) / T
    rhs = (-(T - 1) / 2 * np.log(2 * np.pi * sigma2) - 0.5 * np.log(T)
           - 0.5 * (y - mu) @ Mproj @ (y - mu) / sigma2)
    assert np.isclose(lhs, rhs)
    assert np.allclose(A @ A.T, 2 * np.eye(q) - np.eye(q, k=1) - np.eye(q, k=-1))

    # 10. covariance conditionals
    nu_n, S_n = inverse_wishart_conditional(q + 3.0, np.eye(q), fx["R"])
    assert nu_n == q + 3.0 + fx["n"] and np.allclose(S_n, np.eye(q) + fx["R"].T @ fx["R"])
    a_n, b_n = difference_iid_conditional(2.0, 0.5, fx["R"], np.linalg.inv(A @ A.T))
    assert np.isclose(a_n, 2.0 + fx["n"] * q / 2)

    # 11. the measured bias of the level model (see plan appendix A.1)
    assert np.isclose(random_intercept_level_bias(2.0, 2, 0.25, 2.0, 0.5), 0.22, atol=0.01)
    assert np.isclose(random_intercept_level_bias(2.0, 6, 0.25, 2.0, 0.5), 0.08, atol=0.01)

    print(f"dose_reference: all self-checks passed ({len(trees)} trees enumerated).")


if __name__ == "__main__":
    run_self_checks()
