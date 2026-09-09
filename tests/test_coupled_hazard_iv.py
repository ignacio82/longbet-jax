"""Tests for coupled joint hazard-outcome IV model."""

import numpy as np
import pytest

from longbet._coupled_hazard_iv import (
    CoupledHazardConfig,
    CoupledHazardIV,
    CoupledHazardResult,
)


def test_coupled_hazard_iv_basic():
    np.random.seed(42)
    n = 60
    t_len = 5

    z = np.zeros((n, t_len))
    z[:30, 1:] = 1.0

    d = np.zeros((n, t_len))
    # Treated adopt earlier
    d[:30, 1:] = (np.random.rand(30, 4) < 0.7).astype(float)
    # Ensure absorbing
    d = np.maximum.accumulate(d, axis=1)

    y = 1.2 * d + np.random.normal(0, 0.5, size=(n, t_len))

    cfg = CoupledHazardConfig(num_sweeps=60, num_burnin=20, seed=123)
    model = CoupledHazardIV(cfg)
    res = model.fit(y, d, z)

    assert isinstance(res, CoupledHazardResult)
    assert len(res.beta_post) == 40
    assert len(res.rho_post) == 40
    assert not res.summary_table.empty
    assert res.beta_ci[0] < res.beta_ci[1]


def test_coupled_hazard_iv_endogenous_recovery():
    """Verify recovery when unobserved confounding (rho != 0) is present."""
    np.random.seed(42)
    n = 200
    t_len = 5

    # Instrument: encouraged at period 1
    z = np.zeros((n, t_len))
    z[:100, 1:] = 1.0

    true_rho = 0.5
    true_beta = 1.5
    sigma_y = 0.8

    # Generate correlated errors (eps_y, eps_d)
    cov_mat = np.array([[sigma_y**2, true_rho * sigma_y], [true_rho * sigma_y, 1.0]])
    errors = np.random.multivariate_normal([0, 0], cov_mat, size=(n, t_len))
    eps_y = errors[:, :, 0]
    eps_d = errors[:, :, 1]

    d = np.zeros((n, t_len))
    for t in range(t_len):
        if t == 0:
            continue
        # Active risk set: d[:, t-1] == 0
        risk = (d[:, t - 1] == 0)
        # Latent adoption utility
        u = -0.5 + 1.2 * z[:, t] + eps_d[:, t]
        adopting_now = risk & (u > 0)
        d[adopting_now, t:] = 1.0

    y = true_beta * d + eps_y

    cfg = CoupledHazardConfig(num_sweeps=100, num_burnin=30, seed=42)
    res = CoupledHazardIV(cfg).fit(y, d, z)

    # Beta should be close to true_beta and CI covers true_beta
    assert np.isclose(res.beta_mean, true_beta, atol=0.25)
    assert res.beta_ci[0] <= true_beta <= res.beta_ci[1]
    # Instrument relevance should be strongly detected
    assert res.xi_inclusion_prob > 0.8
    # Rho should be estimated positive
    assert res.rho_mean > 0.05


def test_coupled_hazard_null_instrument():
    """Verify spike-and-slab turns off when instrument has no relevance."""
    np.random.seed(555)
    n = 80
    t_len = 4

    z = np.zeros((n, t_len))
    z[:40, 1:] = 1.0  # Independent instrument

    # Adoption happens independently of z
    d = np.zeros((n, t_len))
    for t in range(1, t_len):
        risk = (d[:, t - 1] == 0)
        # Adoption purely random, z has zero effect
        adopt = risk & (np.random.rand(n) < 0.2)
        d[adopt, t:] = 1.0

    y = 0.5 * d + np.random.normal(0, 0.5, size=(n, t_len))

    cfg = CoupledHazardConfig(num_sweeps=80, num_burnin=20, seed=42)
    res = CoupledHazardIV(cfg).fit(y, d, z)

    # Under complete null, inclusion probability should drop
    assert res.xi_inclusion_prob < 0.5
