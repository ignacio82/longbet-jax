"""Independent input, probability, and distributional ordered-probit oracles."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from longbet import LongBetConfig
from longbet._ordinal import (category_probabilities, prepare_ordinal,
    sample_cutpoints, sample_cutpoints_marginalized, sample_ordinal_latents,
    sample_truncated_normal)


@pytest.mark.parametrize("K", [2, 3, 5])
@pytest.mark.parametrize("observed", [[0], [1], [0, 1]])
def test_preparation_empty_categories(K, observed):
    y = np.array([*observed, np.nan])
    prepared = prepare_ordinal(y, K)
    np.testing.assert_array_equal(prepared.labels[:-1], observed)
    assert prepared.labels[-1] == 0
    assert not prepared.obs_mask[-1]
    assert prepared.cutpoints.shape == (K - 2,)
    assert np.isfinite(prepared.cutpoints).all()
    assert (np.diff(np.r_[0, prepared.cutpoints]) > 0).all()


def test_preparation_sign_and_no_recoding():
    prepared = prepare_ordinal([0]*8 + [2, 4], 5)
    assert prepared.offset == pytest.approx(-0.8416212335729143)
    assert stats.norm.cdf(-prepared.offset) == pytest.approx(.8)
    np.testing.assert_array_equal(prepared.labels, [0]*8 + [2, 4])
    # The binary compatibility branch must use precisely the legacy calculation.
    for y in ([0]*10, [1]*10, [0]*8 + [1]*2):
        assert prepare_ordinal(y, 2).offset == float(stats.norm.ppf(
            np.clip(float(np.mean(y)), 1e-4, 1-1e-4)))


@pytest.mark.parametrize("K", [None, True, np.bool_(True), 1, 0, -3, 2., "3", np.nan])
def test_bad_category_count(K):
    with pytest.raises(ValueError, match="num_categories"):
        LongBetConfig(outcome="ordinal", num_categories=K)
    with pytest.raises(ValueError, match="num_categories"):
        prepare_ordinal([0, 1], K)


@pytest.mark.parametrize("scale", [True, np.bool_(True), 0, -1, np.inf, np.nan, "5"])
def test_bad_prior_scale(scale):
    with pytest.raises(ValueError, match="cutpoint_prior_scale"):
        LongBetConfig(cutpoint_prior_scale=scale)


def test_configuration_contract():
    for outcome in ("continuous", "binary"):
        with pytest.raises(ValueError, match="num_categories"):
            LongBetConfig(outcome=outcome, num_categories=3)
    with pytest.raises(ValueError, match="outcome"):
        LongBetConfig(outcome="nominal")
    cfg = LongBetConfig(outcome="ordinal", num_categories=np.int64(3))
    assert type(cfg.num_categories) is int


@pytest.mark.parametrize("y", [[0, 1.1], [-1, 2], [0, 3], [np.inf, 1],
                               [-np.inf, 0], [np.nan], ["0", "1"], [1j, 0]])
def test_bad_labels(y):
    with pytest.raises(ValueError):
        prepare_ordinal(y, 3)


@pytest.mark.parametrize("a,b", [(-np.inf, np.inf), (-1., 1.), (-.1, .2),
    (-2., 3.), (0., np.inf), (8., 9.), (12., np.inf), (-9., -8.),
    (-np.inf, -12.), (1., 1.0001), (-.0001, .0002)])
def test_interval_distribution(a, b):
    n = 30000
    draws = np.asarray(jax.jit(sample_truncated_normal)(
        jax.random.key(147), jnp.full(n, a), jnp.full(n, b)))
    assert np.isfinite(draws).all()
    assert (draws > a).all() and (draws < b).all()
    # Probability integral transform is independently evaluated by SciPy.
    # D=0.018 has a DKW tail bound below 8e-9 at this sample size.
    uniforms = stats.truncnorm.cdf(draws.astype(np.float64), a, b)
    assert stats.kstest(uniforms, "uniform").statistic < .018
    mean, var = stats.truncnorm.stats(a, b, moments="mv")
    assert abs(draws.astype(np.float64).mean() - mean) < 6*np.sqrt(var/n) + 2e-6


def test_jit_vmap_varying_latent_means_scales_and_missing_placeholders():
    n = 12000
    labels = jnp.arange(n) % 4
    mask = (jnp.arange(n) % 11) != 0
    mean = jnp.where(mask, jnp.sin(jnp.arange(n)), jnp.nan)
    sd = jnp.where(mask, .3 + (jnp.arange(n) % 3), jnp.nan)
    keys = jax.random.split(jax.random.key(843), 2)
    cuts = jnp.array([[1., 2.5], [.2, 1.3]])
    def sample(key, cp):
        return sample_ordinal_latents(key, labels, mean, sd, cp, mask, jnp.full(n, jnp.nan))
    draws = np.asarray(jax.jit(jax.vmap(sample))(keys, cuts))
    for c in range(2):
        full = np.r_[-np.inf, 0, np.array(cuts[c]), np.inf]
        sel = np.asarray(mask)
        lo = full[np.asarray(labels)[sel]]
        hi = full[np.asarray(labels)[sel] + 1]
        assert (draws[c, sel] > lo).all() and (draws[c, sel] < hi).all()
        assert (draws[c, ~sel] == 0).all()
        m, s = np.asarray(mean)[sel], np.asarray(sd)[sel]
        u = stats.truncnorm.cdf(draws[c, sel], (lo-m)/s, (hi-m)/s, loc=m, scale=s)
        assert stats.kstest(u, "uniform").statistic < .025


@pytest.mark.parametrize("lower,upper", [(1., 1.), (2., 1.), (np.nan, 2.),
    (1., np.nextafter(np.float32(1), np.float32(2)))])
def test_invalid_intervals_fail(lower, upper):
    with pytest.raises(Exception, match="ordinal numerical failure"):
        jax.jit(sample_truncated_normal)(jax.random.key(1), lower, upper).block_until_ready()


@pytest.mark.parametrize("top_empty", [False, True])
def test_cutpoint_conditional_is_normal_prior(top_empty):
    z = jnp.array([-1., .2, 1., 4.])
    labels = jnp.array([0, 1, 1, 2])
    mask = jnp.array([True, True, True, not top_empty])
    keys = jax.random.split(jax.random.key(904), 20000)
    draws = np.asarray(jax.jit(jax.vmap(lambda k: sample_cutpoints(
        k, z, labels, jnp.array([2.]), mask, 1.5)))(keys))[:, 0]
    hi = np.inf if top_empty else 4.
    u = stats.truncnorm.cdf(draws, 1/1.5, hi/1.5, scale=1.5)
    assert stats.kstest(u, "uniform").statistic < .022
    assert (draws > 1).all() and (draws < hi).all()


def test_empty_adjacent_categories_use_current_neighbor():
    @jax.jit
    def run(key):
        def step(cp, k):
            cp = sample_cutpoints(k, jnp.array([-.5]), jnp.array([0]), cp,
                                  jnp.array([True]), 2.)
            return cp, cp
        return jax.lax.scan(step, jnp.array([.2, .5, .9]), jax.random.split(key, 15000))[1]
    draws = np.asarray(run(jax.random.key(921)))[1000:]
    assert (np.diff(np.c_[np.zeros(len(draws)), draws], axis=1) > 0).all()
    # With only category zero observed, the posterior is exactly the ordered
    # half-normal prior. Sum-of-squares / scale^2 has expectation K-2.
    assert np.mean(np.sum((draws/2)**2, axis=1)) == pytest.approx(3., abs=.15)


def test_probabilities_pair_draws_and_tail_accuracy():
    means = np.array([[0, 1, -12, 12], [1, -1, -8, 8.]])
    cuts = np.array([[1.], [2.]])
    p = category_probabilities(means, cuts)
    assert p.shape == (2, 4, 3)
    np.testing.assert_allclose(p[0, 0], [.5, stats.norm.cdf(1)-.5, stats.norm.sf(1)])
    np.testing.assert_allclose(p.sum(-1), 1., atol=3e-16)
    assert (p >= 0).all() and np.isfinite(p).all()
    assert p[0, 2, 1] > 0  # Direct CDF subtraction would return zero here.
    np.testing.assert_allclose(p[0, 2, 1], stats.norm.sf(12)-stats.norm.sf(13), rtol=3e-14)
    p1 = category_probabilities(means+.2, cuts)
    assert ((p1-p) @ np.arange(3) >= -1e-15).all()
    np.testing.assert_allclose((p1-p).sum(-1), 0., atol=3e-16)
    binary = category_probabilities(means, np.empty((2, 0)))
    np.testing.assert_allclose(binary[:, :, 1], stats.norm.cdf(means), rtol=3e-14)


def test_sample_cutpoints_marginalized_contract():
    key = jax.random.key(123)
    # K=2 returns empty array
    k2 = sample_cutpoints_marginalized(key, jnp.empty(0, jnp.float32), jnp.array([0, 1]),
                                       jnp.zeros(2), 1.0, jnp.ones(2, bool), 5.0)
    assert k2.shape == (0,)

    # K=4 produces positive strictly ordered thresholds
    cp_init = jnp.array([1.0, 2.0], jnp.float32)
    labels = jnp.array([0, 1, 2, 3, 0, 1], jnp.float32)
    mean = jnp.array([-0.5, 0.2, 1.1, 2.3, -0.1, 0.4], jnp.float32)
    mask = jnp.array([True, True, True, True, True, False])
    cp_new = jax.jit(sample_cutpoints_marginalized)(key, cp_init, labels, mean, 1.0, mask, 5.0)
    assert cp_new.shape == (2,)
    assert (cp_new > 0).all()
    assert cp_new[0] < cp_new[1]
    assert np.isfinite(cp_new).all()

    # Masked values cannot affect the draw
    labels_alt = labels.at[5].set(999.0)
    cp_alt = jax.jit(sample_cutpoints_marginalized)(key, cp_init, labels_alt, mean, 1.0, mask, 5.0)
    np.testing.assert_array_equal(cp_new, cp_alt)

