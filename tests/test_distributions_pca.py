"""Tests for the GaussianPCA (low-rank + diagonal Gaussian) base distribution."""

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import pytest

from gauss_flows import GaussianPCA, gaussianization_flow


def _reference_log_prob(dist: GaussianPCA, x):
    return jstats.multivariate_normal.logpdf(x, dist.loc, dist.covariance())


@pytest.mark.parametrize("latent_dim", [2, 5, 10])
def test_log_prob_matches_full_covariance_reference(key, latent_dim):
    k_init, k_x = jr.split(key)
    dist = GaussianPCA(k_init, event_shape=(10,), latent_dim=latent_dim)
    x = jr.normal(k_x, (10,))
    actual = dist.log_prob(x)
    expected = _reference_log_prob(dist, x)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_log_prob_batched(key):
    dist = GaussianPCA(key, event_shape=(6,), latent_dim=3)
    xs = jr.normal(jr.fold_in(key, 1), (32, 6))
    lp = dist.log_prob(xs)
    assert lp.shape == (32,)
    expected = jnp.asarray([_reference_log_prob(dist, x) for x in xs])
    assert jnp.allclose(lp, expected, atol=1e-5)


def test_sample_shape_and_finite(key):
    dist = GaussianPCA(key, event_shape=(5,), latent_dim=2)
    samples = dist.sample(jr.fold_in(key, 1), sample_shape=(16,))
    assert samples.shape == (16, 5)
    assert jnp.all(jnp.isfinite(samples))


def test_sample_moments_match_covariance(key):
    """Empirical mean/cov from 20k samples match loc and W W^T + sigma^2 I."""
    d, k = 5, 2
    dist = GaussianPCA(key, event_shape=(d,), latent_dim=k, loc=0.0, W_init_scale=0.5)
    samples = dist.sample(jr.fold_in(key, 1), sample_shape=(20_000,))
    empirical_mean = samples.mean(axis=0)
    centered = samples - empirical_mean
    empirical_cov = centered.T @ centered / samples.shape[0]
    assert jnp.allclose(empirical_mean, dist.loc, atol=0.05)
    assert jnp.allclose(empirical_cov, dist.covariance(), atol=0.1)


def test_as_base_dist_in_gaussianization_flow(key):
    k_base, k_flow, k_sample = jr.split(key, 3)
    base = GaussianPCA(k_base, event_shape=(4,), latent_dim=2)
    flow = gaussianization_flow(k_flow, n_dims=4, n_layers=2, base_dist=base)
    samples = flow.sample(k_sample, (8,))
    log_probs = flow.log_prob(samples)
    assert samples.shape == (8, 4)
    assert log_probs.shape == (8,)
    assert jnp.all(jnp.isfinite(log_probs))


def test_latent_dim_bounds(key):
    with pytest.raises(ValueError, match="latent_dim"):
        GaussianPCA(key, event_shape=(3,), latent_dim=0)
    with pytest.raises(ValueError, match="latent_dim"):
        GaussianPCA(key, event_shape=(3,), latent_dim=4)


def test_only_1d_event_shape(key):
    with pytest.raises(ValueError, match="1D event"):
        GaussianPCA(key, event_shape=(3, 2), latent_dim=1)


def test_custom_loc(key):
    loc = jnp.array([1.0, -2.0, 3.0])
    dist = GaussianPCA(key, event_shape=(3,), latent_dim=2, loc=loc)
    assert jnp.allclose(dist.loc, loc)


def test_log_prob_stable_for_very_negative_log_sigma(key):
    """log_sigma deeply negative shouldn't make sigma^2 underflow to zero
    and break Cholesky / Woodbury — the eps floor keeps log_prob finite.
    """
    import equinox as eqx

    dist = GaussianPCA(key, event_shape=(4,), latent_dim=2)
    dist = eqx.tree_at(lambda d: d.log_sigma, dist, jnp.asarray(-50.0))
    x = jr.normal(jr.fold_in(key, 1), (4,))
    lp = dist.log_prob(x)
    assert jnp.isfinite(lp)


def test_eps_must_be_non_negative(key):
    with pytest.raises(ValueError, match="eps"):
        GaussianPCA(key, event_shape=(3,), latent_dim=2, eps=-1.0)
