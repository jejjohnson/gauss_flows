"""Tests for information-theoretic measures."""

import jax.numpy as jnp

import gauss_flows


def test_entropy_returns_scalar(key):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    h = gauss_flows.entropy(flow, n_samples=200, key=key)
    assert h.shape == ()
    assert jnp.isfinite(h)


def test_entropy_positive(key):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    h = gauss_flows.entropy(flow, n_samples=200, key=key)
    assert h > 0


def test_total_correlation_returns_scalar(key):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    tc = gauss_flows.total_correlation(flow, n_samples=200, key=key)
    assert tc.shape == ()
    assert jnp.isfinite(tc)


def test_kl_divergence_self_is_zero(key, key2):
    """KL(P || P) should be approximately zero."""
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    kl = gauss_flows.kl_divergence(flow, flow, n_samples=1000, key=key2)
    assert jnp.isfinite(kl)
    assert jnp.abs(kl) < 1.0  # Should be close to zero


def test_negentropy_nonnegative(key):
    """Negentropy should be non-negative (Gaussian has minimum entropy)."""
    from flowjax.distributions import Normal

    gauss = Normal(jnp.zeros(2))
    # For a Gaussian, negentropy ≈ 0
    j = gauss_flows.negentropy(gauss, n_samples=500, key=key)
    assert jnp.isfinite(j)
    assert j >= -0.5  # Allow for MC noise


def test_mutual_information_returns_scalar(key, key2):
    from flowjax.distributions import Normal

    dist_x = Normal(jnp.zeros(2))
    dist_y = Normal(jnp.zeros(2))
    dist_xy = Normal(jnp.zeros(4))
    mi = gauss_flows.mutual_information(dist_xy, dist_x, dist_y, n_samples=200, key=key)
    assert mi.shape == ()
    assert jnp.isfinite(mi)
