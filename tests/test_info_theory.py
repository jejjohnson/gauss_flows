"""Tests for information-theoretic measures."""

import jax
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


# ---------------------------------------------------------------------------
# Analytical Gaussian closed forms
# ---------------------------------------------------------------------------


def test_gaussian_entropy_matches_known():
    """Entropy of an identity-covariance Gaussian: d/2 * (1 + log(2*pi))."""
    cov = jnp.eye(3)
    h = gauss_flows.gaussian_entropy(cov)
    expected = 3 / 2 * (1 + jnp.log(2 * jnp.pi))
    assert jnp.allclose(h, expected, atol=1e-5)


def test_gaussian_total_correlation_diagonal_is_zero():
    """A diagonal covariance has zero total correlation."""
    cov = jnp.diag(jnp.array([1.0, 2.0, 0.5]))
    tc = gauss_flows.gaussian_total_correlation(cov)
    assert jnp.allclose(tc, 0.0, atol=1e-5)


def test_gaussian_mutual_information_independent_is_zero():
    """Block-diagonal joint covariance has zero mutual information."""
    cov = jnp.eye(4)
    mi = gauss_flows.gaussian_mutual_information(cov, dim_x=2)
    assert jnp.allclose(mi, 0.0, atol=1e-5)


def test_gaussian_mutual_information_positive_when_correlated():
    cov = jnp.eye(4).at[0, 2].set(0.8).at[2, 0].set(0.8)
    mi = gauss_flows.gaussian_mutual_information(cov, dim_x=2)
    assert mi > 0


def test_gaussian_kl_divergence_self_is_zero():
    cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])
    mu = jnp.array([0.5, -0.2])
    kl = gauss_flows.gaussian_kl_divergence(mu, cov, mu, cov)
    assert jnp.allclose(kl, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# RBIG-way (total-correlation reduction) measures
# ---------------------------------------------------------------------------


def test_information_reduction_identical_is_zero(key):
    x = jax.random.normal(key, (500, 3))
    red = gauss_flows.information_reduction(x, x)
    assert jnp.allclose(red, 0.0, atol=1e-6)


def test_total_correlation_reduction_matches_analytical(key):
    """RBIG-way TC of a correlated Gaussian should track the analytical value."""
    cov = (
        jnp.eye(4)
        .at[0, 2]
        .set(0.8)
        .at[2, 0]
        .set(0.8)
        .at[1, 3]
        .set(0.5)
        .at[3, 1]
        .set(0.5)
    )
    chol = jnp.linalg.cholesky(cov)
    x = jax.random.normal(key, (2000, 4)) @ chol.T
    flow = gauss_flows.fit_rbig(x, n_layers=8, n_components=8)
    tc = gauss_flows.total_correlation_reduction(flow, x)
    tc_true = gauss_flows.gaussian_total_correlation(cov)
    assert jnp.isfinite(tc)
    assert jnp.abs(tc - tc_true) < 0.2


def test_entropy_reduction_returns_finite(key):
    x = jax.random.normal(key, (1000, 3))
    flow = gauss_flows.fit_rbig(x, n_layers=6, n_components=8)
    h = gauss_flows.entropy_reduction(flow, x)
    assert h.shape == ()
    assert jnp.isfinite(h)


def test_kl_divergence_reduction_self_is_small(key):
    x = jax.random.normal(key, (1500, 2))
    flow = gauss_flows.fit_rbig(x, n_layers=6, n_components=8)
    kl = gauss_flows.kl_divergence_reduction(flow, x)
    assert jnp.isfinite(kl)
    assert jnp.abs(kl) < 0.2


def test_kl_divergence_reduction_positive_for_shift(key, key2):
    x = jax.random.normal(key, (1500, 2))
    x_shift = jax.random.normal(key2, (1500, 2)) + jnp.array([1.0, 0.0])
    flow = gauss_flows.fit_rbig(x, n_layers=6, n_components=8)
    kl = gauss_flows.kl_divergence_reduction(flow, x_shift)
    assert kl > 0
