"""Tests for spherical base distributions."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Identity
from flowjax.distributions import Transformed
from scipy.special import gammaln, iv
from scipy.stats import vonmises_fisher

from gauss_flows import UniformOnSphere, VonMisesFisher


def _log_surface_area_sphere(d: int) -> float:
    half = 0.5 * (d + 1)
    return float(jnp.log(2.0) + half * jnp.log(jnp.pi) - gammaln(half))


def _normalize(x):
    return x / jnp.linalg.norm(x)


@pytest.mark.parametrize("d", [1, 2, 3, 7])
def test_uniform_on_sphere_sample_shape_and_norm(key, d):
    dist = UniformOnSphere(d)
    samples = dist.sample(jr.fold_in(key, d), sample_shape=(128,))
    assert samples.shape == (128, d + 1)
    assert jnp.allclose(jnp.linalg.norm(samples, axis=-1), 1.0, atol=1e-6)


@pytest.mark.parametrize("d", [1, 2, 3, 7])
def test_uniform_on_sphere_log_prob_matches_surface_area(key, d):
    dist = UniformOnSphere(d)
    x = _normalize(jr.normal(jr.fold_in(key, d), (d + 1,)))
    expected = -_log_surface_area_sphere(d)
    assert jnp.allclose(dist.log_prob(x), expected, atol=0.0)


@pytest.mark.parametrize("d", [1, 2, 3, 7])
def test_uniform_on_sphere_sample_mean_near_zero(key, d):
    dist = UniformOnSphere(d)
    samples = dist.sample(jr.fold_in(key, 100 + d), sample_shape=(20_000,))
    assert jnp.allclose(samples.mean(axis=0), 0.0, atol=2e-2)


def test_uniform_on_sphere_works_with_transformed_distribution(key):
    dist = UniformOnSphere(2)
    transformed = Transformed(dist, Identity(dist.shape))
    sample = transformed.sample(key)
    assert sample.shape == (3,)
    assert jnp.isfinite(transformed.log_prob(sample))


@pytest.mark.parametrize("d", [1, 2, 3, 7])
@pytest.mark.parametrize("concentration", [0.0, 0.1, 1.0, 10.0])
def test_von_mises_fisher_log_prob_matches_scipy(key, d, concentration):
    mean = _normalize(
        jr.normal(jr.fold_in(key, 10 * d + int(concentration * 10)), (d + 1,))
    )
    x = _normalize(jr.normal(jr.fold_in(key, 999 + d), (d + 1,)))
    dist = VonMisesFisher(mean, concentration)
    if concentration == 0.0:
        expected = -_log_surface_area_sphere(d)
    else:
        expected = vonmises_fisher(mu=jnp.asarray(mean), kappa=concentration).logpdf(
            jnp.asarray(x)
        )
    assert jnp.allclose(dist.log_prob(x), expected, atol=1e-5)


@pytest.mark.parametrize("d, concentration", [(1, 1.0), (2, 5.0), (7, 10.0)])
def test_von_mises_fisher_sample_mean_matches_theory(key, d, concentration):
    mean = _normalize(jr.normal(jr.fold_in(key, d), (d + 1,)))
    dist = VonMisesFisher(mean, concentration)
    samples = dist.sample(jr.fold_in(key, 500 + d), sample_shape=(20_000,))
    empirical_mean = samples.mean(axis=0)
    nu = 0.5 * (d - 1)
    expected_mrl = (
        0.0
        if concentration == 0.0
        else iv(nu + 1.0, concentration) / iv(nu, concentration)
    )
    expected_mean = jnp.asarray(expected_mrl) * mean
    assert jnp.allclose(empirical_mean, expected_mean, atol=1e-2)


def test_von_mises_fisher_zero_concentration_matches_uniform(key):
    mean = jnp.array([0.0, 0.0, 1.0])
    vmf = VonMisesFisher(mean, 0.0)
    uniform = UniformOnSphere(2)
    x = uniform.sample(key)
    assert jnp.allclose(vmf.log_prob(x), uniform.log_prob(x), atol=1e-6)


def test_von_mises_fisher_jit_vmap_and_grad_smoke(key):
    mean = jnp.array([0.0, 0.0, 1.0])
    dist = VonMisesFisher(mean, 3.0)
    xs = dist.sample(jr.fold_in(key, 1), sample_shape=(16,))

    sample_fn = jax.jit(lambda k: dist.sample(k, (8,)))
    log_prob_fn = jax.jit(jax.vmap(dist.log_prob))
    assert sample_fn(jr.fold_in(key, 2)).shape == (8, 3)
    assert jnp.all(jnp.isfinite(log_prob_fn(xs)))

    grads = eqx.filter_grad(lambda vmf: vmf.log_prob(xs[0]))(dist)
    assert jnp.all(jnp.isfinite(grads.mean))
    assert jnp.isfinite(grads.concentration)


def test_von_mises_fisher_vmap_over_batch_of_distributions(key):
    """Constructor must be trace-safe so `vmap` across a batch of VMFs works."""
    k_means, _k_kappas, k_x = jr.split(key, 3)
    raw_means = jr.normal(k_means, (5, 3))
    means = raw_means / jnp.linalg.norm(raw_means, axis=-1, keepdims=True)
    concentrations = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    x = _normalize(jr.normal(k_x, (3,)))

    def log_prob_single(mean, concentration):
        return VonMisesFisher(mean, concentration).log_prob(x)

    log_probs = jax.vmap(log_prob_single)(means, concentrations)
    assert log_probs.shape == (5,)
    assert jnp.all(jnp.isfinite(log_probs))


def test_von_mises_fisher_promotes_integer_mean_to_float():
    """Integer-typed ``mean`` must not crash `jnp.finfo` calls downstream."""
    dist = VonMisesFisher(jnp.array([0, 0, 1]), 2.0)
    assert jnp.issubdtype(dist.mean.dtype, jnp.floating)
    assert jnp.issubdtype(dist.concentration.dtype, jnp.floating)
    x = jnp.array([0.0, 0.0, 1.0])
    assert jnp.isfinite(dist.log_prob(x))


def test_von_mises_fisher_zero_mean_propagates_nan(key):
    """Zero-vector ``mean`` is an invalid caller input; surface it as NaN."""
    dist = VonMisesFisher(jnp.zeros(3), 2.0)
    assert jnp.all(jnp.isnan(dist.mean))
    x = _normalize(jr.normal(jr.fold_in(key, 0), (3,)))
    assert jnp.isnan(dist.log_prob(x))


def test_von_mises_fisher_negative_concentration_propagates_nan(key):
    """Negative ``concentration`` is invalid; log_prob and samples must be NaN."""
    dist = VonMisesFisher(jnp.array([0.0, 0.0, 1.0]), -1.5)
    assert jnp.isnan(dist.concentration)
    x = jnp.array([0.0, 0.0, 1.0])
    assert jnp.isnan(dist.log_prob(x))
    # The Wood sampler is run with a safe placeholder κ so the while-loop
    # terminates, then tainted so the output propagates NaN.
    sample = dist.sample(jr.fold_in(key, 1))
    assert jnp.all(jnp.isnan(sample))


@pytest.mark.parametrize("concentration", [50.0, 200.0, 1000.0])
def test_von_mises_fisher_log_prob_large_kappa_matches_scipy(key, concentration):
    """Hybrid series/asymptotic Bessel must stay accurate well past κ ≈ 50."""
    mean = jnp.array([0.0, 0.0, 1.0])
    x = _normalize(jr.normal(jr.fold_in(key, int(concentration)), (3,)))
    dist = VonMisesFisher(mean, concentration)
    expected = vonmises_fisher(mu=jnp.asarray(mean), kappa=concentration).logpdf(
        jnp.asarray(x)
    )
    assert jnp.allclose(dist.log_prob(x), expected, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("d, concentration", [(21, 30.0), (51, 100.0), (101, 100.0)])
def test_von_mises_fisher_high_d_log_prob_is_finite(key, d, concentration):
    """High-ν VMFs must not trip the Hankel asymptotic inside its divergence
    region: Bessel branch selection has to gate on both κ and ν, or log_prob
    returns NaN for perfectly valid parameters.
    """
    mean = jnp.zeros(d + 1).at[0].set(1.0)
    x = _normalize(jr.normal(jr.fold_in(key, d), (d + 1,)))
    dist = VonMisesFisher(mean, concentration)
    log_prob = dist.log_prob(x)
    assert jnp.isfinite(log_prob)
    expected = vonmises_fisher(mu=jnp.asarray(mean), kappa=concentration).logpdf(
        jnp.asarray(x)
    )
    assert jnp.allclose(log_prob, expected, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize(
    "d, concentration",
    [(41, 800.0), (51, 500.0), (51, 1000.0)],
)
def test_von_mises_fisher_high_d_series_regime_matches_scipy(key, d, concentration):
    """High-ν, moderate-κ falls under the Hankel threshold so the Bessel
    series has to carry the accuracy burden. The series peak
    ``k ≈ (√(ν² + κ²) − ν) / 2`` lands above 256 for d ≥ ~41 at κ of a few
    hundred; truncating to 256 terms silently undercounts log I by 100+
    log-units. Regression test against scipy to pin down the new
    truncation depth.
    """
    mean = jnp.zeros(d + 1).at[0].set(1.0)
    x = _normalize(jr.normal(jr.fold_in(key, d + int(concentration)), (d + 1,)))
    dist = VonMisesFisher(mean, concentration)
    expected = vonmises_fisher(mu=jnp.asarray(mean), kappa=concentration).logpdf(
        jnp.asarray(x)
    )
    assert jnp.allclose(dist.log_prob(x), expected, atol=1e-2, rtol=1e-3)


def test_von_mises_fisher_sample_at_large_kappa_is_finite(key):
    """Stable Wood ``b`` must keep high-κ sampling from hanging or NaN-ing."""
    dist = VonMisesFisher(jnp.array([0.0, 0.0, 1.0]), 500.0)
    samples = dist.sample(jr.fold_in(key, 0), sample_shape=(32,))
    assert samples.shape == (32, 3)
    assert jnp.all(jnp.isfinite(samples))
    # Concentrated samples should cluster near the mean direction.
    assert (samples @ jnp.array([0.0, 0.0, 1.0])).min() > 0.5
