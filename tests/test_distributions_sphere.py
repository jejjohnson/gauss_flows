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
    samples = dist.sample(jr.fold_in(key, 100 + d), sample_shape=(100_000,))
    assert jnp.allclose(samples.mean(axis=0), 0.0, atol=1e-2)


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
