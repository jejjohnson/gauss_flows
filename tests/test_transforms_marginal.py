"""Tests for marginal transforms."""

import jax.numpy as jnp
import jax.random as jr

from gauss_flows import (
    InverseGaussCDF,
    MixtureGaussianCDF,
    MixtureLogisticCDF,
    RQSplineMarginal,
)


def test_mixture_gaussian_cdf_shape(key):
    shape = (4,)
    transform = MixtureGaussianCDF(n_components=4, shape=shape)
    x = jr.normal(key, shape)
    y, log_det = transform.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_mixture_gaussian_cdf_finite(key):
    shape = (2,)
    transform = MixtureGaussianCDF(n_components=4, shape=shape)
    x = jr.normal(key, shape)
    y, log_det = transform.transform_and_log_det(x)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.isfinite(log_det)


def test_mixture_logistic_cdf_shape(key):
    shape = (4,)
    transform = MixtureLogisticCDF(n_components=4, shape=shape)
    x = jr.normal(key, shape)
    y, log_det = transform.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_mixture_logistic_cdf_finite(key):
    shape = (2,)
    transform = MixtureLogisticCDF(n_components=4, shape=shape)
    x = jr.normal(key, shape)
    y, log_det = transform.transform_and_log_det(x)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.isfinite(log_det)


def test_rqspline_marginal_forward_inverse(key):
    shape = (4,)
    transform = RQSplineMarginal(n_bins=8, shape=shape)
    x = jr.normal(key, shape) * 0.5  # Keep within spline interval
    y, log_det_f = transform.transform_and_log_det(x)
    x_rec, log_det_i = transform.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-4)


def test_rqspline_marginal_shape(key):
    shape = (4,)
    transform = RQSplineMarginal(n_bins=8, shape=shape)
    x = jr.normal(key, shape) * 0.5
    y, log_det = transform.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_inverse_gauss_cdf_forward_inverse(key):
    shape = (4,)
    transform = InverseGaussCDF(shape)
    # Input should be in (0, 1) for InverseGaussCDF
    u = jnp.array([0.1, 0.3, 0.7, 0.9])
    y, log_det_f = transform.transform_and_log_det(u)
    u_rec, log_det_i = transform.inverse_and_log_det(y)
    assert jnp.allclose(u, u_rec, atol=1e-5)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-5)
