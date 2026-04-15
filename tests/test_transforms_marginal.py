"""Tests for marginal transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats

from gauss_flows import (
    HistogramCDF,
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


def test_histogram_cdf_forward_inverse(key):
    shape = (3,)
    data = jr.normal(key, (5000, shape[0]))
    transform = HistogramCDF(n_bins=64, shape=shape).fit(data)
    x = jr.normal(key, shape)
    y, log_det_f = transform.transform_and_log_det(x)
    x_rec, log_det_i = transform.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-3)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-5)


def test_histogram_cdf_matches_gaussian_cdf(key):
    shape = (1,)
    data = jr.normal(key, (20000, shape[0]))
    transform = HistogramCDF(n_bins=256, shape=shape).fit(data)
    points = jnp.linspace(-2.0, 2.0, 21)[:, None]

    def _single_cdf(x_point):
        y, _ = transform.transform_and_log_det(x_point)
        return y[0]

    cdf_vals = jax.vmap(_single_cdf)(points)
    reference = jstats.norm.cdf(points[:, 0])
    assert jnp.mean(jnp.abs(cdf_vals - reference)) < 0.02


def test_histogram_cdf_idempotent_fit(key):
    shape = (2,)
    data = jr.normal(key, (1000, shape[0]))
    transform = HistogramCDF(n_bins=32, shape=shape)
    fitted_once = transform.fit(data)
    fitted_twice = fitted_once.fit(data)
    assert jnp.allclose(fitted_once.bin_edges, fitted_twice.bin_edges)
    assert jnp.allclose(fitted_once.bin_pdf, fitted_twice.bin_pdf)


def test_histogram_cdf_out_of_range():
    shape = (1,)
    transform = HistogramCDF(n_bins=8, shape=shape)
    data = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    fitted = transform.fit(data)
    y, log_det = fitted.transform_and_log_det(jnp.array([-5.0]))
    assert jnp.allclose(y, jnp.array([0.0]))
    assert log_det == -jnp.inf


def test_histogram_cdf_boundary_log_det_consistent():
    """``x == edges[0/−1]`` in forward and ``y == 0/1`` in inverse are both
    in-range (finite log_det). ``y < 0`` or ``y > 1`` is still -inf."""
    shape = (1,)
    transform = HistogramCDF(n_bins=8, shape=shape)
    data = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    fitted = transform.fit(data)

    # Forward at left-most edge: inside, finite log_det.
    _, log_det_f_left = fitted.transform_and_log_det(fitted.bin_edges[0, 0:1])
    assert jnp.isfinite(log_det_f_left)

    # Inverse at y == 0: also inside (boundary), finite log_det (strict < now).
    _, log_det_i_zero = fitted.inverse_and_log_det(jnp.array([0.0]))
    assert jnp.isfinite(log_det_i_zero)

    # Inverse at y == 1: same — finite log_det.
    _, log_det_i_one = fitted.inverse_and_log_det(jnp.array([1.0]))
    assert jnp.isfinite(log_det_i_one)

    # Inverse at y > 1: truly out of range, log_det is -inf.
    _, log_det_i_above = fitted.inverse_and_log_det(jnp.array([1.5]))
    assert log_det_i_above == -jnp.inf
