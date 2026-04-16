"""Tests for the orthogonal convolutional exponential transform."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from gauss_flows import OrthogonalConvExponential


def test_convexp_forward_inverse(key):
    shape = (4, 4, 3)
    transform = OrthogonalConvExponential(
        key,
        shape=shape,
        kernel_size=3,
        n_terms=10,
        n_power_iterations=3,
    )
    x = jr.normal(key, shape)
    y, log_det = transform.transform_and_log_det(x)
    x_rec, log_det_inv = transform.inverse_and_log_det(y)
    assert y.shape == shape
    assert log_det.shape == ()
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-4)


def test_convexp_logdet_matches_jacobian(key):
    key_kernel, key_input = jr.split(key)
    shape = (2, 2, 1)
    transform = OrthogonalConvExponential(
        key_kernel,
        shape=shape,
        kernel_size=3,
        n_terms=8,
        n_power_iterations=4,
    )
    x = jr.normal(key_input, shape)

    def flat_forward(z):
        y, _ = transform.transform_and_log_det(z)
        return y.ravel()

    y, log_det = transform.transform_and_log_det(x)
    jac = jax.jacobian(flat_forward)(x).reshape((x.size, x.size))
    sign, logabsdet = jnp.linalg.slogdet(jac)
    assert sign != 0.0
    assert y.shape == shape
    assert jnp.allclose(log_det, logabsdet, atol=1e-4)
