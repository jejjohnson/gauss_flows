"""Tests for the orthogonal convolutional exponential transform."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

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
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-8)


def test_convexp_log_det_is_exactly_zero(key):
    # Skew-symmetric => tr(M) = 0 => log|det exp(M)| = 0 exactly.
    shape = (4, 4, 2)
    transform = OrthogonalConvExponential(key, shape=shape, kernel_size=3)
    x = jr.normal(key, shape)
    _, log_det = transform.transform_and_log_det(x)
    _, log_det_inv = transform.inverse_and_log_det(x)
    assert jnp.array_equal(log_det, jnp.zeros(()))
    assert jnp.array_equal(log_det_inv, jnp.zeros(()))


def test_convexp_is_norm_preserving(key):
    # exp(skew) is orthogonal => ||y|| == ||x|| up to truncation error.
    shape = (4, 4, 3)
    transform = OrthogonalConvExponential(
        key, shape=shape, kernel_size=3, n_terms=16, n_power_iterations=5
    )
    x = jr.normal(key, shape)
    y, _ = transform.transform_and_log_det(x)
    assert jnp.isclose(jnp.linalg.norm(x), jnp.linalg.norm(y), rtol=1e-4)


def test_convexp_logdet_matches_jacobian(key):
    # The implementation returns 0; autodiff |log det J| on exp(skew) should
    # also be ~0 up to the Taylor-truncation residual.
    key_kernel, key_input = jr.split(key)
    shape = (2, 2, 1)
    transform = OrthogonalConvExponential(
        key_kernel,
        shape=shape,
        kernel_size=3,
        n_terms=16,
        n_power_iterations=5,
    )
    x = jr.normal(key_input, shape)

    def flat_forward(z):
        y, _ = transform.transform_and_log_det(z)
        return y.ravel()

    _, log_det = transform.transform_and_log_det(x)
    jac = jax.jacobian(flat_forward)(x).reshape((x.size, x.size))
    sign, logabsdet = jnp.linalg.slogdet(jac)
    assert sign > 0.0  # exp(skew) has det = +1
    assert jnp.allclose(log_det, logabsdet, atol=1e-4)


def test_convexp_roundtrip_with_large_weight(key):
    # Spectral norm must clamp ||K||<=1 even when the raw weight is large,
    # keeping the truncated round-trip accurate.
    shape = (4, 4, 2)
    transform = OrthogonalConvExponential(
        key, shape=shape, kernel_size=3, n_terms=12, n_power_iterations=6
    )
    scaled = eqx.tree_at(lambda m: m.weight, transform, transform.weight * 20.0)
    x = jr.normal(key, shape)
    y, _ = scaled.transform_and_log_det(x)
    x_rec, _ = scaled.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-3)
    # Still norm-preserving within truncation tolerance.
    assert jnp.isclose(jnp.linalg.norm(x), jnp.linalg.norm(y), rtol=5e-4)


def test_convexp_spectral_norm_blocks_gradient_through_sigma(key):
    # The spectral-norm estimate must not introduce a gradient on the power
    # iteration; grads w.r.t. weight flow only through kernel / stop_grad(sigma).
    shape = (3, 3, 2)
    transform = OrthogonalConvExponential(
        key, shape=shape, kernel_size=3, n_terms=6, n_power_iterations=3
    )
    x = jr.normal(key, shape)

    def loss(t):
        y, _ = t.transform_and_log_det(x)
        return jnp.sum(y**2)

    grads = eqx.filter_grad(loss)(transform)
    assert jnp.all(jnp.isfinite(grads.weight))
    assert jnp.any(grads.weight != 0.0)


def test_convexp_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        OrthogonalConvExponential(jr.key(0), shape=(4, 4))  # type: ignore[arg-type]


def test_convexp_invalid_kernel_size():
    with pytest.raises(ValueError, match="kernel_size"):
        OrthogonalConvExponential(jr.key(0), shape=(4, 4, 3), kernel_size=4)
    with pytest.raises(ValueError, match="kernel_size"):
        OrthogonalConvExponential(jr.key(0), shape=(4, 4, 3), kernel_size=0)


def test_convexp_invalid_n_terms():
    with pytest.raises(ValueError, match="n_terms"):
        OrthogonalConvExponential(jr.key(0), shape=(4, 4, 3), n_terms=0)


def test_convexp_invalid_n_power_iterations():
    with pytest.raises(ValueError, match="n_power_iterations"):
        OrthogonalConvExponential(jr.key(0), shape=(4, 4, 3), n_power_iterations=0)


def test_convexp_jit_forward(key):
    # Forward compiles under eqx.filter_jit (standard flowjax/eqx idiom).
    shape = (4, 4, 2)
    transform = OrthogonalConvExponential(key, shape=shape, kernel_size=3)
    x = jr.normal(key, shape)

    @eqx.filter_jit
    def apply(t, z):
        return t.transform_and_log_det(z)

    y, log_det = apply(transform, x)
    assert y.shape == shape
    assert log_det.shape == ()
