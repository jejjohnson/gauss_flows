"""Tests for rotation transforms."""

import jax.numpy as jnp
import jax.random as jr

from gauss_flows import FixedRotation, HouseholderRotation, OrthogonalRotation


def test_householder_rotation_forward_inverse(key):
    shape = (4,)
    rotation = HouseholderRotation(n_reflections=4, shape=shape)
    x = jr.normal(key, shape)
    y, log_det_f = rotation.transform_and_log_det(x)
    x_rec, log_det_i = rotation.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det_f, 0.0, atol=1e-5)
    assert jnp.allclose(log_det_i, 0.0, atol=1e-5)


def test_householder_rotation_shape(key):
    shape = (4,)
    rotation = HouseholderRotation(n_reflections=4, shape=shape)
    x = jr.normal(key, shape)
    y, log_det = rotation.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_orthogonal_rotation_forward_inverse(key):
    shape = (4,)
    rotation = OrthogonalRotation(shape=shape)
    x = jr.normal(key, shape)
    y, log_det_f = rotation.transform_and_log_det(x)
    x_rec, _log_det_i = rotation.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.allclose(log_det_f, 0.0, atol=1e-5)


def test_orthogonal_rotation_volume_preserving(key):
    shape = (4,)
    rotation = OrthogonalRotation(shape=shape)
    x = jr.normal(key, shape)
    _, log_det = rotation.transform_and_log_det(x)
    assert jnp.allclose(log_det, 0.0, atol=1e-4)


def test_fixed_rotation_forward_inverse(key):
    shape = (4,)
    # Use a random orthogonal matrix
    W = jr.normal(key, (4, 4))
    Q, _ = jnp.linalg.qr(W)
    rotation = FixedRotation(Q)
    x = jr.normal(key, shape)
    y, log_det_f = rotation.transform_and_log_det(x)
    x_rec, log_det_i = rotation.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-5)


def test_fixed_rotation_orthogonal_matrix_preserves_volume(key):
    W = jr.normal(key, (4, 4))
    Q, _ = jnp.linalg.qr(W)
    rotation = FixedRotation(Q)
    x = jr.normal(key, (4,))
    _, log_det = rotation.transform_and_log_det(x)
    assert jnp.allclose(log_det, 0.0, atol=1e-4)
