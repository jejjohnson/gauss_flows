"""Tests for classic (VI-only) planar and Sylvester normalizing flow bijections."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gauss_flows import PlanarFlow, SylvesterFlow


def _log_det_from_jacobian(fn, x):
    """Reference log-abs-det via explicit Jacobian of fn at x."""
    J = jax.jacobian(fn)(x)
    return jnp.log(jnp.abs(jnp.linalg.det(J)))


def test_planar_log_det_matches_jacobian(key):
    shape = (5,)
    k_init, k_x = jr.split(key)
    flow = PlanarFlow(k_init, shape=shape, scale_init=0.3)
    x = jr.normal(k_x, shape)
    _, log_det = flow.transform_and_log_det(x)
    expected = _log_det_from_jacobian(lambda z: flow.transform_and_log_det(z)[0], x)
    assert jnp.allclose(log_det, expected, atol=1e-5)


def test_planar_shape(key):
    shape = (4,)
    flow = PlanarFlow(key, shape=shape)
    x = jr.normal(jr.fold_in(key, 1), shape)
    y, log_det = flow.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_planar_inverse_raises(key):
    flow = PlanarFlow(key, shape=(3,))
    y = jnp.zeros((3,))
    with pytest.raises(NotImplementedError, match="PlanarFlow"):
        flow.inverse_and_log_det(y)


def test_planar_invertibility_projection_keeps_det_positive(key):
    """u_hat projection ensures 1 + u_hat.w * (1 - tanh²) >= 0."""
    shape = (4,)
    # Force |u|, |w| large to exercise the projection
    flow = PlanarFlow(key, shape=shape, scale_init=5.0)
    x = jr.normal(jr.fold_in(key, 2), shape)
    _, log_det = flow.transform_and_log_det(x)
    assert jnp.isfinite(log_det)


def test_sylvester_log_det_matches_jacobian_full_rank(key):
    shape = (5,)
    k_init, k_x = jr.split(key)
    flow = SylvesterFlow(k_init, shape=shape, rank=5, scale_init=0.3)
    x = jr.normal(k_x, shape)
    _, log_det = flow.transform_and_log_det(x)
    expected = _log_det_from_jacobian(lambda z: flow.transform_and_log_det(z)[0], x)
    assert jnp.allclose(log_det, expected, atol=1e-5)


def test_sylvester_log_det_matches_jacobian_low_rank(key):
    shape = (6,)
    k_init, k_x = jr.split(key)
    flow = SylvesterFlow(k_init, shape=shape, rank=3, scale_init=0.3)
    x = jr.normal(k_x, shape)
    _, log_det = flow.transform_and_log_det(x)
    expected = _log_det_from_jacobian(lambda z: flow.transform_and_log_det(z)[0], x)
    assert jnp.allclose(log_det, expected, atol=1e-5)


def test_sylvester_shape(key):
    shape = (4,)
    flow = SylvesterFlow(key, shape=shape, rank=2)
    x = jr.normal(jr.fold_in(key, 1), shape)
    y, log_det = flow.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_sylvester_inverse_raises(key):
    flow = SylvesterFlow(key, shape=(3,))
    y = jnp.zeros((3,))
    with pytest.raises(NotImplementedError, match="SylvesterFlow"):
        flow.inverse_and_log_det(y)


def test_sylvester_rank_bounds(key):
    with pytest.raises(ValueError, match="rank"):
        SylvesterFlow(key, shape=(3,), rank=0)
    with pytest.raises(ValueError, match="rank"):
        SylvesterFlow(key, shape=(3,), rank=4)


def test_sylvester_Q_orthonormal_columns(key):
    shape = (6,)
    flow = SylvesterFlow(key, shape=shape, rank=3, scale_init=0.5)
    Q = flow._build_Q()
    assert Q.shape == (6, 3)
    assert jnp.allclose(Q.T @ Q, jnp.eye(3), atol=1e-5)
