"""Tests for periodic and circular transforms."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from gauss_flows import CircularRQSplineCoupling, PeriodicShift, PeriodicWrap


def test_periodic_wrap_roundtrip():
    wrap = PeriodicWrap(ind=(0,), shape=(2,))
    x = jnp.array([1.5 * jnp.pi, -1.0])
    y, log_det = wrap.transform_and_log_det(x)
    x_rec, inv_log_det = wrap.inverse_and_log_det(y)
    expected = jnp.array([-0.5 * jnp.pi, -1.0])
    assert jnp.allclose(y, expected)
    assert jnp.allclose(x_rec, expected)
    assert jnp.allclose(log_det + inv_log_det, 0.0)


def test_periodic_shift_roundtrip():
    shift = PeriodicShift(ind=(1,), shape=(3,), shift_init=0.4)
    x = jnp.array([0.2, -jnp.pi + 0.1, 0.3])
    y, log_det = shift.transform_and_log_det(x)
    x_rec, inv_log_det = shift.inverse_and_log_det(y)
    assert jnp.allclose(x_rec, x)
    assert jnp.allclose(log_det + inv_log_det, 0.0)


def test_circular_rqspline_coupling_roundtrip(key):
    shape = (4,)
    coupling = CircularRQSplineCoupling(
        key,
        shape=shape,
        periodic_dims=(1, 3),
        n_bins=4,
        nn_width=16,
        nn_depth=1,
    )
    x = jr.normal(key, shape)
    y, log_det = coupling.transform_and_log_det(x)
    x_rec, inv_log_det = coupling.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.allclose(log_det + inv_log_det, 0.0, atol=1e-4)


def test_circular_coupling_wrap_continuity(key):
    coupling = CircularRQSplineCoupling(
        key,
        shape=(2,),
        periodic_dims=(0, 1),
        n_bins=4,
        nn_width=16,
        nn_depth=1,
    )
    dist = coupling.as_distribution()
    x = jr.normal(key, (8, 2))
    wrapped_shift = x + 2 * jnp.pi
    log_p = dist.log_prob(x)
    log_p_shift = dist.log_prob(wrapped_shift)
    assert jnp.allclose(log_p, log_p_shift, atol=1e-3)
