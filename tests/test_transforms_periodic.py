"""Tests for periodic and circular transforms."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import paramax as px
import pytest
from flowjax.bijections import Chain, Permute

from gauss_flows import (
    CircularRationalQuadraticSpline,
    CircularRQSplineCoupling,
    PeriodicShift,
    PeriodicWrap,
)


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


def test_periodic_shift_multi_index_init():
    shift = PeriodicShift(ind=(0, 2), shape=(3,), shift_init=jnp.array([0.3, -0.7]))
    x = jnp.array([0.1, 0.5, -0.2])
    y, _ = shift.transform_and_log_det(x)
    # Periodic dims advanced by shift_init; dim 1 unchanged.
    assert jnp.allclose(y[1], x[1])
    assert not jnp.allclose(y[0], x[0])
    assert not jnp.allclose(y[2], x[2])


def test_periodic_shift_grad_flows_through_shift():
    shift = PeriodicShift(ind=(0,), shape=(2,), shift_init=0.0)
    x = jnp.array([0.5, 0.1])

    def loss(s):
        out, _ = s.transform_and_log_det(x)
        return jnp.sum(out**2)

    grads = eqx.filter_grad(loss)(shift)
    assert jnp.any(jnp.abs(grads.shift) > 0)


def test_circular_spline_boundary_c1_continuity(key):
    """Tied endpoint derivatives → derivative continuous across the wrap join."""
    spline = CircularRationalQuadraticSpline(knots=8, bound=jnp.pi)
    # Perturb the underlying parameters so we test a non-identity spline.
    perturbed = jax.tree_util.tree_map(
        lambda leaf: (
            leaf + jr.normal(key, leaf.shape) * 0.5
            if eqx.is_inexact_array(leaf)
            else leaf
        ),
        spline,
    )
    unwrapped = px.unwrap(perturbed)
    eps = 1e-5

    deriv_fn = jax.grad(lambda x: unwrapped.transform_and_log_det(x)[0])
    d_lo = deriv_fn(jnp.array(-jnp.pi + eps))
    d_hi = deriv_fn(jnp.array(jnp.pi - eps))
    # Tied endpoints → boundary derivatives match to numerical precision.
    assert jnp.allclose(d_lo, d_hi, atol=1e-3)


def test_circular_spline_roundtrip(key):
    spline = CircularRationalQuadraticSpline(knots=8, bound=jnp.pi)
    perturbed = jax.tree_util.tree_map(
        lambda leaf: (
            leaf + jr.normal(key, leaf.shape) * 0.3
            if eqx.is_inexact_array(leaf)
            else leaf
        ),
        spline,
    )
    unwrapped = px.unwrap(perturbed)
    xs = jnp.linspace(-jnp.pi + 1e-3, jnp.pi - 1e-3, 16)
    for x in xs:
        y, ld_f = unwrapped.transform_and_log_det(x)
        x_rec, ld_i = unwrapped.inverse_and_log_det(y)
        assert jnp.allclose(x, x_rec, atol=1e-4)
        assert jnp.allclose(ld_f + ld_i, 0.0, atol=1e-4)


def test_circular_rqspline_coupling_roundtrip(key):
    shape = (4,)
    coupling = CircularRQSplineCoupling(
        key,
        shape=shape,
        n_bins=4,
        nn_width=16,
        nn_depth=1,
    )
    coupling = px.unwrap(coupling)
    x = jr.uniform(key, shape, minval=-jnp.pi, maxval=jnp.pi)
    y, log_det = coupling.transform_and_log_det(x)
    x_rec, inv_log_det = coupling.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.allclose(log_det + inv_log_det, 0.0, atol=1e-4)


def test_circular_coupling_density_continuous_at_boundary(key):
    """Forward log_det is continuous across the ±π wrap join."""
    coupling = CircularRQSplineCoupling(
        key,
        shape=(2,),
        n_bins=8,
        nn_width=16,
        nn_depth=1,
    )
    coupling = px.unwrap(coupling)
    eps = 1e-3

    # log-det at a point just inside the lower bound vs the upper bound.
    _, ld_lo = coupling.transform_and_log_det(jnp.array([-jnp.pi + eps, -jnp.pi + eps]))
    _, ld_hi = coupling.transform_and_log_det(jnp.array([jnp.pi - eps, jnp.pi - eps]))
    # Without tied endpoints this gap is O(0.1-1.0); tied endpoints -> O(1e-3).
    assert jnp.allclose(ld_lo, ld_hi, atol=1e-2)


def test_circular_coupling_chain_roundtrip(key):
    """Stacked CircularRQSplineCoupling + Permute layers roundtrip exactly."""
    shape = (4,)
    layer_keys = jr.split(key, 3)
    layers = []
    for lk in layer_keys:
        bk, pk = jr.split(lk)
        layers.append(
            CircularRQSplineCoupling(bk, shape=shape, n_bins=4, nn_width=16, nn_depth=1)
        )
        layers.append(Permute(jr.permutation(pk, jnp.arange(shape[0]))))
    chain = px.unwrap(Chain(layers))
    x = jr.uniform(key, shape, minval=-jnp.pi, maxval=jnp.pi)
    y, ld_f = chain.transform_and_log_det(x)
    x_rec, ld_i = chain.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.allclose(ld_f + ld_i, 0.0, atol=1e-4)


def test_circular_coupling_rejects_invalid_split(key):
    with pytest.raises(ValueError, match="untransformed_dim must be in"):
        CircularRQSplineCoupling(key, shape=(4,), untransformed_dim=0, n_bins=4)
    with pytest.raises(ValueError, match="untransformed_dim must be in"):
        CircularRQSplineCoupling(key, shape=(4,), untransformed_dim=4, n_bins=4)
