"""Tests for coupling transforms."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from gauss_flows import ActNorm1D, AffineCoupling, DeepSigmoidCoupling, RQSplineCoupling


def test_act_norm_1d_forward_inverse(key):
    shape = (4,)
    act_norm = ActNorm1D(shape)
    x = jr.normal(key, shape)
    y, log_det = act_norm.transform_and_log_det(x)
    x_rec, log_det_inv = act_norm.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_act_norm_1d_shape(key):
    shape = (4,)
    act_norm = ActNorm1D(shape)
    x = jr.normal(key, shape)
    y, log_det = act_norm.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_affine_coupling_forward_inverse(key):
    shape = (4,)
    coupling = AffineCoupling(key, shape, nn_width=16, nn_depth=1)
    x = jr.normal(key, shape)
    y, log_det_f = coupling.transform_and_log_det(x)
    x_rec, log_det_i = coupling.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-5)


def test_affine_coupling_shape(key):
    shape = (4,)
    coupling = AffineCoupling(key, shape, nn_width=16, nn_depth=1)
    x = jr.normal(key, shape)
    y, log_det = coupling.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_rqspline_coupling_forward_inverse(key):
    shape = (4,)
    coupling = RQSplineCoupling(key, shape, n_bins=4, nn_width=16, nn_depth=1)
    x = jr.normal(key, shape)
    y, log_det_f = coupling.transform_and_log_det(x)
    x_rec, log_det_i = coupling.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-4)


def test_deep_sigmoid_coupling_forward_inverse(key):
    shape = (4,)
    coupling = DeepSigmoidCoupling(key, shape, nn_width=16, nn_depth=1)
    x = jr.normal(key, shape)
    y, log_det_f = coupling.transform_and_log_det(x)
    x_rec, log_det_i = coupling.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-6)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-6)


def test_deep_sigmoid_coupling_logdet_matches_jacobian(key):
    shape = (4,)
    coupling = DeepSigmoidCoupling(key, shape, n_components=4, nn_width=32, nn_depth=2)
    x = jr.normal(key, shape)

    def _forward(inp):
        y, _ = coupling.transform_and_log_det(inp)
        return y

    _, log_det = coupling.transform_and_log_det(x)
    jac = jax.jacobian(_forward)(x)
    _, logabsdet = jnp.linalg.slogdet(jac)
    assert jnp.allclose(log_det, logabsdet, atol=1e-5)


class _ConstantConditioner(eqx.Module):
    params: jnp.ndarray

    def __call__(self, _x):
        return self.params


def test_deep_sigmoid_coupling_has_curvature(key):
    shape = (2,)
    deep = DeepSigmoidCoupling(key, shape, n_components=3, nn_width=8, nn_depth=1)
    affine = AffineCoupling(key, shape, nn_width=8, nn_depth=1)

    deep_params = deep._coupling.conditioner(jnp.zeros((shape[0] // 2,)))
    affine_params = affine._coupling.conditioner(jnp.zeros((shape[0] // 2,)))

    deep_conditioner = _ConstantConditioner(jnp.linspace(-1.0, 1.0, deep_params.size))
    affine_conditioner = _ConstantConditioner(
        jnp.linspace(-0.5, 0.5, affine_params.size)
    )

    deep = eqx.tree_at(lambda c: c._coupling.conditioner, deep, deep_conditioner)
    affine = eqx.tree_at(lambda c: c._coupling.conditioner, affine, affine_conditioner)

    def _transformed_value(coupling, x_scalar):
        y, _ = coupling.transform_and_log_det(jnp.array([0.0, x_scalar]))
        return y[1]

    deep_curvature = jax.grad(jax.grad(lambda x: _transformed_value(deep, x)))(0.0)
    affine_curvature = jax.grad(jax.grad(lambda x: _transformed_value(affine, x)))(0.0)

    assert jnp.abs(deep_curvature) > 1e-3
    assert jnp.allclose(affine_curvature, 0.0, atol=1e-6)
