"""Tests for coupling transforms."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Chain

from gauss_flows import (
    ActNorm1D,
    AffineCoupling,
    BatchNorm,
    DeepSigmoidCoupling,
    RQSplineCoupling,
)


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


def test_batch_norm_requires_stats_source(key):
    """Calling transform/inverse on a bare BatchNorm raises a clear error."""
    shape = (4,)
    bn = BatchNorm(shape)
    x = jr.normal(key, shape)
    with pytest.raises(RuntimeError, match="BatchNorm needs a statistics source"):
        bn.transform_and_log_det(x)
    with pytest.raises(RuntimeError, match="BatchNorm needs a statistics source"):
        bn.inverse_and_log_det(x)


def test_batch_norm_single_event_scalar_log_det(key):
    """Single-event call returns a scalar log_det (flowjax convention)."""
    shape = (4,)
    batch = jr.normal(key, (16, *shape))
    bn = BatchNorm(shape).with_batch_stats_from_data(batch)
    x = batch[0]
    y, log_det = bn.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_batch_norm_training_uses_batch_stats(key):
    """With batch stats set, vmap over the batch gives expected normalisation."""
    shape = (4,)
    batch = jr.normal(key, (16, *shape))
    bn = BatchNorm(shape, eps=1e-4).with_batch_stats_from_data(batch)
    y, log_det = jax.vmap(bn.transform_and_log_det)(batch)
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    scale = jnp.exp(bn.log_gamma) / jnp.sqrt(batch_var + bn.eps)
    expected_y = (batch - batch_mean) * scale + bn.beta
    expected_log_det = jnp.sum(bn.log_gamma - 0.5 * jnp.log(batch_var + bn.eps))
    assert jnp.allclose(y, expected_y, atol=1e-5)
    assert jnp.allclose(log_det, jnp.broadcast_to(expected_log_det, (16,)), atol=1e-5)


def test_batch_norm_eval_roundtrip_with_running_stats(key):
    """With running avg enabled, forward∘inverse is exact and log_det matches."""
    shape = (3,)
    batch = jr.normal(key, (8, *shape))
    bn = (
        BatchNorm(shape)
        .update_running_stats_from_batch(batch)
        .with_running_average(True)
    )
    y, log_det = jax.vmap(bn.transform_and_log_det)(batch)
    x_rec, log_det_inv = jax.vmap(bn.inverse_and_log_det)(y)
    expected_log_det = jnp.sum(bn.log_gamma - 0.5 * jnp.log(bn.running_var + bn.eps))
    assert jnp.allclose(batch, x_rec, atol=1e-5)
    assert jnp.allclose(log_det, jnp.broadcast_to(expected_log_det, (8,)), atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_batch_norm_running_stats_update():
    """EMA update formula matches (1 - momentum) * old + momentum * batch."""
    shape = (2,)
    bn = BatchNorm(shape, momentum=0.5)
    batch_mean = jnp.array([1.0, -1.0])
    batch_var = jnp.array([2.0, 3.0])
    updated = bn.update_running_stats(batch_mean, batch_var)
    assert jnp.allclose(updated.running_mean, jnp.array([0.5, -0.5]))
    assert jnp.allclose(updated.running_var, jnp.array([1.5, 2.0]))
    # Original is unchanged (immutable update).
    assert jnp.allclose(bn.running_mean, jnp.zeros_like(bn.running_mean))
    assert jnp.allclose(bn.running_var, jnp.ones_like(bn.running_var))


def test_batch_norm_deep_stack_with_act_norm(key):
    """BatchNorm composes with ActNorm1D in a 10-layer Chain (eval mode)."""
    shape = (3,)
    stats_batch = jr.normal(key, (32, *shape))
    layers = []
    for _ in range(10):
        layers.append(ActNorm1D(shape))
        bn = BatchNorm(shape).update_running_stats_from_batch(stats_batch)
        bn = bn.with_running_average(True)
        layers.append(bn)
    chain = Chain(layers)
    x = jr.normal(jr.fold_in(key, 1), (32, *shape))
    y, log_det = jax.vmap(chain.transform_and_log_det)(x)
    x_rec, log_det_inv = jax.vmap(chain.inverse_and_log_det)(y)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_batch_norm_pytree_structure_is_stable():
    """with_batch_stats_from_data must not change the pytree structure."""
    shape = (3,)
    bn_default = BatchNorm(shape)
    bn_with_stats = bn_default.with_batch_stats_from_data(jnp.zeros((8, *shape)))
    # Same tree structure → same treedef → jit caches are shared.
    _, def_a = jax.tree_util.tree_flatten(bn_default)
    _, def_b = jax.tree_util.tree_flatten(bn_with_stats)
    assert def_a == def_b
