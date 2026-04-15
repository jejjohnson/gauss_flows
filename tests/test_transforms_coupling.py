"""Tests for coupling transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
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
    y, _log_det_f = coupling.transform_and_log_det(x)
    x_rec, _log_det_i = coupling.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)


def test_batch_norm_training_uses_batch_stats(key):
    shape = (4,)
    batch = jr.normal(key, (16, *shape))
    batch_norm = BatchNorm(shape, eps=1e-4).with_batch_stats_from_data(batch)
    y, log_det = jax.vmap(batch_norm.transform_and_log_det)(batch)
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    scale = jnp.exp(batch_norm.log_gamma) / jnp.sqrt(batch_var + batch_norm.eps)
    expected_y = (batch - batch_mean) * scale + batch_norm.beta
    expected_log_det = jnp.sum(
        batch_norm.log_gamma - 0.5 * jnp.log(batch_var + batch_norm.eps)
    )
    expected_log_det = jnp.broadcast_to(expected_log_det, batch.shape[:1])
    assert jnp.allclose(y, expected_y, atol=1e-5)
    assert jnp.allclose(log_det, expected_log_det, atol=1e-5)


def test_batch_norm_eval_roundtrip_with_running_stats(key):
    shape = (3,)
    batch = jr.normal(key, (8, *shape))
    batch_norm = (
        BatchNorm(shape)
        .update_running_stats_from_batch(batch)
        .with_running_average(True)
    )
    y, log_det = jax.vmap(batch_norm.transform_and_log_det)(batch)
    x_rec, log_det_inv = jax.vmap(batch_norm.inverse_and_log_det)(y)
    expected_log_det = jnp.sum(
        batch_norm.log_gamma - 0.5 * jnp.log(batch_norm.running_var + batch_norm.eps)
    )
    expected_log_det = jnp.broadcast_to(expected_log_det, batch.shape[:1])
    assert jnp.allclose(batch, x_rec, atol=1e-5)
    assert jnp.allclose(log_det, expected_log_det, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, jnp.zeros_like(log_det), atol=1e-5)


def test_batch_norm_running_stats_update():
    shape = (2,)
    batch_norm = BatchNorm(shape, momentum=0.5)
    batch_mean = jnp.array([1.0, -1.0])
    batch_var = jnp.array([2.0, 3.0])
    updated = batch_norm.update_running_stats(batch_mean, batch_var)
    assert jnp.allclose(updated.running_mean, jnp.array([0.5, -0.5]))
    assert jnp.allclose(updated.running_var, jnp.array([1.5, 2.0]))
    assert jnp.allclose(
        batch_norm.running_mean, jnp.zeros_like(batch_norm.running_mean)
    )
    assert jnp.allclose(batch_norm.running_var, jnp.ones_like(batch_norm.running_var))


def test_batch_norm_deep_stack_with_act_norm(key):
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
    assert jnp.allclose(log_det + log_det_inv, jnp.zeros_like(log_det), atol=1e-5)
