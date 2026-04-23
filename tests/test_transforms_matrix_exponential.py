"""Tests for the closed-form matrix exponential flow."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Chain

from gauss_flows import MatrixExponential, TimeIdentity, TimeTanh, pack_time_control


@pytest.mark.parametrize("use_bias", [False, True])
def test_matrix_exponential_is_identity_at_zero(key, use_bias):
    bijection = MatrixExponential(key, shape=(4,), use_bias=use_bias)
    x = jr.normal(jr.fold_in(key, 1), (4,))

    y, log_det = bijection.transform_and_log_det(x, pack_time_control(0.0))

    assert jnp.allclose(y, x, atol=1e-6)
    assert jnp.allclose(log_det, 0.0, atol=1e-6)


@pytest.mark.parametrize("use_bias", [False, True])
def test_matrix_exponential_roundtrip_and_closed_form_logdet(key, use_bias):
    bijection = MatrixExponential(key, shape=(5,), use_bias=use_bias)
    x = jr.normal(jr.fold_in(key, 1), (5,))
    condition = pack_time_control(-0.7)

    y, log_det_fwd = bijection.transform_and_log_det(x, condition)
    x_rec, log_det_inv = bijection.inverse_and_log_det(y, condition)
    expected = -0.7 * jnp.trace(bijection.W)

    assert jnp.allclose(x_rec, x, atol=1e-5)
    assert jnp.allclose(log_det_fwd, expected, atol=1e-6)
    assert jnp.allclose(log_det_inv, -expected, atol=1e-6)


def test_matrix_exponential_logdet_matches_jacobian(key):
    bijection = MatrixExponential(
        key,
        shape=(4,),
        time_bias_net=TimeTanh(jr.fold_in(key, 1), embedding_dim=8),
    )
    x = jr.normal(jr.fold_in(key, 2), (4,))
    condition = pack_time_control(0.6)

    def forward(inp):
        y, _ = bijection.transform_and_log_det(inp, condition)
        return y

    _, log_det = bijection.transform_and_log_det(x, condition)
    jac = jax.jacrev(forward)(x)
    _, logabsdet = jnp.linalg.slogdet(jac)

    assert jnp.allclose(log_det, logabsdet, atol=1e-6)


def test_matrix_exponential_jit_and_grad(key):
    bijection = MatrixExponential(
        key,
        shape=(4,),
        time_bias_net=TimeTanh(jr.fold_in(key, 1), embedding_dim=8),
    )
    x = jr.normal(jr.fold_in(key, 2), (4,))
    condition = pack_time_control(0.8)

    eager_y, eager_log_det = bijection.transform_and_log_det(x, condition)
    jit_y, jit_log_det = eqx.filter_jit(bijection.transform_and_log_det)(x, condition)
    assert jnp.allclose(jit_y, eager_y, atol=1e-6)
    assert jnp.allclose(jit_log_det, eager_log_det, atol=1e-6)

    def loss_fn(model):
        y, log_det = model.transform_and_log_det(x, condition)
        return jnp.sum(y**2) + log_det

    grads = eqx.filter_grad(loss_fn)(bijection)
    time_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(grads.time_bias_net)
        if eqx.is_array(leaf)
    ]

    assert jnp.all(jnp.isfinite(grads.W))
    assert grads.b is not None
    assert jnp.all(jnp.isfinite(grads.b))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in time_leaves)
    assert jnp.any(jnp.abs(grads.W) > 0)
    assert jnp.any(jnp.abs(grads.b) > 0)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in time_leaves)


def test_matrix_exponential_vmap_matches_loop(key):
    bijection = MatrixExponential(key, shape=(3,))
    x = jr.normal(jr.fold_in(key, 1), (5, 3))
    t = jnp.linspace(-0.8, 0.8, 5)
    condition = pack_time_control(t)

    batched_y, batched_log_det = jax.vmap(
        bijection.transform_and_log_det,
        in_axes=(0, 0),
    )(x, condition)
    loop_results = [
        bijection.transform_and_log_det(x_i, c_i)
        for x_i, c_i in zip(x, condition, strict=True)
    ]
    loop_y, loop_log_det = zip(*loop_results, strict=True)

    assert jnp.allclose(batched_y, jnp.stack(loop_y), atol=1e-6)
    assert jnp.allclose(batched_log_det, jnp.stack(loop_log_det), atol=1e-6)


def test_matrix_exponential_diagonal_special_case():
    lambdas = jnp.array([0.25, -0.5, 0.75])
    bias = jnp.array([1.0, -2.0, 0.5])
    x = jnp.array([2.0, -1.0, 0.25])
    t = 0.4
    bijection = MatrixExponential(jr.key(0), shape=(3,), time_bias_net=TimeIdentity())
    bijection = eqx.tree_at(lambda model: model.W, bijection, jnp.diag(lambdas))
    bijection = eqx.tree_at(lambda model: model.b, bijection, bias)

    y, log_det = bijection.transform_and_log_det(x, pack_time_control(t))

    expected_y = jnp.exp(lambdas * t) * x + t * bias
    expected_log_det = t * jnp.sum(lambdas)
    assert jnp.allclose(y, expected_y, atol=1e-6)
    assert jnp.allclose(log_det, expected_log_det, atol=1e-6)


def test_matrix_exponential_chain_roundtrip_and_logdet_additivity(key):
    keys = jr.split(key, 3)
    layers = [MatrixExponential(layer_key, shape=(3,)) for layer_key in keys]
    chain = Chain(layers)
    x = jr.normal(jr.fold_in(key, 1), (3,))
    condition = pack_time_control(0.5)

    y_chain, log_det_chain = chain.transform_and_log_det(x, condition)
    x_rec, log_det_inv = chain.inverse_and_log_det(y_chain, condition)

    y_manual = x
    log_det_manual = jnp.zeros(())
    for layer in layers:
        y_manual, layer_log_det = layer.transform_and_log_det(y_manual, condition)
        log_det_manual = log_det_manual + layer_log_det

    assert jnp.allclose(y_chain, y_manual, atol=1e-6)
    assert jnp.allclose(log_det_chain, log_det_manual, atol=1e-6)
    assert jnp.allclose(x_rec, x, atol=1e-5)
    assert jnp.allclose(log_det_chain + log_det_inv, 0.0, atol=1e-5)


def test_matrix_exponential_chain_vmap_over_times(key):
    keys = jr.split(key, 3)
    layers = [MatrixExponential(k, shape=(3,), use_bias=False) for k in keys]
    chain = Chain(layers)
    x = jr.normal(jr.fold_in(key, 1), (4, 3))
    t = jnp.linspace(-0.5, 0.5, 4)
    condition = pack_time_control(t)

    y, log_det = jax.vmap(chain.transform_and_log_det)(x, condition)
    x_rec, log_det_inv = jax.vmap(chain.inverse_and_log_det)(y, condition)
    trace_sum = sum(jnp.trace(layer.W) for layer in layers)
    expected_log_det = t * trace_sum

    assert jnp.allclose(x_rec, x, atol=1e-5)
    assert jnp.allclose(log_det, expected_log_det, atol=1e-6)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_matrix_exponential_requires_1d_shape():
    with pytest.raises(ValueError, match="1D"):
        MatrixExponential(jr.key(0), shape=(3, 3))


def test_matrix_exponential_requires_condition(key):
    bij = MatrixExponential(key, shape=(3,))
    x = jnp.zeros((3,))
    with pytest.raises(ValueError, match="condition to be provided"):
        bij.transform_and_log_det(x)
    with pytest.raises(ValueError, match="condition to be provided"):
        bij.inverse_and_log_det(x)


def test_matrix_exponential_rejects_bias_gate_without_bias():
    with pytest.raises(ValueError, match="time_bias_net was provided"):
        MatrixExponential(
            jr.key(0),
            shape=(3,),
            use_bias=False,
            time_bias_net=TimeIdentity(),
        )
