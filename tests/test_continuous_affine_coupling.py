"""Tests for continuous-time affine coupling and packed conditions."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from flowjax.distributions import Normal, Transformed

from gauss_flows import (
    ContinuousAffineCoupling,
    TimeFourier,
    TimeIdentity,
    TimeTanh,
    pack_time_control,
    unpack_time_control,
)


@pytest.mark.parametrize("control_dim", [0, 1, 3, 8])
def test_pack_unpack_time_control_roundtrip(control_dim):
    t = 0.75
    control = None if control_dim == 0 else jnp.arange(control_dim, dtype=float)

    condition = pack_time_control(t, control)
    t_rec, control_rec = unpack_time_control(condition, control_dim=control_dim)

    assert t_rec.shape == (1,)
    assert jnp.allclose(t_rec, jnp.array([t]))
    if control_dim == 0:
        assert control_rec is None
    else:
        assert control_rec.shape == (control_dim,)
        assert jnp.allclose(control_rec, control)


def test_pack_time_control_batched():
    t = jnp.array([0.0, 0.5, 1.0])
    control = jnp.arange(6, dtype=float).reshape(3, 2)

    condition = pack_time_control(t, control)
    t_rec, control_rec = unpack_time_control(condition, control_dim=2)

    assert condition.shape == (3, 3)
    assert t_rec.shape == (3, 1)
    assert jnp.allclose(t_rec[:, 0], t)
    assert jnp.allclose(control_rec, control)


@pytest.mark.parametrize(
    "time_net",
    [
        TimeIdentity(),
        TimeTanh(jr.key(0), embedding_dim=8),
        TimeFourier(jr.key(1), embedding_dim=8),
    ],
)
def test_time_nets_preserve_zero_at_origin(time_net):
    assert jnp.asarray(time_net(0.0)) == 0.0

    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(time_net) if eqx.is_array(leaf)
    ]
    if leaves:
        opt = optax.adam(1e-2)
        opt_state = opt.init(eqx.filter(time_net, eqx.is_inexact_array))

        def loss_fn(net):
            return (net(1.0) - 1.0) ** 2

        for _ in range(5):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(time_net)
            assert jnp.isfinite(loss)
            updates, opt_state = opt.update(grads, opt_state)
            time_net = eqx.apply_updates(time_net, updates)

    assert jnp.asarray(time_net(0.0)) == 0.0


@pytest.mark.parametrize("control_dim", [0, 3])
def test_continuous_affine_coupling_roundtrip_and_logdet(key, control_dim):
    coupling = ContinuousAffineCoupling(
        key,
        shape=(6,),
        control_dim=control_dim,
        n_hidden=16,
        n_layers=1,
        time_embedding_dim=8,
    )
    x = jr.normal(jr.fold_in(key, 1), (6,))
    control = (
        None if control_dim == 0 else jr.normal(jr.fold_in(key, 2), (control_dim,))
    )
    condition = pack_time_control(0.7, control)

    y, log_det_f = coupling.transform_and_log_det(x, condition)
    x_rec, log_det_i = coupling.inverse_and_log_det(y, condition)

    assert y.shape == x.shape
    assert log_det_f.shape == ()
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-5)


@pytest.mark.parametrize(
    "make_time_net",
    [
        lambda key: TimeIdentity(),
        lambda key: TimeTanh(key, embedding_dim=8),
        lambda key: TimeFourier(key, embedding_dim=8),
    ],
)
def test_continuous_affine_coupling_accepts_custom_time_net(key, make_time_net):
    time_net = make_time_net(jr.fold_in(key, 7))
    coupling = ContinuousAffineCoupling(
        key,
        shape=(4,),
        control_dim=1,
        n_hidden=16,
        n_layers=1,
        time_net=time_net,
    )
    assert coupling.time_net is time_net

    x = jr.normal(jr.fold_in(key, 1), (4,))
    condition_zero = pack_time_control(0.0, jnp.array([0.25]))
    y0, log_det0 = coupling.transform_and_log_det(x, condition_zero)
    assert jnp.allclose(y0, x, atol=1e-6)
    assert jnp.allclose(log_det0, 0.0, atol=1e-6)

    condition = pack_time_control(0.6, jnp.array([0.25]))
    y, _ = coupling.transform_and_log_det(x, condition)
    x_rec, _ = coupling.inverse_and_log_det(y, condition)
    assert jnp.allclose(x, x_rec, atol=1e-5)


def test_continuous_affine_coupling_passive_half_invariant_under_training(key):
    """Passive dims must never be written by the layer, even after gradient
    updates. Regression test for the case where the coupling mask was stored
    as a trainable (floating) pytree leaf and Optax could drift it away from
    ``{0, 1}``, silently breaking the bijection.
    """
    coupling = ContinuousAffineCoupling(
        key,
        shape=(4,),
        control_dim=1,
        n_hidden=16,
        n_layers=1,
        time_embedding_dim=8,
    )
    x = jr.normal(jr.fold_in(key, 1), (4,))
    condition = pack_time_control(0.5, jnp.array([0.3]))

    opt = optax.adam(1e-1)
    opt_state = opt.init(eqx.filter(coupling, eqx.is_inexact_array))

    def loss_fn(model):
        y, log_det = model.transform_and_log_det(x, condition)
        return jnp.sum(y**2) + log_det

    for _ in range(5):
        grads = eqx.filter_grad(loss_fn)(coupling)
        updates, opt_state = opt.update(grads, opt_state)
        coupling = eqx.apply_updates(coupling, updates)

    y, _ = coupling.transform_and_log_det(x, condition)
    passive_dim = coupling.untransformed_dim
    assert jnp.array_equal(y[:passive_dim], x[:passive_dim])


def test_continuous_affine_coupling_is_identity_at_t_zero(key):
    coupling = ContinuousAffineCoupling(key, shape=(4,), control_dim=2)
    x = jr.normal(jr.fold_in(key, 1), (4,))
    control = jr.normal(jr.fold_in(key, 2), (2,))
    condition = pack_time_control(0.0, control)

    y, log_det = coupling.transform_and_log_det(x, condition)

    assert jnp.allclose(y, x, atol=1e-6)
    assert jnp.allclose(log_det, 0.0, atol=1e-6)


def test_continuous_affine_coupling_logdet_matches_jacobian(key):
    coupling = ContinuousAffineCoupling(
        key,
        shape=(4,),
        control_dim=1,
        n_hidden=16,
        n_layers=1,
        time_embedding_dim=8,
    )
    x = jr.normal(jr.fold_in(key, 1), (4,))
    condition = pack_time_control(0.6, jnp.array([0.25]))

    def forward(inp):
        y, _ = coupling.transform_and_log_det(inp, condition)
        return y

    _, log_det = coupling.transform_and_log_det(x, condition)
    jac = jax.jacrev(forward)(x)
    _, logabsdet = jnp.linalg.slogdet(jac)

    assert jnp.allclose(log_det, logabsdet, atol=1e-5)


def test_continuous_affine_coupling_jit_and_grad(key):
    coupling = ContinuousAffineCoupling(
        key,
        shape=(4,),
        control_dim=2,
        n_hidden=16,
        n_layers=1,
        time_embedding_dim=8,
    )
    x = jr.normal(jr.fold_in(key, 1), (4,))
    condition = pack_time_control(0.8, jnp.array([0.5, -0.25]))

    eager_y, eager_log_det = coupling.transform_and_log_det(x, condition)
    jit_fn = eqx.filter_jit(coupling.transform_and_log_det)
    jit_y, jit_log_det = jit_fn(x, condition)
    assert jnp.allclose(jit_y, eager_y, atol=1e-6)
    assert jnp.allclose(jit_log_det, eager_log_det, atol=1e-6)

    def loss_fn(model):
        y, log_det = model.transform_and_log_det(x, condition)
        return jnp.sum(y**2) + log_det

    grads = eqx.filter_grad(loss_fn)(coupling)
    nn_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads.nn) if eqx.is_array(leaf)
    ]
    time_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads.time_net) if eqx.is_array(leaf)
    ]
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in nn_leaves + time_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in nn_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in time_leaves)


def test_continuous_affine_coupling_time_only_path(key):
    coupling = ContinuousAffineCoupling(key, shape=(4,), control_dim=0)
    x = jr.normal(jr.fold_in(key, 1), (4,))
    condition = pack_time_control(0.5)

    y, log_det = coupling.transform_and_log_det(x, condition)
    x_rec, log_det_inv = coupling.inverse_and_log_det(y, condition)

    assert y.shape == x.shape
    assert jnp.isfinite(log_det)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_continuous_affine_coupling_vmap_matches_loop(key):
    coupling = ContinuousAffineCoupling(key, shape=(4,), control_dim=1)
    x = jr.normal(jr.fold_in(key, 1), (5, 4))
    t = jnp.linspace(0.1, 0.9, 5)
    control = jr.normal(jr.fold_in(key, 2), (5, 1))
    condition = pack_time_control(t, control)

    batched_y, batched_log_det = jax.vmap(
        coupling.transform_and_log_det,
        in_axes=(0, 0),
    )(x, condition)
    loop_y, loop_log_det = zip(
        *(
            coupling.transform_and_log_det(xi, ci)
            for xi, ci in zip(x, condition, strict=True)
        ),
        strict=True,
    )

    assert jnp.allclose(batched_y, jnp.stack(loop_y), atol=1e-6)
    assert jnp.allclose(batched_log_det, jnp.stack(loop_log_det), atol=1e-6)


def test_continuous_affine_coupling_training_sanity(key):
    n_samples = 64
    t = jnp.linspace(0.1, 1.0, n_samples)
    control = jnp.sin(t)[:, None]
    x1 = jr.normal(jr.fold_in(key, 1), (n_samples,))
    noise = 0.1 * jr.normal(jr.fold_in(key, 2), (n_samples,))
    x2 = (1.0 + 0.25 * t) * x1 + 0.5 * t + 0.3 * control[:, 0] + noise
    data = jnp.stack((x1, x2), axis=-1)
    condition = pack_time_control(t, control)

    flow = Transformed(
        Normal(jnp.zeros(2)),
        ContinuousAffineCoupling(
            jr.fold_in(key, 3),
            shape=(2,),
            control_dim=1,
            n_hidden=16,
            n_layers=1,
            time_embedding_dim=8,
        ),
    )

    @eqx.filter_jit
    def loss_fn(model):
        return -jnp.mean(model.log_prob(data, condition=condition))

    opt = optax.adam(5e-2)
    opt_state = opt.init(eqx.filter(flow, eqx.is_inexact_array))
    initial_loss = loss_fn(flow)

    for _ in range(50):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
        assert jnp.isfinite(loss)
        updates, opt_state = opt.update(grads, opt_state)
        flow = eqx.apply_updates(flow, updates)

    final_loss = loss_fn(flow)
    assert jnp.isfinite(final_loss)
    # Require a measurable margin so numerics drift can't silently flake the
    # test, while still checking that training moved the loss meaningfully.
    assert final_loss < initial_loss - 0.05

    x_probe = jr.normal(jr.fold_in(key, 4), (2,))
    y0, log_det0 = flow.bijection.transform_and_log_det(
        x_probe,
        pack_time_control(0.0, jnp.array([0.5])),
    )
    assert jnp.allclose(y0, x_probe, atol=1e-6)
    assert jnp.allclose(log_det0, 0.0, atol=1e-6)
