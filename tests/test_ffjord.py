"""Tests for the FFJORD continuous normalizing flow."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Chain
from flowjax.distributions import Normal, Transformed

from gauss_flows import FFJORD, DiffeqConcat, DiffeqMLP, TimeIdentity, pack_time_control


def test_diffeq_nets_preserve_event_shape(key):
    x = jr.normal(jr.fold_in(key, 1), (4,))
    cond = pack_time_control(0.5, jnp.array([1.0, -1.0]))

    mlp = DiffeqMLP(key, in_dim=4, control_dim=2, hidden=(8, 8))
    concat = DiffeqConcat(
        jr.fold_in(key, 2),
        in_dim=4,
        control_dim=2,
        hidden=(8, 8),
        time_net=TimeIdentity(),
    )

    assert mlp(0.25, x, cond).shape == x.shape
    assert concat(0.25, x, cond).shape == x.shape


@pytest.mark.slow
@pytest.mark.parametrize("control_dim", [0, 2])
def test_ffjord_roundtrip_and_logdet_cancel(key, control_dim):
    vf = DiffeqMLP(
        jr.fold_in(key, 1),
        in_dim=3,
        control_dim=control_dim,
        hidden=(16, 16),
    )
    bij = FFJORD(
        key,
        shape=(3,),
        vector_field=vf,
        control_dim=control_dim,
        divergence_mode="exact",
        solver="tsit5",
        adjoint="direct",
        rtol=1e-6,
        atol=1e-6,
    )
    x = jr.normal(jr.fold_in(key, 2), (3,))
    control = (
        None if control_dim == 0 else jr.normal(jr.fold_in(key, 3), (control_dim,))
    )
    condition = pack_time_control(0.7, control)

    y, log_det_f = bij.transform_and_log_det(x, condition)
    x_rec, log_det_i = bij.inverse_and_log_det(y, condition)

    assert y.shape == x.shape
    assert log_det_f.shape == ()
    assert jnp.allclose(x_rec, x, atol=1e-4)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-4)


@pytest.mark.slow
def test_ffjord_exact_logdet_matches_jacobian(key):
    bij = FFJORD(
        key,
        shape=(2,),
        vector_field=DiffeqMLP(
            jr.fold_in(key, 1), in_dim=2, control_dim=1, hidden=(8, 8)
        ),
        control_dim=1,
        divergence_mode="exact",
        solver="tsit5",
        adjoint="direct",
        rtol=1e-6,
        atol=1e-6,
    )
    x = jr.normal(jr.fold_in(key, 2), (2,))
    condition = pack_time_control(0.4, jnp.array([0.25]))

    def forward(inp):
        y, _ = bij.transform_and_log_det(inp, condition)
        return y

    _, log_det = bij.transform_and_log_det(x, condition)
    jacobian = jax.jacrev(forward)(x)
    _, logabsdet = jnp.linalg.slogdet(jacobian)

    # ODE integration plus an outer jacobian-of-solver call accumulates more
    # numerical error than the pure round-trip tests above.
    assert jnp.allclose(log_det, logabsdet, atol=1e-3)


@pytest.mark.slow
def test_ffjord_hutchinson_converges_toward_exact(key):
    vf = DiffeqMLP(jr.fold_in(key, 1), in_dim=2, control_dim=1, hidden=(8, 8))
    exact = FFJORD(
        key,
        shape=(2,),
        vector_field=vf,
        control_dim=1,
        divergence_mode="exact",
        solver="tsit5",
        adjoint="direct",
        rtol=1e-6,
        atol=1e-6,
    )
    approx = FFJORD(
        jr.fold_in(key, 9),
        shape=(2,),
        vector_field=vf,
        control_dim=1,
        divergence_mode="hutchinson",
        n_hutchinson_samples=1024,
        solver="tsit5",
        adjoint="direct",
        rtol=1e-6,
        atol=1e-6,
    )
    x = jr.normal(jr.fold_in(key, 2), (2,))
    condition = pack_time_control(0.4, jnp.array([0.25]))

    _, exact_log_det = exact.transform_and_log_det(x, condition)
    _, approx_log_det = approx.transform_and_log_det(x, condition)

    # Even with 1024 probes, Hutchinson retains Monte Carlo variance, so this
    # tolerance is intentionally looser than the exact-mode checks above.
    assert jnp.allclose(approx_log_det, exact_log_det, atol=1e-2)


@pytest.mark.slow
def test_ffjord_jit_grad_and_distribution_integration(key):
    bij = FFJORD(
        key,
        shape=(2,),
        vector_field=DiffeqConcat(
            jr.fold_in(key, 1),
            in_dim=2,
            control_dim=1,
            hidden=(8, 8),
            time_net=TimeIdentity(),
        ),
        control_dim=1,
        divergence_mode="hutchinson",
        n_hutchinson_samples=8,
        solver="tsit5",
        adjoint="recursive_checkpoint",
    )
    x = jr.normal(jr.fold_in(key, 2), (2,))
    condition = pack_time_control(0.8, jnp.array([0.5]))

    eager_y, eager_log_det = bij.transform_and_log_det(x, condition)
    jit_fn = eqx.filter_jit(bij.transform_and_log_det)
    jit_y, jit_log_det = jit_fn(x, condition)
    assert jnp.allclose(jit_y, eager_y, atol=1e-6)
    assert jnp.allclose(jit_log_det, eager_log_det, atol=1e-6)

    def loss_fn(model):
        y, log_det = model.transform_and_log_det(x, condition)
        return jnp.sum(y**2) + log_det

    grads = eqx.filter_grad(loss_fn)(bij)
    grad_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(grads.vector_field)
        if eqx.is_array(leaf)
    ]
    assert grad_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in grad_leaves)

    flow = Transformed(Normal(jnp.zeros(2)), bij)
    log_prob = flow.log_prob(x, condition=condition)
    sample = flow.sample(jr.fold_in(key, 3), condition=condition)
    assert jnp.isfinite(log_prob)
    assert sample.shape == x.shape


@pytest.mark.slow
def test_ffjord_chain_roundtrip_and_logdet_additivity(key):
    layers = [
        FFJORD(
            jr.fold_in(key, i),
            shape=(2,),
            vector_field=DiffeqMLP(
                jr.fold_in(key, 10 + i), in_dim=2, control_dim=1, hidden=(8, 8)
            ),
            control_dim=1,
            divergence_mode="exact",
            solver="tsit5",
            adjoint="direct",
            rtol=1e-6,
            atol=1e-6,
        )
        for i in range(2)
    ]
    chain = Chain(layers)
    x = jr.normal(jr.fold_in(key, 20), (2,))
    condition = pack_time_control(0.3, jnp.array([0.5]))

    y, log_det_chain = chain.transform_and_log_det(x, condition)
    x_rec, log_det_inv = chain.inverse_and_log_det(y, condition)

    y_manual = x
    log_det_manual = jnp.zeros(())
    for layer in layers:
        y_manual, layer_log_det = layer.transform_and_log_det(y_manual, condition)
        log_det_manual = log_det_manual + layer_log_det

    assert jnp.allclose(y, y_manual, atol=1e-4)
    assert jnp.allclose(log_det_chain, log_det_manual, atol=1e-4)
    assert jnp.allclose(x_rec, x, atol=1e-3)
    assert jnp.allclose(log_det_chain + log_det_inv, 0.0, atol=1e-3)


def test_ffjord_rejects_invalid_init():
    with pytest.raises(ValueError, match="1D event shapes"):
        FFJORD(jr.key(0), shape=(2, 2))
    with pytest.raises(ValueError, match="positive dimension"):
        FFJORD(jr.key(0), shape=(0,))
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        FFJORD(jr.key(0), shape=(2,), control_dim=-1)
    with pytest.raises(ValueError, match="n_hutchinson_samples must be positive"):
        FFJORD(jr.key(0), shape=(2,), n_hutchinson_samples=0)
    with pytest.raises(ValueError, match="t_span endpoints must differ"):
        FFJORD(jr.key(0), shape=(2,), t_span=(0.5, 0.5))
    with pytest.raises(ValueError, match="divergence_mode must be"):
        FFJORD(jr.key(0), shape=(2,), divergence_mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="probe_distribution must be"):
        FFJORD(jr.key(0), shape=(2,), probe_distribution="bogus")  # type: ignore[arg-type]
