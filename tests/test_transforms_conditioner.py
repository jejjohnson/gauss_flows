"""Tests for the FiLM-style Conditioner wrapper."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gauss_flows import (
    AffineCoupling,
    Conditioner,
    HouseholderRotation,
    OrthogonalRotation,
)


def test_shape_contracts_and_metadata(key):
    inner = OrthogonalRotation(shape=(4,))
    bij = Conditioner(key, inner=inner, cond_shape=(2,))
    assert bij.shape == (4,)
    assert bij.cond_shape == (2,)

    x = jr.normal(jr.fold_in(key, 1), (4,))
    cond = jnp.array([0.5, -0.5])
    y, log_det = bij.transform_and_log_det(x, cond)
    assert y.shape == (4,)
    assert log_det.shape == ()
    assert jnp.isfinite(log_det)


def test_roundtrip_and_logdet_cancel(key):
    inner = OrthogonalRotation(shape=(4,))
    bij = Conditioner(key, inner=inner, cond_shape=(3,))
    x = jr.normal(jr.fold_in(key, 1), (4,))
    cond = jr.normal(jr.fold_in(key, 2), (3,))
    y, log_det_f = bij.transform_and_log_det(x, cond)
    x_rec, log_det_i = bij.inverse_and_log_det(y, cond)
    assert jnp.allclose(x_rec, x, atol=1e-5)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-5)


def test_logdet_matches_jacobian_slogdet(key):
    inner = OrthogonalRotation(shape=(3,))
    bij = Conditioner(key, inner=inner, cond_shape=(2,))
    x = jr.normal(jr.fold_in(key, 1), (3,))
    cond = jnp.array([0.4, -0.3])

    def fwd(inp):
        y, _ = bij.transform_and_log_det(inp, cond)
        return y

    _, log_det = bij.transform_and_log_det(x, cond)
    jac = jax.jacrev(fwd)(x)
    _, slogdet_logdet = jnp.linalg.slogdet(jac)
    assert jnp.allclose(log_det, slogdet_logdet, atol=1e-5)


def test_condition_actually_affects_output(key):
    inner = HouseholderRotation(n_reflections=2, shape=(4,))
    bij = Conditioner(key, inner=inner, cond_shape=(2,))
    x = jr.normal(jr.fold_in(key, 1), (4,))
    y_a, _ = bij.transform_and_log_det(x, jnp.array([1.0, 0.0]))
    y_b, _ = bij.transform_and_log_det(x, jnp.array([0.0, 1.0]))
    assert not jnp.allclose(y_a, y_b, atol=1e-3)


def test_jit_and_grad(key):
    inner = OrthogonalRotation(shape=(4,))
    bij = Conditioner(key, inner=inner, cond_shape=(3,))
    x = jr.normal(jr.fold_in(key, 1), (4,))
    cond = jr.normal(jr.fold_in(key, 2), (3,))

    @eqx.filter_jit
    def loss(model, x, cond):
        y, log_det = model.transform_and_log_det(x, cond)
        return jnp.sum(y * y) + log_det

    val = loss(bij, x, cond)
    assert jnp.isfinite(val)

    grads = eqx.filter_grad(loss)(bij, x, cond)
    grad_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(grads.context_net)
        if eqx.is_array(leaf)
    ]
    assert grad_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in grad_leaves)


def test_user_supplied_context_net(key):
    """A user-supplied context_net is validated and accepted."""
    d = 4

    class IdentityNet(eqx.Module):
        bias: jax.Array

        def __call__(self, condition):
            # Condition is shape (D_ctx=2*D); pass through with bias.
            return condition + self.bias

    net = IdentityNet(bias=jnp.zeros(2 * d))
    inner = OrthogonalRotation(shape=(d,))
    bij = Conditioner(key, inner=inner, cond_shape=(2 * d,), context_net=net)
    x = jr.normal(jr.fold_in(key, 1), (d,))
    cond = jr.normal(jr.fold_in(key, 2), (2 * d,))
    y, log_det = bij.transform_and_log_det(x, cond)
    assert y.shape == (d,) and log_det.shape == ()


def test_rejects_inner_with_non_none_cond_shape(key):
    inner = AffineCoupling(jr.fold_in(key, 0), shape=(4,), cond_dim=3)
    assert inner.cond_shape == (3,)
    with pytest.raises(ValueError, match="strictly unconditional"):
        Conditioner(key, inner=inner, cond_shape=(3,))


def test_rejects_user_context_net_with_wrong_output_dim(key):
    d = 4

    class WrongNet(eqx.Module):
        def __call__(self, condition):
            return jnp.zeros(d)  # wrong: should be 2 * d

    inner = OrthogonalRotation(shape=(d,))
    with pytest.raises(ValueError, match=r"\(2 \* inner.shape\[0\],\)"):
        Conditioner(key, inner=inner, cond_shape=(2,), context_net=WrongNet())


def test_rejects_non_1d_cond_shape(key):
    inner = OrthogonalRotation(shape=(4,))
    with pytest.raises(ValueError, match="1D cond_shape"):
        Conditioner(key, inner=inner, cond_shape=(2, 2))


def test_rejects_zero_cond_dim(key):
    inner = OrthogonalRotation(shape=(4,))
    with pytest.raises(ValueError, match="positive dimension"):
        Conditioner(key, inner=inner, cond_shape=(0,))


def test_rejects_non_1d_inner_shape(key):
    """The current Conditioner contract is 1D-only (matches the rest of the package)."""

    class FakeMultiDim(eqx.Module):
        shape: tuple[int, ...] = (2, 2)
        cond_shape: tuple[int, ...] | None = None

        def transform_and_log_det(self, x, condition=None):
            return x, jnp.zeros(())

        def inverse_and_log_det(self, y, condition=None):
            return y, jnp.zeros(())

    inner = FakeMultiDim()
    with pytest.raises(ValueError, match="1D inner event shapes"):
        Conditioner(key, inner=inner, cond_shape=(2,))
