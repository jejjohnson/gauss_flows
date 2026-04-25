"""Tests for ConditionalDiagGaussian."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import pytest

from gauss_flows import ConditionalDiagGaussian


def test_shape_contracts(key):
    base = ConditionalDiagGaussian(key, event_shape=(3,), cond_shape=(2,))
    cond = jnp.array([0.5, -0.5])
    assert base.shape == (3,)
    assert base.cond_shape == (2,)

    x = base.sample(jr.fold_in(key, 1), condition=cond)
    assert x.shape == (3,)

    log_p = base.log_prob(x, condition=cond)
    assert log_p.shape == ()
    assert jnp.isfinite(log_p)


def test_sample_shape_with_leading_dims(key):
    base = ConditionalDiagGaussian(key, event_shape=(4,), cond_shape=(2,))
    cond = jnp.array([1.0, -1.0])
    samples = base.sample(jr.fold_in(key, 1), sample_shape=(7,), condition=cond)
    assert samples.shape == (7, 4)


def test_log_prob_matches_diag_normal_with_fixed_params(key):
    """Check log-prob against jax.scipy.stats.norm with a hand-set net."""
    d = 3

    fixed_loc = jnp.array([0.5, -0.25, 1.0])
    fixed_log_scale = jnp.array([0.0, jnp.log(2.0), jnp.log(0.5)])
    fixed_out = jnp.concatenate([fixed_loc, fixed_log_scale])

    class ConstantNet(eqx.Module):
        out: jax.Array

        def __call__(self, _condition):
            return self.out

    base = ConditionalDiagGaussian(
        key,
        event_shape=(d,),
        cond_shape=(2,),
        param_net=ConstantNet(fixed_out),
    )

    x = jr.normal(jr.fold_in(key, 1), (d,))
    cond = jnp.zeros(2)
    expected = jnp.sum(jstats.norm.logpdf(x, fixed_loc, jnp.exp(fixed_log_scale)))
    assert jnp.allclose(base.log_prob(x, condition=cond), expected, atol=1e-5)


def test_param_net_split_uses_first_half_for_loc(key):
    """The first D outputs are loc, the second D are log_scale (not interleaved)."""
    d = 4
    # Set loc=ones (first half) and log_scale=zeros (second half).
    out = jnp.concatenate([jnp.ones(d), jnp.zeros(d)])

    class ConstantNet(eqx.Module):
        out: jax.Array

        def __call__(self, _condition):
            return self.out

    base = ConditionalDiagGaussian(
        key,
        event_shape=(d,),
        cond_shape=(1,),
        param_net=ConstantNet(out),
    )
    cond = jnp.zeros(1)
    sample = base.sample(jr.fold_in(key, 1), condition=cond)
    # Sample = loc + exp(log_scale) * z = 1 + z, mean ≈ 1.
    big_sample = base.sample(jr.fold_in(key, 2), sample_shape=(2000,), condition=cond)
    assert jnp.allclose(jnp.mean(big_sample), 1.0, atol=0.1)
    # Variance = exp(2 * log_scale) = 1.
    assert jnp.allclose(jnp.var(big_sample), 1.0, atol=0.15)
    assert sample.shape == (d,)


def test_jit_and_grad(key):
    base = ConditionalDiagGaussian(key, event_shape=(3,), cond_shape=(2,))
    cond = jnp.array([0.5, -0.5])
    x = base.sample(jr.fold_in(key, 1), condition=cond)

    eager = base.log_prob(x, condition=cond)
    jit_lp = eqx.filter_jit(base.log_prob)(x, condition=cond)
    assert jnp.allclose(eager, jit_lp, atol=1e-6)

    def loss_fn(model):
        x_local = jr.normal(jr.fold_in(key, 7), (3,))
        return jnp.sum(model.log_prob(x_local, condition=cond) ** 2)

    grads = eqx.filter_grad(loss_fn)(base)
    grad_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(grads.param_net)
        if eqx.is_array(leaf)
    ]
    assert grad_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in grad_leaves)


def test_vmap_over_conditions(key):
    base = ConditionalDiagGaussian(key, event_shape=(3,), cond_shape=(2,))
    conds = jr.normal(jr.fold_in(key, 1), (5, 2))
    xs = jr.normal(jr.fold_in(key, 2), (5, 3))
    log_ps = jax.vmap(base.log_prob, in_axes=(0, 0))(xs, conds)
    assert log_ps.shape == (5,)
    assert jnp.all(jnp.isfinite(log_ps))


def test_rejects_non_1d_event_shape(key):
    with pytest.raises(ValueError, match="1D event shapes"):
        ConditionalDiagGaussian(key, event_shape=(2, 2), cond_shape=(2,))


def test_rejects_non_1d_cond_shape(key):
    with pytest.raises(ValueError, match="1D condition shapes"):
        ConditionalDiagGaussian(key, event_shape=(3,), cond_shape=(2, 2))


def test_rejects_zero_event_dim(key):
    with pytest.raises(ValueError, match="positive dimension"):
        ConditionalDiagGaussian(key, event_shape=(0,), cond_shape=(2,))


def test_rejects_zero_cond_dim(key):
    with pytest.raises(ValueError, match="positive dimension"):
        ConditionalDiagGaussian(key, event_shape=(3,), cond_shape=(0,))


def test_rejects_user_param_net_with_wrong_output_dim(key):
    """A user-supplied param_net whose output ≠ 2 * D must raise at init."""
    d = 3

    class WrongNet(eqx.Module):
        def __call__(self, condition):
            return jnp.zeros(d)  # wrong: should be 2 * d

    with pytest.raises(ValueError, match=r"\(2 \* event_dim,\)"):
        ConditionalDiagGaussian(
            key, event_shape=(d,), cond_shape=(2,), param_net=WrongNet()
        )
