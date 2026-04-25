"""cond_dim parametrize sweep across all 5 coupling layers."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gauss_flows import (
    AffineCoupling,
    DeepSigmoidCoupling,
    GINCoupling,
    MixtureGaussianCDFCoupling,
    RQSplineCoupling,
)


COUPLING_FACTORIES = [
    pytest.param(lambda k, **kw: AffineCoupling(k, shape=(4,), **kw), id="affine"),
    pytest.param(lambda k, **kw: RQSplineCoupling(k, shape=(4,), **kw), id="spline"),
    pytest.param(
        lambda k, **kw: DeepSigmoidCoupling(k, shape=(4,), n_components=4, **kw),
        id="deepsigmoid",
    ),
    pytest.param(
        lambda k, **kw: MixtureGaussianCDFCoupling(k, shape=(4,), n_components=4, **kw),
        id="mixture_cdf",
    ),
    pytest.param(lambda k, **kw: GINCoupling(k, shape=(4,), **kw), id="gin"),
]


@pytest.mark.parametrize("make", COUPLING_FACTORIES)
def test_unconditional_default_unchanged(key, make):
    """cond_dim=None gives the original unconditional behavior."""
    layer = make(key)
    assert layer.cond_shape is None
    x = jr.normal(jr.fold_in(key, 1), (4,))
    y, log_det = layer.transform_and_log_det(x)
    assert y.shape == (4,)
    assert log_det.shape == ()


@pytest.mark.parametrize("make", COUPLING_FACTORIES)
def test_conditional_shape_contracts(key, make):
    """cond_dim=3 declares cond_shape=(3,) and accepts a (3,) condition."""
    layer = make(key, cond_dim=3)
    assert layer.cond_shape == (3,)
    x = jr.normal(jr.fold_in(key, 1), (4,))
    cond = jr.normal(jr.fold_in(key, 2), (3,))
    y, log_det = layer.transform_and_log_det(x, cond)
    assert y.shape == (4,)
    assert log_det.shape == ()


@pytest.mark.parametrize("make", COUPLING_FACTORIES)
def test_conditional_roundtrip(key, make):
    layer = make(key, cond_dim=3)
    x = jr.normal(jr.fold_in(key, 1), (4,))
    cond = jr.normal(jr.fold_in(key, 2), (3,))
    y, log_det_f = layer.transform_and_log_det(x, cond)
    x_rec, log_det_i = layer.inverse_and_log_det(y, cond)
    assert jnp.allclose(x_rec, x, atol=1e-4)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-4)


@pytest.mark.parametrize("make", COUPLING_FACTORIES)
def test_condition_actually_changes_output(key, make):
    """Sanity: passing different conditions must produce different outputs."""
    layer = make(key, cond_dim=3)
    x = jr.normal(jr.fold_in(key, 1), (4,))
    cond_a = jr.normal(jr.fold_in(key, 2), (3,))
    cond_b = jr.normal(jr.fold_in(key, 3), (3,))
    y_a, _ = layer.transform_and_log_det(x, cond_a)
    y_b, _ = layer.transform_and_log_det(x, cond_b)
    # GIN's centered log-scale + shift contributes via the active half;
    # the passive half is identity-passed regardless of condition. So
    # comparing on the active half only is the right check for GIN; for
    # the other layers, the inner MLP's per-dim output also varies.
    assert not jnp.allclose(y_a, y_b, atol=1e-3)


@pytest.mark.parametrize("make", COUPLING_FACTORIES)
def test_unconditional_layer_inside_conditional_flow(key, make):
    """An unconditional layer must silently ignore a SurVAEFlow-threaded condition.

    flowjax's inner Coupling concatenates any non-None condition, which
    would crash the inner MLP if our wrapper let it through. Verify our
    cond-dropping guard kicks in: an unconditional layer accepts (and
    ignores) a non-None condition without crashing.
    """
    layer = make(key)
    assert layer.cond_shape is None
    x = jr.normal(jr.fold_in(key, 1), (4,))
    spurious_cond = jr.normal(jr.fold_in(key, 2), (5,))
    # Should not crash; condition is silently dropped.
    y, _ = layer.transform_and_log_det(x, spurious_cond)
    assert y.shape == (4,)


@pytest.mark.parametrize("make", COUPLING_FACTORIES)
def test_jit_and_grad_through_conditional_layer(key, make):
    layer = make(key, cond_dim=3)
    x = jr.normal(jr.fold_in(key, 1), (4,))
    cond = jr.normal(jr.fold_in(key, 2), (3,))

    @eqx.filter_jit
    def f(layer, x, cond):
        y, log_det = layer.transform_and_log_det(x, cond)
        return jnp.sum(y * y) + log_det

    val = f(layer, x, cond)
    assert jnp.isfinite(val)

    grads = eqx.filter_grad(f)(layer, x, cond)
    grad_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads) if eqx.is_array(leaf)
    ]
    assert grad_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)


@pytest.mark.parametrize("make", COUPLING_FACTORIES)
def test_rejects_zero_or_negative_cond_dim(key, make):
    with pytest.raises(ValueError, match="cond_dim must be a positive int or None"):
        make(key, cond_dim=0)
    with pytest.raises(ValueError, match="cond_dim must be a positive int or None"):
        make(key, cond_dim=-1)


def test_four_layer_affine_stack_is_conditional_end_to_end(key):
    """Acceptance criterion from #28: condition flows through a 4-layer stack."""
    from flowjax.bijections import Chain, Permute

    n_dims = 4
    cond_dim = 3
    layers = []
    perm = jr.permutation(jr.fold_in(key, 100), jnp.arange(n_dims))
    for i in range(4):
        layers.append(
            AffineCoupling(jr.fold_in(key, i), shape=(n_dims,), cond_dim=cond_dim)
        )
        if i < 3:
            layers.append(Permute(perm))
    chain = Chain(layers).merge_chains()

    x = jr.normal(jr.fold_in(key, 200), (n_dims,))
    cond_a = jr.normal(jr.fold_in(key, 201), (cond_dim,))
    cond_b = jr.normal(jr.fold_in(key, 202), (cond_dim,))

    y_a, log_det_a = chain.transform_and_log_det(x, cond_a)
    y_b, log_det_b = chain.transform_and_log_det(x, cond_b)
    x_rec_a, log_det_inv_a = chain.inverse_and_log_det(y_a, cond_a)

    # Roundtrip preserves x under matching conditions
    assert jnp.allclose(x_rec_a, x, atol=1e-4)
    assert jnp.allclose(log_det_a + log_det_inv_a, 0.0, atol=1e-4)
    # Different conditions visibly differ at the output
    assert not jnp.allclose(y_a, y_b, atol=1e-3)
    assert not jnp.allclose(log_det_a, log_det_b, atol=1e-4)
