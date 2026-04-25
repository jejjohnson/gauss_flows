"""End-to-end tests for SurVAEFlow's conditional path with PR-A bases."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as ndist
import pytest
from flowjax.distributions import Normal

from gauss_flows import (
    AffineCoupling,
    ClassCondDiagGaussian,
    ConditionalDiagGaussian,
    Conditioner,
    NumpyroBase,
    OrthogonalRotation,
    SurVAEFlow,
)


# (event_shape, cond_shape) the chain expects.
EVENT_SHAPE = (4,)
COND_SHAPE = (3,)


def _make_base_only(key):
    """Conditional base + unconditional couplings."""
    base = ConditionalDiagGaussian(
        jr.fold_in(key, 1), event_shape=EVENT_SHAPE, cond_shape=COND_SHAPE
    )
    transforms = [AffineCoupling(jr.fold_in(key, 2), shape=EVENT_SHAPE)]
    return SurVAEFlow(base, transforms)


def _make_coupling_only(key):
    """Unconditional base + conditional couplings."""
    base = Normal(jnp.zeros(EVENT_SHAPE))
    transforms = [
        AffineCoupling(jr.fold_in(key, 1), shape=EVENT_SHAPE, cond_dim=COND_SHAPE[0]),
    ]
    return SurVAEFlow(base, transforms)


def _make_both(key):
    """Conditional base + conditional couplings sharing the same condition."""
    base = ConditionalDiagGaussian(
        jr.fold_in(key, 1), event_shape=EVENT_SHAPE, cond_shape=COND_SHAPE
    )
    transforms = [
        AffineCoupling(jr.fold_in(key, 2), shape=EVENT_SHAPE, cond_dim=COND_SHAPE[0]),
    ]
    return SurVAEFlow(base, transforms)


def _make_classcond_base(key):
    """ClassCondDiagGaussian (scalar int condition) + unconditional couplings."""
    base = ClassCondDiagGaussian(
        jr.fold_in(key, 1), n_classes=3, event_shape=EVENT_SHAPE
    )
    transforms = [AffineCoupling(jr.fold_in(key, 2), shape=EVENT_SHAPE)]
    return SurVAEFlow(base, transforms)


def _make_numpyro_factory_base(key):
    """NumpyroBase factory mode + unconditional couplings.

    The factory produces an event-shape-(4,) Normal whose loc is the mean
    of the (3,) condition broadcast to (4,) — keeps EVENT_SHAPE and
    COND_SHAPE distinct so the test exercises the cond/event split.
    """

    def factory(c):
        loc = jnp.mean(c) * jnp.ones(EVENT_SHAPE)
        return ndist.Normal(loc, 1.0).to_event(1)

    base = NumpyroBase(
        dist_factory=factory, event_shape=EVENT_SHAPE, cond_shape=COND_SHAPE
    )
    transforms = [AffineCoupling(jr.fold_in(key, 2), shape=EVENT_SHAPE)]
    return SurVAEFlow(base, transforms)


# ---------------------------------------------------------------------------
# Three SurVAEFlow conditioning configurations (from the user's question on #28).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_flow",
    [
        pytest.param(_make_base_only, id="base_only"),
        pytest.param(_make_coupling_only, id="coupling_only"),
        pytest.param(_make_both, id="both"),
    ],
)
def test_survae_conditional_three_configurations(key, make_flow):
    """{base only, coupling only, both} all produce finite log-probs + samples."""
    flow = make_flow(key)
    cond = jr.normal(jr.fold_in(key, 100), COND_SHAPE)
    x = flow.sample(jr.fold_in(key, 101), condition=cond)
    assert x.shape == EVENT_SHAPE
    log_p = flow.log_prob(x, jr.fold_in(key, 102), condition=cond)
    assert log_p.shape == ()
    assert jnp.isfinite(log_p)

    # Different conditions ⇒ different log-probs (genuine conditional flow).
    cond_b = jr.normal(jr.fold_in(key, 200), COND_SHAPE)
    log_p_b = flow.log_prob(x, jr.fold_in(key, 102), condition=cond_b)
    assert not jnp.allclose(log_p, log_p_b, atol=1e-4)


@pytest.mark.parametrize(
    "make_flow",
    [
        pytest.param(_make_base_only, id="base_only"),
        pytest.param(_make_coupling_only, id="coupling_only"),
        pytest.param(_make_both, id="both"),
    ],
)
def test_grad_flows_through_conditional_chain(key, make_flow):
    flow = make_flow(key)
    cond = jr.normal(jr.fold_in(key, 100), COND_SHAPE)
    x = jr.normal(jr.fold_in(key, 101), EVENT_SHAPE)

    @eqx.filter_jit
    def loss(model, x, cond):
        return -model.log_prob(x, jr.fold_in(key, 999), condition=cond)

    val = loss(flow, x, cond)
    assert jnp.isfinite(val)

    grads = eqx.filter_grad(loss)(flow, x, cond)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if eqx.is_array(leaf)]
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in leaves)


# ---------------------------------------------------------------------------
# PR-A base coverage inside SurVAEFlow.
# ---------------------------------------------------------------------------


def test_classcond_base_inside_survae(key):
    flow = _make_classcond_base(key)
    label = jnp.int32(2)
    x = flow.sample(jr.fold_in(key, 101), condition=label)
    assert x.shape == EVENT_SHAPE
    log_p = flow.log_prob(x, jr.fold_in(key, 102), condition=label)
    assert log_p.shape == ()
    assert jnp.isfinite(log_p)


def test_numpyro_base_inside_survae(key):
    flow = _make_numpyro_factory_base(key)
    cond = jr.normal(jr.fold_in(key, 100), COND_SHAPE)
    x = flow.sample(jr.fold_in(key, 101), condition=cond)
    assert x.shape == EVENT_SHAPE
    log_p = flow.log_prob(x, jr.fold_in(key, 102), condition=cond)
    assert log_p.shape == ()
    assert jnp.isfinite(log_p)


# ---------------------------------------------------------------------------
# Conditioner inside SurVAEFlow alongside other conditional layers.
# ---------------------------------------------------------------------------


def test_conditioner_in_mixed_conditional_chain(key):
    """A Conditioner-wrapped rotation composes cleanly with a conditional coupling."""
    rot = OrthogonalRotation(shape=EVENT_SHAPE)
    flow = SurVAEFlow(
        base_dist=Normal(jnp.zeros(EVENT_SHAPE)),
        transforms=[
            Conditioner(jr.fold_in(key, 1), inner=rot, cond_shape=COND_SHAPE),
            AffineCoupling(
                jr.fold_in(key, 2), shape=EVENT_SHAPE, cond_dim=COND_SHAPE[0]
            ),
        ],
    )
    cond = jr.normal(jr.fold_in(key, 100), COND_SHAPE)
    x = flow.sample(jr.fold_in(key, 101), condition=cond)
    assert x.shape == EVENT_SHAPE
    log_p = flow.log_prob(x, jr.fold_in(key, 102), condition=cond)
    assert log_p.shape == ()
    assert jnp.isfinite(log_p)


# ---------------------------------------------------------------------------
# AC #1 from issue #28: condition flows cleanly through a 4-layer AffineCoupling
# stack with no perf regression. We test the correctness leg here; perf is
# implicit in not crashing or going non-finite under jit.
# ---------------------------------------------------------------------------


def test_four_layer_conditional_affine_stack_in_survae(key):
    from flowjax.bijections import Permute

    n_dims = EVENT_SHAPE[0]
    cond_dim = COND_SHAPE[0]
    perm = jr.permutation(jr.fold_in(key, 0), jnp.arange(n_dims))
    transforms = []
    for i in range(4):
        transforms.append(
            AffineCoupling(jr.fold_in(key, i + 1), shape=(n_dims,), cond_dim=cond_dim)
        )
        if i < 3:
            transforms.append(Permute(perm))

    flow = SurVAEFlow(
        base_dist=Normal(jnp.zeros(EVENT_SHAPE)),
        transforms=transforms,
    )
    cond = jr.normal(jr.fold_in(key, 100), (cond_dim,))
    x = flow.sample(jr.fold_in(key, 101), condition=cond)
    assert x.shape == EVENT_SHAPE
    log_p = flow.log_prob(x, jr.fold_in(key, 102), condition=cond)
    assert jnp.isfinite(log_p)

    # jit + grad through the whole stack
    @eqx.filter_jit
    def loss(model, x, cond):
        return -model.log_prob(x, jr.fold_in(key, 999), condition=cond)

    grads = eqx.filter_grad(loss)(flow, x, cond)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if eqx.is_array(leaf)]
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
