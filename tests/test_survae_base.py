"""Tests for the SurVAE foundation: AbstractSurjection + SurVAEFlow."""

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Affine, Chain
from flowjax.distributions import Normal, Transformed
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows import (
    AbstractStochastic,
    AbstractSurjection,
    SurVAEFlow,
)
from gauss_flows._src._protocols import ConditionalDistribution
from gauss_flows._src.transforms.base import _IdentitySurjection


# ---------------------------------------------------------------------------
# Test fixtures: minimal concrete surjections for each lower_bound category.
# ---------------------------------------------------------------------------


class _DeterministicSurjection(AbstractSurjection):
    """Inference-style surjection: exact log_prob (no lower bound)."""

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = False
    lower_bound: ClassVar[bool] = False

    def forward_and_log_det(
        self, x: ArrayLike, key: PRNGKeyArray, cond: Array | None = None
    ) -> tuple[Array, Array]:
        del key, cond
        return jnp.asarray(x), jnp.zeros(())

    def inverse_and_log_det(
        self, z: ArrayLike, key: PRNGKeyArray, cond: Array | None = None
    ) -> tuple[Array, Array]:
        del key, cond
        return jnp.asarray(z), jnp.zeros(())


class _LowerBoundSurjection(AbstractSurjection):
    """Generative-style surjection: contributes a lower bound."""

    stochastic_forward: ClassVar[bool] = True
    stochastic_inverse: ClassVar[bool] = False
    lower_bound: ClassVar[bool] = True

    def forward_and_log_det(
        self, x: ArrayLike, key: PRNGKeyArray, cond: Array | None = None
    ) -> tuple[Array, Array]:
        del key, cond
        return jnp.asarray(x), jnp.zeros(())

    def inverse_and_log_det(
        self, z: ArrayLike, key: PRNGKeyArray, cond: Array | None = None
    ) -> tuple[Array, Array]:
        del key, cond
        return jnp.asarray(z), jnp.zeros(())


class _IdentityStochastic(AbstractStochastic):
    """Stochastic-both-ways: contributes a lower bound (inherited)."""

    def forward_and_log_det(
        self, x: ArrayLike, key: PRNGKeyArray, cond: Array | None = None
    ) -> tuple[Array, Array]:
        del key, cond
        return jnp.asarray(x), jnp.zeros(())

    def inverse_and_log_det(
        self, z: ArrayLike, key: PRNGKeyArray, cond: Array | None = None
    ) -> tuple[Array, Array]:
        del key, cond
        return jnp.asarray(z), jnp.zeros(())


# ---------------------------------------------------------------------------
# AbstractSurjection ABC behaviour
# ---------------------------------------------------------------------------


def test_abstract_surjection_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractSurjection()  # type: ignore[abstract]


def test_identity_surjection_is_concrete(key):
    surj = _IdentitySurjection()
    x = jr.normal(key, (3,))
    z, ld_f = surj.forward_and_log_det(x, key)
    x_back, ld_i = surj.inverse_and_log_det(z, key)
    assert jnp.allclose(z, x)
    assert jnp.allclose(x_back, x)
    assert ld_f == 0.0
    assert ld_i == 0.0


def test_abstract_stochastic_inherits_flags():
    assert AbstractStochastic.stochastic_forward is True
    assert AbstractStochastic.stochastic_inverse is True
    assert AbstractStochastic.lower_bound is True


# ---------------------------------------------------------------------------
# SurVAEFlow.lower_bound flag derivation
# ---------------------------------------------------------------------------


def test_lower_bound_bijection_only(key):
    base = Normal(jnp.zeros(2))
    flow = SurVAEFlow(base, [Affine(loc=jnp.zeros(2))])
    assert flow.lower_bound is False


def test_lower_bound_with_inference_surjection(key):
    base = Normal(jnp.zeros(2))
    flow = SurVAEFlow(base, [Affine(loc=jnp.zeros(2)), _DeterministicSurjection()])
    assert flow.lower_bound is False


def test_lower_bound_with_generative_surjection(key):
    base = Normal(jnp.zeros(2))
    flow = SurVAEFlow(base, [Affine(loc=jnp.zeros(2)), _LowerBoundSurjection()])
    assert flow.lower_bound is True


def test_lower_bound_with_stochastic_transform(key):
    base = Normal(jnp.zeros(2))
    flow = SurVAEFlow(base, [Affine(loc=jnp.zeros(2)), _IdentityStochastic()])
    assert flow.lower_bound is True


# ---------------------------------------------------------------------------
# Regression: bijection-only SurVAEFlow matches flowjax.Transformed
# ---------------------------------------------------------------------------


def _make_bijection_chain(key):
    k1, k2 = jr.split(key)
    return [
        Affine(loc=jr.normal(k1, (3,)), scale=jnp.exp(jr.normal(k2, (3,)) * 0.3)),
        Affine(
            loc=jr.normal(jr.fold_in(key, 1), (3,)),
            scale=jnp.exp(jr.normal(jr.fold_in(key, 2), (3,)) * 0.3),
        ),
    ]


def test_log_prob_matches_flowjax_transformed_for_bijection_only_chain(key):
    base = Normal(jnp.zeros(3))
    bijections = _make_bijection_chain(key)
    survae = SurVAEFlow(base, bijections)
    reference = Transformed(base, Chain(bijections))

    xs = jr.normal(jr.fold_in(key, 10), (8, 3))
    actual = jnp.asarray(
        [survae.log_prob(x, jr.fold_in(key, i)) for i, x in enumerate(xs)]
    )
    expected = reference.log_prob(xs)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_sample_shape_for_bijection_only_chain(key):
    base = Normal(jnp.zeros(3))
    flow = SurVAEFlow(base, _make_bijection_chain(key))
    single = flow.sample(jr.fold_in(key, 1))
    assert single.shape == (3,)
    batched = flow.sample(jr.fold_in(key, 2), sample_shape=(7,))
    assert batched.shape == (7, 3)


def test_sample_log_prob_finite_with_identity_surjection(key):
    base = Normal(jnp.zeros(3))
    flow = SurVAEFlow(base, [Affine(loc=jnp.zeros(3)), _IdentitySurjection()])
    samples = flow.sample(jr.fold_in(key, 1), sample_shape=(8,))
    log_probs = jnp.asarray(
        [flow.log_prob(x, jr.fold_in(key, i)) for i, x in enumerate(samples)]
    )
    assert samples.shape == (8, 3)
    assert log_probs.shape == (8,)
    assert jnp.all(jnp.isfinite(log_probs))


# ---------------------------------------------------------------------------
# ConditionalDistribution structural protocol
# ---------------------------------------------------------------------------


def test_conditional_distribution_protocol_runtime_checkable():
    class _Stub:
        def sample(self, key, *, condition):
            return condition

        def log_prob(self, value, *, condition):
            return jnp.zeros(())

    assert isinstance(_Stub(), ConditionalDistribution)
    # A bare object lacking the required methods should fail the check.

    class _Missing:
        def sample(self, key, *, condition):
            return condition

    assert not isinstance(_Missing(), ConditionalDistribution)
