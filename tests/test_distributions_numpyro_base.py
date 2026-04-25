"""Tests for NumpyroBase."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as ndist
import pytest

from gauss_flows import NumpyroBase


def test_unconditional_normal_shape_contracts(key):
    inner = ndist.Normal(0.0, 1.0).expand((3,)).to_event(1)
    base = NumpyroBase(dist=inner)
    assert base.shape == (3,)
    assert base.cond_shape is None

    x = base.sample(jr.fold_in(key, 1))
    assert x.shape == (3,)
    log_p = base.log_prob(x)
    assert log_p.shape == ()
    assert jnp.isfinite(log_p)


def test_unconditional_log_prob_matches_numpyro(key):
    inner = ndist.Normal(0.5, 2.0).expand((4,)).to_event(1)
    base = NumpyroBase(dist=inner)
    x = jr.normal(jr.fold_in(key, 1), (4,))
    expected = inner.log_prob(x)
    got = base.log_prob(x)
    assert jnp.allclose(got, expected, atol=1e-6)


def test_unconditional_multivariate_normal(key):
    inner = ndist.MultivariateNormal(jnp.zeros(3), covariance_matrix=jnp.eye(3))
    base = NumpyroBase(dist=inner)
    assert base.shape == (3,)
    x = jr.normal(jr.fold_in(key, 1), (3,))
    assert jnp.allclose(base.log_prob(x), inner.log_prob(x), atol=1e-6)


def test_conditional_factory_log_prob_varies_with_condition(key):
    def factory(c):
        return ndist.Normal(c, 1.0).to_event(1)

    base = NumpyroBase(dist_factory=factory, event_shape=(3,), cond_shape=(3,))
    x = jnp.zeros(3)
    lp_a = base.log_prob(x, condition=jnp.zeros(3))
    lp_b = base.log_prob(x, condition=jnp.ones(3) * 2.0)
    assert lp_a > lp_b  # x=0 is more likely under N(0, 1) than under N(2, 1)


def test_conditional_factory_sample_mean_tracks_condition(key):
    def factory(c):
        return ndist.Normal(c, 0.5).to_event(1)

    base = NumpyroBase(dist_factory=factory, event_shape=(2,), cond_shape=(2,))
    cond = jnp.array([1.5, -0.7])
    samples = base.sample(jr.fold_in(key, 1), sample_shape=(2000,), condition=cond)
    assert samples.shape == (2000, 2)
    assert jnp.allclose(jnp.mean(samples, axis=0), cond, atol=0.05)


def test_jit_unconditional(key):
    base = NumpyroBase(dist=ndist.Normal(0.0, 1.0).expand((3,)).to_event(1))
    x = jr.normal(jr.fold_in(key, 1), (3,))
    jit_lp = eqx.filter_jit(base.log_prob)(x)
    assert jnp.allclose(jit_lp, base.log_prob(x), atol=1e-6)


def test_vmap_conditional_log_prob(key):
    def factory(c):
        return ndist.Normal(c, 1.0).to_event(1)

    base = NumpyroBase(dist_factory=factory, event_shape=(2,), cond_shape=(2,))
    xs = jr.normal(jr.fold_in(key, 1), (5, 2))
    cs = jr.normal(jr.fold_in(key, 2), (5, 2))
    log_ps = jax.vmap(base.log_prob, in_axes=(0, 0))(xs, cs)
    assert log_ps.shape == (5,)
    assert jnp.all(jnp.isfinite(log_ps))


def test_eqx_module_factory_is_trainable(key):
    """A parameterized eqx.Module factory must remain visible to filter_grad.

    The wrapper must NOT mark _factory as static — that would freeze any
    learnable parameters carried by an eqx.Module factory.
    """

    class TrainableFactory(eqx.Module):
        loc_param: jax.Array

        def __call__(self, condition):
            return ndist.Normal(self.loc_param + condition, 1.0).to_event(1)

    fac = TrainableFactory(loc_param=jnp.array([0.5, -0.5, 1.0]))
    base = NumpyroBase(dist_factory=fac, event_shape=(3,), cond_shape=(3,))

    def loss(model, c, x):
        return -model.log_prob(x, condition=c)

    grads = eqx.filter_grad(loss)(base, jnp.zeros(3), jnp.zeros(3))
    assert grads._factory is not None
    grad_loc = grads._factory.loc_param
    assert jnp.all(jnp.isfinite(grad_loc))
    # x = 0, condition = 0, so loc_param sits at the mean; grad of log p wrt
    # loc_param is loc_param itself for unit-variance Normal.
    assert jnp.allclose(grad_loc, fac.loc_param, atol=1e-5)


def test_expand_to_event_recipe_from_docstring(key):
    """The docstring's `.expand(shape).to_event(rank)` recipe works."""
    inner = ndist.Normal(0.0, 1.0).expand((4,)).to_event(1)
    assert inner.batch_shape == ()
    assert inner.event_shape == (4,)
    base = NumpyroBase(dist=inner)
    assert base.shape == (4,)
    x = jnp.zeros(4)
    assert jnp.allclose(base.log_prob(x), inner.log_prob(x), atol=1e-6)


def test_rejects_neither_dist_nor_factory():
    with pytest.raises(ValueError, match="exactly one of"):
        NumpyroBase()


def test_rejects_both_dist_and_factory():
    with pytest.raises(ValueError, match="exactly one of"):
        NumpyroBase(
            dist=ndist.Normal(0.0, 1.0).to_event(0),
            dist_factory=lambda c: ndist.Normal(c, 1.0),
        )


def test_rejects_factory_without_event_shape():
    with pytest.raises(ValueError, match="both 'event_shape' and 'cond_shape'"):
        NumpyroBase(dist_factory=lambda c: ndist.Normal(c, 1.0), cond_shape=(2,))


def test_rejects_factory_without_cond_shape():
    with pytest.raises(ValueError, match="both 'event_shape' and 'cond_shape'"):
        NumpyroBase(dist_factory=lambda c: ndist.Normal(c, 1.0), event_shape=(2,))


def test_rejects_dist_with_nontrivial_batch_shape():
    inner = ndist.Normal(jnp.zeros(4), 1.0)  # batch_shape=(4,), event_shape=()
    with pytest.raises(ValueError, match="batch_shape == \\(\\)"):
        NumpyroBase(dist=inner)


def test_rejects_event_shape_mismatch_in_dist_mode():
    inner = ndist.Normal(0.0, 1.0).expand((3,)).to_event(1)  # event_shape=(3,)
    with pytest.raises(ValueError, match="does not match"):
        NumpyroBase(dist=inner, event_shape=(5,))


def test_rejects_cond_shape_in_dist_mode():
    inner = ndist.Normal(0.0, 1.0).expand((3,)).to_event(1)
    with pytest.raises(ValueError, match="cond_shape must be None"):
        NumpyroBase(dist=inner, cond_shape=(2,))
