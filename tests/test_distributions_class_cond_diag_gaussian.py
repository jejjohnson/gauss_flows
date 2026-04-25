"""Tests for ClassCondDiagGaussian (covers the GlowBase configuration)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import pytest

from gauss_flows import ClassCondDiagGaussian


def test_shape_contracts_and_metadata(key):
    base = ClassCondDiagGaussian(key, n_classes=4, event_shape=(3,))
    assert base.shape == (3,)
    assert base.cond_shape == ()
    assert base.n_classes == 4
    assert base.loc.shape == (4, 3)
    assert base.log_scale_raw.shape == (4, 3)

    label = jnp.int32(2)
    x = base.sample(jr.fold_in(key, 1), condition=label)
    assert x.shape == (3,)
    log_p = base.log_prob(x, condition=label)
    assert log_p.shape == ()
    assert jnp.isfinite(log_p)


def test_sample_with_leading_dims(key):
    base = ClassCondDiagGaussian(key, n_classes=4, event_shape=(2,))
    samples = base.sample(jr.fold_in(key, 1), sample_shape=(8,), condition=jnp.int32(0))
    assert samples.shape == (8, 2)


def test_log_prob_at_default_init_matches_unit_normal(key):
    """At init, loc=0, log_scale_raw=0 → effectively N(0, I)."""
    base = ClassCondDiagGaussian(key, n_classes=3, event_shape=(4,))
    x = jr.normal(jr.fold_in(key, 1), (4,))
    expected = jnp.sum(jstats.norm.logpdf(x, 0.0, 1.0))
    got = base.log_prob(x, condition=jnp.int32(1))
    assert jnp.allclose(got, expected, atol=1e-5)


def test_per_class_params_distinct_after_setting(key):
    """log_prob differs across classes once per-class params diverge."""
    d = 3
    base = ClassCondDiagGaussian(key, n_classes=3, event_shape=(d,))
    new_loc = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    )
    base = eqx.tree_at(lambda b: b.loc, base, new_loc)

    x = jnp.array([2.0, 0.0, 0.0])
    lp0 = base.log_prob(x, condition=jnp.int32(0))
    lp1 = base.log_prob(x, condition=jnp.int32(1))
    lp2 = base.log_prob(x, condition=jnp.int32(2))
    # Class 1 has loc that matches x → highest log-prob.
    assert lp1 > lp0
    assert lp1 > lp2


def test_learn_mean_false_zeros_out_loc(key):
    """When learn_mean=False the stored loc is ignored at evaluation."""
    base = ClassCondDiagGaussian(key, n_classes=3, event_shape=(2,), learn_mean=False)
    new_loc = jnp.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    base = eqx.tree_at(lambda b: b.loc, base, new_loc)

    # Despite the non-zero stored loc, evaluation should treat mean as zero.
    x = jnp.zeros(2)
    lp = base.log_prob(x, condition=jnp.int32(0))
    expected = jnp.sum(jstats.norm.logpdf(x, 0.0, 1.0))
    assert jnp.allclose(lp, expected, atol=1e-5)


def test_logscale_factor_scales_effective_log_scale(key):
    """log_scale_raw[c] * logscale_factor is the effective log-scale."""
    base = ClassCondDiagGaussian(
        key, n_classes=2, event_shape=(2,), logscale_factor=3.0
    )
    new_lsr = jnp.array([[0.5, 0.5], [0.0, 0.0]])
    base = eqx.tree_at(lambda b: b.log_scale_raw, base, new_lsr)

    # Effective log_scale at class 0 = 0.5 * 3.0 = 1.5
    x = jnp.zeros(2)
    expected = jnp.sum(jstats.norm.logpdf(x, 0.0, jnp.exp(1.5)))
    got = base.log_prob(x, condition=jnp.int32(0))
    assert jnp.allclose(got, expected, atol=1e-5)


def test_glow_recipe_smoke(key):
    """The Glow base configuration constructs and runs end-to-end."""
    base = ClassCondDiagGaussian(
        key,
        n_classes=10,
        event_shape=(5,),
        learn_mean=False,
        logscale_factor=3.0,
    )
    label = jnp.int32(7)
    x = base.sample(jr.fold_in(key, 1), condition=label)
    log_p = base.log_prob(x, condition=label)
    assert x.shape == (5,)
    assert jnp.isfinite(log_p)


def test_jit_and_vmap_over_labels(key):
    base = ClassCondDiagGaussian(key, n_classes=4, event_shape=(3,))
    labels = jnp.arange(4, dtype=jnp.int32)
    xs = jr.normal(jr.fold_in(key, 1), (4, 3))

    @eqx.filter_jit
    def batch_log_prob(xs_, labels_):
        return jax.vmap(base.log_prob, in_axes=(0, 0))(xs_, labels_)

    jit_lp = batch_log_prob(xs, labels)
    assert jit_lp.shape == (4,)
    assert jnp.all(jnp.isfinite(jit_lp))


def test_grad_flows_to_per_class_params(key):
    base = ClassCondDiagGaussian(key, n_classes=3, event_shape=(3,))
    label = jnp.int32(1)
    x = jnp.ones(3)

    def loss_fn(model):
        return -model.log_prob(x, condition=label)

    grads = eqx.filter_grad(loss_fn)(base)
    # Only the row for class 1 should receive a gradient.
    assert jnp.all(jnp.isfinite(grads.loc))
    assert jnp.all(jnp.isfinite(grads.log_scale_raw))
    assert jnp.any(jnp.abs(grads.loc[1]) > 0)
    assert jnp.allclose(grads.loc[0], 0.0)
    assert jnp.allclose(grads.loc[2], 0.0)


def test_rejects_non_scalar_condition_via_internal_api(key):
    """The internal _resolve_params guard rejects non-scalar conditions.

    The public `log_prob`/`sample` API auto-vmaps over a leading batch
    dimension (flowjax handles this for us, with correct per-class
    semantics). The guard is defence-in-depth for any code path that
    bypasses the public API and calls _resolve_params/_log_prob directly.
    """
    base = ClassCondDiagGaussian(key, n_classes=4, event_shape=(3,))
    with pytest.raises(ValueError, match="scalar integer class label"):
        base._resolve_params(jnp.array([1, 2]))
    with pytest.raises(ValueError, match="scalar integer class label"):
        base._log_prob(jnp.zeros(3), condition=jnp.array([1, 2]))


def test_public_api_auto_vmaps_batched_condition(key):
    """Verify flowjax auto-vmaps over batched conditions correctly."""
    base = ClassCondDiagGaussian(key, n_classes=4, event_shape=(3,))
    new_loc = jnp.array([[0.0, 0, 0], [5.0, 0, 0], [-5.0, 0, 0], [0, 0, 0]])
    base = eqx.tree_at(lambda m: m.loc, base, new_loc)
    x = jnp.array([5.0, 0.0, 0.0])
    # Should produce per-class log_probs matching individual scalar calls.
    out = base.log_prob(x, condition=jnp.array([1, 2]))
    assert out.shape == (2,)
    expected_1 = base.log_prob(x, condition=jnp.int32(1))
    expected_2 = base.log_prob(x, condition=jnp.int32(2))
    assert jnp.allclose(out[0], expected_1)
    assert jnp.allclose(out[1], expected_2)


def test_rejects_zero_n_classes(key):
    with pytest.raises(ValueError, match="n_classes must be positive"):
        ClassCondDiagGaussian(key, n_classes=0, event_shape=(3,))


def test_rejects_zero_event_dim(key):
    with pytest.raises(ValueError, match="positive"):
        ClassCondDiagGaussian(key, n_classes=2, event_shape=(0,))


def test_rejects_nonpositive_logscale_factor(key):
    with pytest.raises(ValueError, match="logscale_factor must be positive"):
        ClassCondDiagGaussian(key, n_classes=2, event_shape=(3,), logscale_factor=0.0)
