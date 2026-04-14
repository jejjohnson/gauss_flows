"""Tests for the GaussianMixture trainable base distribution."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import pytest
from jax.scipy.special import logsumexp

from gauss_flows import GaussianMixture, gaussianization_flow


def _reference_log_prob(dist: GaussianMixture, x):
    """Oracle GMM log-prob via jstats.multivariate_normal.logpdf."""
    log_weights = jax.nn.log_softmax(dist.log_weights)
    L = dist._build_L()

    def _component(loc_k, L_k):
        Sigma = L_k @ L_k.T
        return jstats.multivariate_normal.logpdf(x, loc_k, Sigma)

    log_pdfs = jax.vmap(_component)(dist.loc, L)
    return logsumexp(log_weights + log_pdfs)


def test_diagonal_sample_shape(key):
    dist = GaussianMixture(key, n_components=3, event_shape=(4,))
    samples = dist.sample(jr.fold_in(key, 1), sample_shape=(8,))
    assert samples.shape == (8, 4)
    assert jnp.all(jnp.isfinite(samples))


def test_full_sample_shape(key):
    dist = GaussianMixture(key, n_components=3, event_shape=(4,), diagonal=False)
    samples = dist.sample(jr.fold_in(key, 1), sample_shape=(8,))
    assert samples.shape == (8, 4)
    assert jnp.all(jnp.isfinite(samples))


def test_diagonal_log_prob_matches_reference(key):
    dist = GaussianMixture(key, n_components=3, event_shape=(4,))
    x = jr.normal(jr.fold_in(key, 1), (4,))
    actual = dist.log_prob(x)
    expected = _reference_log_prob(dist, x)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_full_log_prob_matches_reference(key):
    dist = GaussianMixture(key, n_components=3, event_shape=(4,), diagonal=False)
    # Give strict_tril non-zero entries to exercise off-diagonal terms.
    new_strict = jr.normal(jr.fold_in(key, 2), dist.strict_tril.shape) * 0.3
    dist = eqx.tree_at(lambda d: d.strict_tril, dist, new_strict)

    x = jr.normal(jr.fold_in(key, 3), (4,))
    actual = dist.log_prob(x)
    expected = _reference_log_prob(dist, x)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_sample_log_prob_roundtrip_finite(key):
    dist = GaussianMixture(key, n_components=3, event_shape=(3,))
    samples = dist.sample(jr.fold_in(key, 1), sample_shape=(16,))
    log_probs = dist.log_prob(samples)
    assert log_probs.shape == (16,)
    assert jnp.all(jnp.isfinite(log_probs))


def test_weights_respected(key):
    """Heavy-weighted component should dominate a sample-based estimate."""
    dist = GaussianMixture(key, n_components=2, event_shape=(2,))
    # Pin locs far apart so the nearest-component assignment is unambiguous.
    dist = eqx.tree_at(lambda d: d.loc, dist, jnp.array([[-10.0, 0.0], [10.0, 0.0]]))
    # Skew weights strongly toward component 0.
    dist = eqx.tree_at(lambda d: d.log_weights, dist, jnp.array([5.0, -5.0]))
    samples = dist.sample(jr.fold_in(key, 1), sample_shape=(2000,))
    d0 = jnp.linalg.norm(samples - dist.loc[0], axis=-1)
    d1 = jnp.linalg.norm(samples - dist.loc[1], axis=-1)
    fraction_near_c0 = jnp.mean(d0 < d1)
    assert fraction_near_c0 > 0.95


def test_fits_known_gmm_via_log_prob_improvement(key):
    """Crude fit test: after a few steps of max-likelihood the model should
    assign strictly higher average log-prob than a poorly-initialised one.
    Avoids a full optimisation dependency; we just check local improvement.
    """
    target_locs = jnp.array([[-3.0, 0.0], [3.0, 0.0]])
    n = 500
    k_sample = jr.fold_in(key, 1)
    comp = jr.bernoulli(k_sample, 0.5, (n,)).astype(jnp.int32)
    noise = jr.normal(jr.fold_in(key, 2), (n, 2)) * 0.5
    data = target_locs[comp] + noise

    init = GaussianMixture(key, n_components=2, event_shape=(2,), loc_init_scale=3.0)

    # Fit locs directly to empirical means for each component under hard
    # assignment. This is a single-step E/M, good enough to beat init.
    def hard_assign(d):
        L = d._build_L()

        def _component(loc_k, L_k):
            Sigma = L_k @ L_k.T
            return jstats.multivariate_normal.logpdf(data, loc_k, Sigma)

        log_pdfs = jax.vmap(_component)(d.loc, L)
        return jnp.argmax(log_pdfs, axis=0)

    assign = hard_assign(init)

    # Guard against empty assignments: keep the prior loc for any component
    # that received zero points (otherwise jnp.mean of an empty slice is NaN
    # and the test fails for reasons unrelated to GaussianMixture itself).
    def _safe_mean(mask, prior_loc):
        count = jnp.sum(mask)
        masked_sum = jnp.sum(jnp.where(mask[:, None], data, 0.0), axis=0)
        return jnp.where(count > 0, masked_sum / jnp.maximum(count, 1), prior_loc)

    new_loc = jnp.stack(
        [
            _safe_mean(assign == 0, init.loc[0]),
            _safe_mean(assign == 1, init.loc[1]),
        ]
    )
    fitted = eqx.tree_at(lambda d: d.loc, init, new_loc)
    mean_lp_init = jnp.mean(init.log_prob(data))
    mean_lp_fit = jnp.mean(fitted.log_prob(data))
    assert mean_lp_fit > mean_lp_init + 0.1


def test_as_base_dist_in_gaussianization_flow(key):
    k_base, k_flow, k_sample = jr.split(key, 3)
    base = GaussianMixture(k_base, n_components=3, event_shape=(3,))
    flow = gaussianization_flow(k_flow, n_dims=3, n_layers=2, base_dist=base)
    samples = flow.sample(k_sample, (8,))
    log_probs = flow.log_prob(samples)
    assert samples.shape == (8, 3)
    assert log_probs.shape == (8,)
    assert jnp.all(jnp.isfinite(log_probs))


def test_event_shape_mismatch_raises(key):
    k_base, k_flow = jr.split(key)
    base = GaussianMixture(k_base, n_components=2, event_shape=(4,))
    with pytest.raises(ValueError, match=r"base_dist\.shape"):
        gaussianization_flow(k_flow, n_dims=3, n_layers=2, base_dist=base)


def test_conditional_base_dist_raises(key):
    """gaussianization_flow doesn't thread a condition, so reject conditional bases."""

    # Forge a conditional version of GaussianMixture by overriding cond_shape
    # at the class level in a tiny subclass — easier than building a full
    # conditional distribution from scratch.
    class _ConditionalMixture(GaussianMixture):
        cond_shape: tuple[int, ...] | None = (1,)  # type: ignore[assignment]

    conditional = _ConditionalMixture(key, n_components=2, event_shape=(3,))
    assert conditional.cond_shape == (1,)
    with pytest.raises(ValueError, match="Conditional base distributions"):
        gaussianization_flow(
            jr.fold_in(key, 1), n_dims=3, n_layers=2, base_dist=conditional
        )


def test_n_components_must_be_positive(key):
    with pytest.raises(ValueError, match="n_components"):
        GaussianMixture(key, n_components=0, event_shape=(2,))


def test_only_1d_event_shape(key):
    with pytest.raises(ValueError, match="1D event"):
        GaussianMixture(key, n_components=2, event_shape=(2, 3))
