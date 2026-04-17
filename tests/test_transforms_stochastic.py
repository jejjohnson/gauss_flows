"""Tests for :class:`VAE` — SurVAE stochastic transform."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from flowjax.distributions import Normal
from jax import Array
from jaxtyping import PRNGKeyArray

from gauss_flows import VAE, SurVAEFlow


class DiagGaussianConditional(eqx.Module):
    """Minimal reparameterized conditional Gaussian ``N(Wc+b, softplus(log_s)^2·I)``.

    Matches the :class:`ConditionalDistribution` protocol and is
    differentiable end-to-end through the reparameterization trick. Used as
    both encoder and decoder for the VAE tests; small enough to keep the
    smoke test fast.
    """

    weight: Array
    bias: Array
    log_scale: Array

    def __init__(self, key: PRNGKeyArray, in_dim: int, out_dim: int):
        w_key, b_key = jr.split(key)
        self.weight = jr.normal(w_key, (out_dim, in_dim)) * 0.1
        self.bias = jr.normal(b_key, (out_dim,)) * 0.1
        self.log_scale = jnp.full((out_dim,), -0.3)

    def _mean(self, condition: Array) -> Array:
        return self.weight @ condition + self.bias

    def _scale(self) -> Array:
        return jax.nn.softplus(self.log_scale) + 1e-3

    def sample(self, key: PRNGKeyArray, *, condition: Array) -> Array:
        mean = self._mean(jnp.asarray(condition))
        scale = self._scale()
        eps = jr.normal(key, mean.shape)
        return mean + scale * eps  # reparameterized

    def log_prob(self, value: Array, *, condition: Array) -> Array:
        mean = self._mean(jnp.asarray(condition))
        scale = self._scale()
        var = scale**2
        log_norm = -0.5 * jnp.log(2 * jnp.pi * var)
        quad = -0.5 * ((jnp.asarray(value) - mean) ** 2) / var
        return jnp.sum(log_norm + quad)


def test_vae_forward_returns_encoder_sample_and_elbo_term(key):
    """Forward must return the sampled z and exactly log p(x|z) - log q(z|x).

    Recomputing the identity from the *same* z is tautological; what matters
    is that the class returns z from `encoder.sample` and the ELBO term
    evaluated at that same z, which this test pins down.
    """
    encoder = DiagGaussianConditional(jr.fold_in(key, 0), in_dim=2, out_dim=2)
    decoder = DiagGaussianConditional(jr.fold_in(key, 1), in_dim=2, out_dim=2)
    vae = VAE(encoder, decoder)
    x = jnp.array([0.3, -0.7])

    fwd_key = jr.fold_in(key, 2)
    z, log_det = vae.forward_and_log_det(x, fwd_key)
    z_expected = encoder.sample(fwd_key, condition=x)
    expected = decoder.log_prob(x, condition=z_expected) - encoder.log_prob(
        z_expected, condition=x
    )

    assert z.shape == x.shape
    assert jnp.allclose(z, z_expected)
    assert jnp.allclose(log_det, expected)


def test_vae_forward_elbo_matches_closed_form_gaussian_kl(key):
    """Monte-Carlo ``E[log p(x|z) - log q(z|x)]`` must match the closed-form
    ``log p(x|Wμ_q+b, Σ_p) - 0.5·tr(W Σ_q Wᵀ Σ_p⁻¹) + H(q)`` for Gaussian q/p.

    Concretely, with ``q(z|x) = N(μ_q, diag(s_q²))``,
    ``p(x|z) = N(Wz+b, diag(s_p²))``, the expectation decomposes as::

        E[log p(x|z)]   = log N(x; Wμ_q+b, diag(s_p²)) − 0.5·tr(W diag(s_q²) Wᵀ / s_p²)
        E[log q(z|x)]  = -H(q) = -0.5·Σ log(2πe s_q²)

        E[log p(x|z) - log q(z|x)] = E[log p(x|z)] + H(q)

    With enough samples the MC estimate lands within 3-sigma of this value,
    verifying both the sampling and the log-prob paths are wired correctly.
    """
    enc_key, dec_key, sample_key = jr.split(key, 3)
    in_dim, out_dim = 2, 2
    encoder = DiagGaussianConditional(enc_key, in_dim=in_dim, out_dim=out_dim)
    decoder = DiagGaussianConditional(dec_key, in_dim=out_dim, out_dim=in_dim)
    vae = VAE(encoder, decoder)
    x = jnp.array([0.5, -0.2])

    n_samples = 4096
    keys = jr.split(sample_key, n_samples)
    # one forward per key; take the log_det only.
    _, log_dets = jax.vmap(lambda k: vae.forward_and_log_det(x, k))(keys)
    mc_estimate = jnp.mean(log_dets)

    # Closed form for Gaussian q, p.
    mu_q = encoder._mean(x)
    s_q = encoder._scale()
    mu_p = decoder._mean(mu_q)  # E_q[Wz+b] = W μ_q + b
    s_p = decoder._scale()
    var_p = s_p**2
    # E[log p(x|z)] under q:
    log_p_at_mean = jnp.sum(
        -0.5 * jnp.log(2 * jnp.pi * var_p) - 0.5 * (x - mu_p) ** 2 / var_p
    )
    trace_term = 0.5 * jnp.sum(
        (decoder.weight @ (s_q**2 * decoder.weight.T)).diagonal() / var_p
    )
    expected_log_p = log_p_at_mean - trace_term
    # Entropy of q: H(q) = 0.5 Σ log(2π e s_q²)
    entropy_q = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * s_q**2))
    # E[log q(z|x)] = -H(q)
    expected_log_q = -entropy_q
    closed_form = expected_log_p - expected_log_q

    # 3-sigma MC tolerance on 4096 samples is ~ O(0.05) for this problem.
    assert jnp.allclose(mc_estimate, closed_form, atol=0.1)


def test_vae_inverse_samples_and_zero_log_det(key):
    encoder = DiagGaussianConditional(jr.fold_in(key, 0), in_dim=2, out_dim=2)
    decoder = DiagGaussianConditional(jr.fold_in(key, 1), in_dim=2, out_dim=2)
    vae = VAE(encoder, decoder)
    z = jnp.array([1.0, -1.0])
    x, log_det = vae.inverse_and_log_det(z, jr.fold_in(key, 3))
    assert x.shape == z.shape
    assert log_det == pytest.approx(0.0)


def test_vae_inverse_matches_decoder_sample(key):
    """Inverse must delegate to ``decoder.sample`` with the same key."""
    encoder = DiagGaussianConditional(jr.fold_in(key, 0), in_dim=2, out_dim=2)
    decoder = DiagGaussianConditional(jr.fold_in(key, 1), in_dim=2, out_dim=2)
    vae = VAE(encoder, decoder)
    z = jnp.array([0.25, 0.75])
    inv_key = jr.fold_in(key, 4)
    x, _ = vae.inverse_and_log_det(z, inv_key)
    expected = decoder.sample(inv_key, condition=z)
    assert jnp.allclose(x, expected)


def test_vae_cond_shape_is_none(key):
    """``VAE`` is unconditional at the chain level — matches every other
    unconditional transform in the package."""
    encoder = DiagGaussianConditional(jr.fold_in(key, 0), in_dim=2, out_dim=2)
    decoder = DiagGaussianConditional(jr.fold_in(key, 1), in_dim=2, out_dim=2)
    vae = VAE(encoder, decoder)
    assert vae.cond_shape is None


@pytest.mark.slow
def test_vae_svi_smoke_trains_below_fixed_prior_baseline(key):
    """SVI sanity check: trained VAE ELBO should beat the *fixed* prior baseline.

    Key correctness points:
      - The ``Normal`` base_dist has trainable parameters in flowjax, so we
        compute the baseline *before* training against the original fixed
        prior, not against the drifted post-training prior.
      - The training step is wrapped in ``eqx.filter_jit`` so 500 iterations
        are not 500 Python-level retraces.
      - We compare the mean of the last segment to the first, rather than
        requiring strictly monotone segment means (which is flaky under
        Adam with ``n=256`` batches).
    """
    latent_dim = 2
    data = jr.normal(jr.fold_in(key, 10), (256, latent_dim)) + jnp.array([1.0, -1.0])

    encoder = DiagGaussianConditional(
        jr.fold_in(key, 20), in_dim=latent_dim, out_dim=latent_dim
    )
    decoder = DiagGaussianConditional(
        jr.fold_in(key, 30), in_dim=latent_dim, out_dim=latent_dim
    )
    flow = SurVAEFlow(Normal(jnp.zeros(latent_dim)), [VAE(encoder, decoder)])

    # Baseline from the fixed standard-normal prior, BEFORE any updates.
    fixed_prior = Normal(jnp.zeros(latent_dim))
    baseline = -jnp.mean(jax.vmap(fixed_prior.log_prob)(data))

    def loss_fn(model, k):
        return -jnp.mean(model.log_prob(data, k))

    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, k):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, k)
        updates, opt_state = optim.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    train_keys = jr.split(jr.fold_in(key, 999), 500)
    losses = []
    for step_key in train_keys:
        flow, opt_state, loss = step(flow, opt_state, step_key)
        losses.append(loss)
    losses = jnp.stack(losses)

    assert jnp.isfinite(losses).all()
    # Trained VAE should beat the fixed-prior baseline.
    assert losses[-1] < baseline
    # Compare mean of the last segment to the first — robust to Adam noise.
    segment_means = losses.reshape(5, 100).mean(axis=1)
    assert segment_means[-1] < segment_means[0] - 1e-2
