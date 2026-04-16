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
        cond = jnp.asarray(condition)
        mean = self._mean(cond)
        scale = self._scale()
        eps = jr.normal(key, mean.shape)
        return mean + scale * eps

    def log_prob(self, value: Array, *, condition: Array) -> Array:
        val = jnp.asarray(value)
        cond = jnp.asarray(condition)
        mean = self._mean(cond)
        scale = self._scale()
        var = scale**2
        log_norm = -0.5 * jnp.log(2 * jnp.pi * var)
        quad = -0.5 * ((val - mean) ** 2) / var
        return jnp.sum(log_norm + quad)


def test_vae_forward_log_det_matches_expected(key):
    encoder = DiagGaussianConditional(jr.fold_in(key, 0), in_dim=2, out_dim=2)
    decoder = DiagGaussianConditional(jr.fold_in(key, 1), in_dim=2, out_dim=2)
    vae = VAE(encoder, decoder)
    x = jnp.array([0.3, -0.7])
    z, log_det = vae.forward_and_log_det(x, jr.fold_in(key, 2))
    expected = decoder.log_prob(x, condition=z) - encoder.log_prob(z, condition=x)
    assert z.shape == x.shape
    assert jnp.allclose(log_det, expected)


def test_vae_inverse_samples_and_zero_log_det(key):
    encoder = DiagGaussianConditional(jr.fold_in(key, 0), in_dim=2, out_dim=2)
    decoder = DiagGaussianConditional(jr.fold_in(key, 1), in_dim=2, out_dim=2)
    vae = VAE(encoder, decoder)
    z = jnp.array([1.0, -1.0])
    x, log_det = vae.inverse_and_log_det(z, jr.fold_in(key, 3))
    assert x.shape == z.shape
    assert log_det == pytest.approx(0.0)


def test_vae_svi_smoke_train_elbo_improves(key):
    latent_dim = 2
    data = jr.normal(jr.fold_in(key, 10), (256, latent_dim)) + jnp.array([1.0, -1.0])

    encoder = DiagGaussianConditional(
        jr.fold_in(key, 20), in_dim=latent_dim, out_dim=latent_dim
    )
    decoder = DiagGaussianConditional(
        jr.fold_in(key, 30), in_dim=latent_dim, out_dim=latent_dim
    )
    flow = SurVAEFlow(Normal(jnp.zeros(latent_dim)), [VAE(encoder, decoder)])

    def loss_fn(model, k):
        return -jnp.mean(model.log_prob(data, k))

    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(flow, eqx.is_inexact_array))
    train_keys = jr.split(jr.fold_in(key, 999), 500)

    losses = []
    for step_key in train_keys:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow, step_key)
        updates, opt_state = optim.update(grads, opt_state)
        flow = eqx.apply_updates(flow, updates)
        losses.append(loss)
    losses = jnp.stack(losses)
    baseline = -jnp.mean(flow.base_dist.log_prob(data))
    assert jnp.isfinite(losses).all()
    assert losses[-1] < baseline
    segment_means = losses.reshape(5, 100).mean(axis=1)
    assert jnp.all(segment_means[1:] <= segment_means[:-1] + 1e-4)
