"""Tests for the FlowGuide NumPyro variational guide."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

import gauss_flows
from gauss_flows import FlowGuide

# ---------------------------------------------------------------------------
# Helpers / simple models
# ---------------------------------------------------------------------------


def scalar_model(obs=None):
    """Model with a single scalar latent variable."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(mu, 1.0), obs=obs)


def multivariate_model(obs=None):
    """Model with a 3-D multivariate latent variable."""
    mu = numpyro.sample(
        "mu",
        dist.MultivariateNormal(jnp.zeros(3), jnp.eye(3)),
    )
    numpyro.sample(
        "obs",
        dist.MultivariateNormal(mu, jnp.eye(3)),
        obs=obs,
    )


def two_site_model(obs=None):
    """Model with two independent scalar latent variables."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=obs)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_flow_guide_creates():
    """FlowGuide can be instantiated with a simple model."""
    guide = FlowGuide(scalar_model)
    assert guide is not None


def test_flow_guide_creates_with_kwargs():
    """FlowGuide accepts flow_factory and flow_kwargs."""
    guide = FlowGuide(
        scalar_model,
        flow_factory=gauss_flows.gaussianization_flow,
        flow_kwargs={"n_layers": 2, "n_components": 4},
    )
    assert guide is not None


def test_flow_guide_custom_factory():
    """FlowGuide works with coupling_gaussianization_flow as factory."""
    guide = FlowGuide(
        scalar_model,
        flow_factory=gauss_flows.coupling_gaussianization_flow,
        flow_kwargs={"n_layers": 2},
    )
    assert guide is not None


# ---------------------------------------------------------------------------
# SVI smoke tests
# ---------------------------------------------------------------------------


def test_flow_guide_svi_runs(key):
    """SVI with FlowGuide completes without error (smoke test)."""
    obs = jnp.array([1.0])
    guide = FlowGuide(scalar_model, flow_kwargs={"n_layers": 2, "n_components": 4})
    svi = SVI(scalar_model, guide, Adam(1e-3), loss=Trace_ELBO())
    result = svi.run(key, num_steps=5, obs=obs, progress_bar=False)
    assert result.losses.shape == (5,)
    assert jnp.all(jnp.isfinite(result.losses))


def test_flow_guide_svi_multidim(key):
    """SVI with FlowGuide works for a multi-dimensional latent space."""
    obs = jnp.zeros(3)
    guide = FlowGuide(
        multivariate_model, flow_kwargs={"n_layers": 2, "n_components": 4}
    )
    svi = SVI(multivariate_model, guide, Adam(1e-3), loss=Trace_ELBO())
    result = svi.run(key, num_steps=5, obs=obs, progress_bar=False)
    assert result.losses.shape == (5,)
    assert jnp.all(jnp.isfinite(result.losses))


def test_flow_guide_svi_two_sites(key):
    """SVI with FlowGuide works when the model has two latent sites."""
    obs = jnp.array([1.0])
    guide = FlowGuide(two_site_model, flow_kwargs={"n_layers": 2, "n_components": 4})
    svi = SVI(two_site_model, guide, Adam(1e-3), loss=Trace_ELBO())
    result = svi.run(key, num_steps=5, obs=obs, progress_bar=False)
    assert result.losses.shape == (5,)
    assert jnp.all(jnp.isfinite(result.losses))


def test_flow_guide_svi_coupling_factory(key):
    """SVI with FlowGuide works when coupling_gaussianization_flow is used."""
    obs = jnp.array([1.0])
    guide = FlowGuide(
        scalar_model,
        flow_factory=gauss_flows.coupling_gaussianization_flow,
        flow_kwargs={"n_layers": 2},
    )
    svi = SVI(scalar_model, guide, Adam(1e-3), loss=Trace_ELBO())
    result = svi.run(key, num_steps=5, obs=obs, progress_bar=False)
    assert result.losses.shape == (5,)
    assert jnp.all(jnp.isfinite(result.losses))


# ---------------------------------------------------------------------------
# sample_posterior tests
# ---------------------------------------------------------------------------


def test_flow_guide_sample_posterior_shape(key, key2):
    """sample_posterior returns samples with the expected shape."""
    obs = jnp.array([1.0])
    guide = FlowGuide(scalar_model, flow_kwargs={"n_layers": 2, "n_components": 4})
    svi = SVI(scalar_model, guide, Adam(1e-3), loss=Trace_ELBO())
    result = svi.run(key, num_steps=5, obs=obs, progress_bar=False)

    posterior_samples = guide.sample_posterior(
        key2, result.params, obs=obs, sample_shape=(10,)
    )
    assert "mu" in posterior_samples
    assert posterior_samples["mu"].shape == (10,)
    assert jnp.all(jnp.isfinite(posterior_samples["mu"]))


def test_flow_guide_sample_posterior_multidim(key, key2):
    """sample_posterior returns correctly shaped samples for vector latents."""
    obs = jnp.zeros(3)
    guide = FlowGuide(
        multivariate_model, flow_kwargs={"n_layers": 2, "n_components": 4}
    )
    svi = SVI(multivariate_model, guide, Adam(1e-3), loss=Trace_ELBO())
    result = svi.run(key, num_steps=5, obs=obs, progress_bar=False)

    posterior_samples = guide.sample_posterior(
        key2, result.params, obs=obs, sample_shape=(10,)
    )
    assert "mu" in posterior_samples
    assert posterior_samples["mu"].shape == (10, 3)
    assert jnp.all(jnp.isfinite(posterior_samples["mu"]))
