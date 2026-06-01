"""Tests for flow constructors."""

import jax.numpy as jnp
import jax.random as jr
import pytest

import gauss_flows


def test_gaussianization_flow_creates_distribution(key):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    assert flow.shape == (2,)


@pytest.mark.integration
def test_gaussianization_flow_log_prob(key, data_2d):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    log_probs = flow.log_prob(data_2d)
    assert log_probs.shape == (100,)
    assert jnp.all(jnp.isfinite(log_probs))


@pytest.mark.integration
def test_gaussianization_flow_sample(key):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    samples = flow.sample(key, (10,))
    assert samples.shape == (10, 2)
    assert jnp.all(jnp.isfinite(samples))


def test_coupling_gaussianization_flow_creates_distribution(key):
    flow = gauss_flows.coupling_gaussianization_flow(
        key, n_dims=4, n_layers=2, n_bins=4
    )
    assert flow.shape == (4,)


@pytest.mark.integration
def test_coupling_gaussianization_flow_log_prob(key, data_4d):
    flow = gauss_flows.coupling_gaussianization_flow(
        key, n_dims=4, n_layers=2, n_bins=4
    )
    log_probs = flow.log_prob(data_4d)
    assert log_probs.shape == (100,)
    assert jnp.all(jnp.isfinite(log_probs))


def test_iterative_rbig_creates_distribution(key):
    flow = gauss_flows.iterative_rbig(key, n_dims=2, n_layers=2, n_components=4)
    assert flow.shape == (2,)


@pytest.mark.integration
def test_iterative_rbig_log_prob(key, data_2d):
    flow = gauss_flows.iterative_rbig(key, n_dims=2, n_layers=2, n_components=4)
    log_probs = flow.log_prob(data_2d)
    assert log_probs.shape == (100,)


@pytest.mark.integration
def test_gaussianization_flow_orthogonal_rotation(key):
    flow = gauss_flows.gaussianization_flow(
        key, n_dims=2, n_layers=2, n_components=4, rotation="orthogonal"
    )
    assert flow.shape == (2,)
    log_probs = flow.log_prob(jr.normal(key, (10, 2)))
    assert jnp.all(jnp.isfinite(log_probs))
