"""Tests for NumPyro compatibility."""

import jax.numpy as jnp
import pytest

import gauss_flows


def test_flowdist_creates(key):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    flow_dist = gauss_flows.FlowDist(flow)
    assert flow_dist is not None


def test_flowdist_sample(key):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    flow_dist = gauss_flows.FlowDist(flow)
    samples = flow_dist.sample(key, (10,))
    assert samples.shape == (10, 2)
    assert jnp.all(jnp.isfinite(samples))


@pytest.mark.integration
def test_flowdist_log_prob(key, data_2d):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    flow_dist = gauss_flows.FlowDist(flow)
    log_probs = flow_dist.log_prob(data_2d)
    assert log_probs.shape == (100,)
    assert jnp.all(jnp.isfinite(log_probs))


def test_flowdist_event_shape(key):
    flow = gauss_flows.gaussianization_flow(key, n_dims=4, n_layers=2, n_components=4)
    flow_dist = gauss_flows.FlowDist(flow)
    assert flow_dist.event_shape == (4,)
