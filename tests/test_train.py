"""Tests for training utilities."""

import pytest

import gauss_flows


@pytest.mark.slow
def test_fit_gaussianization_flow(key, data_2d):
    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    trained_flow, losses = gauss_flows.fit_gaussianization_flow(
        key,
        flow,
        data_2d,
        max_epochs=3,
        batch_size=32,
        show_progress=False,
    )
    assert "train" in losses
    assert "val" in losses
    assert len(losses["train"]) > 0
    assert trained_flow.shape == flow.shape


def test_fit_gaussianization_flow_returns_correct_type(key, data_2d):
    from flowjax.distributions import Transformed

    flow = gauss_flows.gaussianization_flow(key, n_dims=2, n_layers=2, n_components=4)
    trained_flow, _losses = gauss_flows.fit_gaussianization_flow(
        key,
        flow,
        data_2d,
        max_epochs=2,
        batch_size=32,
        show_progress=False,
    )
    assert isinstance(trained_flow, Transformed)
