"""Shared fixtures for gauss_flows tests."""

import jax.random as jr
import pytest


@pytest.fixture
def key():
    return jr.key(42)


@pytest.fixture
def key2():
    return jr.key(123)


@pytest.fixture
def data_2d(key):
    """Small 2D Gaussian dataset for testing."""
    return jr.normal(key, (100, 2))


@pytest.fixture
def data_4d(key):
    """Small 4D Gaussian dataset for testing."""
    return jr.normal(key, (100, 4))
