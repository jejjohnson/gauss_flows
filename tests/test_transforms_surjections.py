"""Tests for simple surjections and stochastic transforms."""

import math

import jax.numpy as jnp
import jax.random as jr

from gauss_flows import (
    SimpleAbsSurjection,
    SimpleMaxPoolSurjection2d,
    SimpleSortSurjection,
    StochasticPermutation,
)


class _ZeroDecoder:
    def sample(self, key, *, condition):
        return jnp.zeros((*condition.shape, 3))

    def log_prob(self, value, *, condition):
        del value, condition
        return jnp.zeros(())


class _ReplayDecoder:
    def __init__(self):
        self.residuals = None

    def sample(self, key, *, condition):
        del key, condition
        if self.residuals is None:
            raise ValueError("Residuals must be set before sampling.")
        return self.residuals

    def log_prob(self, value, *, condition):
        del value, condition
        return jnp.zeros(())


def _moment_checks(x, tol=0.05):
    flat = x.reshape(-1)
    assert jnp.allclose(jnp.mean(flat), 0.0, atol=tol)
    assert jnp.allclose(jnp.std(flat), 1.0, atol=tol)


def test_simple_abs_log_det_and_shape(key):
    surj = SimpleAbsSurjection((4,))
    x = jr.normal(key, (256, 4))
    z, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    x_back, log_det_inv = surj.inverse_and_log_det(z, jr.fold_in(key, 2))
    expected = -4 * math.log(2.0)
    assert z.shape == x.shape
    assert x_back.shape == x.shape
    assert jnp.allclose(log_det, expected)
    assert jnp.allclose(log_det_inv, expected)
    _moment_checks(x_back)


def test_simple_sort_log_det(key):
    surj = SimpleSortSurjection((6,), axis=0)
    x = jr.normal(key, (512, 6))
    z, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    x_back, log_det_inv = surj.inverse_and_log_det(z, jr.fold_in(key, 2))
    expected = -math.lgamma(6 + 1)
    assert jnp.allclose(log_det, expected)
    assert jnp.allclose(log_det_inv, expected)
    assert jnp.all(z[..., :-1] <= z[..., 1:])
    _moment_checks(x_back, tol=0.05)


def test_stochastic_permutation_roundtrip(key):
    surj = StochasticPermutation((5,), axis=0)
    x = jr.normal(key, (256, 5))
    z, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    x_back, log_det_inv = surj.inverse_and_log_det(z, jr.fold_in(key, 2))
    assert log_det.shape == (256,)
    assert log_det_inv.shape == (256,)
    assert jnp.allclose(log_det, 0.0)
    assert jnp.allclose(log_det_inv, 0.0)
    _moment_checks(x_back, tol=0.06)


def test_simple_maxpool_log_det_and_shape(key):
    decoder = _ZeroDecoder()
    surj = SimpleMaxPoolSurjection2d(decoder)
    x = jr.normal(key, (4, 4, 1))
    z, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    expected = -4 * math.log(4.0)
    assert z.shape == (2, 2, 1)
    assert jnp.allclose(log_det, expected)


def test_simple_maxpool_deconstruct_construct_known_inputs(key):
    decoder = _ZeroDecoder()
    surj = SimpleMaxPoolSurjection2d(decoder)
    x = jr.uniform(key, (4, 4, 1))
    z, residuals, k = surj._deconstruct_x(x)
    x_rec = surj._construct_x(z, residuals, k)
    assert x_rec.shape == x.shape
    assert jnp.allclose(x_rec, x)


def test_simple_maxpool_distribution_roundtrip(key):
    decoder = _ReplayDecoder()
    surj = SimpleMaxPoolSurjection2d(decoder)
    x = jr.normal(key, (64, 4, 4, 1))
    z, residuals, _ = surj._deconstruct_x(x)
    decoder.residuals = residuals
    x_back, _log_det = surj.inverse_and_log_det(z, jr.fold_in(key, 1))
    assert x_back.shape == x.shape
    _moment_checks(x_back, tol=0.1)
