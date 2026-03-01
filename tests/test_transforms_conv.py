"""Tests for conv transforms."""

import jax.numpy as jnp
import jax.random as jr

from gauss_flows import ActNorm, HaarWavelet, Invertible1x1Conv, Squeeze


def test_act_norm_forward_inverse(key):
    shape = (4,)
    act_norm = ActNorm(shape)
    x = jr.normal(key, shape)
    y, log_det = act_norm.transform_and_log_det(x)
    x_rec, log_det_inv = act_norm.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_act_norm_shape(key):
    act_norm = ActNorm((8,))
    x = jr.normal(key, (8,))
    y, log_det = act_norm.transform_and_log_det(x)
    assert y.shape == (8,)
    assert log_det.shape == ()


def test_invertible1x1conv_forward_inverse(key):
    n_channels = 4
    conv = Invertible1x1Conv(key, n_channels)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    x_rec, log_det_inv = conv.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_invertible1x1conv_shape(key):
    n_channels = 4
    conv = Invertible1x1Conv(key, n_channels)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    assert y.shape == (n_channels,)
    assert log_det.shape == ()


def test_haar_wavelet_forward_inverse(key):
    shape = (8,)
    haar = HaarWavelet(shape)
    x = jr.normal(key, shape)
    y, _log_det = haar.transform_and_log_det(x)
    x_rec, _log_det_inv = haar.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)


def test_squeeze_forward_inverse(key):
    shape = (4,)
    squeeze = Squeeze(shape)
    x = jr.normal(key, shape)
    y, log_det = squeeze.transform_and_log_det(x)
    x_rec, log_det_inv = squeeze.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)
    assert log_det.shape == ()
