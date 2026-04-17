"""Tests for conv transforms."""

import jax.numpy as jnp
import jax.random as jr

from gauss_flows import (
    ActNorm,
    HaarWavelet,
    Invertible1x1Conv,
    Orthogonal1x1Conv,
    Squeeze,
)


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
    # Call the underlying (unwrapped) methods to bypass flowjax's shape-check,
    # since Squeeze changes the array shape (volume-preserving reshape).
    y, log_det = Squeeze.transform_and_log_det.__wrapped__(squeeze, x)
    assert y.shape == (2, 2)  # 1D squeeze: (n,) → (n//2, 2)
    x_rec, log_det_inv = Squeeze.inverse_and_log_det.__wrapped__(squeeze, y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)
    assert log_det.shape == ()


def test_orthogonal1x1conv_learnable_forward_inverse(key):
    """Test learnable orthogonal 1x1 conv forward and inverse."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels, fixed=False)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    x_rec, log_det_inv = conv.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_orthogonal1x1conv_learnable_shape(key):
    """Test learnable orthogonal 1x1 conv output shapes."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels, fixed=False)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    assert y.shape == (n_channels,)
    assert log_det.shape == ()


def test_orthogonal1x1conv_learnable_volume_preserving(key):
    """Test that learnable orthogonal 1x1 conv has log-det = 0."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels, fixed=False)
    x = jr.normal(key, (n_channels,))
    _, log_det_f = conv.transform_and_log_det(x)
    assert jnp.allclose(log_det_f, 0.0, atol=1e-5)


def test_orthogonal1x1conv_fixed_forward_inverse(key):
    """Test fixed orthogonal 1x1 conv forward and inverse."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels, fixed=True)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    x_rec, log_det_inv = conv.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_orthogonal1x1conv_fixed_is_identity(key):
    """Test that fixed mode starts as identity."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels, fixed=True)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    # Fixed mode initializes as identity
    assert jnp.allclose(x, y, atol=1e-6)
    assert jnp.allclose(log_det, 0.0, atol=1e-6)


def test_orthogonal1x1conv_fixed_volume_preserving(key):
    """Test that fixed orthogonal 1x1 conv has log-det = 0."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels, fixed=True)
    x = jr.normal(key, (n_channels,))
    _, log_det = conv.transform_and_log_det(x)
    assert jnp.allclose(log_det, 0.0, atol=1e-5)


def test_orthogonal1x1conv_orthogonality(key):
    """Test that the weight matrix is actually orthogonal (Q @ Q.T ≈ I)."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels, fixed=False)
    Q = conv._get_rotation_matrix()
    # Check Q @ Q.T = I
    I = jnp.eye(n_channels)
    assert jnp.allclose(Q @ Q.T, I, atol=1e-5)
    assert jnp.allclose(Q.T @ Q, I, atol=1e-5)


def test_orthogonal1x1conv_preserves_norm(key):
    """Test that orthogonal 1x1 conv preserves vector norm."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels, fixed=False)
    x = jr.normal(key, (n_channels,))
    y, _ = conv.transform_and_log_det(x)
    # Orthogonal transformation preserves norm
    assert jnp.allclose(jnp.linalg.norm(x), jnp.linalg.norm(y), atol=1e-5)
