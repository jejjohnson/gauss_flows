"""Convolutional transforms for Gaussianization flows.

These bijections implement invertible convolution operations used in
image-based normalizing flows (e.g., Glow-style architectures).
"""

from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


class Invertible1x1Conv(AbstractBijection):
    """Invertible 1x1 convolution as used in Glow (https://arxiv.org/abs/1807.03039).

    Parameterizes an invertible linear mixing across channels/features using
    an LU decomposition to ensure invertibility and efficient log-det computation.
    The weight matrix is implicitly W = L @ U where L is lower triangular with
    unit diagonal and U is upper triangular with positive diagonal (stored as
    log_diag_u). Log-det is computed cheaply as sum(log_diag_u).

    Args:
        key: JAX random key.
        n_channels: Number of channels/features.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    lower_off: Array  # lower-triangular off-diagonal entries of L
    upper_off: Array  # upper-triangular off-diagonal entries of U
    log_diag_u: Array  # log of diagonal of U (ensures non-zero det)

    def __init__(self, key: PRNGKeyArray, n_channels: int):
        import jax.random as jr

        self.shape = (n_channels,)
        n = n_channels
        # Initialize near identity: small random off-diagonals, unit diagonal
        k1, k2 = jr.split(key)
        n_off = n * (n - 1) // 2
        self.lower_off = jr.normal(k1, (n_off,)) * 0.01
        self.upper_off = jr.normal(k2, (n_off,)) * 0.01
        self.log_diag_u = jnp.zeros(n)

    def _get_weight(self) -> Array:
        n = self.shape[0]
        idx_lower = jnp.tril_indices(n, k=-1)
        idx_upper = jnp.triu_indices(n, k=1)
        L = jnp.eye(n).at[idx_lower].set(self.lower_off)
        # Build U: zero off-diagonal, then set upper triangle and diagonal explicitly
        U = jnp.zeros((n, n)).at[idx_upper].set(self.upper_off)
        U = U.at[jnp.diag_indices(n)].set(jnp.exp(self.log_diag_u))
        return L @ U

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        W = self._get_weight()
        y = W @ jnp.asarray(x)
        log_det = jnp.sum(self.log_diag_u)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        W = self._get_weight()
        x = jnp.linalg.solve(W, jnp.asarray(y))
        log_det = -jnp.sum(self.log_diag_u)
        return x, log_det


class ActNorm(AbstractBijection):
    """Activation normalization (ActNorm).

    Performs per-channel normalization with learnable location and scale.
    For images, operates on the channel dimension.

    Args:
        shape: Shape of the input.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    log_scale: Array

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        n_dims = shape[-1]
        self.loc = jnp.zeros(n_dims)
        self.log_scale = jnp.zeros(n_dims)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        from jax.nn import softplus

        scale = softplus(self.log_scale) + 1e-5
        y = (jnp.asarray(x) - self.loc) / scale
        # Multiply by number of non-channel positions (spatial dims) so that
        # the log-det accounts for each broadcasted application of the scale.
        n_spatial = (
            int(jnp.prod(jnp.array(self.shape[:-1]))) if len(self.shape) > 1 else 1
        )
        log_det = -n_spatial * jnp.sum(jnp.log(scale))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        from jax.nn import softplus

        scale = softplus(self.log_scale) + 1e-5
        x = jnp.asarray(y) * scale + self.loc
        n_spatial = (
            int(jnp.prod(jnp.array(self.shape[:-1]))) if len(self.shape) > 1 else 1
        )
        log_det = n_spatial * jnp.sum(jnp.log(scale))
        return x, log_det


class HaarWavelet(AbstractBijection):
    """Haar wavelet transform bijection.

    Implements the 1D Haar wavelet transform as a bijection. This is used
    in multi-scale flow architectures to factorize spatial information.

    Args:
        shape: Shape of the input. The last dimension must be divisible by 2.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...]):
        if shape[-1] % 2 != 0:
            raise ValueError("Last dimension must be divisible by 2 for HaarWavelet.")
        self.shape = shape

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x = jnp.asarray(x)
        # Haar wavelet: split into even/odd, compute averages and differences
        even = x[..., ::2]
        odd = x[..., 1::2]
        avg = (even + odd) / 2.0
        diff = (even - odd) / 2.0
        y = jnp.concatenate([avg, diff], axis=-1)
        # Log det: both avg and diff scale by 1/2, giving -n*log(2) total.
        log_det = -x.shape[-1] * jnp.log(2.0)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        y = jnp.asarray(y)
        n = y.shape[-1] // 2
        avg = y[..., :n]
        diff = y[..., n:]
        even = avg + diff
        odd = avg - diff
        # Interleave even and odd
        x = jnp.stack([even, odd], axis=-1).reshape((*y.shape[:-1], y.shape[-1]))
        log_det = x.shape[-1] * jnp.log(2.0)
        return x, log_det


class Squeeze(AbstractBijection):
    """Squeeze operation for image-like inputs.

    Reshapes spatial dimensions into channels, halving the spatial resolution
    and quadrupling the channels. Commonly used in multi-scale flow architectures.

    Args:
        shape: Shape of the input (H, W, C) or (N, C).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    out_shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        if len(shape) == 1:
            # 1D case: split into two halves → (n//2, 2)
            n = shape[0]
            if n % 2 != 0:
                raise ValueError("1D squeeze requires even dimension size.")
            self.out_shape = (n // 2, 2)
        elif len(shape) == 2:
            h, c = shape
            if h % 2 != 0:
                raise ValueError("Height must be divisible by 2 for Squeeze.")
            self.out_shape = (h // 2, c * 2)
        else:
            self.out_shape = shape

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        # Volume-preserving: log det = 0
        return jnp.asarray(x).reshape(self.out_shape), jnp.zeros(())

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        return jnp.asarray(y).reshape(self.shape), jnp.zeros(())


__all__ = [
    "ActNorm",
    "HaarWavelet",
    "Invertible1x1Conv",
    "Squeeze",
]
