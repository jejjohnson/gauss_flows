"""Convolutional transforms for Gaussianization flows.

These bijections implement invertible convolution operations used in
image-based normalizing flows (e.g., Glow-style architectures).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array, lax
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


class OrthogonalConvExponential(AbstractBijection):
    """Orthogonal convolutional exponential bijection.

    Approximates the matrix exponential of a convolution operator using a
    truncated power series. The kernel is spectral-normalized to keep the
    operator norm ≤ 1, which stabilizes the series and preserves
    invertibility via kernel negation.

    Shape:
        Input/output: (H, W, C) single event (no batch). log_det is scalar.

    Example:
        >>> import jax.random as jr
        >>> key = jr.key(0)
        >>> transform = OrthogonalConvExponential(key, shape=(8, 8, 3))
        >>> x = jr.normal(key, (8, 8, 3))
        >>> y, log_det = transform.transform_and_log_det(x)
        >>> x_rec, log_det_inv = transform.inverse_and_log_det(y)
        >>> jnp.allclose(x, x_rec, atol=1e-4)
        DeviceArray(True, dtype=bool)
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    kernel: Array  # (k_h, k_w, C, C) in HWIO format
    n_terms: int
    n_power_iterations: int

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, int, int],
        *,
        kernel_size: int = 3,
        n_terms: int = 8,
        n_power_iterations: int = 2,
    ):
        import jax.random as jr

        if len(shape) != 3:
            raise ValueError("OrthogonalConvExponential expects shape (H, W, C).")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to have a central pixel.")
        if n_terms < 1:
            raise ValueError("n_terms must be positive.")
        if n_power_iterations < 1:
            raise ValueError("n_power_iterations must be positive.")

        self.shape = shape
        c = shape[-1]
        self.n_terms = int(n_terms)
        self.n_power_iterations = int(n_power_iterations)
        self.kernel = jr.normal(key, (kernel_size, kernel_size, c, c)) * 0.05

    def _conv2d(self, x: Array, kernel: Array) -> Array:
        # x: (H, W, C) -> x_b: (1, H, W, C)
        x_b = x[None, ...]
        y = lax.conv_general_dilated(
            x_b,
            kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        # y: (1, H, W, C) -> (H, W, C)
        return y[0]

    def _conv2d_transpose(self, x: Array, kernel: Array) -> Array:
        # x: (H, W, C) -> x_b: (1, H, W, C)
        x_b = x[None, ...]
        y = lax.conv_transpose(
            lhs=x_b,
            rhs=kernel,
            strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        # y: (1, H, W, C) -> (H, W, C)
        return y[0]

    def _spectral_normalize(self, kernel: Array) -> Array:
        h, w, c = self.shape
        u0 = jnp.ones((h, w, c)) / jnp.sqrt(float(h * w * c))

        def body(_, state):
            u_vec, v_vec = state
            v_vec = self._conv2d_transpose(u_vec, kernel)
            v_vec = v_vec / (jnp.linalg.norm(v_vec) + 1e-6)
            u_vec = self._conv2d(v_vec, kernel)
            u_vec = u_vec / (jnp.linalg.norm(u_vec) + 1e-6)
            return u_vec, v_vec

        u, v = lax.fori_loop(0, self.n_power_iterations, body, (u0, u0))
        conv_v = self._conv2d(v, kernel)
        sigma = jnp.abs(jnp.vdot(u, conv_v))
        scale = jnp.maximum(sigma, 1.0)
        return kernel / (scale + 1e-6)

    def _convolution_exponential(self, x: Array, kernel: Array) -> Array:
        product = x
        result = x

        def body(carry, i):
            prod, acc = carry
            prod = self._conv2d(prod, kernel) / (i + 1)
            acc = acc + prod
            return (prod, acc), None

        (product, result), _ = lax.scan(
            body,
            (product, result),
            jnp.arange(self.n_terms),
        )
        return result

    def _log_det(self, kernel: Array) -> Array:
        h, w, _ = self.shape
        k_h, k_w, _, _ = kernel.shape
        center_h = k_h // 2
        center_w = k_w // 2
        center_slice = kernel[center_h, center_w, :, :]
        diag_trace = jnp.trace(center_slice)
        return jnp.asarray(h * w * diag_trace)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x_arr = jnp.asarray(x)
        kernel_sn = self._spectral_normalize(self.kernel)
        y = self._convolution_exponential(x_arr, kernel_sn)
        log_det = self._log_det(kernel_sn)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        y_arr = jnp.asarray(y)
        kernel_sn = self._spectral_normalize(self.kernel)
        x = self._convolution_exponential(y_arr, -kernel_sn)
        log_det = -self._log_det(kernel_sn)
        return x, log_det


__all__ = [
    "ActNorm",
    "HaarWavelet",
    "Invertible1x1Conv",
    "OrthogonalConvExponential",
    "Squeeze",
]
