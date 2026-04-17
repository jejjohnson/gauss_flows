"""Orthogonal 1x1 convolution for Gaussianization flows."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


class Orthogonal1x1Conv(AbstractBijection):
    """Orthogonal 1x1 convolution via Cayley parameterization.

    Parameterizes an orthogonal linear mixing across channels/features using
    the Cayley map of skew-symmetric matrices. The weight matrix Q is orthogonal
    and has log-det = 0, making it volume-preserving. This is useful for
    Gaussianization flows where orthogonal transformations are preferred.

    The Cayley map constructs an orthogonal matrix from a skew-symmetric matrix A:
    Q = (I - A)(I + A)^{-1}, where A^T = -A.

    Args:
        key: JAX random key (unused if fixed=True).
        n_channels: Number of channels/features.
        fixed: If True, initializes as identity and makes the transform non-trainable.
               If False (default), uses learnable Cayley parameterization.

    Shape:
        ``x``: ``(n_channels,)`` → ``y``: ``(n_channels,)``
        ``log_det``: ``()`` (always 0)

    Example:
        >>> import jax.random as jr
        >>> from gauss_flows import Orthogonal1x1Conv
        >>> key = jr.PRNGKey(0)
        >>> # Learnable orthogonal 1x1 conv
        >>> conv = Orthogonal1x1Conv(key, n_channels=4)
        >>> x = jr.normal(key, (4,))
        >>> y, log_det = conv.transform_and_log_det(x)
        >>> assert y.shape == (4,) and log_det == 0.0
        >>> # Fixed (identity) orthogonal 1x1 conv
        >>> conv_fixed = Orthogonal1x1Conv(key, n_channels=4, fixed=True)
        >>> y_fixed, _ = conv_fixed.transform_and_log_det(x)
        >>> assert jnp.allclose(x, y_fixed)
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    skew_params: Array | None  # Upper triangle of skew-symmetric matrix (None if fixed)
    fixed_matrix: Array | None  # Fixed orthogonal matrix (None if learnable)

    def __init__(self, key: PRNGKeyArray, n_channels: int, *, fixed: bool = False):
        self.shape = (n_channels,)
        if fixed:
            # Fixed mode: use identity matrix
            self.skew_params = None
            self.fixed_matrix = jnp.eye(n_channels)
        else:
            # Learnable mode: initialize with small random skew-symmetric parameters
            n_params = n_channels * (n_channels - 1) // 2
            self.skew_params = jr.normal(key, (n_params,)) * 0.01
            self.fixed_matrix = None

    def _get_rotation_matrix(self) -> Array:
        """Construct the orthogonal rotation matrix."""
        if self.fixed_matrix is not None:
            # Fixed mode: return the fixed matrix
            return self.fixed_matrix

        # Learnable mode: Cayley map
        n_dims = self.shape[0]
        A = jnp.zeros((n_dims, n_dims))
        idx = jnp.tril_indices(n_dims, k=-1)
        A = A.at[idx].set(self.skew_params)
        A = A - A.T  # make skew-symmetric: A^T = -A
        eye = jnp.eye(n_dims)
        # Cayley map: Q = (I - A)(I + A)^{-1}
        # Compute as: Q = solve(I + A, I - A)^T
        Q = jnp.linalg.solve(eye + A, eye - A).T
        return Q

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        Q = self._get_rotation_matrix()
        y = Q @ jnp.asarray(x)
        # Orthogonal matrices have det = ±1, so log|det| = 0
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        Q = self._get_rotation_matrix()
        # Inverse of orthogonal matrix is its transpose
        x = Q.T @ jnp.asarray(y)
        return x, jnp.zeros(())


__all__ = ["Orthogonal1x1Conv"]
