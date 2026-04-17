"""Invertible 1x1 convolution (Glow-style, LU-parameterised)."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
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


__all__ = ["Invertible1x1Conv"]
