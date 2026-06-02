"""Invertible 1x1 convolution (Glow-style, LU-parameterised)."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


class Invertible1x1Conv(AbstractBijection):
    """Invertible 1×1 convolution with LU-parameterised weight (Glow).

    Mixes channels/features with an invertible linear map ``y = W·x``,
    where ``W = L·U`` is stored in LU form: ``L`` is lower-triangular with
    unit diagonal and ``U`` is upper-triangular with strictly positive
    diagonal (parameterised as ``exp(log_diag_u)``). This guarantees
    invertibility and reduces the log-determinant to the cheap
    ``∑ log_diag_u``, avoiding an explicit Jacobian. Introduced as the 1×1
    convolution of Glow (Kingma & Dhariwal 2018,
    https://arxiv.org/abs/1807.03039).

    Operates on a single ``(n_channels,)`` event (the channel/feature
    vector). Callers vmap over any spatial or batch axes.

    Args:
        key: PRNG key for near-identity initialisation (small off-diagonals).
        n_channels: Number of channels / feature dimensions.

    Shape:
        - transform_and_log_det: ``(n_channels,)`` → ``(n_channels,)``, scalar log_det
        - inverse_and_log_det:   ``(n_channels,)`` → ``(n_channels,)``, scalar log_det

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import Invertible1x1Conv
        >>> conv = Invertible1x1Conv(jr.key(0), n_channels=4)
        >>> x = jr.normal(jr.key(1), (4,))
        >>> y, log_det = conv.transform_and_log_det(x)
        >>> y.shape
        (4,)
        >>> x_rec, log_det_inv = conv.inverse_and_log_det(y)
        >>> bool(jnp.allclose(x, x_rec, atol=1e-5))
        True
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
