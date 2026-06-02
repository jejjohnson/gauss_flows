"""Volume-preserving squeeze / reshape bijection."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jaxtyping import ArrayLike


class Squeeze(AbstractBijection):
    """Volume-preserving squeeze that trades spatial extent for channels.

    Reshapes a single event so that resolution along the leading axis is halved
    and the trailing (channel) axis is correspondingly enlarged, exposing more
    structure to subsequent channel-wise layers. The map is a pure reshape, so
    it is exactly invertible and volume-preserving (``log_det = 0``). Commonly
    used to build multi-scale normalizing-flow architectures (Dinh et al. 2017).

    Three regimes by rank of ``shape``:

    - 1-D ``(n,)`` with ``n`` even → ``(n // 2, 2)``: split into adjacent pairs.
    - 2-D ``(H, C)`` with ``H`` even → ``(H // 2, 2·C)``: fold pairs of rows
      into the channel axis.
    - Higher rank: pass through unchanged (identity reshape).

    Args:
        shape: Event shape ``(n,)`` or ``(H, C)``; the leading axis must be even
            in those two cases. Higher-rank shapes are accepted but unchanged.

    Raises:
        ValueError: If the leading axis of a 1-D or 2-D ``shape`` is not even.

    Shape:
        - transform_and_log_det: ``shape`` → ``out_shape``, scalar log_det (0)
        - inverse_and_log_det:   ``out_shape`` → ``shape``, scalar log_det (0)

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import Squeeze
        >>> layer = Squeeze(shape=(4, 3))
        >>> y, log_det = layer.transform_and_log_det(jnp.ones((4, 3)))
        >>> y.shape, float(log_det)
        ((2, 6), 0.0)
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


__all__ = ["Squeeze"]
