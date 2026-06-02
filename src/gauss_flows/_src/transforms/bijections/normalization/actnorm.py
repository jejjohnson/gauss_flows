"""Activation normalization (ActNorm).

``ActNorm`` handles image-like inputs ``(..., C)`` and broadcasts the scale
across any leading spatial axes. ``ActNorm1D`` is the 1D-only specialisation
used inside the coupling-flow stack where there are no spatial axes.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jax.nn import softplus
from jaxtyping import ArrayLike


class ActNorm(AbstractBijection):
    """Activation normalization (ActNorm) for image-shaped events.

    Per-channel affine normalization with learnable location ``loc`` and
    positive scale ``softplus(log_scale) + 1e-5``. The scale and location are
    indexed by the trailing (channel) axis and broadcast across any leading
    spatial axes, so the log-det of a single event is the per-channel log-scale
    sum multiplied by the number of spatial positions. Introduced as a
    replacement for batch normalization in Glow (Kingma & Dhariwal 2018).

    Args:
        shape: Event shape ``(..., C)``; the last axis is the channel axis.
            For example ``(H, W, C)`` for an image or ``(C,)`` for a plain
            channel vector.

    Shape:
        - transform_and_log_det: ``(..., C)`` → ``(..., C)``, scalar log_det
        - inverse_and_log_det:   ``(..., C)`` → ``(..., C)``, scalar log_det

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import ActNorm
        >>> layer = ActNorm(shape=(4, 4, 3))
        >>> y, log_det = layer.transform_and_log_det(jnp.ones((4, 4, 3)))
        >>> y.shape, log_det.shape
        ((4, 4, 3), ())
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
        scale = softplus(self.log_scale) + 1e-5
        x = jnp.asarray(y) * scale + self.loc
        n_spatial = (
            int(jnp.prod(jnp.array(self.shape[:-1]))) if len(self.shape) > 1 else 1
        )
        log_det = n_spatial * jnp.sum(jnp.log(scale))
        return x, log_det


class ActNorm1D(AbstractBijection):
    """Activation normalization (ActNorm) for 1-D events.

    The 1-D specialisation of :class:`ActNorm` used inside the coupling-flow
    stack where there are no spatial axes. Applies a per-dimension affine map
    ``y = (x − loc) / scale`` with positive ``scale = softplus(log_scale) +
    1e-5``. Unlike batch normalization, the parameters are learned and not
    data-dependent at inference time (Kingma & Dhariwal 2018).

    Args:
        shape: Event shape ``(n_dims,)``. Must be 1-D.

    Raises:
        ValueError: If ``shape`` is not 1-D.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)``, scalar log_det
        - inverse_and_log_det:   ``(n_dims,)`` → ``(n_dims,)``, scalar log_det

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import ActNorm1D
        >>> layer = ActNorm1D(shape=(3,))
        >>> y, log_det = layer.transform_and_log_det(jnp.ones((3,)))
        >>> y.shape, log_det.shape
        ((3,), ())
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    log_scale: Array

    def __init__(self, shape: tuple[int, ...]):
        if len(shape) != 1:
            raise ValueError("ActNorm1D only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        self.loc = jnp.zeros(n_dims)
        self.log_scale = jnp.zeros(n_dims)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        scale = softplus(self.log_scale) + 1e-5
        y = (jnp.asarray(x) - self.loc) / scale
        log_det = -jnp.sum(jnp.log(scale))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        scale = softplus(self.log_scale) + 1e-5
        x = jnp.asarray(y) * scale + self.loc
        log_det = jnp.sum(jnp.log(scale))
        return x, log_det


__all__ = ["ActNorm", "ActNorm1D"]
