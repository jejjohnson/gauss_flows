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
    """Activation normalization for 1D inputs.

    Performs per-dimension affine transformation with learnable location and
    log-scale parameters. Similar to batch normalization but with learned
    parameters that are not data-dependent at inference time.

    Args:
        shape: Shape of the input (n_dims,).
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
