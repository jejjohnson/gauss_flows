"""Elementwise inverse Gaussian CDF (probit)."""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from flowjax.bijections import AbstractBijection
from jaxtyping import ArrayLike


class InverseGaussCDF(AbstractBijection):
    """Apply the inverse Gaussian CDF (probit function) element-wise.

    This maps uniform marginals to Gaussian marginals using the probit function.
    It is typically used after a CDF transform.

    Args:
        shape: Shape of the input.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x = jnp.asarray(x)
        x_clipped = jnp.clip(x, 1e-6, 1 - 1e-6)
        y = jax.scipy.special.ndtri(x_clipped)
        log_det = jnp.sum(-jstats.norm.logpdf(y))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        y = jnp.asarray(y)
        x = jax.scipy.special.ndtr(y)
        log_det = jnp.sum(jstats.norm.logpdf(y))
        return x, log_det


__all__ = ["InverseGaussCDF"]
