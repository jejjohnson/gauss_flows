"""Stochastic transforms (both directions stochastic)."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms.base import AbstractStochastic


class StochasticPermutation(AbstractStochastic):
    """Uniform random permutation along one axis (zero log-det).

    Operates on a single event with ``x.shape == shape``. Both directions draw
    an independent uniform permutation of ``shape[axis]`` and apply it along
    ``axis``. Log-det is zero: the forward and inverse sample from the same
    uniform distribution over the symmetric group.
    """

    shape: tuple[int, ...]
    axis: int
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...], axis: int = 0):
        if not shape:
            raise ValueError("Shape must be non-empty for StochasticPermutation.")
        self.shape = shape
        self.axis = axis % len(shape)

    def _permute(self, key: PRNGKeyArray, x: Array) -> Array:
        perm = jr.permutation(key, self.shape[self.axis])
        return jnp.take(x, perm, axis=self.axis)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del cond
        z = self._permute(key, jnp.asarray(x))
        return z, jnp.zeros(())

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del cond
        x = self._permute(key, jnp.asarray(z))
        return x, jnp.zeros(())


__all__ = ["StochasticPermutation"]
