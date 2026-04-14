"""Stochastic transforms (both directions stochastic)."""

from __future__ import annotations

import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms.base import AbstractStochastic


class StochasticPermutation(AbstractStochastic):
    """Uniform random permutation in both directions (zero log-det)."""

    shape: tuple[int, ...]
    axis: int
    cond_shape: ClassVar[None] = None
    _axis_size: int

    def __init__(self, shape: tuple[int, ...], axis: int = 0):
        if not shape:
            raise ValueError("Shape must be non-empty for StochasticPermutation.")
        self.shape = shape
        self.axis = axis % len(shape)
        self._axis_size = shape[self.axis]

    def _sample_shape(self, arr: Array) -> tuple[int, ...]:
        sample_ndim = arr.ndim - len(self.shape)
        if sample_ndim < 0:
            raise ValueError("Input has fewer dimensions than the configured shape.")
        return arr.shape[:sample_ndim]

    def _flatten_batch(self, arr: Array) -> tuple[Array, tuple[int, ...]]:
        sample_shape = self._sample_shape(arr)
        batch = int(math.prod(sample_shape)) if sample_shape else 1
        flat = arr.reshape((batch, *self.shape))
        return flat, sample_shape

    def _permute_single(self, key: PRNGKeyArray, values: Array) -> Array:
        perm = jr.permutation(key, self._axis_size)
        reshape_dims = (
            (1,) * self.axis + (self._axis_size,) + (1,) * (values.ndim - self.axis - 1)
        )
        perm_expanded = perm.reshape(reshape_dims)
        return jnp.take_along_axis(values, perm_expanded, axis=self.axis)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del cond
        x_arr = jnp.asarray(x)
        flat_x, sample_shape = self._flatten_batch(x_arr)
        keys = jr.split(key, flat_x.shape[0])
        z_flat = jax.vmap(self._permute_single)(keys, flat_x)
        z = z_flat.reshape(sample_shape + self.shape)
        log_det = jnp.zeros(sample_shape)
        return z, log_det

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del cond
        z_arr = jnp.asarray(z)
        flat_z, sample_shape = self._flatten_batch(z_arr)
        keys = jr.split(key, flat_z.shape[0])
        x_flat = jax.vmap(self._permute_single)(keys, flat_z)
        x = x_flat.reshape(sample_shape + self.shape)
        log_det = jnp.zeros(sample_shape)
        return x, log_det


__all__ = ["StochasticPermutation"]
