"""Simple SurVAE surjections without neural encoders/decoders."""

from __future__ import annotations

import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src._protocols import ConditionalDistribution
from gauss_flows._src.transforms.base import AbstractSurjection


class SimpleAbsSurjection(AbstractSurjection):
    """Elementwise absolute-value surjection with random sign inverse."""

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    _log_det: float

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        n_elems = math.prod(shape)
        self._log_det = -n_elems * math.log(2.0)

    def _sample_shape(self, arr: Array) -> tuple[int, ...]:
        sample_ndim = arr.ndim - len(self.shape)
        if sample_ndim < 0:
            raise ValueError("Input has fewer dimensions than the configured shape.")
        return arr.shape[:sample_ndim]

    def _log_det_broadcast(self, sample_shape: tuple[int, ...]) -> Array:
        return jnp.full(sample_shape, self._log_det)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del key, cond
        x_arr = jnp.asarray(x)
        z = jnp.abs(x_arr)
        log_det = self._log_det_broadcast(self._sample_shape(x_arr))
        return z, log_det

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del cond
        z_arr = jnp.asarray(z)
        signs = jr.bernoulli(key, 0.5, shape=z_arr.shape)
        signs = jnp.where(signs, 1.0, -1.0)
        x = signs * z_arr
        log_det = self._log_det_broadcast(self._sample_shape(z_arr))
        return x, log_det


class SimpleSortSurjection(AbstractSurjection):
    """Sort along one axis and invert via a random permutation."""

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    shape: tuple[int, ...]
    axis: int
    cond_shape: ClassVar[None] = None
    _axis_size: int
    _log_det: float

    def __init__(self, shape: tuple[int, ...], axis: int = 0):
        if not shape:
            raise ValueError("Shape must be non-empty for SimpleSortSurjection.")
        self.shape = shape
        self.axis = axis % len(shape)
        self._axis_size = shape[self.axis]
        self._log_det = -math.lgamma(self._axis_size + 1)

    def _sample_shape(self, arr: Array) -> tuple[int, ...]:
        sample_ndim = arr.ndim - len(self.shape)
        if sample_ndim < 0:
            raise ValueError("Input has fewer dimensions than the configured shape.")
        return arr.shape[:sample_ndim]

    def _log_det_broadcast(self, sample_shape: tuple[int, ...]) -> Array:
        return jnp.full(sample_shape, self._log_det)

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
        del key, cond
        x_arr = jnp.asarray(x)
        axis_in_x = x_arr.ndim - len(self.shape) + self.axis
        z = jnp.sort(x_arr, axis=axis_in_x)
        log_det = self._log_det_broadcast(self._sample_shape(x_arr))
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
        log_det = self._log_det_broadcast(sample_shape)
        return x, log_det


class SimpleMaxPoolSurjection2d(AbstractSurjection):
    """2D max-pooling surjection with learned residual decoder."""

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    decoder: ConditionalDistribution
    pool_size: int
    cond_shape: ClassVar[None] = None

    def __init__(self, decoder: ConditionalDistribution, pool_size: int = 2):
        if pool_size <= 0:
            raise ValueError("pool_size must be positive.")
        self.decoder = decoder
        self.pool_size = pool_size

    @property
    def _pool_area(self) -> int:
        return self.pool_size * self.pool_size

    def _validate_spatial(self, x: Array) -> tuple[int, int, int]:
        if x.ndim < 3:
            raise ValueError("SimpleMaxPoolSurjection2d expects (..., H, W, C) inputs.")
        h, w, c = x.shape[-3:]
        if h % self.pool_size != 0 or w % self.pool_size != 0:
            raise ValueError("Spatial dimensions must be divisible by pool_size.")
        return h, w, c

    def _squeeze(self, x: Array) -> Array:
        _h, _w, _c = self._validate_spatial(x)
        return rearrange(
            x,
            "... (h ph) (w pw) c -> ... h w c (ph pw)",
            ph=self.pool_size,
            pw=self.pool_size,
        )

    def _unsqueeze(self, xs: Array) -> Array:
        return rearrange(
            xs,
            "... h w c (ph pw) -> ... (h ph) (w pw) c",
            ph=self.pool_size,
            pw=self.pool_size,
        )

    def _deconstruct_x(self, x: Array) -> tuple[Array, Array, Array]:
        xs = self._squeeze(x)
        z = xs.max(axis=-1)
        k = jnp.argmax(xs, axis=-1)
        mask = jax.nn.one_hot(k, self._pool_area, dtype=bool)
        residuals = xs[~mask].reshape((*z.shape, self._pool_area - 1))
        x_residuals = z[..., None] - residuals
        return z, x_residuals, k

    def _construct_x(self, z: Array, x_residuals: Array, k: Array) -> Array:
        others = z[..., None] - x_residuals
        flat_z = z.reshape((-1,))
        flat_k = k.reshape((-1,))
        flat_others = others.reshape((-1, self._pool_area - 1))

        def _assemble(z_val: Array, k_val: Array, others_row: Array) -> Array:
            combined = jnp.concatenate([others_row, jnp.array([z_val])])
            idx = jnp.arange(self._pool_area)
            shift = (idx > k_val).astype(idx.dtype)
            gather_idx = idx - shift
            gather_idx = jnp.where(idx == k_val, self._pool_area - 1, gather_idx)
            return combined[gather_idx]

        xs_flat = jax.vmap(_assemble)(flat_z, flat_k, flat_others)
        xs = xs_flat.reshape((*z.shape, self._pool_area))
        return self._unsqueeze(xs)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del key, cond
        x_arr = jnp.asarray(x)
        z, x_residuals, _k = self._deconstruct_x(x_arr)
        num_positions = z.shape[-3] * z.shape[-2] * z.shape[-1]
        ldj_k = -math.log(self._pool_area) * num_positions
        log_det = self.decoder.log_prob(x_residuals, condition=z) + ldj_k
        return z, log_det

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del cond
        z_arr = jnp.asarray(z)
        key_k, key_res = jr.split(key)
        k = jr.randint(key_k, z_arr.shape, 0, self._pool_area)
        x_residuals = self.decoder.sample(key_res, condition=z_arr)
        x = self._construct_x(z_arr, x_residuals, k)
        num_positions = z_arr.shape[-3] * z_arr.shape[-2] * z_arr.shape[-1]
        ldj_k = -math.log(self._pool_area) * num_positions
        log_det = self.decoder.log_prob(x_residuals, condition=z_arr) + ldj_k
        return x, log_det


__all__ = [
    "SimpleAbsSurjection",
    "SimpleMaxPoolSurjection2d",
    "SimpleSortSurjection",
]
