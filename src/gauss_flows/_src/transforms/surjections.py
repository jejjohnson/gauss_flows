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
    """Elementwise absolute-value surjection with a random-sign inverse.

    Inference-style surjection (Nielsen et al. 2020 §3.1): the deterministic
    forward ``z = |x|`` is paired with a uniform Bernoulli sign inverse, and
    the log-det contribution ``-D · log 2`` is the entropy of the sign
    distribution.
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    @property
    def _log_det(self) -> float:
        return -math.prod(self.shape) * math.log(2.0)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del key, cond
        z = jnp.abs(jnp.asarray(x))
        return z, jnp.asarray(self._log_det)

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
        return x, jnp.asarray(self._log_det)


class SimpleSortSurjection(AbstractSurjection):
    """Sort along one axis; invert via independent per-slice permutations.

    ``jnp.sort(x, axis=axis)`` sorts each 1D slice along ``axis`` independently,
    so the inverse must draw an independent uniform permutation of
    ``shape[axis]`` for each such slice. With
    ``n_slices = prod(shape) // shape[axis]``, the log-det contribution is
    ``-n_slices · log(shape[axis]!)``.
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    shape: tuple[int, ...]
    axis: int
    cond_shape: ClassVar[None] = None
    _pack_pattern: str
    _unpack_pattern: str

    def __init__(self, shape: tuple[int, ...], axis: int = 0):
        if not shape:
            raise ValueError("Shape must be non-empty for SimpleSortSurjection.")
        self.shape = shape
        self.axis = axis % len(shape)
        names = [f"d{i}" for i in range(len(shape))]
        target = names[self.axis]
        others = [n for i, n in enumerate(names) if i != self.axis]
        all_dims = " ".join(names)
        if others:
            other_dims = " ".join(others)
            self._pack_pattern = f"{all_dims} -> {target} ({other_dims})"
            self._unpack_pattern = f"{target} ({other_dims}) -> {all_dims}"
        else:
            self._pack_pattern = f"{all_dims} -> {target}"
            self._unpack_pattern = f"{target} -> {all_dims}"

    @property
    def _axis_size(self) -> int:
        return self.shape[self.axis]

    @property
    def _n_slices(self) -> int:
        return math.prod(self.shape) // self._axis_size

    @property
    def _log_det(self) -> float:
        return -self._n_slices * math.lgamma(self._axis_size + 1)

    def _unpack_sizes(self) -> dict[str, int]:
        return {f"d{i}": s for i, s in enumerate(self.shape) if i != self.axis}

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del key, cond
        z = jnp.sort(jnp.asarray(x), axis=self.axis)
        return z, jnp.asarray(self._log_det)

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del cond
        z_arr = jnp.asarray(z)
        if len(self.shape) == 1:
            perm = jr.permutation(key, self._axis_size)
            x = z_arr[perm]
        else:
            z_flat = rearrange(z_arr, self._pack_pattern)
            keys = jr.split(key, self._n_slices)

            def _permute_slice(k: PRNGKeyArray, col: Array) -> Array:
                perm = jr.permutation(k, self._axis_size)
                return col[perm]

            x_flat = jax.vmap(_permute_slice, in_axes=(0, 1), out_axes=1)(keys, z_flat)
            x = rearrange(x_flat, self._unpack_pattern, **self._unpack_sizes())
        return x, jnp.asarray(self._log_det)


class SimpleMaxPoolSurjection2d(AbstractSurjection):
    """2D max-pooling surjection with a learned residual decoder.

    Operates on a single image event ``x`` with ``shape = (H, W, C)``. Forward
    returns the pooled maxima ``z`` with spatial dims ``(H/pool_size, W/pool_size)``
    and accumulates ``decoder.log_prob(residuals | z) - N · log(pool_area)``
    where ``N`` is the number of pooled positions. The inverse draws a fresh
    ``k`` (which pool location held the max) uniformly and fresh residuals from
    the decoder.
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    shape: tuple[int, ...]
    decoder: ConditionalDistribution
    pool_size: int
    cond_shape: ClassVar[None] = None

    def __init__(
        self,
        shape: tuple[int, ...],
        decoder: ConditionalDistribution,
        pool_size: int = 2,
    ):
        if pool_size <= 0:
            raise ValueError("pool_size must be positive.")
        if len(shape) != 3:
            raise ValueError(
                "SimpleMaxPoolSurjection2d requires a (H, W, C) event shape."
            )
        h, w, _ = shape
        if h % pool_size != 0 or w % pool_size != 0:
            raise ValueError("Spatial dimensions must be divisible by pool_size.")
        self.shape = shape
        self.decoder = decoder
        self.pool_size = pool_size

    @property
    def _pool_area(self) -> int:
        return self.pool_size * self.pool_size

    @property
    def _num_positions(self) -> int:
        h, w, c = self.shape
        return (h // self.pool_size) * (w // self.pool_size) * c

    @property
    def _ldj_k(self) -> float:
        return -math.log(self._pool_area) * self._num_positions

    def _squeeze(self, x: Array) -> Array:
        return rearrange(
            x,
            "(h ph) (w pw) c -> h w c (ph pw)",
            ph=self.pool_size,
            pw=self.pool_size,
        )

    def _unsqueeze(self, xs: Array) -> Array:
        return rearrange(
            xs,
            "h w c (ph pw) -> (h ph) (w pw) c",
            ph=self.pool_size,
            pw=self.pool_size,
        )

    def _deconstruct_x(self, x: Array) -> tuple[Array, Array, Array]:
        xs = self._squeeze(x)
        z = xs.max(axis=-1)
        k = jnp.argmax(xs, axis=-1)
        # Static gather of the (pool_area - 1) non-argmax positions per pool —
        # boolean masking on a runtime mask isn't traceable under jit/vmap
        # (NonConcreteBooleanIndexError).
        idx = jnp.arange(self._pool_area - 1)
        gather_idx = jnp.where(idx < k[..., None], idx, idx + 1)
        residuals = jnp.take_along_axis(xs, gather_idx, axis=-1)
        x_residuals = z[..., None] - residuals
        return z, x_residuals, k

    def _construct_x(self, z: Array, x_residuals: Array, k: Array) -> Array:
        # ``k`` is marginalised out of the log-det via the ``-N·log(pool_area)``
        # entropy term; here it only decides which slot the pooled max ``z``
        # gets slotted into when we reassemble the pool-sized window.
        others = z[..., None] - x_residuals
        flat_z = rearrange(z, "h w c -> (h w c)")
        flat_k = rearrange(k, "h w c -> (h w c)")
        flat_others = rearrange(others, "h w c p -> (h w c) p")

        def _assemble(z_val: Array, k_val: Array, others_row: Array) -> Array:
            combined = jnp.concatenate([others_row, jnp.array([z_val])])
            idx = jnp.arange(self._pool_area)
            shift = (idx > k_val).astype(idx.dtype)
            gather_idx = idx - shift
            gather_idx = jnp.where(idx == k_val, self._pool_area - 1, gather_idx)
            return combined[gather_idx]

        xs_flat = jax.vmap(_assemble)(flat_z, flat_k, flat_others)
        xs = rearrange(
            xs_flat,
            "(h w c) p -> h w c p",
            h=z.shape[0],
            w=z.shape[1],
            c=z.shape[2],
        )
        return self._unsqueeze(xs)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del key, cond
        z, x_residuals, _k = self._deconstruct_x(jnp.asarray(x))
        log_det = self.decoder.log_prob(x_residuals, condition=z) + self._ldj_k
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
        log_det = self.decoder.log_prob(x_residuals, condition=z_arr) + self._ldj_k
        return x, log_det


__all__ = [
    "SimpleAbsSurjection",
    "SimpleMaxPoolSurjection2d",
    "SimpleSortSurjection",
]
