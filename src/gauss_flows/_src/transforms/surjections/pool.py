"""2D max-pooling surjection with a learned residual decoder."""

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


class SimpleMaxPoolSurjection2d(AbstractSurjection):
    """2D max-pooling surjection with a learned residual decoder.

    Inference-style surjection on a single image event ``x ∈ ℝ^{H × W × C}``.
    Each pool window of shape ``(pool_size, pool_size)`` collapses to its max,
    and the ``pool_area − 1`` non-max values are scored by a conditional
    decoder. With ``pool_area = pool_size²`` and
    ``N = (H/pool_size) · (W/pool_size) · C`` pooled positions:

        forward:  z[h, w, c]  = max(x[h·ps:(h+1)·ps, w·ps:(w+1)·ps, c])
                  residuals   = z − {non-max values per pool}    (≥ 0)
                  log_det     = log q(residuals | z) − N · log(pool_area)

        inverse:  k       ~ Uniform({0..pool_area−1})^N          (argmax slot)
                  resid   ~ q(· | z)                              (decoder)
                  x       = unpool(z, resid, k)

    The ``−N · log(pool_area)`` term is the entropy of the uniform prior on the
    argmax location ``k``; it is marginalised out of the log-det. The decoder
    contribution is exact under the residual parameterisation.

    Args:
        shape: Image event shape ``(H, W, C)``. Both spatial dims must be
            divisible by ``pool_size``.
        decoder: Object satisfying `ConditionalDistribution` —
            ``sample(key, *, condition=z)`` must produce an array of shape
            ``(H/ps, W/ps, C, pool_area − 1)`` and ``log_prob(value, *,
            condition=z)`` must return a scalar.
        pool_size: Spatial pool size (default ``2`` → 2×2 windows).

    Shape:
        - Input  ``x``:  ``(H, W, C)``
        - Output ``z``:  ``(H/pool_size, W/pool_size, C)``
        - ``log_det``:   scalar (shape ``()``)

    Examples:
        ``my_decoder`` must implement the `ConditionalDistribution`
        protocol (see ``_src/_protocols.py``); it maps a pooled image
        ``z`` of shape ``(4, 4, 3)`` to residuals of shape ``(4, 4, 3, 3)``:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from gauss_flows import SimpleMaxPoolSurjection2d

        surj = SimpleMaxPoolSurjection2d((8, 8, 3), my_decoder, pool_size=2)
        z, log_det = surj.forward_and_log_det(jnp.zeros((8, 8, 3)), jr.key(0))
        # z.shape == (4, 4, 3); residuals.shape == (4, 4, 3, 3) (pool_area-1).
        ```
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
        # x: (H, W, C) -> (H/ps, W/ps, C, pool_area)
        return rearrange(
            x,
            "(h ph) (w pw) c -> h w c (ph pw)",
            ph=self.pool_size,
            pw=self.pool_size,
        )

    def _unsqueeze(self, xs: Array) -> Array:
        # xs: (H/ps, W/ps, C, pool_area) -> (H, W, C)
        return rearrange(
            xs,
            "h w c (ph pw) -> (h ph) (w pw) c",
            ph=self.pool_size,
            pw=self.pool_size,
        )

    def _deconstruct_x(self, x: Array) -> tuple[Array, Array, Array]:
        """x → (z, x_residuals, k); shapes documented inline."""
        # xs: (H/ps, W/ps, C, pool_area)
        xs = self._squeeze(x)
        # z, k: (H/ps, W/ps, C); k is the argmax slot in [0, pool_area).
        z = xs.max(axis=-1)
        k = jnp.argmax(xs, axis=-1)
        # Static gather of the (pool_area - 1) non-argmax positions per pool —
        # boolean masking on a runtime mask isn't traceable under jit/vmap
        # (NonConcreteBooleanIndexError). gather_idx[..., i] = i if i < k else
        # i + 1 skips the argmax slot in O(1).
        idx = jnp.arange(self._pool_area - 1)
        gather_idx = jnp.where(idx < k[..., None], idx, idx + 1)
        # residuals: (H/ps, W/ps, C, pool_area - 1) — non-max values
        residuals = jnp.take_along_axis(xs, gather_idx, axis=-1)
        # x_residuals: same shape, encoded as (max - value) ≥ 0.
        x_residuals = z[..., None] - residuals
        return z, x_residuals, k

    def _construct_x(self, z: Array, x_residuals: Array, k: Array) -> Array:
        """(z, x_residuals, k) → x; the inverse of _deconstruct_x."""
        # ``k`` is marginalised out of the log-det via the ``-N·log(pool_area)``
        # entropy term; here it only decides which slot the pooled max ``z``
        # gets slotted into when we reassemble the pool-sized window.
        # others: (H/ps, W/ps, C, pool_area - 1) — the actual non-max values.
        others = z[..., None] - x_residuals
        # Flatten spatial+channel dims into a single batch for vmap.
        flat_z = rearrange(z, "h w c -> (h w c)")
        flat_k = rearrange(k, "h w c -> (h w c)")
        flat_others = rearrange(others, "h w c p -> (h w c) p")

        def _assemble(z_val: Array, k_val: Array, others_row: Array) -> Array:
            # Place ``others_row`` (length pool_area - 1) into a window of
            # length pool_area, with ``z_val`` slotted at position k_val.
            # combined: (pool_area,) where the last entry is z_val.
            combined = jnp.concatenate([others_row, jnp.array([z_val])])
            idx = jnp.arange(self._pool_area)
            shift = (idx > k_val).astype(idx.dtype)
            gather_idx = idx - shift
            gather_idx = jnp.where(idx == k_val, self._pool_area - 1, gather_idx)
            return combined[gather_idx]

        # xs_flat: ((H/ps)·(W/ps)·C, pool_area)
        xs_flat = jax.vmap(_assemble)(flat_z, flat_k, flat_others)
        # xs: (H/ps, W/ps, C, pool_area)
        xs = rearrange(
            xs_flat,
            "(h w c) p -> h w c p",
            h=z.shape[0],
            w=z.shape[1],
            c=z.shape[2],
        )
        # back to (H, W, C)
        return self._unsqueeze(xs)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Pool x to z; return ``(z, log q(residuals|z) - N·log(pool_area))``."""
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
        """Sample residuals + uniform argmax slot; reconstruct x."""
        del cond
        z_arr = jnp.asarray(z)
        key_k, key_res = jr.split(key)
        # k: (H/ps, W/ps, C) uniform over {0, …, pool_area - 1}.
        k = jr.randint(key_k, z_arr.shape, 0, self._pool_area)
        # x_residuals: (H/ps, W/ps, C, pool_area - 1) from the decoder.
        x_residuals = self.decoder.sample(key_res, condition=z_arr)
        x = self._construct_x(z_arr, x_residuals, k)
        log_det = self.decoder.log_prob(x_residuals, condition=z_arr) + self._ldj_k
        return x, log_det


__all__ = ["SimpleMaxPoolSurjection2d"]
