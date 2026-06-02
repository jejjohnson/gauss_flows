"""Sort along one axis; invert via independent per-slice permutations."""

from __future__ import annotations

import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms.base import AbstractSurjection


class SimpleSortSurjection(AbstractSurjection):
    """Sort along one axis; invert via independent per-slice permutations.

    Inference-style surjection. The forward map sorts each 1D slice along
    ``axis`` independently — ``jnp.sort(x, axis=axis)`` is per-slice — so the
    inverse draws an independent uniform permutation σ_i ∈ S_{shape[axis]} for
    each of the ``n_slices = prod(shape) // shape[axis]`` slices and applies it
    along ``axis``:

        forward:  z = sort(x, axis)                  (deterministic, per-slice)
        inverse:  σ_i ~ Uniform(S_n)  for i ∈ {0, …, n_slices − 1}
                  x_i = z_i ∘ σ_i

    The log-det contribution is the joint entropy of the n_slices independent
    uniform permutation priors:

        log_det = −n_slices · log(shape[axis]!)

    Args:
        shape: Event shape of a single input (no leading batch).
        axis: Axis to sort along (default ``0``). Wrapped modulo ``len(shape)``.

    Shape:
        - Input ``x``:   ``shape``
        - Output ``z``:  ``shape``    (sorted along ``axis``)
        - ``log_det``:   scalar (shape ``()``)

    Examples:
        1D event:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import SimpleSortSurjection
        >>>
        >>> surj = SimpleSortSurjection((4,))
        >>> z, _ = surj.forward_and_log_det(
        ...     jnp.array([3.0, 1.0, 4.0, 1.5]), jr.key(0)
        ... )
        >>> # z == [1., 1.5, 3., 4.]; log_det == -log(4!)

        2D event sorted along axis=0 (each column is its own slice):

        >>> surj2 = SimpleSortSurjection((4, 3), axis=0)
        >>> # n_slices = 3 (the W axis); log_det == -3 * log(4!)
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
        # Build einops patterns that move ``axis`` to the front and collapse the
        # remaining dims into a single ``n_slices`` axis. Stored at init time
        # since the shape is static. Example: shape=(4, 3), axis=0 →
        # pack = "d0 d1 -> d0 (d1)"; unpack = "d0 (d1) -> d0 d1".
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
        """Sort each slice along ``axis``; return ``(z, -n_slices · log(D!))``."""
        del key, cond
        # x: shape -> z: shape (sorted along ``axis``).
        z = jnp.sort(jnp.asarray(x), axis=self.axis)
        return z, jnp.asarray(self._log_det)

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Apply an independent uniform permutation to each slice along ``axis``."""
        del cond
        z_arr = jnp.asarray(z)
        if len(self.shape) == 1:
            # 1D fast path — single slice, single permutation.
            perm = jr.permutation(key, self._axis_size)
            x = z_arr[perm]
        else:
            # z_arr: shape -> z_flat: (axis_size, n_slices).
            z_flat = rearrange(z_arr, self._pack_pattern)
            keys = jr.split(key, self._n_slices)

            def _permute_slice(k: PRNGKeyArray, col: Array) -> Array:
                # col: (axis_size,); out: (axis_size,)
                perm = jr.permutation(k, self._axis_size)
                return col[perm]

            # vmap over n_slices: x_flat has same shape as z_flat.
            x_flat = jax.vmap(_permute_slice, in_axes=(0, 1), out_axes=1)(keys, z_flat)
            # x_flat: (axis_size, n_slices) -> x: shape.
            x = rearrange(x_flat, self._unpack_pattern, **self._unpack_sizes())
        return x, jnp.asarray(self._log_det)


__all__ = ["SimpleSortSurjection"]
