"""Shared helpers for periodic / circular bijections.

Used by :mod:`.shift`, :mod:`.wrap`, and
:mod:`...coupling.circular_spline` — kept in one place so the wrapping
primitives remain consistent across every transform that treats an input
dim as living on the circle.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _wrap_all(x: Array, bound: float) -> Array:
    """Wrap every dim into ``[-bound, bound]``. Fast path used when the
    'wrap-all-dims' case is statically known (e.g. CircularRQSplineCoupling)."""
    return ((x + bound) % (2 * bound)) - bound


def _wrap_angles(x: Array, mask: Array, bound: float) -> Array:
    """Wrap masked dimensions into ``[-bound, bound]``."""
    return jnp.where(mask, _wrap_all(x, bound), x)


def _build_periodic_mask(ind: tuple[int, ...], n_dims: int) -> Array:
    mask = jnp.zeros(n_dims, dtype=bool)
    if ind:
        mask = mask.at[jnp.array(ind)].set(True)
    return mask


__all__ = ["_build_periodic_mask", "_wrap_all", "_wrap_angles"]
