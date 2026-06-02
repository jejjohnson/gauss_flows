"""Learnable circular shift with unit Jacobian."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array

from gauss_flows._src.transforms.bijections.periodic._utils import (
    _build_periodic_mask,
    _wrap_angles,
)


class PeriodicShift(AbstractBijection):
    """Learnable circular shift on selected periodic dimensions.

    Adds a learnable per-index offset to the chosen dimensions and re-wraps the
    result back into ``[−bound, bound]``. The map is a translation on the
    circle, so its Jacobian determinant is 1 and ``log_det = 0``. Non-periodic
    dimensions pass through unchanged.

    Args:
        ind: Indices of periodic dimensions to shift. Must be unique and lie in
            ``[0, n_dims)``.
        shape: Event shape ``(n_dims,)``. Must be 1-D.
        bound: Half-width of the periodic interval. Defaults to ``π``.
        shift_init: Initial shift. Either a scalar (broadcast to every index) or
            a 1-D array of length ``len(ind)``. Defaults to ``0.0``.

    Raises:
        ValueError: If ``shape`` is not 1-D, if any index is out of range or
            duplicated, or if ``shift_init`` has an incompatible shape.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)``, scalar log_det
        - inverse_and_log_det:   ``(n_dims,)`` → ``(n_dims,)``, scalar log_det

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import PeriodicShift
        >>> shift = PeriodicShift(ind=(0,), shape=(2,), shift_init=0.25)
        >>> y, log_det = shift.transform_and_log_det(jnp.array([1.0, 0.5]))
        >>> y.shape, float(log_det)
        ((2,), 0.0)
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    bound: float
    mask: Array  # boolean mask over the full event, used for re-wrapping only
    indices: Array  # integer indices in `ind` order, used for shift scatter
    shift: Array

    def __init__(
        self,
        ind: Iterable[int],
        shape: tuple[int, ...],
        bound: float = jnp.pi,
        shift_init: Array | float = 0.0,
    ):
        if len(shape) != 1:
            raise ValueError("PeriodicShift only supports 1D inputs.")
        n_dims = shape[0]
        ind = tuple(ind)
        if any(i < 0 or i >= n_dims for i in ind):
            raise ValueError("All indices in ind must be within [0, n_dims).")
        if len(set(ind)) != len(ind):
            raise ValueError("ind must not contain duplicate indices.")
        self.shape = shape
        self.bound = float(bound)
        self.mask = _build_periodic_mask(ind, n_dims)
        self.indices = jnp.asarray(ind, dtype=jnp.int32)
        shift_values = jnp.asarray(shift_init, dtype=jnp.float32)
        if shift_values.ndim == 0:
            shift_values = jnp.broadcast_to(shift_values, (len(ind),))
        elif shift_values.shape != (len(ind),):
            raise ValueError(
                "shift_init must be scalar or match the number of indices."
            )
        self.shift = shift_values

    def _apply(self, x: Array, sign: float) -> Array:
        # Scatter via integer indices so the shift values stay aligned with
        # `ind` order (boolean-mask scatter would silently sort by index).
        shift_vec = jnp.zeros_like(x).at[self.indices].set(sign * self.shift)
        return _wrap_angles(x + shift_vec, self.mask, self.bound)

    def transform_and_log_det(self, x: Array, condition=None) -> tuple[Array, Array]:
        return self._apply(jnp.asarray(x), 1.0), jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition=None) -> tuple[Array, Array]:
        return self._apply(jnp.asarray(y), -1.0), jnp.zeros(())


__all__ = ["PeriodicShift"]
