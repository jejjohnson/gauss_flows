"""Periodic and circular transforms for angular data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection, Coupling, RationalQuadraticSpline
from flowjax.distributions import Normal, Transformed
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


def _wrap_angles(x: Array, mask: Array, bound: float) -> Array:
    """Wrap masked dimensions into [-bound, bound]."""
    wrapped = ((x + bound) % (2 * bound)) - bound
    return jnp.where(mask, wrapped, x)


class PeriodicWrap(AbstractBijection):
    """Wrap selected dimensions onto a bounded interval.

    Args:
        ind: Indices of periodic dimensions to wrap.
        shape: Event shape, must be 1D.
        bound: Half-width of the periodic interval. Defaults to ``jnp.pi``.

    Examples:
        Wrap a wind direction into ``[-pi, pi]``::

            wrap = PeriodicWrap(ind=(0,), shape=(2,))
            y, _ = wrap.transform_and_log_det(jnp.array([3.5, 0.2]))

        Wrapping is its own inverse on the circle::

            x = jnp.array([-4.0, 1.0])
            y, _ = wrap.transform_and_log_det(x)
            x_rec, _ = wrap.inverse_and_log_det(y)
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    bound: float
    mask: Array

    def __init__(
        self,
        ind: Iterable[int],
        shape: tuple[int, ...],
        bound: float = jnp.pi,
    ):
        if len(shape) != 1:
            raise ValueError("PeriodicWrap only supports 1D inputs.")
        n_dims = shape[0]
        ind = tuple(ind)
        if any(i < 0 or i >= n_dims for i in ind):
            raise ValueError("All indices in ind must be within [0, n_dims).")
        self.shape = shape
        self.bound = float(bound)
        mask = jnp.zeros(n_dims, dtype=bool)
        if ind:
            mask = mask.at[jnp.array(ind)].set(True)
        self.mask = mask

    def transform_and_log_det(
        self, x: ArrayLike, condition=None
    ) -> tuple[Array, float]:
        x_arr = jnp.asarray(x)
        y = _wrap_angles(x_arr, self.mask, self.bound)
        return y, 0.0

    def inverse_and_log_det(self, y: ArrayLike, condition=None) -> tuple[Array, float]:
        y_arr = jnp.asarray(y)
        x = _wrap_angles(y_arr, self.mask, self.bound)
        return x, 0.0


class PeriodicShift(AbstractBijection):
    """Learnable circular shift on selected periodic dimensions.

    Args:
        ind: Indices of periodic dimensions to shift.
        shape: Event shape, must be 1D.
        bound: Half-width of the periodic interval. Defaults to ``jnp.pi``.
        shift_init: Optional initial shift (scalar or per-index array).

    Examples:
        >>> shift = PeriodicShift(ind=(0,), shape=(2,), shift_init=0.25)
        >>> y, log_det = shift.transform_and_log_det(jnp.array([1.0, 0.5]))
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    bound: float
    mask: Array
    shift: Array

    def __init__(
        self,
        ind: Iterable[int],
        shape: tuple[int, ...],
        bound: float = jnp.pi,
        shift_init: ArrayLike | float = 0.0,
    ):
        if len(shape) != 1:
            raise ValueError("PeriodicShift only supports 1D inputs.")
        n_dims = shape[0]
        ind = tuple(ind)
        if any(i < 0 or i >= n_dims for i in ind):
            raise ValueError("All indices in ind must be within [0, n_dims).")
        self.shape = shape
        self.bound = float(bound)
        mask = jnp.zeros(n_dims, dtype=bool)
        if ind:
            mask = mask.at[jnp.array(ind)].set(True)
        self.mask = mask
        shift_values = jnp.asarray(shift_init)
        if shift_values.ndim == 0:
            shift_values = jnp.broadcast_to(shift_values, (len(ind),))
        elif shift_values.shape != (len(ind),):
            raise ValueError(
                "shift_init must be scalar or match the number of indices."
            )
        self.shift = shift_values

    def transform_and_log_det(
        self, x: ArrayLike, condition=None
    ) -> tuple[Array, float]:
        x_arr = jnp.asarray(x)
        shift_vec = jnp.zeros_like(x_arr)
        shift_vec = shift_vec.at[self.mask].set(self.shift)
        y = _wrap_angles(x_arr + shift_vec, self.mask, self.bound)
        return y, 0.0

    def inverse_and_log_det(self, y: ArrayLike, condition=None) -> tuple[Array, float]:
        y_arr = jnp.asarray(y)
        shift_vec = jnp.zeros_like(y_arr)
        shift_vec = shift_vec.at[self.mask].set(self.shift)
        x = _wrap_angles(y_arr - shift_vec, self.mask, self.bound)
        return x, 0.0


class CircularRQSplineCoupling(AbstractBijection):
    """Coupling layer with circular rational quadratic splines on periodic dims.

    Periodic dimensions are wrapped into ``[-bound, bound]`` before and after the
    spline, ensuring ``log_prob(x) == log_prob(x + 2π)`` on those coordinates.
    Non-periodic dimensions use standard linear tails with the same interval.

    Args:
        key: JAX random key.
        shape: Input shape (n_dims,).
        periodic_dims: Iterable of indices treated as angular coordinates.
        n_bins: Number of spline bins. Defaults to 8.
        bound: Half-width of the circular interval. Defaults to ``jnp.pi``.
        nn_width: Hidden layer width for the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    periodic_mask: Array
    bound: float
    _coupling: AbstractBijection

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        periodic_dims: Iterable[int],
        n_bins: int = 8,
        bound: float = jnp.pi,
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        if len(shape) != 1:
            raise ValueError("CircularRQSplineCoupling only supports 1D inputs.")
        n_dims = shape[0]
        periodic_dims = tuple(periodic_dims)
        if any(i < 0 or i >= n_dims for i in periodic_dims):
            raise ValueError("All periodic_dims must be within [0, n_dims).")

        self.shape = shape
        self.bound = float(bound)
        mask = jnp.zeros(n_dims, dtype=bool)
        if periodic_dims:
            mask = mask.at[jnp.array(periodic_dims)].set(True)
        self.periodic_mask = mask
        untransformed_dim = n_dims // 2

        spline = RationalQuadraticSpline(knots=n_bins, interval=bound)
        self._coupling = Coupling(
            key=key,
            transformer=spline,
            untransformed_dim=untransformed_dim,
            dim=n_dims,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=jnn.relu,
        )

    def transform_and_log_det(
        self, x: ArrayLike, condition=None
    ) -> tuple[Array, Array]:
        x_arr = jnp.asarray(x)
        x_wrapped = _wrap_angles(x_arr, self.periodic_mask, self.bound)
        y, log_det = self._coupling.transform_and_log_det(x_wrapped, condition)
        y = _wrap_angles(y, self.periodic_mask, self.bound)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None) -> tuple[Array, Array]:
        y_arr = jnp.asarray(y)
        y_wrapped = _wrap_angles(y_arr, self.periodic_mask, self.bound)
        x, log_det = self._coupling.inverse_and_log_det(y_wrapped, condition)
        x = _wrap_angles(x, self.periodic_mask, self.bound)
        return x, log_det

    def as_distribution(self, base_key: PRNGKeyArray | None = None) -> Transformed:
        """Utility to wrap the bijection as a FlowJax distribution."""
        if base_key is None:
            base_key = jr.key(0)
        base = Normal(jnp.zeros(self.shape[0]))
        return Transformed(base, self)


__all__ = [
    "CircularRQSplineCoupling",
    "PeriodicShift",
    "PeriodicWrap",
]
