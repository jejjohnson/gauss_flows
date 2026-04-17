"""Canonical projection onto [-bound, bound] for selected periodic dimensions."""

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


class PeriodicWrap(AbstractBijection):
    """Canonical projection of selected dimensions onto ``[-bound, bound]``.

    .. warning::
        This subclasses ``flowjax.bijections.AbstractBijection`` for API
        compatibility but is **not a true bijection on R** — it is a many-to-one
        canonical projection. ``x`` and ``x + 2·bound·k`` collapse to the same
        canonical value, and ``inverse_and_log_det`` returns that same wrapped
        value (not the original ``x``) with ``log_det = 0``. Composing this
        inside a ``flowjax.distributions.Transformed`` or ``SurVAEFlow`` will
        give **incorrect log-densities** if the upstream samples are not already
        in ``[-bound, bound]``, because the change-of-variables formula assumes
        invertibility. Use this layer only as a leading-edge canonicaliser for
        raw angles, never as an inner layer in a density model.

    Shape:
        Input/output: ``(D,)`` (single event). ``log_det`` is a scalar ``Array``.

    Args:
        ind: Indices of periodic dimensions to wrap.
        shape: Event shape, must be 1D.
        bound: Half-width of the periodic interval. Defaults to ``π``.

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import PeriodicWrap
        >>> wrap = PeriodicWrap(ind=(0,), shape=(2,))
        >>> y, log_det = wrap.transform_and_log_det(jnp.array([3.5, 0.2]))
        >>> # y[0] is in [-π, π); y[1] is unchanged.
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
        self.mask = _build_periodic_mask(ind, n_dims)

    def transform_and_log_det(self, x: Array, condition=None) -> tuple[Array, Array]:
        x_arr = jnp.asarray(x)
        y = _wrap_angles(x_arr, self.mask, self.bound)
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition=None) -> tuple[Array, Array]:
        y_arr = jnp.asarray(y)
        x = _wrap_angles(y_arr, self.mask, self.bound)
        return x, jnp.zeros(())


__all__ = ["PeriodicWrap"]
