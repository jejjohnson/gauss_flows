"""Marginal rational quadratic spline bijection.

Applies a per-dim RQ spline independently — the elementwise counterpart
of :class:`RQSplineCoupling`.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
from flowjax.bijections import AbstractBijection, RationalQuadraticSpline, Vmap
from jaxtyping import ArrayLike


class RQSplineMarginal(AbstractBijection):
    """Marginal Gaussianization via rational quadratic splines.

    Applies a rational quadratic spline independently to each dimension.

    Args:
        n_bins: Number of spline bins.
        shape: Shape of the input (n_dims,).
        interval: Interval for the spline. Defaults to 5.0.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    spline: AbstractBijection

    def __init__(self, n_bins: int, shape: tuple[int, ...], interval: float = 5.0):
        if len(shape) != 1:
            raise ValueError("RQSplineMarginal only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        spline = eqx.filter_vmap(
            lambda: RationalQuadraticSpline(knots=n_bins, interval=interval),
            axis_size=n_dims,
        )()
        self.spline = Vmap(spline, in_axes=eqx.if_array(0))

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        return self.spline.transform_and_log_det(x, condition)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        return self.spline.inverse_and_log_det(y, condition)


__all__ = ["RQSplineMarginal"]
