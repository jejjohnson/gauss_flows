"""Marginal rational quadratic spline bijection.

Applies a per-dim RQ spline independently — the elementwise counterpart
of `RQSplineCoupling`.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
from flowjax.bijections import AbstractBijection, RationalQuadraticSpline, Vmap
from jaxtyping import ArrayLike


class RQSplineMarginal(AbstractBijection):
    """Marginal Gaussianization via rational-quadratic splines.

    Applies a monotonic rational-quadratic spline (Durkan et al. 2019,
    *Neural Spline Flows*) independently to each dimension — the elementwise
    counterpart of `RQSplineCoupling`. Each dim gets its own ``n_bins``
    knots over the symmetric interval ``[−interval, interval]``; outside that
    interval the map is the identity (linear tails). Inputs should lie within
    the interval for the spline to act non-trivially.

    Args:
        n_bins: Number of spline knots (bins) per dimension.
        shape: Event shape ``(n_dims,)``. Only 1-D events are supported.
        interval: Half-width of the symmetric spline interval
            ``[−interval, interval]``. Defaults to ``5.0``.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)``, scalar log_det
        - inverse_and_log_det:   ``(n_dims,)`` → ``(n_dims,)``, scalar log_det

    Examples:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import RQSplineMarginal
        >>> t = RQSplineMarginal(n_bins=8, shape=(3,), interval=5.0)
        >>> x = jnp.array([0.1, -0.5, 1.0])  # inside [−5, 5]
        >>> y, log_det = t.transform_and_log_det(x)
        >>> y.shape
        (3,)
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
