"""Scalar circular rational quadratic spline (Rezende et al. 2020).

Maps ``[-bound, bound] → [-bound, bound]`` with tied endpoint derivatives
so the induced density is C¹-continuous across the wrap join. Scalar
(``shape == ()``): compose inside ``Coupling`` / ``Vmap`` to apply to a
vector of periodic coordinates.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import paramax as px
from flowjax.bijections import AbstractBijection, RationalQuadraticSpline
from jax import Array
from paramax.utils import inv_softplus


class CircularRationalQuadraticSpline(AbstractBijection):
    """Scalar rational quadratic spline with tied endpoint derivatives.

    The spline maps ``[-bound, bound] → [-bound, bound]`` and is C¹-continuous
    when the interval is identified with the circle (``derivative(+bound) ==
    derivative(-bound)``). This makes the resulting density continuous across the
    angular wrap boundary, matching the construction of Rezende et al. 2020,
    *Normalizing Flows on Tori and Spheres*.

    Implementation: wraps :class:`flowjax.bijections.RationalQuadraticSpline` and
    replaces the ``derivatives`` Parameterize with a tied variant — one fewer
    free parameter than the unconstrained spline.

    Args:
        knots: Number of bins.
        bound: Half-width of the periodic interval. Defaults to ``π``.
        min_derivative: Minimum derivative at any knot. Defaults to ``1e-3``.
        min_width: Minimum bin width. Defaults to ``1e-3``.

    Shape:
        Scalar bijection: ``shape == ()``. Use inside a ``Coupling`` or ``Vmap``
        to apply it to a vector of periodic coordinates.

        - transform_and_log_det: ``()`` → ``()``, scalar log_det
        - inverse_and_log_det:   ``()`` → ``()``, scalar log_det

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import CircularRationalQuadraticSpline
        >>> t = CircularRationalQuadraticSpline(knots=8)  # bound = π
        >>> x = jnp.array(0.3)  # scalar angle inside [−π, π]
        >>> y, log_det = t.transform_and_log_det(x)
        >>> y.shape
        ()
    """

    shape: ClassVar[tuple[int, ...]] = ()
    cond_shape: ClassVar[None] = None
    bound: float
    _spline: RationalQuadraticSpline

    def __init__(
        self,
        knots: int,
        bound: float = float(jnp.pi),
        min_derivative: float = 1e-3,
        min_width: float = 1e-3,
    ):
        self.bound = float(bound)
        spline = RationalQuadraticSpline(
            knots=knots,
            interval=bound,
            min_derivative=min_derivative,
            min_width=min_width,
        )
        # K free derivatives (one per knot position on the circle, with d[0] tied
        # to d[K]). Resolved shape (K+2,) matches flowjax's allocation; the final
        # slot is unused by flowjax's indexing but kept for shape compatibility.
        base = jnp.full(knots, inv_softplus(1.0 - min_derivative))

        def _resolve(arr: Array) -> Array:
            # Use axis=-1 so this broadcasts cleanly when the underlying
            # parameter array is vmapped (e.g. inside a Coupling's per-dim
            # transformer reconstruction).
            d = jnn.softplus(arr) + min_derivative
            return jnp.concatenate([d, d[..., :1], d[..., :1]], axis=-1)

        self._spline = eqx.tree_at(
            lambda s: s.derivatives,
            spline,
            replace=px.Parameterize(_resolve, base),
        )

    def transform_and_log_det(self, x: Array, condition=None) -> tuple[Array, Array]:
        return self._spline.transform_and_log_det(x, condition)

    def inverse_and_log_det(self, y: Array, condition=None) -> tuple[Array, Array]:
        return self._spline.inverse_and_log_det(y, condition)


__all__ = ["CircularRationalQuadraticSpline"]
