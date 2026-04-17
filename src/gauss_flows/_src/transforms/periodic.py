"""Periodic and circular transforms for angular data.

This module provides bijections suited to data living on the circle / torus:

- :class:`PeriodicWrap` — canonical projection onto ``[-bound, bound]`` (not a
  bijection on R, but useful as a leading-edge layer that canonicalises raw
  angles before downstream flow layers).
- :class:`PeriodicShift` — learnable circular shift with unit Jacobian.
- :class:`CircularRationalQuadraticSpline` — scalar RQS with tied endpoint
  derivatives (Rezende et al. 2020). C¹-continuous on the circle.
- :class:`CircularRQSplineCoupling` — coupling layer whose transformer is a
  circular RQS. **All transformed dims must be periodic** (compose with a
  regular ``RQSplineCoupling`` via ``Chain`` for mixed inputs).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import paramax as px
from flowjax.bijections import AbstractBijection, Coupling, RationalQuadraticSpline
from jax import Array
from jaxtyping import PRNGKeyArray
from paramax.utils import inv_softplus


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


class PeriodicShift(AbstractBijection):
    """Learnable circular shift on selected periodic dimensions.

    Adds a learnable per-index offset and re-wraps into ``[-bound, bound]``.
    Determinant is 1, so ``log_det = 0``.

    Shape:
        Input/output: ``(D,)`` (single event). ``log_det`` is a scalar ``Array``.

    Args:
        ind: Indices of periodic dimensions to shift.
        shape: Event shape, must be 1D.
        bound: Half-width of the periodic interval. Defaults to ``π``.
        shift_init: Optional initial shift. Either a scalar (broadcast to every
            index) or a 1D array of length ``len(ind)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import PeriodicShift
        >>> shift = PeriodicShift(ind=(0,), shape=(2,), shift_init=0.25)
        >>> y, log_det = shift.transform_and_log_det(jnp.array([1.0, 0.5]))
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

    Shape:
        Scalar bijection: ``shape == ()``. Use inside a ``Coupling`` or ``Vmap``
        to apply it to a vector of periodic coordinates.

    Args:
        knots: Number of bins.
        bound: Half-width of the periodic interval. Defaults to ``π``.
        min_derivative: Minimum derivative at any knot. Defaults to ``1e-3``.
        min_width: Minimum bin width. Defaults to ``1e-3``.
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


class CircularRQSplineCoupling(AbstractBijection):
    """Coupling layer whose transformer is a circular rational quadratic spline.

    Implements the construction of Rezende et al. 2020 for densities on the
    n-torus. **All input dims are assumed periodic** with period ``2·bound``.
    For mixed periodic/linear inputs, compose this with a regular
    :class:`RQSplineCoupling` via ``flowjax.bijections.Chain``.

    Two ingredients combine to make the resulting log-density continuous across
    the wrap join:

    1. The per-dim transformer is :class:`CircularRationalQuadraticSpline`
       (tied endpoint derivatives → C¹ across the join).
    2. The conditioner sees ``[sin(x_cond), cos(x_cond)]`` features rather than
       raw angles, so it cannot distinguish ``-π+ε`` from ``+π-ε``.

    Shape:
        Input/output: ``(n_dims,)`` (single event). ``log_det`` is a scalar
        ``Array``. The transformed slice is ``input[untransformed_dim:]``.

    Args:
        key: JAX random key for the conditioner MLP.
        shape: Input shape ``(n_dims,)``. Every dim is treated as periodic.
        n_bins: Number of spline bins. Defaults to 8.
        bound: Half-width of the circular interval. Defaults to ``π``.
        untransformed_dim: Number of leading dims left unchanged (used as
            conditioner inputs). Defaults to ``n_dims // 2``.
        nn_width: Hidden layer width for the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import CircularRQSplineCoupling
        >>> key = jr.key(0)
        >>> coupling = CircularRQSplineCoupling(key, shape=(4,), n_bins=8)
        >>> y, log_det = coupling.transform_and_log_det(jr.uniform(key, (4,)))

    Reference:
        Rezende et al. 2020, *Normalizing Flows on Tori and Spheres*.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    bound: float
    untransformed_dim: int
    _inner: AbstractBijection

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        n_bins: int = 8,
        bound: float = float(jnp.pi),
        untransformed_dim: int | None = None,
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        if len(shape) != 1:
            raise ValueError("CircularRQSplineCoupling only supports 1D inputs.")
        n_dims = shape[0]
        if n_dims < 2:
            raise ValueError("CircularRQSplineCoupling requires n_dims >= 2.")

        if untransformed_dim is None:
            untransformed_dim = n_dims // 2
        if not 0 < untransformed_dim < n_dims:
            raise ValueError(
                f"untransformed_dim must be in (0, n_dims); got {untransformed_dim}."
            )

        self.shape = shape
        self.bound = float(bound)
        self.untransformed_dim = int(untransformed_dim)

        spline = CircularRationalQuadraticSpline(knots=n_bins, bound=bound)
        # Use cond_dim = 2 * untransformed_dim to feed Fourier features
        # ([sin(x_cond), cos(x_cond)]) into the conditioner MLP. Setting the
        # inner Coupling's untransformed_dim=0 means we manage the split
        # ourselves and pass the periodic conditioning via `condition`.
        n_trans = n_dims - untransformed_dim
        self._inner = Coupling(
            key=key,
            transformer=spline,
            untransformed_dim=0,
            dim=n_trans,
            cond_dim=2 * untransformed_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=jnn.relu,
        )

    def _fourier_features(self, x_cond: Array) -> Array:
        # x_cond: (untransformed_dim,) -> features: (2 * untransformed_dim,)
        return jnp.concatenate([jnp.sin(x_cond), jnp.cos(x_cond)])

    def transform_and_log_det(self, x: Array, condition=None) -> tuple[Array, Array]:
        x_arr = jnp.asarray(x)
        x_wrapped = _wrap_all(x_arr, self.bound)
        x_cond = x_wrapped[: self.untransformed_dim]
        x_trans = x_wrapped[self.untransformed_dim :]
        y_trans, log_det = self._inner.transform_and_log_det(
            x_trans, self._fourier_features(x_cond)
        )
        y = jnp.concatenate([x_cond, y_trans])
        return y, log_det

    def inverse_and_log_det(self, y: Array, condition=None) -> tuple[Array, Array]:
        y_arr = jnp.asarray(y)
        y_wrapped = _wrap_all(y_arr, self.bound)
        y_cond = y_wrapped[: self.untransformed_dim]
        y_trans = y_wrapped[self.untransformed_dim :]
        x_trans, log_det = self._inner.inverse_and_log_det(
            y_trans, self._fourier_features(y_cond)
        )
        x = jnp.concatenate([y_cond, x_trans])
        return x, log_det


__all__ = [
    "CircularRQSplineCoupling",
    "CircularRationalQuadraticSpline",
    "PeriodicShift",
    "PeriodicWrap",
]
