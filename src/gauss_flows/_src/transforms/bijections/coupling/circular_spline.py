"""Circular rational quadratic spline coupling layer (Rezende et al. 2020).

Coupling on the n-torus: all input dims are periodic and the conditioner
sees ``[sin(x), cos(x)]`` features so it cannot distinguish ``-π+ε`` from
``+π-ε``.
"""

from __future__ import annotations

from typing import ClassVar

import jax.nn as jnn
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection, Coupling
from jax import Array
from jaxtyping import PRNGKeyArray

from gauss_flows._src.transforms.bijections.elementwise.circular_spline import (
    CircularRationalQuadraticSpline,
)
from gauss_flows._src.transforms.bijections.periodic._utils import _wrap_all


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


__all__ = ["CircularRQSplineCoupling"]
