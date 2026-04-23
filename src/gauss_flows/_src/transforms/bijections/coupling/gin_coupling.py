"""Volume-preserving affine coupling layer (GIN; Sorrenson et al. 2020)."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


def _center_log_scale(log_scale: Array) -> Array:
    """Center log-scales so ``Σᵢ log_scaleᵢ = 0`` in exact arithmetic.

    In float32 the centering leaves ~1e-7 drift per layer; :class:`GINCoupling`
    returns a hardcoded zero ``log_det`` rather than ``jnp.sum`` of this tensor
    to keep the volume preservation drift-free through stacked layers.
    """
    return log_scale - jnp.mean(log_scale)


class GINCoupling(AbstractBijection):
    """Volume-preserving affine coupling from Sorrenson et al. (2020).

    This matches :class:`AffineCoupling` except the active-half log-scales are
    centred before exponentiation:

    ``y_active = x_active · exp(log_scale − mean(log_scale)) + shift``

    so the transformed block has unit determinant and the layer returns
    ``log_det = 0`` exactly. We hardcode ``jnp.zeros(())`` rather than
    returning ``jnp.sum(log_scale_centered)`` because the centering still
    leaves ~1e-7 float32 drift per layer; compounded across a stack of
    ``Chain``-ed GIN blocks that drift breaks the volume-preservation that
    is the whole point of this bijection.

    The active half must have at least two dimensions for centering to
    actually constrain a scale — a 1-element ``log_scale`` is centred to
    zero and degenerates to a pure shift. We therefore require
    ``shape[0] >= 3`` (smallest ``dim`` with ``transformed_dim = 2``).

    Reference:
        Sorrenson, Rother, Köthe (2020), *Disentanglement by Nonlinear ICA
        with General Incompressible-flow Networks (GIN)*.

    Args:
        key: JAX random key for conditioner MLP initialisation.
        shape: Event shape ``(n_dims,)``.
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.

    Shape:
        - Input ``x``: ``(n_dims,)``
        - Output ``y``: ``(n_dims,)``
        - ``log_det``: scalar ``()``, always exactly zero

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import GINCoupling
        >>> coupling = GINCoupling(jr.key(0), shape=(4,), nn_width=16, nn_depth=1)
        >>> x = jr.normal(jr.key(1), (4,))
        >>> y, log_det = coupling.transform_and_log_det(x)
        >>> x_rec, log_det_inv = coupling.inverse_and_log_det(y)
        >>> assert jnp.allclose(x, x_rec)
        >>> assert jnp.array_equal(log_det, jnp.array(0.0, dtype=log_det.dtype))
        >>> assert jnp.array_equal(log_det_inv, jnp.array(0.0, dtype=log_det_inv.dtype))
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    untransformed_dim: int = eqx.field(static=True)
    nn: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        if len(shape) != 1:
            raise ValueError("GINCoupling only supports 1D inputs.")
        if shape[0] < 3:
            raise ValueError(
                "GINCoupling requires shape[0] >= 3 so the active half has "
                "at least two dimensions; otherwise centering collapses the "
                f"log_scale to zero and the layer is a pure shift. Got "
                f"shape={shape}."
            )

        dim = shape[0]
        transformed_dim = dim - dim // 2
        self.shape = shape
        self.untransformed_dim = dim // 2
        self.nn = eqx.nn.MLP(
            in_size=self.untransformed_dim,
            out_size=2 * transformed_dim,
            width_size=nn_width,
            depth=nn_depth,
            activation=jnn.relu,
            key=key,
        )

    def _active_params(self, passive: Array) -> tuple[Array, Array]:
        shift, log_scale = jnp.split(self.nn(passive), 2)
        return shift, _center_log_scale(log_scale)

    def transform_and_log_det(
        self,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        del condition
        x = jnp.asarray(x)
        passive = x[: self.untransformed_dim]
        active = x[self.untransformed_dim :]
        shift, log_scale = self._active_params(passive)
        active_y = active * jnp.exp(log_scale) + shift
        y = jnp.concatenate((passive, active_y))
        # Use y.dtype rather than x.dtype: if x is integer, jnp.exp promotes
        # active_y to float and y.dtype is the right thing to match.
        log_det = jnp.zeros((), dtype=y.dtype)
        return y, log_det

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        del condition
        y = jnp.asarray(y)
        passive = y[: self.untransformed_dim]
        active_y = y[self.untransformed_dim :]
        shift, log_scale = self._active_params(passive)
        active_x = (active_y - shift) * jnp.exp(-log_scale)
        x = jnp.concatenate((passive, active_x))
        # Mirror the forward-direction fix: base log_det.dtype on the output x,
        # which is promoted to float by jnp.exp even when the input y is int.
        log_det = jnp.zeros((), dtype=x.dtype)
        return x, log_det


__all__ = ["GINCoupling"]
