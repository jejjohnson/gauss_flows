"""Volume-preserving affine coupling layer (GIN; Sorrenson et al. 2020)."""

from __future__ import annotations

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
        cond_dim: If not ``None``, the layer expects a 1-D ``condition`` of
            shape ``(cond_dim,)`` at call time, concatenated onto the inner
            MLP's input alongside the passive half of ``x``. Defaults to
            ``None``.
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.

    Shape:
        - transform_and_log_det: (n_dims,) → (n_dims,), scalar log_det (≡ 0)
        - inverse_and_log_det:   (n_dims,) → (n_dims,), scalar log_det (≡ 0)
        (with ``cond_dim`` set, also takes a ``condition`` of shape ``(cond_dim,)``)

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
    cond_shape: tuple[int, ...] | None
    untransformed_dim: int = eqx.field(static=True)
    nn: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        *,
        cond_dim: int | None = None,
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
        if cond_dim is not None and cond_dim < 1:
            raise ValueError(
                f"cond_dim must be a positive int or None; got {cond_dim}."
            )

        dim = shape[0]
        transformed_dim = dim - dim // 2
        self.shape = shape
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        self.untransformed_dim = dim // 2
        # The MLP consumes [passive_half, condition] when cond_dim is set,
        # mirroring how flowjax's Coupling threads condition into its inner
        # MLP for the other coupling layers in this package.
        in_size = self.untransformed_dim + (cond_dim or 0)
        self.nn = eqx.nn.MLP(
            in_size=in_size,
            out_size=2 * transformed_dim,
            width_size=nn_width,
            depth=nn_depth,
            activation=jnn.relu,
            key=key,
        )

    def _active_params(
        self, passive: Array, condition: Array | None
    ) -> tuple[Array, Array]:
        if self.cond_shape is None:
            mlp_in = passive
        else:
            mlp_in = jnp.concatenate(
                [passive, jnp.asarray(condition, dtype=passive.dtype)]
            )
        shift, log_scale = jnp.split(self.nn(mlp_in), 2)
        return shift, _center_log_scale(log_scale)

    def transform_and_log_det(
        self,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Forward map ``x → y`` with volume-preserving ``log_det ≡ 0``.

        Args:
            x: Single event of shape ``(n_dims,)``.
            condition: Context of shape ``(cond_dim,)`` when conditional;
                ``None`` otherwise.

        Returns:
            Tuple ``(y, log_det)`` with ``y`` of shape ``(n_dims,)`` and a
            scalar ``log_det`` that is exactly zero.
        """
        x = jnp.asarray(x)
        passive = x[: self.untransformed_dim]
        active = x[self.untransformed_dim :]
        shift, log_scale = self._active_params(passive, condition)
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
        """Inverse map ``y → x`` with volume-preserving ``log_det ≡ 0``.

        Args:
            y: Single event of shape ``(n_dims,)``.
            condition: Context of shape ``(cond_dim,)`` when conditional;
                ``None`` otherwise.

        Returns:
            Tuple ``(x, log_det)`` with ``x`` of shape ``(n_dims,)`` and a
            scalar ``log_det`` that is exactly zero.
        """
        y = jnp.asarray(y)
        passive = y[: self.untransformed_dim]
        active_y = y[self.untransformed_dim :]
        shift, log_scale = self._active_params(passive, condition)
        active_x = (active_y - shift) * jnp.exp(-log_scale)
        x = jnp.concatenate((passive, active_x))
        # Mirror the forward-direction fix: base log_det.dtype on the output x,
        # which is promoted to float by jnp.exp even when the input y is int.
        log_det = jnp.zeros((), dtype=x.dtype)
        return x, log_det


__all__ = ["GINCoupling"]
