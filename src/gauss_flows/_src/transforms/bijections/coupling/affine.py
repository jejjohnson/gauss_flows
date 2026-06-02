"""Affine coupling layer."""

from __future__ import annotations

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection, Coupling
from flowjax.bijections.affine import Affine
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


class AffineCoupling(AbstractBijection):
    """Affine coupling layer.

    Splits the input into two halves along its single (1D) axis. The first
    ``n_dims // 2`` dims pass through unchanged and feed an inner conditioner
    MLP that emits per-dim affine parameters ``(shift, log_scale)`` for the
    remaining dims, which are transformed as ``y = x · exp(log_scale) + shift``.
    With ``cond_dim`` set, the same MLP additionally consumes an external
    context vector concatenated onto the first-half input.

    Wraps `flowjax.bijections.Coupling` with an
    `flowjax.bijections.affine.Affine` transformer; this class adds the
    convention that an unconditional layer (``cond_dim=None``) drops any
    incoming ``condition`` so a base-only condition cannot leak into the inner
    MLP.

    Args:
        key: JAX random key for conditioner MLP initialisation.
        shape: Event shape ``(n_dims,)``. Only 1D inputs are supported.
        cond_dim: If not ``None``, the layer expects a 1-D ``condition`` of
            shape ``(cond_dim,)`` at call time and concatenates it onto the
            inner MLP's input. Defaults to ``None`` (unconditional).
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.

    Raises:
        ValueError: If ``shape`` is not 1D, or ``cond_dim`` is a non-positive int.

    Shape:
        - transform_and_log_det: (n_dims,) → (n_dims,), scalar log_det
        - inverse_and_log_det:   (n_dims,) → (n_dims,), scalar log_det
        (with ``cond_dim`` set, also takes a ``condition`` of shape ``(cond_dim,)``)

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import AffineCoupling
        >>> t = AffineCoupling(jr.key(0), shape=(4,))
        >>> x = jr.normal(jr.key(1), (4,))
        >>> y, log_det = t.transform_and_log_det(x)
        >>> y.shape
        (4,)
        >>> x_rec, log_det_inv = t.inverse_and_log_det(y)
        >>> bool(jnp.allclose(x, x_rec, atol=1e-5))
        True
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    _coupling: AbstractBijection

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
            raise ValueError("AffineCoupling only supports 1D inputs.")
        if cond_dim is not None and cond_dim < 1:
            raise ValueError(
                f"cond_dim must be a positive int or None; got {cond_dim}."
            )
        n_dims = shape[0]
        self.shape = shape
        self.cond_shape = None if cond_dim is None else (cond_dim,)

        affine = Affine()
        self._coupling = Coupling(
            key=key,
            transformer=affine,
            untransformed_dim=n_dims // 2,
            dim=n_dims,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Forward map ``x → y`` and its scalar log-determinant.

        Args:
            x: Single event of shape ``(n_dims,)``.
            condition: Context of shape ``(cond_dim,)`` when the layer is
                conditional; ignored (dropped) when ``cond_dim=None``.

        Returns:
            Tuple ``(y, log_det)`` with ``y`` of shape ``(n_dims,)`` and a
            scalar ``log_det``.
        """
        # Drop the condition when the layer is unconditional. flowjax's inner
        # Coupling unconditionally concatenates any non-None condition with
        # the untransformed half, which would crash the inner MLP if we let
        # a SurVAEFlow base-only condition leak into an unconditional layer.
        if self.cond_shape is None:
            condition = None
        return self._coupling.transform_and_log_det(jnp.asarray(x), condition)

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Inverse map ``y → x`` and its scalar log-determinant.

        Args:
            y: Single event of shape ``(n_dims,)``.
            condition: Context of shape ``(cond_dim,)`` when the layer is
                conditional; ignored (dropped) when ``cond_dim=None``.

        Returns:
            Tuple ``(x, log_det)`` with ``x`` of shape ``(n_dims,)`` and a
            scalar ``log_det``.
        """
        if self.cond_shape is None:
            condition = None
        return self._coupling.inverse_and_log_det(jnp.asarray(y), condition)


__all__ = ["AffineCoupling"]
