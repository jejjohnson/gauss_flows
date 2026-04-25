"""Affine coupling layer."""

from __future__ import annotations

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection, Coupling
from flowjax.bijections.affine import Affine
from jaxtyping import ArrayLike, PRNGKeyArray


class AffineCoupling(AbstractBijection):
    """Affine coupling layer.

    Splits input into two halves: the first half is unchanged (the
    *data-dependent* conditioning input), and the second half is
    transformed by an affine function whose parameters are produced by an
    inner MLP. With ``cond_dim > 0`` the same MLP additionally consumes an
    external context vector, concatenated to the first-half input.

    Args:
        key: JAX random key.
        shape: Shape of the input ``(n_dims,)``.
        cond_dim: If not ``None``, the layer expects a 1-D ``condition`` of
            shape ``(cond_dim,)`` at call time and concatenates it onto the
            inner MLP's input. Defaults to ``None`` (unconditional).
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.
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

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        # Drop the condition when the layer is unconditional. flowjax's inner
        # Coupling unconditionally concatenates any non-None condition with
        # the untransformed half, which would crash the inner MLP if we let
        # a SurVAEFlow base-only condition leak into an unconditional layer.
        if self.cond_shape is None:
            condition = None
        return self._coupling.transform_and_log_det(jnp.asarray(x), condition)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        if self.cond_shape is None:
            condition = None
        return self._coupling.inverse_and_log_det(jnp.asarray(y), condition)


__all__ = ["AffineCoupling"]
