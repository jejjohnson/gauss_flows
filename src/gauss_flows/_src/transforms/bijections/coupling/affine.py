"""Affine coupling layer."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection, Coupling
from flowjax.bijections.affine import Affine
from jaxtyping import ArrayLike, PRNGKeyArray


class AffineCoupling(AbstractBijection):
    """Affine coupling layer.

    Splits input into two halves: the first half is unchanged (condition),
    and the second half is transformed by an affine function parameterized
    by the first half.

    Args:
        key: JAX random key.
        shape: Shape of the input (n_dims,).
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    _coupling: AbstractBijection

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        if len(shape) != 1:
            raise ValueError("AffineCoupling only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape

        affine = Affine()
        self._coupling = Coupling(
            key=key,
            transformer=affine,
            untransformed_dim=n_dims // 2,
            dim=n_dims,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        return self._coupling.transform_and_log_det(jnp.asarray(x), condition)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        return self._coupling.inverse_and_log_det(jnp.asarray(y), condition)


__all__ = ["AffineCoupling"]
