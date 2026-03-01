"""Coupling layer transforms for Gaussianization flows.

These bijections implement various coupling-based architectures used in
normalizing flows for density estimation.
"""

from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections import (
    AbstractBijection,
    Coupling,
    RationalQuadraticSpline,
)
from flowjax.bijections.affine import Affine
from jax import Array
from jax.nn import softplus
from jaxtyping import ArrayLike, PRNGKeyArray


class ActNorm1D(AbstractBijection):
    """Activation normalization for 1D inputs.

    Performs per-dimension affine transformation with learnable location and
    log-scale parameters. Similar to batch normalization but with learned
    parameters that are not data-dependent at inference time.

    Args:
        shape: Shape of the input (n_dims,).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    log_scale: Array

    def __init__(self, shape: tuple[int, ...]):
        if len(shape) != 1:
            raise ValueError("ActNorm1D only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        self.loc = jnp.zeros(n_dims)
        self.log_scale = jnp.zeros(n_dims)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        scale = softplus(self.log_scale) + 1e-5
        y = (jnp.asarray(x) - self.loc) / scale
        log_det = -jnp.sum(jnp.log(scale))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        scale = softplus(self.log_scale) + 1e-5
        x = jnp.asarray(y) * scale + self.loc
        log_det = jnp.sum(jnp.log(scale))
        return x, log_det


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


class RQSplineCoupling(AbstractBijection):
    """Rational quadratic spline coupling layer.

    Like AffineCoupling, but uses a rational quadratic spline as the transformer.

    Args:
        key: JAX random key.
        shape: Shape of the input (n_dims,).
        n_bins: Number of spline bins. Defaults to 8.
        interval: Spline interval. Defaults to 5.0.
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
        n_bins: int = 8,
        interval: float = 5.0,
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        if len(shape) != 1:
            raise ValueError("RQSplineCoupling only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape

        spline = RationalQuadraticSpline(knots=n_bins, interval=interval)
        self._coupling = Coupling(
            key=key,
            transformer=spline,
            untransformed_dim=n_dims // 2,
            dim=n_dims,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        return self._coupling.transform_and_log_det(jnp.asarray(x), condition)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        return self._coupling.inverse_and_log_det(jnp.asarray(y), condition)


class DeepSigmoidCoupling(AbstractBijection):
    """Affine coupling layer with a configurable number of conditioner components.

    This is an affine coupling layer. The ``n_components`` parameter is reserved
    for a future deep-sigmoidal transformer; currently the layer uses an Affine
    transformer regardless of that value.

    Args:
        key: JAX random key.
        shape: Shape of the input (n_dims,).
        n_components: Reserved for future use (deep sigmoid components). Defaults to 8.
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
        n_components: int = 8,
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        if len(shape) != 1:
            raise ValueError("DeepSigmoidCoupling only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape

        # Affine transformer; n_components is reserved for a future deep sigmoid
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


__all__ = [
    "ActNorm1D",
    "AffineCoupling",
    "DeepSigmoidCoupling",
    "RQSplineCoupling",
]
