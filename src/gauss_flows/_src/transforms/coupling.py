"""Coupling layer transforms for Gaussianization flows.

These bijections implement various coupling-based architectures used in
normalizing flows for density estimation.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
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


class BatchNorm(AbstractBijection):
    """Invertible batch normalization layer for 1D inputs.

    Training mode uses per-batch statistics for normalization. Evaluation mode
    uses exponential running statistics stored as non-static buffers; call
    :meth:`update_running_stats_from_batch` outside of ``jit`` to refresh them
    and set ``use_running_average=True`` for exact round-trips at inference.
    In training mode, exact forward/inverse round-trips only hold when using the
    same batch statistics (set via :meth:`with_batch_stats_from_data`).

    Args:
        shape: Input shape ``(n_dims,)``.
        momentum: Exponential moving-average factor for running statistics.
        eps: Numerical stability term added to the variance.
        use_running_average: Whether to use running statistics (evaluation).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    momentum: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    use_running_average: bool
    log_gamma: Array
    beta: Array
    running_mean: Array = eqx.field(static=False)
    running_var: Array = eqx.field(static=False)
    batch_stats: tuple[Array, Array] | None = eqx.field(static=False)

    def __init__(
        self,
        shape: tuple[int, ...],
        momentum: float = 0.1,
        eps: float = 1e-5,
        *,
        use_running_average: bool = False,
    ):
        if len(shape) != 1:
            raise ValueError("BatchNorm only supports 1D inputs.")
        if not 0.0 < momentum < 1.0:
            raise ValueError(f"momentum must be in (0, 1), got {momentum}.")
        self.shape = shape
        self.momentum = momentum
        self.eps = eps
        self.use_running_average = use_running_average
        n_dims = shape[0]
        self.log_gamma = jnp.zeros(n_dims)
        self.beta = jnp.zeros(n_dims)
        self.running_mean = jnp.zeros(n_dims)
        self.running_var = jnp.ones(n_dims)
        self.batch_stats = None

    def _batch_axes(self, x: Array) -> tuple[int, ...]:
        leading = x.ndim - len(self.shape)
        return tuple(range(leading)) if leading > 0 else ()

    def _batch_stats(self, x: Array) -> tuple[Array, Array]:
        axes = self._batch_axes(x)
        mean = jnp.mean(x, axis=axes)
        var = jnp.var(x, axis=axes)
        return mean, var

    def _broadcast_log_det(self, log_det: Array, ref: Array) -> Array:
        leading_shape = ref.shape[: ref.ndim - len(self.shape)]
        if not leading_shape:
            return log_det
        return jnp.broadcast_to(log_det, leading_shape)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x_arr = jnp.asarray(x)
        if self.use_running_average:
            mean, var = self.running_mean, self.running_var
        elif self.batch_stats is not None:
            mean, var = self.batch_stats
        else:
            mean, var = self._batch_stats(x_arr)
        scale = jnp.exp(self.log_gamma) / jnp.sqrt(var + self.eps)
        y = (x_arr - mean) * scale + self.beta
        log_det = jnp.sum(self.log_gamma - 0.5 * jnp.log(var + self.eps))
        return y, self._broadcast_log_det(log_det, x_arr)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        y_arr = jnp.asarray(y)
        if self.use_running_average:
            mean, var = self.running_mean, self.running_var
        elif self.batch_stats is not None:
            mean, var = self.batch_stats
        else:
            mean, var = self._batch_stats(y_arr)
        scale = jnp.exp(-self.log_gamma) * jnp.sqrt(var + self.eps)
        x = (y_arr - self.beta) * scale + mean
        log_det = jnp.sum(-self.log_gamma + 0.5 * jnp.log(var + self.eps))
        return x, self._broadcast_log_det(log_det, y_arr)

    def update_running_stats(
        self, batch_mean: ArrayLike, batch_var: ArrayLike
    ) -> BatchNorm:
        """Return a copy with running stats updated (call outside ``jit``)."""

        new_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * jnp.asarray(batch_mean)
        new_var = (1 - self.momentum) * self.running_var + self.momentum * jnp.asarray(
            batch_var
        )
        return eqx.tree_at(
            lambda m: (m.running_mean, m.running_var), self, (new_mean, new_var)
        )

    def update_running_stats_from_batch(self, x: ArrayLike) -> BatchNorm:
        """Compute batch statistics from ``x`` and update running buffers."""

        batch_mean, batch_var = self._batch_stats(jnp.asarray(x))
        return self.update_running_stats(batch_mean, batch_var)

    def with_batch_stats_from_data(self, x: ArrayLike) -> BatchNorm:
        """Return a copy using batch statistics computed from ``x``."""

        batch_mean, batch_var = self._batch_stats(jnp.asarray(x))
        return eqx.tree_at(lambda m: m.batch_stats, self, (batch_mean, batch_var))

    def with_running_average(self, use_running_average: bool = True) -> BatchNorm:
        """Return a copy toggling the use of running statistics."""

        return eqx.tree_at(lambda m: m.use_running_average, self, use_running_average)


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
    "BatchNorm",
    "DeepSigmoidCoupling",
    "RQSplineCoupling",
]
