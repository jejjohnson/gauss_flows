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
from jax import Array, lax
from jax.nn import sigmoid, softplus
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


class _DeepSigmoidTransformer(AbstractBijection):
    """Monotone dense sigmoid transformer for a single dimension."""

    n_components: int = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    bias: Array
    log_base_scale: Array
    log_amplitudes: Array
    log_slopes: Array
    shifts: Array

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.shape = ()
        self.bias = jnp.array(0.0)
        self.log_base_scale = jnp.array(0.0)
        self.log_amplitudes = jnp.zeros((n_components,))
        self.log_slopes = jnp.zeros((n_components,))
        self.shifts = jnp.zeros((n_components,))

    def _parameters(self):
        base_scale = softplus(self.log_base_scale) + 1e-4
        amplitudes = softplus(self.log_amplitudes) + 1e-4
        slopes = softplus(self.log_slopes) + 1e-4
        return base_scale, amplitudes, slopes

    def _forward(self, x: Array):
        base_scale, amplitudes, slopes = self._parameters()
        preactivations = slopes * x + self.shifts
        sig = sigmoid(preactivations)
        y = base_scale * x + self.bias + jnp.sum(amplitudes * sig, axis=-1)
        dy_dx = base_scale + jnp.sum(amplitudes * slopes * sig * (1.0 - sig), axis=-1)
        return y, dy_dx

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        y, dy_dx = self._forward(jnp.asarray(x))
        log_det = jnp.log(dy_dx)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        y = jnp.asarray(y)
        base_scale, amplitudes, _ = self._parameters()
        approx_center = (y - self.bias - 0.5 * jnp.sum(amplitudes)) / base_scale
        span = 10.0
        lo = approx_center - span
        hi = approx_center + span

        def _maybe_expand(i, bounds):
            lo_val, hi_val = bounds
            width = span * (2.0**i)
            y_lo, _ = self._forward(lo_val)
            y_hi, _ = self._forward(hi_val)
            expand_down = y_lo > y
            expand_up = y_hi < y
            return (
                jnp.where(expand_down, lo_val - width, lo_val),
                jnp.where(expand_up, hi_val + width, hi_val),
            )

        lo, hi = lax.fori_loop(0, 6, _maybe_expand, (lo, hi))

        def _bisect(_i, bounds):
            lo_val, hi_val = bounds
            mid = 0.5 * (lo_val + hi_val)
            y_mid, _ = self._forward(mid)
            go_up = y_mid < y
            return (
                jnp.where(go_up, mid, lo_val),
                jnp.where(go_up, hi_val, mid),
            )

        lo, hi = lax.fori_loop(0, 50, _bisect, (lo, hi))
        x = 0.5 * (lo + hi)
        _, dy_dx = self._forward(x)
        log_det = -jnp.log(dy_dx)
        return x, log_det


class DeepSigmoidCoupling(AbstractBijection):
    """Monotone deep-sigmoid coupling layer.

    Replaces the affine transformer with a deep dense sigmoid flow (DDSF) style
    monotone network composed of sigmoid activations and positive weights. The
    ``n_components`` parameter sets the number of sigmoid units in the monotone
    transformer, controlling the expressiveness of the conditional transform.

    Args:
        key: JAX random key.
        shape: Shape of the input (n_dims,).
        n_components: Number of sigmoid components in the transformer. Defaults to 8.
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

        transformer = _DeepSigmoidTransformer(n_components=n_components)
        self._coupling = Coupling(
            key=key,
            transformer=transformer,
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
