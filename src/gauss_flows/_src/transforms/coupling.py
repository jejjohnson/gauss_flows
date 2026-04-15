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
    """Monotone single-layer sigmoidal transformer for a single dimension.

    Implements the per-dim transformer

        y    = base_scale·x + bias + Σᵢ amplitudesᵢ · σ(slopesᵢ·x + shiftsᵢ)
        dy/dx = base_scale + Σᵢ amplitudesᵢ · slopesᵢ · σ(·) · (1 − σ(·))

    with ``base_scale``, ``amplitudes``, ``slopes`` constrained positive
    (softplus + 1e-4) so ``dy/dx > 0`` everywhere → strictly monotone, hence
    invertible. Matches the *single-layer* sigmoid flow of Huang et al.
    (2018) NAF, §3.2; it is **not** the stacked DDSF ("deep dense sigmoid
    flow") variant from the same paper. Single-layer DSF has limited
    expressiveness for multi-modal targets and the loss surface tends to
    have a strong attractor at the affine local minimum (see PR #45).
    """

    n_components: int = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    bias: Array
    log_base_scale: Array
    log_amplitudes: Array
    log_slopes: Array
    shifts: Array

    def __init__(self, n_components: int):
        # All-zero defaults are just shape declarations: the parent
        # `Coupling` overrides every leaf at runtime with the conditioner's
        # output, so these values are never used after construction.
        self.n_components = n_components
        self.shape = ()
        self.bias = jnp.array(0.0)
        self.log_base_scale = jnp.array(0.0)
        self.log_amplitudes = jnp.zeros((n_components,))
        self.log_slopes = jnp.zeros((n_components,))
        self.shifts = jnp.zeros((n_components,))

    def _parameters(self):
        # Positive constraint via softplus + small floor for numerical safety.
        base_scale = softplus(self.log_base_scale) + 1e-4
        amplitudes = softplus(self.log_amplitudes) + 1e-4
        slopes = softplus(self.log_slopes) + 1e-4
        return base_scale, amplitudes, slopes

    def _forward(self, x: Array):
        # x: scalar -> y: scalar, dy/dx: scalar.
        base_scale, amplitudes, slopes = self._parameters()
        preactivations = slopes * x + self.shifts  # (n_components,)
        sig = sigmoid(preactivations)  # (n_components,)
        y = base_scale * x + self.bias + jnp.sum(amplitudes * sig, axis=-1)
        dy_dx = base_scale + jnp.sum(amplitudes * slopes * sig * (1.0 - sig), axis=-1)
        return y, dy_dx

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        y, dy_dx = self._forward(jnp.asarray(x))
        log_det = jnp.log(dy_dx)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        # Solve y = self._forward(x) for x by bracketed bisection. We use a
        # geometric expansion to bracket the root from a linear-approx seed,
        # then 50 bisections for ~1e-12 resolution.
        y = jnp.asarray(y)
        base_scale, amplitudes, _ = self._parameters()
        approx_center = (y - self.bias - 0.5 * jnp.sum(amplitudes)) / base_scale
        # Initial half-width: 10 covers ~3 std for N(0,1) inputs.
        lo = approx_center - 10.0
        hi = approx_center + 10.0

        def _maybe_expand(_i, bounds):
            # Geometric: each non-bracketing iter doubles the bracket. With 20
            # iters of 2x growth from width-20, max bracket width is ~20·2²⁰ ≈
            # 2·10⁷ — handles any reasonable y. Two forward evals per iter.
            lo_val, hi_val = bounds
            y_lo, _ = self._forward(lo_val)
            y_hi, _ = self._forward(hi_val)
            half = 0.5 * (hi_val - lo_val)
            expand_down = y_lo > y
            expand_up = y_hi < y
            return (
                jnp.where(expand_down, lo_val - 2.0 * half, lo_val),
                jnp.where(expand_up, hi_val + 2.0 * half, hi_val),
            )

        lo, hi = lax.fori_loop(0, 20, _maybe_expand, (lo, hi))

        def _bisect(_i, bounds):
            # bounds: (lo, hi) each shape ()
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
    """Coupling layer with a monotone single-layer sigmoidal transformer.

    Replaces ``AffineCoupling``'s affine transformer with the
    :class:`_DeepSigmoidTransformer` — a per-dim monotone function
    parametrised by ``n_components`` sigmoid units (single-layer DSF, Huang
    et al. 2018 NAF §3.2). The transformer is *strictly more expressive*
    than affine in the function-class sense: it has nonzero second derivative
    where Affine has zero, and can in principle approximate any smooth
    monotone CDF.

    In practice, single-layer DSF inherits the optimisation difficulties
    Huang et al. flagged: the loss surface around the conditioner's default
    initialisation is dominated by an affine-equivalent basin, and SGD often
    fails to escape it on multi-modal / heavy-tailed targets within a few
    hundred steps. For challenging targets, prefer ``RQSplineCoupling`` (see
    flowjax) or stacked DSF chains.

    Args:
        key: JAX random key for conditioner MLP initialisation.
        shape: Event shape ``(n_dims,)``.
        n_components: Number of sigmoid units in the per-dim transformer.
            More components → more capacity, more parameters per dim.
            Defaults to 8.
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.

    Shape:
        - Input  ``x``:  ``(n_dims,)`` (with ``n_dims`` even-friendly; the
          coupling splits into ``n_dims // 2`` kept + transformed halves).
        - Output ``y``:  ``(n_dims,)``
        - ``log_det``:   scalar (sum over transformed dims of ``log(dy/dx)``)

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import DeepSigmoidCoupling
        >>>
        >>> coupling = DeepSigmoidCoupling(
        ...     key=jr.key(0), shape=(4,), n_components=8, nn_width=32, nn_depth=2,
        ... )
        >>> y, log_det = coupling.transform_and_log_det(jr.normal(jr.key(1), (4,)))
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
