"""Distributions on the unit sphere ``S^d ⊂ ℝ^{d+1}``."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import AbstractDistribution
from jax import Array
from jax.scipy.special import gammaln, logsumexp
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms._sphere_utils import tangent_basis


_LOG_2PI = jnp.log(2.0 * jnp.pi)
# Truncated power-series expansion of log I_nu(x); accurate while the series
# peak (near k ≈ x / 2) stays below _VMF_SERIES_TERMS. For larger x we fall
# back to the Hankel-style large-argument asymptotic instead.
_VMF_SERIES_TERMS = 256
# Crossover between the power series and the asymptotic expansion. Both
# approximations are accurate around this value for ``nu`` in the small-integer
# half-integer range that arises for spheres up to ``d ~ 20``.
_VMF_ASYMPTOTIC_THRESHOLD = 50.0
_ON_SPHERE_ATOL = 1e-5


def _normalize_to_sphere(x: Array) -> Array:
    norm = jnp.maximum(jnp.linalg.norm(x), jnp.finfo(x.dtype).tiny)
    return x / norm


def _log_surface_area_sphere(d: int) -> Array:
    half = 0.5 * (d + 1)
    return jnp.log(2.0) + half * jnp.log(jnp.pi) - gammaln(half)


def _log_bessel_iv_series(nu: Array, x: Array) -> Array:
    x_safe = jnp.maximum(x, jnp.finfo(x.dtype).tiny)
    ks = jnp.arange(_VMF_SERIES_TERMS, dtype=x.dtype)
    log_terms = (
        ks * jnp.log((x_safe * x_safe) / 4.0)
        - gammaln(ks + 1.0)
        - gammaln(ks + nu + 1.0)
    )
    return nu * jnp.log(x_safe / 2.0) + logsumexp(log_terms)


def _log_bessel_iv_asymptotic(nu: Array, x: Array) -> Array:
    # Hankel expansion:
    #   I_nu(x) ~ e^x / sqrt(2 pi x) * (1 - (mu - 1)/(8x) + ...)  with mu = 4 nu^2
    mu = 4.0 * nu * nu
    z = 8.0 * x
    term1 = (mu - 1.0) / z
    term2 = (mu - 1.0) * (mu - 9.0) / (2.0 * z * z)
    term3 = (mu - 1.0) * (mu - 9.0) * (mu - 25.0) / (6.0 * z * z * z)
    correction = 1.0 - term1 + term2 - term3
    return x - 0.5 * jnp.log(2.0 * jnp.pi * x) + jnp.log(correction)


def _log_bessel_iv(nu: Array, x: Array) -> Array:
    x = jnp.asarray(x)
    # Use a safe input on each branch so the inactive path can't emit NaN/-inf
    # into the ``jnp.where`` gradient. Actual small-x/negative-x handling is
    # done by the outer ``where`` on the real ``x``.
    threshold = jnp.asarray(_VMF_ASYMPTOTIC_THRESHOLD, dtype=x.dtype)
    use_asymptotic = x > threshold
    x_series = jnp.where(use_asymptotic, threshold, x)
    x_asymptotic = jnp.where(use_asymptotic, x, threshold)
    series = _log_bessel_iv_series(nu, x_series)
    asymptotic = _log_bessel_iv_asymptotic(nu, x_asymptotic)
    hybrid = jnp.where(use_asymptotic, asymptotic, series)
    return jnp.where(x > 0, hybrid, jnp.where(nu == 0.0, 0.0, -jnp.inf))


class UniformOnSphere(AbstractDistribution):
    """Uniform distribution on ``S^d ⊂ ℝ^{d+1}``.

    Args:
        d: Intrinsic sphere dimension. Samples live in ``ℝ^{d+1}``.

    Shape:
        Event shape: ``(d+1,)``.
        Sample shape: arbitrary leading dimensions.
        Samples satisfy ``||x||₂ = 1`` up to numerical precision.

    Example:
        >>> import jax.random as jr
        >>> from gauss_flows import UniformOnSphere
        >>> dist = UniformOnSphere(d=2)
        >>> dist.sample(jr.key(0), (4,)).shape
        (4, 3)
    """

    d: int = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, d: int):
        if d < 1:
            raise ValueError(f"d must be positive, got {d}.")
        self.d = d
        self.shape = (d + 1,)

    def _sample(self, key: PRNGKeyArray, condition: Array | None = None) -> Array:
        del condition
        raw = jr.normal(key, self.shape)
        return _normalize_to_sphere(raw)

    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        del condition
        x = jnp.asarray(x)
        on_sphere = jnp.isclose(jnp.linalg.norm(x), 1.0, atol=_ON_SPHERE_ATOL)
        log_density = -_log_surface_area_sphere(self.d)
        return jnp.where(on_sphere, log_density, -jnp.inf)


class VonMisesFisher(AbstractDistribution):
    """Von Mises–Fisher distribution on ``S^d ⊂ ℝ^{d+1}``.

    Args:
        mean: Mean direction in ``ℝ^{d+1}``. Normalized internally. Integer
            inputs are promoted to a floating dtype.
        concentration: Non-negative scalar concentration ``κ``.

    Shape:
        Event shape: ``(d+1,)`` where ``d = mean.shape[0] - 1``.
        Sample shape: arbitrary leading dimensions.
        ``log_prob`` is evaluated on single points on the sphere.

    Invalid inputs (zero-vector ``mean`` or negative ``concentration``) are
    propagated as ``NaN`` through ``log_prob`` / ``sample`` rather than being
    rejected at construction, so the constructor stays ``jit``/``vmap``-safe
    (e.g. for a mixture of VMFs).

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import VonMisesFisher
        >>> dist = VonMisesFisher(jnp.array([0.0, 0.0, 1.0]), 3.0)
        >>> dist.sample(jr.key(0), (2,)).shape
        (2, 3)
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    mean: Array
    concentration: Array

    def __init__(self, mean: ArrayLike, concentration: ArrayLike):
        mean_array = jnp.asarray(mean)
        if mean_array.ndim != 1:
            raise ValueError("mean must have shape (d+1,).")
        if mean_array.shape[0] < 2:
            raise ValueError("mean must live in at least two ambient dimensions.")

        # Promote integer inputs to a floating dtype so downstream `jnp.finfo`
        # calls and rejection-sampling arithmetic work regardless of how the
        # user typed `mean`.
        float_dtype = jnp.result_type(mean_array.dtype, float)
        mean_array = mean_array.astype(float_dtype)
        concentration_array = jnp.asarray(concentration, dtype=float_dtype)
        if concentration_array.shape != ():
            raise ValueError("concentration must be a scalar.")

        # NaN-propagate invalid inputs so errors surface downstream instead of
        # silently producing finite-but-wrong numbers. We can't raise here
        # without breaking vmap-over-κ patterns.
        norm = jnp.linalg.norm(mean_array)
        safe_norm = jnp.where(norm > 0, norm, jnp.asarray(1.0, dtype=float_dtype))
        normalized_mean = jnp.where(
            norm > 0,
            mean_array / safe_norm,
            jnp.full_like(mean_array, jnp.nan),
        )
        concentration_array = jnp.where(
            concentration_array >= 0,
            concentration_array,
            jnp.asarray(jnp.nan, dtype=float_dtype),
        )

        self.shape = mean_array.shape
        self.mean = normalized_mean
        self.concentration = concentration_array

    @property
    def d(self) -> int:
        return self.shape[0] - 1

    def _log_normalizer(self) -> Array:
        nu = 0.5 * (self.d - 1)
        concentration = self.concentration
        concentration_safe = jnp.maximum(
            concentration,
            jnp.finfo(concentration.dtype).tiny,
        )
        uniform_log_density = -_log_surface_area_sphere(self.d)
        vmf_log_density = (
            nu * jnp.log(concentration_safe)
            - 0.5 * (self.d + 1) * _LOG_2PI
            - _log_bessel_iv(nu, concentration_safe)
        )
        # `κ > 0` is false for both κ == 0 and κ == NaN. Adding `0 · κ` below
        # taints the uniform branch with NaN when κ is NaN so invalid inputs
        # don't silently return the uniform density.
        log_density = jnp.where(concentration > 0, vmf_log_density, uniform_log_density)
        return log_density + 0.0 * concentration

    def _sample_w(self, key: PRNGKeyArray) -> Array:
        p = self.shape[0]
        m = p - 1.0
        concentration = self.concentration
        # Wood's acceptance test hangs the while-loop for non-positive or NaN
        # κ, so we run the sampler with a safe placeholder κ and taint the
        # output afterwards to propagate any NaN/invalid input.
        safe_concentration = jnp.where(
            (concentration > 0) & ~jnp.isnan(concentration),
            concentration,
            jnp.asarray(1.0, dtype=concentration.dtype),
        )
        # Algebraically equivalent to
        #   ``(-2κ + sqrt(4κ² + m²)) / m``
        # but avoids catastrophic cancellation for large κ.
        denom = 2.0 * safe_concentration + jnp.sqrt(
            4.0 * safe_concentration * safe_concentration + m * m
        )
        b = m / denom
        x0 = (1.0 - b) / (1.0 + b)
        c = safe_concentration * x0 + m * jnp.log1p(-(x0 * x0))
        alpha = 0.5 * m

        def cond_fn(state):
            _, _, accepted = state
            return ~accepted

        def body_fn(state):
            loop_key, accepted_w, _ = state
            loop_key, key_beta, key_uniform = jr.split(loop_key, 3)
            z = jr.beta(key_beta, alpha, alpha)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = jr.uniform(key_uniform, minval=jnp.finfo(concentration.dtype).tiny)
            accept = safe_concentration * w + m * jnp.log1p(-x0 * w) - c >= jnp.log(u)
            return loop_key, jnp.where(accept, w, accepted_w), accept

        _, w, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (key, jnp.asarray(0.0, dtype=concentration.dtype), jnp.asarray(False)),
        )
        return w + 0.0 * concentration

    def _sample(self, key: PRNGKeyArray, condition: Array | None = None) -> Array:
        del condition

        def sample_uniform(sample_key: PRNGKeyArray) -> Array:
            return _normalize_to_sphere(jr.normal(sample_key, self.shape))

        def sample_vmf(sample_key: PRNGKeyArray) -> Array:
            key_w, key_v = jr.split(sample_key)
            w = self._sample_w(key_w)
            v = _normalize_to_sphere(jr.normal(key_v, (self.d,)))
            basis = tangent_basis(self.mean)
            tangent_part = basis @ v
            scale = jnp.sqrt(jnp.maximum(1.0 - w * w, 0.0))
            sample = w * self.mean + scale * tangent_part
            return _normalize_to_sphere(sample)

        return jax.lax.cond(self.concentration == 0, sample_uniform, sample_vmf, key)

    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        del condition
        x = jnp.asarray(x)
        on_sphere = jnp.isclose(jnp.linalg.norm(x), 1.0, atol=_ON_SPHERE_ATOL)
        log_density = self._log_normalizer() + self.concentration * jnp.dot(
            self.mean, x
        )
        return jnp.where(on_sphere, log_density, -jnp.inf)


__all__ = ["UniformOnSphere", "VonMisesFisher"]
