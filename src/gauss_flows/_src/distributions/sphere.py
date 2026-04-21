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
_VMF_SERIES_TERMS = 256


def _normalize_to_sphere(x: Array) -> Array:
    norm = jnp.maximum(jnp.linalg.norm(x), jnp.finfo(x.dtype).tiny)
    return x / norm


def _log_surface_area_sphere(d: int) -> Array:
    half = 0.5 * (d + 1)
    return jnp.log(2.0) + half * jnp.log(jnp.pi) - gammaln(half)


def _log_bessel_iv(nu: Array, x: Array) -> Array:
    x = jnp.asarray(x, dtype=float)
    x_safe = jnp.maximum(x, jnp.finfo(x.dtype).tiny)
    ks = jnp.arange(_VMF_SERIES_TERMS, dtype=x.dtype)
    log_terms = (
        ks * jnp.log((x_safe * x_safe) / 4.0)
        - gammaln(ks + 1.0)
        - gammaln(ks + nu + 1.0)
    )
    log_series = nu * jnp.log(x_safe / 2.0) + logsumexp(log_terms)
    return jnp.where(x > 0, log_series, jnp.where(nu == 0.0, 0.0, -jnp.inf))


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
        on_sphere = jnp.isclose(jnp.linalg.norm(x), 1.0, atol=1e-6)
        log_density = -_log_surface_area_sphere(self.d)
        return jnp.where(on_sphere, log_density, -jnp.inf)


class VonMisesFisher(AbstractDistribution):
    """Von Mises–Fisher distribution on ``S^d ⊂ ℝ^{d+1}``.

    Args:
        mean: Mean direction in ``ℝ^{d+1}``. Normalized internally.
        concentration: Non-negative scalar concentration ``κ``.

    Shape:
        Event shape: ``(d+1,)`` where ``d = mean.shape[0] - 1``.
        Sample shape: arbitrary leading dimensions.
        ``log_prob`` is evaluated on single points on the sphere.

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
        mean_array = jnp.asarray(mean, dtype=float)
        if mean_array.ndim != 1:
            raise ValueError("mean must have shape (d+1,).")
        if mean_array.shape[0] < 2:
            raise ValueError("mean must live in at least two ambient dimensions.")
        mean_norm = jnp.linalg.norm(mean_array)
        if float(mean_norm) == 0.0:
            raise ValueError("mean must be non-zero.")
        concentration_array = jnp.asarray(concentration, dtype=float)
        if concentration_array.shape != ():
            raise ValueError("concentration must be a scalar.")
        if float(concentration_array) < 0.0:
            raise ValueError("concentration must be non-negative.")
        self.shape = mean_array.shape
        self.mean = mean_array / mean_norm
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
        return jnp.where(concentration > 0, vmf_log_density, uniform_log_density)

    def _sample_w(self, key: PRNGKeyArray) -> Array:
        p = self.shape[0]
        m = p - 1.0
        concentration = self.concentration
        b = (-2.0 * concentration + jnp.sqrt(4.0 * concentration**2 + m**2)) / m
        x0 = (1.0 - b) / (1.0 + b)
        c = concentration * x0 + m * jnp.log1p(-(x0 * x0))
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
            accept = concentration * w + m * jnp.log1p(-x0 * w) - c >= jnp.log(u)
            return loop_key, jnp.where(accept, w, accepted_w), accept

        _, w, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (key, jnp.asarray(0.0, dtype=concentration.dtype), jnp.asarray(False)),
        )
        return w

    def _sample(self, key: PRNGKeyArray, condition: Array | None = None) -> Array:
        del condition
        uniform = UniformOnSphere(self.d)

        def sample_uniform(sample_key: PRNGKeyArray) -> Array:
            return uniform._sample(sample_key)

        def sample_vmf(sample_key: PRNGKeyArray) -> Array:
            key_w, key_v = jr.split(sample_key)
            w = self._sample_w(key_w)
            v = _normalize_to_sphere(jr.normal(key_v, (self.d,)))
            basis = tangent_basis(self.mean)  # basis: (d+1, d)
            tangent_part = basis @ v  # v: (d,) -> tangent_part: (d+1,)
            scale = jnp.sqrt(jnp.maximum(1.0 - w * w, 0.0))
            sample = w * self.mean + scale * tangent_part
            return _normalize_to_sphere(sample)

        return jax.lax.cond(self.concentration == 0, sample_uniform, sample_vmf, key)

    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        del condition
        x = jnp.asarray(x)
        on_sphere = jnp.isclose(jnp.linalg.norm(x), 1.0, atol=1e-6)
        log_density = self._log_normalizer() + self.concentration * jnp.dot(
            self.mean, x
        )
        return jnp.where(on_sphere, log_density, -jnp.inf)


__all__ = ["UniformOnSphere", "VonMisesFisher"]
