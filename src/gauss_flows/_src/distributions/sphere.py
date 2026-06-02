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
# Truncated power-series expansion of log I_nu(x) used for small arguments
# ``x <= _VMF_ASYMPTOTIC_THRESHOLD``. With 1024 terms the series peak
# k_peak ≈ (sqrt(nu^2 + x^2) - nu) / 2 stays well below the cap for any
# realistic ``nu`` in that range.
_VMF_SERIES_TERMS = 1024
# Crossover between the power series and the asymptotic expansion. Above
# this ``x`` the series is never required and the Debye uniform expansion
# (nu > 0) / Hankel expansion (nu = 0) takes over. Both asymptotic forms
# are numerically stable at any x > threshold for their respective nu
# regimes, so there is no gap.
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


def _log_bessel_iv_hankel(nu: Array, x: Array) -> Array:
    # Classic large-x expansion:
    #   I_nu(x) ~ e^x / sqrt(2 pi x) * (1 - (mu - 1)/(8x) + ...)  with mu = 4 nu^2
    # Used only as the ``nu == 0`` path of the primary asymptotic, where the
    # correction polynomial 1 + 1/(8x) + 9/(128 x^2) + ... is strictly
    # positive; at nu > 0 the Debye uniform expansion takes over because
    # Hankel's corrections diverge once ``x < 2 nu^2``.
    mu = 4.0 * nu * nu
    z = 8.0 * x
    term1 = (mu - 1.0) / z
    term2 = (mu - 1.0) * (mu - 9.0) / (2.0 * z * z)
    term3 = (mu - 1.0) * (mu - 9.0) * (mu - 25.0) / (6.0 * z * z * z)
    correction = 1.0 - term1 + term2 - term3
    return x - 0.5 * jnp.log(2.0 * jnp.pi * x) + jnp.log(correction)


def _log_bessel_iv_debye(nu: Array, x: Array) -> Array:
    # Uniform (Debye) asymptotic expansion, DLMF 10.41.3. Unlike Hankel this
    # is numerically stable for any ``x > 0`` given ``nu > 0`` — no validity
    # region to guard — so it handles both the classical large-x regime and
    # the ``x << nu^2`` regime where Hankel's corrections diverge. Empirical
    # error against scipy is < 1e-3 across the grid
    # (nu, x) in [0.5, 150] cross [5, 40000] with one U_1/nu correction
    # term included.
    nu_safe = jnp.maximum(nu, jnp.finfo(nu.dtype).tiny)
    z = x / nu_safe
    z_sq = z * z
    sqrt_term = jnp.sqrt(1.0 + z_sq)
    # eta(z) = sqrt(1+z^2) + log(z) - log(1 + sqrt(1+z^2))
    eta = sqrt_term + jnp.log(z) - jnp.log1p(sqrt_term)
    leading = nu * eta - 0.5 * jnp.log(2.0 * jnp.pi * nu_safe) - 0.25 * jnp.log1p(z_sq)
    # First-order Debye correction U_1(t)/nu with t = 1/sqrt(1 + z^2).
    t = 1.0 / sqrt_term
    u1 = (3.0 * t - 5.0 * t * t * t) / 24.0
    return leading + jnp.log1p(u1 / nu_safe)


def _log_bessel_iv(nu: Array, x: Array) -> Array:
    x = jnp.asarray(x)
    nu_arr = jnp.asarray(nu, dtype=x.dtype)
    threshold = jnp.asarray(_VMF_ASYMPTOTIC_THRESHOLD, dtype=x.dtype)
    use_asymp = x > threshold
    # Feed a safe input to each branch so the inactive path can't emit
    # NaN/-inf into the ``jnp.where`` gradient.
    x_series = jnp.where(use_asymp, threshold, x)
    x_asymp = jnp.where(use_asymp, x, threshold + 1.0)
    series = _log_bessel_iv_series(nu, x_series)
    # Pick Debye (stable for any x > 0 at nu > 0) when nu > 0, and fall
    # back to Hankel (which is bounded for nu == 0 since no terms diverge)
    # at the single-order S^1 case. ``nu`` is derived from the static
    # ``self.d`` so this branch resolves at trace time for VMF.
    debye = _log_bessel_iv_debye(nu_arr, x_asymp)
    hankel = _log_bessel_iv_hankel(nu_arr, x_asymp)
    asymptotic = jnp.where(nu_arr > 0, debye, hankel)
    hybrid = jnp.where(use_asymp, asymptotic, series)
    return jnp.where(x > 0, hybrid, jnp.where(nu == 0.0, 0.0, -jnp.inf))


class UniformOnSphere(AbstractDistribution):
    """Uniform distribution on the unit sphere ``Sᵈ ⊂ ℝ^{d+1}``.

    Constant density ``1 / A_d`` over the sphere, where ``A_d`` is the
    surface area of ``Sᵈ``; ``log_prob`` returns ``−∞`` for off-sphere
    points. Samples are drawn by normalising an isotropic Gaussian, so
    they satisfy ``‖x‖₂ = 1`` up to numerical precision. Useful as a flow
    base for directional / rotation-invariant data.

    Args:
        d: Intrinsic sphere dimension. Samples live in ``ℝ^{d+1}``.

    Shape:
        - log_prob: ``(d+1,)`` → scalar    (batch via vmap / leading axes)
        - sample:   ``(key, sample_shape)`` → ``(*sample_shape, d+1)``

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
    """Von Mises–Fisher distribution on the unit sphere ``Sᵈ ⊂ ℝ^{d+1}``.

    Density ``p(x) ∝ exp(κ · μᵀx)`` on the sphere: the spherical analogue of
    an isotropic Gaussian, peaked at the mean direction ``μ`` with
    concentration ``κ`` (``κ = 0`` recovers :class:`UniformOnSphere`). The
    normaliser involves a modified Bessel function ``I_ν(κ)``, evaluated
    here via a hybrid power-series / asymptotic expansion. Sampling uses
    Wood's (1994) rejection scheme. Useful as a flow base for concentrated
    directional data.

    Invalid inputs (zero-vector ``mean`` or negative ``concentration``) are
    propagated as ``NaN`` through ``log_prob`` / ``sample`` rather than being
    rejected at construction, so the constructor stays ``jit`` / ``vmap``-safe
    (e.g. inside a mixture of VMFs).

    Args:
        mean: Mean direction ``μ`` in ``ℝ^{d+1}``. Normalised internally;
            integer inputs are promoted to a floating dtype. Sets the event
            dimension ``d = mean.shape[0] − 1``.
        concentration: Non-negative scalar concentration ``κ``.

    Shape:
        - log_prob: ``(d+1,)`` → scalar    (batch via vmap / leading axes)
        - sample:   ``(key, sample_shape)`` → ``(*sample_shape, d+1)``

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
        # Wood's acceptance test hangs the while-loop for any non-finite or
        # non-positive κ (NaN, ±inf, 0, negatives), so we run the sampler
        # with a safe placeholder κ and taint the output afterwards to
        # propagate the invalid input. Beyond a dtype-dependent bound,
        # ``b ≈ m / (4κ)`` rounds below the floating eps and ``x0`` snaps to
        # ``1.0``; then ``c = κ + m log1p(-1) = -inf`` and the acceptance
        # test ``κ w + m log1p(-x0 w) - c`` evaluates to NaN, so the loop
        # never accepts. Clamp finite κ to the safe range; the sampler then
        # returns ``w ≈ 1`` (i.e. ``sample ≈ mean``), which is the correct
        # asymptotic limit as ``κ → ∞``.
        eps = jnp.finfo(concentration.dtype).eps
        kappa_safe_max = jnp.asarray(0.25 / eps, dtype=concentration.dtype)
        is_finite_positive = (concentration > 0) & jnp.isfinite(concentration)
        safe_concentration = jnp.where(
            is_finite_positive,
            jnp.minimum(concentration, kappa_safe_max),
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
        # Non-finite/non-positive κ went through the placeholder branch;
        # NaN the output so callers see the invalid input downstream. Finite
        # κ above the safe clamp just gets its sampler-clamped value (a
        # near-delta) and is considered valid.
        return jnp.where(is_finite_positive, w, jnp.asarray(jnp.nan, dtype=w.dtype))

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
