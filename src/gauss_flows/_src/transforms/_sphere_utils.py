"""Utilities for spherical transforms on ``S^d ⊂ ℝ^{d+1}``."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _safe_normalize(x: Array) -> Array:
    norm = jnp.maximum(jnp.linalg.norm(x), jnp.finfo(x.dtype).tiny)
    return x / norm


def tangent_basis(x: Array) -> Array:
    """Return an orthonormal basis of the tangent space ``T_x S^d``.

    Uses a Householder reflection that maps ``x`` to ``±e₁`` and then takes
    columns ``2..d+1`` of that orthogonal matrix.

    Shape:
        Input: ``(d+1,)``.
        Output: ``(d+1, d)`` with ``Eᵀ E = I`` and ``xᵀ E = 0``.

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import tangent_basis
        >>> x = jnp.array([0.0, 0.0, 1.0])
        >>> tangent_basis(x).shape
        (3, 2)
    """

    x_unit = _safe_normalize(x)
    d_plus_1 = x_unit.shape[0]
    e1 = jnp.eye(d_plus_1, dtype=x_unit.dtype)[:, 0]
    sign = jnp.where(x_unit[0] >= 0, 1.0, -1.0).astype(x_unit.dtype)
    v = x_unit + sign * e1
    v_norm2 = jnp.maximum(jnp.dot(v, v), jnp.finfo(x_unit.dtype).tiny)
    householder = (
        jnp.eye(d_plus_1, dtype=x_unit.dtype) - 2.0 * jnp.outer(v, v) / v_norm2
    )
    return householder[:, 1:]


def expmap_sphere(
    x: Array,
    v: Array,
) -> Array:
    """Map a tangent vector at ``x`` to the sphere via the exponential map.

    ``expmap(x, v) = x cos(||v||) + v sin(||v||) / ||v||`` with the
    ``sin(r) / r`` factor evaluated via ``sinc`` for stability at ``r → 0``.

    Shape:
        ``x`` and ``v`` are single events of shape ``(d+1,)``.
        Returns a single point on the sphere with shape ``(d+1,)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import expmap_sphere
        >>> x = jnp.array([0.0, 0.0, 1.0])
        >>> v = jnp.array([0.1, 0.0, 0.0])
        >>> expmap_sphere(x, v).shape
        (3,)
    """

    x_unit = _safe_normalize(x)
    v_tangent = v - x_unit * jnp.dot(x_unit, v)
    radius = jnp.linalg.norm(v_tangent)
    scale = jnp.sinc(radius / jnp.pi)
    y = jnp.cos(radius) * x_unit + scale * v_tangent
    return _safe_normalize(y)


def logmap_sphere(
    x: Array,
    y: Array,
) -> Array:
    """Return the tangent vector at ``x`` whose exp-map reaches ``y``.

    Shape:
        ``x`` and ``y`` are single points on the sphere with shape ``(d+1,)``.
        Returns a tangent vector in ``T_x S^d`` with shape ``(d+1,)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import expmap_sphere, logmap_sphere
        >>> x = jnp.array([0.0, 0.0, 1.0])
        >>> v = jnp.array([0.1, 0.0, 0.0])
        >>> y = expmap_sphere(x, v)
        >>> logmap_sphere(x, y).shape
        (3,)
    """

    x_unit = _safe_normalize(x)
    y_unit = _safe_normalize(y)
    cosine = jnp.clip(jnp.dot(x_unit, y_unit), -1.0, 1.0)
    tangent = y_unit - cosine * x_unit
    tangent_sq = jnp.sum(tangent * tangent)
    # Double-where on ``tangent_sq`` so ``jnp.sqrt`` and the division have
    # finite gradients even at ``y == x`` (where ``tangent == 0``). The
    # ``arctan2`` form avoids the ``arccos`` boundary singularity at
    # ``cosine == ±1`` (whose derivative is ``-∞``) which would otherwise leak
    # NaN via ``0 · ∞`` in the inactive branch of the ``where``.
    eps_sq = jnp.asarray(jnp.finfo(tangent.dtype).eps ** 2, dtype=tangent.dtype)
    is_nonzero = tangent_sq > eps_sq
    safe_tangent_sq = jnp.where(is_nonzero, tangent_sq, 1.0)
    safe_norm = jnp.sqrt(safe_tangent_sq)
    angle = jnp.arctan2(safe_norm, cosine)
    scale = jnp.where(is_nonzero, angle / safe_norm, 1.0)
    return scale * tangent


__all__ = ["expmap_sphere", "logmap_sphere", "tangent_basis"]
