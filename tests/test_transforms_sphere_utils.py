"""Tests for spherical geometry helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from gauss_flows import UniformOnSphere, expmap_sphere, logmap_sphere, tangent_basis


def test_tangent_basis_is_orthonormal_and_orthogonal(key):
    d = 3
    xs = UniformOnSphere(d).sample(key, (8,))
    basis = jax.vmap(tangent_basis)(xs)
    identity = jnp.tile(jnp.eye(d), (xs.shape[0], 1, 1))
    assert basis.shape == (xs.shape[0], d + 1, d)
    assert jnp.allclose(jnp.swapaxes(basis, -1, -2) @ basis, identity, atol=1e-6)
    assert jnp.allclose(jnp.einsum("bi,bij->bj", xs, basis), 0.0, atol=1e-6)


def test_expmap_and_logmap_roundtrip(key):
    xs = UniformOnSphere(2).sample(key, (16,))
    basis = jax.vmap(tangent_basis)(xs)
    coeffs = jr.normal(jr.fold_in(key, 1), (16, 2)) * 0.2
    vs = jnp.einsum("bij,bj->bi", basis, coeffs)
    ys = jax.vmap(expmap_sphere)(xs, vs)
    vs_back = jax.vmap(logmap_sphere)(xs, ys)
    assert jnp.allclose(vs_back, vs, atol=1e-5)
    assert jnp.allclose(jax.vmap(expmap_sphere)(xs, vs_back), ys, atol=1e-5)


def test_expmap_and_logmap_roundtrip_large_angle(key):
    """Roundtrip must survive angles well away from the sinc-small regime.

    The spherical log map is only injective for ``||v|| < π``, so we clamp
    the sampled tangent vectors below that injectivity radius before testing
    the roundtrip identity.
    """
    xs = UniformOnSphere(2).sample(key, (16,))
    basis = jax.vmap(tangent_basis)(xs)
    coeffs = jr.normal(jr.fold_in(key, 1), (16, 2)) * 1.5
    vs = jnp.einsum("bij,bj->bi", basis, coeffs)
    norms = jnp.linalg.norm(vs, axis=-1, keepdims=True)
    max_radius = jnp.pi * 0.95
    vs = vs * jnp.minimum(1.0, max_radius / jnp.maximum(norms, 1e-12))
    ys = jax.vmap(expmap_sphere)(xs, vs)
    vs_back = jax.vmap(logmap_sphere)(xs, ys)
    assert jnp.allclose(vs_back, vs, atol=1e-4)


def test_logmap_near_antipodal_uses_generic_shortest_geodesic(key):
    """Near-antipodal (not exact) points must stay on the generic branch.

    The antipodal fallback is only defined up to a free direction in
    ``T_x S^d``, so triggering it for points that still have a well-defined
    shortest-geodesic tangent would break ``expmap(x, logmap(x, y)) ≈ y``.
    """
    x = UniformOnSphere(2).sample(key)
    basis = tangent_basis(x)
    # Build a y at geodesic distance ~0.99 π from x (cosine ≈ -0.9995).
    near_antipode_tangent = basis @ jnp.array([0.99 * jnp.pi, 0.0])
    y = expmap_sphere(x, near_antipode_tangent)
    v = logmap_sphere(x, y)
    assert jnp.allclose(v, near_antipode_tangent, atol=1e-5)
    y_back = expmap_sphere(x, v)
    assert jnp.allclose(y_back, y, atol=1e-5)


def test_logmap_at_antipodal_point_returns_pi_length_tangent(key):
    """Log map at ``y == -x`` returns a π-length vector in a tangent direction.

    The log map is set-valued at antipodes; we pin down the deterministic
    fallback to ``π · tangent_basis(x)[:, 0]`` which is (a) a valid tangent
    vector at ``x``, and (b) has the correct geodesic length.
    """
    x = UniformOnSphere(2).sample(key)
    y = -x
    v = logmap_sphere(x, y)
    assert jnp.allclose(jnp.linalg.norm(v), jnp.pi, atol=1e-5)
    # Must be orthogonal to x (tangent condition).
    assert jnp.allclose(jnp.dot(x, v), 0.0, atol=1e-5)


def test_logmap_grad_at_coincident_point_is_finite(key):
    """`grad(logmap)` at x == y must not leak NaN via the 0/0 double-where."""
    x = UniformOnSphere(2).sample(key)

    def loss(y):
        return jnp.sum(logmap_sphere(x, y))

    grad_at_coincident = jax.grad(loss)(x)
    assert jnp.all(jnp.isfinite(grad_at_coincident))


def test_sphere_utils_jit_vmap_and_grad_smoke(key):
    d = 2
    x = UniformOnSphere(d).sample(key)
    basis = tangent_basis(x)
    v = basis @ jnp.array([0.2, -0.1])
    xs = UniformOnSphere(d).sample(jr.fold_in(key, 1), (8,))
    vs = jax.vmap(lambda xi: tangent_basis(xi) @ jnp.array([0.05, -0.03]))(xs)

    tangent_basis_fn = jax.jit(jax.vmap(tangent_basis))
    exp_fn = jax.jit(jax.vmap(expmap_sphere))
    log_fn = jax.jit(jax.vmap(logmap_sphere))

    ys = exp_fn(xs, vs)
    assert tangent_basis_fn(xs).shape == (xs.shape[0], d + 1, d)
    assert log_fn(xs, ys).shape == (xs.shape[0], d + 1)

    grad = jax.grad(lambda vec: jnp.sum(expmap_sphere(x, vec)))(v)
    assert jnp.all(jnp.isfinite(grad))
