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
