"""Tests for MixtureGaussianCDFCoupling."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from gauss_flows import MixtureGaussianCDFCoupling
from gauss_flows._src.transforms.bijections.coupling.mixture_cdf import (
    _clamp_log_scale,
    _invert_clamp_log_scale,
)


class TestMixtureCouplingShapes:
    def test_shape_matches_input(self, key):
        bij = MixtureGaussianCDFCoupling(key, shape=(6,), n_components=4)
        x = jnp.arange(6.0)
        y, log_det = bij.transform_and_log_det(x)
        assert y.shape == (6,)
        assert log_det.shape == ()

    def test_identity_half_unchanged(self, key):
        bij = MixtureGaussianCDFCoupling(key, shape=(4,), n_components=4)
        x = jnp.array([0.5, -0.3, 1.2, -1.0])
        y, _ = bij.transform_and_log_det(x)
        # First n_dims // 2 pass through untouched.
        assert jnp.allclose(y[:2], x[:2])
        # Transformed half actually moves.
        assert not jnp.allclose(y[2:], x[2:], atol=1e-3)

    def test_requires_1d_shape(self, key):
        with pytest.raises(ValueError, match="only supports 1D"):
            MixtureGaussianCDFCoupling(key, shape=(2, 2))

    def test_requires_min_two_dims(self, key):
        with pytest.raises(ValueError, match="n_dims >= 2"):
            MixtureGaussianCDFCoupling(key, shape=(1,))


class TestMixtureCouplingRoundtrip:
    @pytest.mark.parametrize("n_dims", [2, 4, 8])
    @pytest.mark.parametrize("n_components", [2, 4, 8])
    def test_forward_inverse_roundtrip(self, key, n_dims, n_components):
        bij = MixtureGaussianCDFCoupling(
            key, shape=(n_dims,), n_components=n_components
        )
        x = jr.normal(jr.key(0), (n_dims,))
        y, log_det_fwd = bij.transform_and_log_det(x)
        x_rec, log_det_inv = bij.inverse_and_log_det(y)
        assert jnp.allclose(x, x_rec, atol=1e-4)
        assert jnp.allclose(log_det_fwd + log_det_inv, 0.0, atol=1e-5)

    def test_batched_roundtrip(self, key):
        import jax

        bij = MixtureGaussianCDFCoupling(key, shape=(4,), n_components=6)
        xs = jr.normal(jr.key(1), (32, 4))
        ys, _ = jax.vmap(bij.transform_and_log_det)(xs)
        xs_rec, _ = jax.vmap(bij.inverse_and_log_det)(ys)
        assert jnp.allclose(xs, xs_rec, atol=1e-4)


class TestMixtureCouplingLogScaleClamp:
    def test_clamp_bounded(self):
        raw = jnp.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        bound = 3.0
        clamped = _clamp_log_scale(raw, bound)
        assert jnp.all(clamped >= -bound)
        assert jnp.all(clamped <= bound)
        # At raw=0 → clamped=0.
        assert jnp.allclose(clamped[2], 0.0)

    def test_clamp_inverse_roundtrip(self):
        bound = 3.0
        raw = jnp.array([-2.5, -0.7, 0.0, 0.5, 2.9])
        clamped = _clamp_log_scale(raw, bound)
        raw_rec = _invert_clamp_log_scale(clamped, bound)
        assert jnp.allclose(raw, raw_rec, atol=1e-5)


class TestMixtureCouplingTraining:
    def test_log_det_finite_under_large_inputs(self, key):
        bij = MixtureGaussianCDFCoupling(key, shape=(4,), n_components=4)
        x = jnp.array([10.0, -10.0, 5.0, -5.0])
        y, log_det = bij.transform_and_log_det(x)
        assert jnp.all(jnp.isfinite(y))
        assert jnp.isfinite(log_det)

    def test_gradient_flows(self, key):
        import equinox as eqx
        import jax

        bij = MixtureGaussianCDFCoupling(key, shape=(4,), n_components=4)
        x = jr.normal(jr.key(2), (4,))

        def loss_fn(bij):
            _, log_det = bij.transform_and_log_det(x)
            return -log_det

        grad = eqx.filter_grad(loss_fn)(bij)
        # Gradient tree must contain a nonzero inexact array somewhere.
        inexact = eqx.filter(grad, eqx.is_inexact_array)
        leaves = jax.tree.leaves(inexact)
        assert any(
            float(jnp.max(jnp.abs(leaf))) > 0.0
            for leaf in leaves
            if leaf is not None
        )
