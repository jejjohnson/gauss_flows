"""Tests for .from_data factories on FixedRotation and MixtureGaussianCDF."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from gauss_flows import FixedRotation, MixtureGaussianCDF


class TestFixedRotationFromData:
    def test_rotation_is_orthogonal(self, data_4d):
        rot = FixedRotation.from_data(data_4d)
        d = data_4d.shape[-1]
        assert rot.matrix.shape == (d, d)
        # Orthogonal: Q Q^T = I.
        should_be_eye = rot.matrix @ rot.matrix.T
        assert jnp.allclose(should_be_eye, jnp.eye(d), atol=1e-5)

    def test_rotation_decorrelates(self, key):
        # Build correlated data from a random linear mixing of N(0, I).
        n = 2000
        d = 3
        z = jr.normal(key, (n, d))
        a = jnp.array([[1.0, 0.8, 0.2], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        x = z @ a.T

        rot = FixedRotation.from_data(x)
        y = jnp.stack([rot.transform_and_log_det(xi)[0] for xi in x])

        # After PCA rotation the covariance should be diagonal.
        cov = jnp.cov(y, rowvar=False)
        off_diag = cov - jnp.diag(jnp.diag(cov))
        assert jnp.max(jnp.abs(off_diag)) < 0.05

    def test_roundtrip(self, data_4d):
        rot = FixedRotation.from_data(data_4d)
        for xi in data_4d[:5]:
            y, _ = rot.transform_and_log_det(xi)
            xi_rec, _ = rot.inverse_and_log_det(y)
            assert jnp.allclose(xi, xi_rec, atol=1e-5)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            FixedRotation.from_data(jnp.zeros((3,)))


class TestMixtureGaussianCDFFromData:
    def test_shape_and_roundtrip(self, data_4d):
        marg = MixtureGaussianCDF.from_data(data_4d, n_components=4)
        d = data_4d.shape[-1]
        assert marg.means.shape == (d, 4)
        assert marg.log_scales.shape == (d, 4)
        assert marg.log_weights.shape == (d, 4)
        # Uniform weights at init (log_weights stays zero).
        assert jnp.allclose(marg.log_weights, 0.0)
        # Roundtrip on one sample (bisection inverse has ~1e-5 precision).
        y, _ = marg.transform_and_log_det(data_4d[0])
        x_rec, _ = marg.inverse_and_log_det(y)
        assert jnp.allclose(data_4d[0], x_rec, atol=1e-4)

    def test_tail_roundtrip_stays_stable(self):
        import jax
        import jax.numpy as jnp
        import numpy as np

        jax.config.update("jax_enable_x64", True)
        x = jnp.asarray(np.random.default_rng(0).standard_normal((2000, 2)))
        marg = MixtureGaussianCDF.from_data(x, n_components=8)
        y = jax.vmap(marg.transform_and_log_det)(x)[0]
        x_rec = jax.vmap(marg.inverse_and_log_det)(y)[0]
        err = jnp.abs(x - x_rec).max(axis=1)
        tail_err = err[jnp.max(jnp.abs(x), axis=1) > 2.5]
        assert jnp.max(err) < 1e-3
        assert jnp.max(tail_err) < 1e-3

    def test_means_match_quantiles(self, data_2d):
        import numpy as np

        k = 5
        marg = MixtureGaussianCDF.from_data(data_2d, n_components=k)
        qs = np.linspace(0.5 / k, 1.0 - 0.5 / k, k)
        for i in range(data_2d.shape[-1]):
            expected = np.quantile(np.asarray(data_2d)[:, i], qs)
            assert jnp.allclose(marg.means[i], expected, atol=1e-5)

    def test_forward_pass_runs(self, data_2d):
        marg = MixtureGaussianCDF.from_data(data_2d, n_components=4)
        y, log_det = marg.transform_and_log_det(data_2d[0])
        assert y.shape == (data_2d.shape[-1],)
        assert log_det.shape == ()
        assert jnp.all(jnp.isfinite(y))
        assert jnp.isfinite(log_det)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            MixtureGaussianCDF.from_data(jnp.zeros((3,)))
