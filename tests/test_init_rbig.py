"""Tests for the RBIG warm-start init (fit_rbig and fit_rbig_coupling)."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gauss_flows import fit_rbig, fit_rbig_coupling


def _skewed_bimodal_2d(key, n: int = 500):
    z = jr.normal(key, (n, 2))
    x = z * jnp.array([2.0, 0.3]) + jnp.array([1.0, -0.5])
    x = jnp.where(x[:, :1] > 0, x + jnp.array([3.0, 0.0]), x)
    return x


class TestFitRbig:
    def test_output_is_transformed(self, key):
        x = jr.normal(key, (200, 3))
        flow = fit_rbig(x, n_layers=2, n_components=4)
        # Duck-type: flowjax Transformed has .log_prob + .sample.
        assert hasattr(flow, "log_prob")
        assert hasattr(flow, "sample")

    def test_log_prob_finite(self, key):
        x = jr.normal(key, (200, 3))
        flow = fit_rbig(x, n_layers=3, n_components=4)
        lp = flow.log_prob(x)
        assert lp.shape == (200,)
        assert jnp.all(jnp.isfinite(lp))

    def test_pushforward_is_standard_normal(self, key):
        x = _skewed_bimodal_2d(key, n=800)
        flow = fit_rbig(x, n_layers=6, n_components=6)

        z = jnp.stack([flow.bijection.inverse(xi) for xi in x[:256]])
        z = np.asarray(z)
        # With 6 RBIG layers on this skewed bimodal, pushforward should be
        # close to N(0, I). Loose tolerance — these are empirical moments.
        assert np.all(np.abs(z.mean(axis=0)) < 0.2)
        assert np.all(np.abs(z.std(axis=0) - 1.0) < 0.2)

    def test_beats_identity_baseline(self, key):
        x = _skewed_bimodal_2d(key, n=500)
        flow = fit_rbig(x, n_layers=4, n_components=4)

        # Baseline: log-likelihood of x under standard N(0, I) — what you
        # get with no flow at all.
        baseline_lp = float(
            (-0.5 * jnp.log(2 * jnp.pi) - 0.5 * x**2).sum(axis=-1).mean()
        )
        fit_lp = float(flow.log_prob(x).mean())
        assert fit_lp > baseline_lp + 0.5

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            fit_rbig(jnp.zeros((3,)))

    def test_reproducible_via_random_state(self, key):
        x = jr.normal(key, (200, 2))
        flow_a = fit_rbig(x, n_layers=2, n_components=4, random_state=7)
        flow_b = fit_rbig(x, n_layers=2, n_components=4, random_state=7)
        lp_a = flow_a.log_prob(x[:32])
        lp_b = flow_b.log_prob(x[:32])
        assert jnp.allclose(lp_a, lp_b)


class TestFitRbigCoupling:
    def test_output_is_transformed(self, key, key2):
        x = jr.normal(key, (200, 4))
        flow = fit_rbig_coupling(x, key2, n_layers=2, n_components=4)
        assert hasattr(flow, "log_prob")
        assert hasattr(flow, "sample")

    def test_log_prob_finite(self, key, key2):
        x = jr.normal(key, (200, 4))
        flow = fit_rbig_coupling(x, key2, n_layers=2, n_components=4)
        lp = flow.log_prob(x)
        assert lp.shape == (200,)
        assert jnp.all(jnp.isfinite(lp))

    def test_beats_identity_baseline(self, key, key2):
        # Skewed 4D data.
        z = jr.normal(key, (500, 4))
        x = z * jnp.array([2.0, 0.3, 1.5, 0.7]) + jnp.array([1.0, -0.5, 0.3, 2.0])

        flow = fit_rbig_coupling(x, key2, n_layers=4, n_components=4)
        baseline_lp = float(
            (-0.5 * jnp.log(2 * jnp.pi) - 0.5 * x**2).sum(axis=-1).mean()
        )
        fit_lp = float(flow.log_prob(x).mean())
        assert fit_lp > baseline_lp

    def test_non_2d_raises(self, key2):
        with pytest.raises(ValueError, match="must be 2-D"):
            fit_rbig_coupling(jnp.zeros((3,)), key2)

    def test_min_dims(self, key2):
        with pytest.raises(ValueError, match="n_dims >= 2"):
            fit_rbig_coupling(jnp.zeros((10, 1)), key2)
