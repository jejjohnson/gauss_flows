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

    def test_narrow_sigmas_preserved_in_init(self, key):
        """Components with fitted sigma in (5e-3, 1e-2) must end up with an
        effective scale close to the fit, not widened to >= 1e-2 by the
        ``softplus + 5e-3`` floor arithmetic.
        """
        import jax
        import jax.nn

        # Build a narrow mixture so GMM fits small sigmas.
        n = 4000
        k1, k2 = jr.split(key)
        comp = jr.randint(k1, (n, 1), 0, 3)
        centers = jnp.array([-1.0, 0.0, 1.0])[comp]
        x = centers + 7e-3 * jr.normal(k2, (n, 1))
        flow = fit_rbig(x, n_layers=1, n_components=3, random_state=0)
        # Extract the MixtureGaussianCDF out of the first block.
        chain = flow.bijection.bijection
        from gauss_flows import MixtureGaussianCDF

        marg = next(b for b in chain.bijections if isinstance(b, MixtureGaussianCDF))
        # Effective per-component scales used at forward time.
        effective_scales = jax.nn.softplus(marg.log_scales) + 5e-3
        # Data has true sigma=7e-3 so the GMM will fit sigmas near that.
        # Pre-fix, the `max(sigma - 5e-3, 5e-3)` floor forced every
        # component to effective scale ~= 1e-2 (ignoring the fit). With
        # the fix, effective scales should track the fitted sigma to
        # well within a factor of two.
        assert float(effective_scales.max()) < 9.5e-3, (
            f"narrow fits were widened to {effective_scales.max():.4f} "
            f"instead of being preserved near the fitted sigma (~7e-3)"
        )


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

    @pytest.mark.parametrize("n_dims", [3, 5, 7])
    def test_odd_dims_rejected(self, key2, n_dims):
        # Odd n_dims breaks the complementary-half assumption behind the
        # [coupling, Flip, coupling, Flip] block: one dim ends up in both
        # b-halves and is transformed twice per block, degrading the
        # warm-start.
        with pytest.raises(ValueError, match="even n_dims"):
            fit_rbig_coupling(jnp.zeros((10, n_dims)), key2)

    def test_survives_gradient_training(self, key, key2):
        """Gradient training on a fit_rbig_coupling flow must leave it a true
        bijection — sampling must round-trip to within float32 epsilon.

        Regression guard: if ``FixedRotation.matrix`` is trainable, Adam
        drifts it off the orthogonal manifold, the inverse stops matching
        the forward, and ``flow.sample`` collapses to a narrow blob.
        """
        import jax
        from flowjax.train import fit_to_data

        x = jr.normal(key, (512, 2))
        flow = fit_rbig_coupling(x, key2, n_layers=2, n_components=4)
        trained, _ = fit_to_data(
            jr.fold_in(key, 99),
            flow,
            x,
            learning_rate=1e-3,
            max_epochs=5,
            max_patience=5,
            batch_size=128,
            val_prop=0.1,
            show_progress=False,
        )
        xs = x[:32]
        zs = jax.vmap(trained.bijection.inverse)(xs)
        xs_rec = jax.vmap(trained.bijection.transform)(zs)
        assert jnp.allclose(xs, xs_rec, atol=1e-4), (
            "Trained flow is not a true bijection — FixedRotation.matrix "
            "may have drifted off the orthogonal manifold."
        )
