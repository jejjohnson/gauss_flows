"""Tests for normalization bijections: Generalized Divisive Normalization (GDN)."""

from __future__ import annotations

import contextlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Chain
from flowjax.distributions import Normal, Transformed

from gauss_flows import (
    FixedRotation,
    GeneralizedDivisiveNormalization,
    GeneralizedDivisiveNormalization1D,
    MixtureGaussianCDF,
)


@contextlib.contextmanager
def x64_enabled():
    """Enable float64 for the duration of a test, then restore (no global leak)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _coupled_gamma(key, c, scale=0.3, bias=-2.0):
    """A modest, stable coupling: small enough that the inverse contracts."""
    return jr.normal(key, (c, c)) * scale + bias


def test_gdn_shape_image(key):
    """Image variant preserves ``(..., C)`` and returns a scalar log_det."""
    layer = GeneralizedDivisiveNormalization(shape=(4, 4, 3))
    x = jr.normal(key, (4, 4, 3))
    y, log_det = layer.transform_and_log_det(x)
    assert y.shape == (4, 4, 3)
    assert log_det.shape == ()


def test_gdn_shape_1d(key):
    """1-D variant preserves ``(D,)`` and rejects non-1-D shapes."""
    layer = GeneralizedDivisiveNormalization1D(shape=(5,))
    x = jr.normal(key, (5,))
    y, log_det = layer.transform_and_log_det(x)
    assert y.shape == (5,)
    assert log_det.shape == ()
    with pytest.raises(ValueError, match="only supports 1D"):
        GeneralizedDivisiveNormalization1D(shape=(4, 4, 3))


def test_gdn_rejects_invalid_controls():
    """Constructor validates fixed-point / floor controls instead of silently
    storing values that would corrupt forward/inverse."""
    with pytest.raises(ValueError, match="beta_floor"):
        GeneralizedDivisiveNormalization1D(shape=(3,), beta_floor=-1e-3)
    with pytest.raises(ValueError, match="inverse_damping"):
        GeneralizedDivisiveNormalization1D(shape=(3,), inverse_damping=0.0)
    with pytest.raises(ValueError, match="inverse_damping"):
        GeneralizedDivisiveNormalization1D(shape=(3,), inverse_damping=1.5)
    with pytest.raises(ValueError, match="beta_floor"):
        GeneralizedDivisiveNormalization(shape=(4, 4, 3), beta_floor=-1.0)


def test_gdn_forward_inverse_1d(key):
    """``f^{-1}(f(x)) ≈ x`` for the 1-D variant; slogdet sign stays positive."""
    layer = GeneralizedDivisiveNormalization1D(shape=(4,))
    layer = eqx.tree_at(lambda t: t.raw_gamma, layer, _coupled_gamma(jr.key(7), 4))
    x = jr.normal(key, (4,))
    y, log_det = layer.transform_and_log_det(x)
    x_rec, log_det_inv = layer.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)
    sign, _ = jnp.linalg.slogdet(
        jax.jacrev(lambda xx: layer.transform_and_log_det(xx)[0])(x)
    )
    assert sign > 0


def test_gdn_forward_inverse_image(key):
    """``f^{-1}(f(x)) ≈ x`` for the image variant across spatial positions."""
    layer = GeneralizedDivisiveNormalization(shape=(3, 3, 2))
    layer = eqx.tree_at(lambda t: t.raw_gamma, layer, _coupled_gamma(jr.key(8), 2))
    x = jr.normal(key, (3, 3, 2))
    y, log_det = layer.transform_and_log_det(x)
    x_rec, log_det_inv = layer.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_gdn_diagonal_gamma_does_not_bound_output(key):
    """A positive diagonal must not cap ``|y_i|``: the diagonal is held at zero.

    With ``gamma_ii > 0`` the forward map would satisfy ``|y_i| < 1/sqrt(gamma_ii)``,
    bounding the range and leaving out-of-box targets with no preimage. Masking the
    diagonal keeps the map onto all of R^D, so far-out targets still round-trip.
    """
    with x64_enabled():
        layer = GeneralizedDivisiveNormalization1D(
            shape=(2,), inverse_tol=1e-12, inverse_max_iters=300
        )
        # Large diagonal (softplus(2) ~ 2.13) would cap |y| < 0.68 if it counted;
        # off-diagonal coupling stays modest.
        raw_gamma = jnp.array([[2.0, -3.0], [-3.0, 2.0]])
        layer = eqx.tree_at(lambda t: t.raw_gamma, layer, raw_gamma)

        _, gamma = layer._params()
        assert jnp.allclose(jnp.diag(gamma), 0.0)

        # Forward of a large input is unbounded (would saturate near 0.68 if the
        # diagonal counted).
        y_big, _ = layer.transform_and_log_det(jnp.array([50.0, 0.0]))
        assert jnp.abs(y_big[0]) > 5.0

        # A target far outside the would-be box still inverts and round-trips.
        y = jnp.array([3.0, -1.5])
        x_rec, _ = layer.inverse_and_log_det(y)
        y_check, _ = layer.transform_and_log_det(x_rec)
        assert jnp.max(jnp.abs(y - y_check)) < 1e-8


def test_gdn_inverse_nan_outside_support(key):
    """Out-of-support targets surface as NaN, not silently-wrong finite values.

    With positive coupling GDN's joint range is bounded; a target outside it has
    no preimage. The inverse verifies ``forward(x*) ≈ y`` and NaNs the result
    otherwise, so ``log_prob`` / ``sample`` flag out-of-support instead of
    returning plausible-but-wrong numbers.
    """
    with x64_enabled():
        layer = GeneralizedDivisiveNormalization1D(
            shape=(2,), inverse_tol=1e-12, inverse_max_iters=300
        )
        # Strong off-diagonal coupling -> small invertible region.
        layer = eqx.tree_at(lambda t: t.raw_gamma, layer, jnp.zeros((2, 2)))

        y_ok = jnp.array([0.3, -0.2])
        x_ok, ld_ok = layer.inverse_and_log_det(y_ok)
        assert jnp.all(jnp.isfinite(x_ok))
        assert jnp.isfinite(ld_ok)
        assert jnp.allclose(layer.transform_and_log_det(x_ok)[0], y_ok, atol=1e-8)

        x_bad, ld_bad = layer.inverse_and_log_det(jnp.array([5.0, -5.0]))
        assert jnp.all(jnp.isnan(x_bad))
        assert jnp.isnan(ld_bad)


def test_gdn_forward_inverse_float64_precision(key):
    """Round-trip is exact to ``< 1e-10`` in float64 with a tight tolerance."""
    with x64_enabled():
        layer = GeneralizedDivisiveNormalization1D(
            shape=(4,), inverse_tol=1e-12, inverse_max_iters=300
        )
        layer = eqx.tree_at(lambda t: t.raw_gamma, layer, _coupled_gamma(jr.key(9), 4))
        x = jr.normal(key, (4,), dtype=jnp.float64)
        y, _ = layer.transform_and_log_det(x)
        x_rec, _ = layer.inverse_and_log_det(y)
        assert jnp.max(jnp.abs(x - x_rec)) < 1e-10


def test_gdn_log_det_matches_autodiff(key):
    """Analytic log_det equals ``slogdet`` of the autodiff Jacobian."""
    layer = GeneralizedDivisiveNormalization1D(shape=(5,))
    layer = eqx.tree_at(lambda t: t.raw_gamma, layer, _coupled_gamma(jr.key(3), 5))
    x = jr.normal(key, (5,))
    _, log_det = layer.transform_and_log_det(x)
    jac = jax.jacrev(lambda xx: layer.transform_and_log_det(xx)[0])(x)
    sign, logabsdet = jnp.linalg.slogdet(jac)
    assert sign > 0
    assert jnp.allclose(log_det, logabsdet, atol=1e-4)


def test_gdn_inverse_grad_matches_implicit(key):
    """``jax.grad`` through ``.inverse`` matches the implicit-function gradient.

    The fixed-point solve is non-differentiable on its own; the ``custom_vjp``
    backpropagates through one linear solve against ``(I − M)``. Checked against
    finite differences for both the target ``y`` and the coupling parameters.
    """
    with x64_enabled():
        layer = GeneralizedDivisiveNormalization1D(
            shape=(4,), inverse_tol=1e-12, inverse_max_iters=300
        )
        layer = eqx.tree_at(lambda t: t.raw_gamma, layer, _coupled_gamma(jr.key(4), 4))
        y = jr.normal(key, (4,), dtype=jnp.float64)

        def obj(yy):
            return layer.inverse_and_log_det(yy)[0].sum()

        g = jax.grad(obj)(y)
        eps = 1e-6
        fd = jnp.array(
            [
                (obj(y.at[i].add(eps)) - obj(y.at[i].add(-eps))) / (2 * eps)
                for i in range(4)
            ]
        )
        assert jnp.any(jnp.abs(g) > 1e-3)  # not the broken zero gradient
        assert jnp.allclose(g, fd, atol=1e-5)

        def pobj(raw_gamma):
            perturbed = eqx.tree_at(lambda t: t.raw_gamma, layer, raw_gamma)
            return perturbed.inverse_and_log_det(y)[0].sum()

        gp = jax.grad(pobj)(layer.raw_gamma)
        fdp = (
            pobj(layer.raw_gamma.at[1, 2].add(eps))
            - pobj(layer.raw_gamma.at[1, 2].add(-eps))
        ) / (2 * eps)
        assert jnp.allclose(gp[1, 2], fdp, atol=1e-5)


def test_gdn_inside_gaussianization_flow(key):
    """GDN composes in a flow: finite ``log_prob`` and a working round-trip."""
    k1, k2 = jr.split(key, 2)
    d = 3
    # Correlated, heavy-ish-tailed synthetic data.
    z = jr.normal(k1, (300, d))
    mix = jnp.array([[1.0, 0.7, 0.2], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    data = z @ mix.T

    bijection = Chain(
        [
            GeneralizedDivisiveNormalization1D(shape=(d,)),
            FixedRotation.from_data(data),
            MixtureGaussianCDF.from_data(data, n_components=6),
        ]
    )
    flow = Transformed(Normal(jnp.zeros(d)), bijection)

    log_probs = flow.log_prob(data)
    assert log_probs.shape == (300,)
    assert jnp.all(jnp.isfinite(log_probs))

    samples = flow.sample(k2, (16,))
    assert samples.shape == (16, d)
    assert jnp.all(jnp.isfinite(samples))


def test_gdn_jit_vmap_grad_smoke(key):
    """jit / vmap / grad all work, in float32 and (scoped) float64."""
    layer = GeneralizedDivisiveNormalization1D(shape=(4,))
    xs = jr.normal(key, (8, 4))

    ys = jax.jit(jax.vmap(lambda x: layer.transform_and_log_det(x)[0]))(xs)
    assert ys.shape == (8, 4)
    assert jnp.all(jnp.isfinite(ys))

    def loss(raw_gamma):
        perturbed = eqx.tree_at(lambda t: t.raw_gamma, layer, raw_gamma)
        fwd = jax.vmap(lambda x: perturbed.transform_and_log_det(x)[1])
        return fwd(xs).sum()

    g = jax.jit(jax.grad(loss))(layer.raw_gamma)
    assert jnp.all(jnp.isfinite(g))

    with x64_enabled():
        layer64 = GeneralizedDivisiveNormalization1D(shape=(4,))
        x64 = jr.normal(key, (4,), dtype=jnp.float64)
        y64, ld64 = layer64.transform_and_log_det(x64)
        assert y64.dtype == jnp.float64
        assert jnp.isfinite(ld64)
