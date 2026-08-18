"""Tests for `normalizing_kalman_filter` and its masked-coupling guard."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as ndist
import pytest
from flowjax.bijections import (
    Affine,
    Chain,
    Coupling,
    Flip,
    Identity,
    Invert,
    RationalQuadraticSpline,
    Vmap,
)
from flowjax.distributions import Normal, Transformed

from gauss_flows import NumpyroBase, normalizing_kalman_filter
from gauss_flows._src.flows.kalman import _mixes_channels


N_STEPS, N_CHANNELS = 8, 4


# ---------------------------------------------------------------------------
# Fixtures: bases and warps
# ---------------------------------------------------------------------------


def _elementwise_warp(n_channels=N_CHANNELS):
    """The recommended warp: a per-channel spline, then a per-channel affine."""
    return Chain(
        [
            # Channel axis: lift the scalar spline to an M-vector BEFORE chaining.
            Vmap(
                RationalQuadraticSpline(knots=6, interval=4),
                in_axes=None,
                axis_size=n_channels,
            ),
            Affine(loc=jnp.zeros(n_channels), scale=jnp.ones(n_channels)),
        ]
    )


def _coupling_warp(key, n_channels=N_CHANNELS, cond_dim=None):
    return Coupling(
        key,
        transformer=Affine(),
        untransformed_dim=n_channels // 2,
        dim=n_channels,
        cond_dim=cond_dim,
        nn_width=8,
        nn_depth=1,
    )


def _unconditional_base(n_steps=N_STEPS, n_channels=N_CHANNELS):
    """Stands in for `NumpyroBase(dist=gaussx.LGSSM(...))`: event shape (T, M)."""
    return Normal(jnp.zeros((n_steps, n_channels)))


def _conditional_base(n_steps=N_STEPS, n_channels=N_CHANNELS):
    """A base that consumes a condition, i.e. the masked case, without gaussx.

    The guard keys on ``cond_shape``, not on the concrete type, so any
    conditional base exercises the same code path a `gaussx.LGSSMFactory`
    would.
    """
    return NumpyroBase(
        dist_factory=lambda mask: (
            ndist.Normal(0.0, 1.0).expand((n_steps, n_channels)).to_event(2)
        ),
        event_shape=(n_steps, n_channels),
        cond_shape=(n_steps, n_channels),
    )


# ---------------------------------------------------------------------------
# Change of variables
# ---------------------------------------------------------------------------


def test_log_prob_matches_base_plus_log_det():
    base = _unconditional_base()
    warp = _elementwise_warp()
    nkf = normalizing_kalman_filter(base, warp)

    y = 0.3 * jr.normal(jr.key(0), (N_STEPS, N_CHANNELS))
    # Per-timestep inverse and log-det, summed over the time axis.
    z, log_det = jax.vmap(warp.inverse_and_log_det)(y)  # (T, M), (T,)
    expected = base.log_prob(z) + log_det.sum()
    assert jnp.abs(nkf.log_prob(y) - expected) < 1e-13


def test_identity_warp_reproduces_the_base_exactly():
    base = _unconditional_base()
    nkf = normalizing_kalman_filter(base, Identity((N_CHANNELS,)))
    y = jr.normal(jr.key(1), (N_STEPS, N_CHANNELS))
    assert jnp.array_equal(nkf.log_prob(y), base.log_prob(y))


def test_round_trip_through_the_nested_vmap_stack():
    warp = _elementwise_warp()
    nkf = normalizing_kalman_filter(_unconditional_base(), warp)
    y = 0.5 * jr.normal(jr.key(2), (N_STEPS, N_CHANNELS))
    z = nkf.bijection.inverse(y)
    assert jnp.allclose(nkf.bijection.transform(z), y, atol=1e-10)


def test_shapes():
    nkf = normalizing_kalman_filter(_unconditional_base(), _elementwise_warp())
    samples = nkf.sample(jr.key(3), (5,))
    assert samples.shape == (5, N_STEPS, N_CHANNELS)
    assert nkf.log_prob(samples).shape == (5,)


def test_gradients_reach_both_the_base_and_the_warp():
    """A frozen warp is the classic silent failure for a parameterised link."""
    base = _unconditional_base()
    warp = _elementwise_warp()
    nkf = normalizing_kalman_filter(base, warp)
    y = 0.3 * jr.normal(jr.key(4), (N_STEPS, N_CHANNELS))

    grads = eqx.filter_grad(lambda m, obs: -m.log_prob(obs))(nkf, y)

    def _norm(tree):
        leaves = [g for g in jax.tree.leaves(tree) if eqx.is_inexact_array(g)]
        assert leaves, "no differentiable leaves found"
        return sum(jnp.sum(g**2) for g in leaves) ** 0.5

    assert _norm(grads.base_dist) > 0.0
    assert _norm(grads.bijection) > 0.0


def test_jit_vmap_over_a_batch_of_series():
    nkf = normalizing_kalman_filter(_unconditional_base(), _elementwise_warp())
    batch = 0.3 * jr.normal(jr.key(5), (16, N_STEPS, N_CHANNELS))
    out = eqx.filter_jit(jax.vmap(nkf.log_prob))(batch)
    assert out.shape == (16,)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_guard_rejects_masked_base_with_unconditional_coupling():
    with pytest.raises(ValueError, match="channel-mixing warp"):
        normalizing_kalman_filter(_conditional_base(), _coupling_warp(jr.key(6)))


def test_guard_allows_masked_base_with_elementwise_warp():
    nkf = normalizing_kalman_filter(_conditional_base(), _elementwise_warp())
    assert nkf.shape == (N_STEPS, N_CHANNELS)


def test_guard_allows_masked_base_with_mask_conditioned_coupling():
    warp = _coupling_warp(jr.key(7), cond_dim=N_CHANNELS)
    assert warp.cond_shape is not None
    nkf = normalizing_kalman_filter(_conditional_base(), warp)
    assert nkf.shape == (N_STEPS, N_CHANNELS)


def test_guard_allows_unmasked_base_with_coupling():
    nkf = normalizing_kalman_filter(_unconditional_base(), _coupling_warp(jr.key(8)))
    assert nkf.shape == (N_STEPS, N_CHANNELS)


def test_guard_is_duck_typed_not_isinstance_based():
    """Any base with a non-None cond_shape trips the guard, gaussx or not.

    The stub below is not a `gaussx.MaskedLGSSM` and shares no base class with
    one, so this asserts the guard reads `cond_shape` rather than a type.
    """

    class _StubConditionalBase(eqx.Module):
        shape: tuple[int, ...] = (N_STEPS, N_CHANNELS)
        cond_shape: tuple[int, ...] | None = (N_STEPS, N_CHANNELS)

    with pytest.raises(ValueError, match="channel-mixing warp"):
        normalizing_kalman_filter(_StubConditionalBase(), _coupling_warp(jr.key(9)))


def test_guard_error_names_the_three_alternatives():
    with pytest.raises(ValueError) as excinfo:
        normalizing_kalman_filter(_conditional_base(), _coupling_warp(jr.key(10)))
    message = str(excinfo.value)
    assert "elementwise warp" in message
    assert "Condition the warp on the mask" in message
    assert "unmasked base" in message


# ---------------------------------------------------------------------------
# Channel-mixing detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "warp_fn",
    [
        lambda: _elementwise_warp(),
        lambda: Identity((N_CHANNELS,)),
        lambda: Vmap(
            RationalQuadraticSpline(knots=6, interval=4),
            in_axes=None,
            axis_size=N_CHANNELS,
        ),
        lambda: Invert(_elementwise_warp()),
    ],
)
def test_elementwise_warps_are_not_flagged(warp_fn):
    assert not _mixes_channels(warp_fn())


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_coupling_warps_are_flagged(seed):
    assert _mixes_channels(_coupling_warp(jr.key(seed)))


def test_chain_containing_a_coupling_is_flagged():
    warp = Chain([_elementwise_warp(), _coupling_warp(jr.key(14))])
    assert _mixes_channels(warp)


def test_unknown_bijection_falls_back_to_the_numerical_probe():
    """A bijection outside the structural allowlist is measured, not trusted."""

    class _RollChannels(eqx.Module):
        """Rotates the channel axis by one: unmistakably channel-mixing."""

        shape: tuple[int, ...] = (N_CHANNELS,)
        cond_shape: tuple[int, ...] | None = None

        def transform_and_log_det(self, x, condition=None):
            return jnp.roll(x, 1), jnp.zeros(())

        def inverse_and_log_det(self, y, condition=None):
            return jnp.roll(y, -1), jnp.zeros(())

        def transform(self, x, condition=None):
            return jnp.roll(x, 1)

    assert _mixes_channels(_RollChannels())


# ---------------------------------------------------------------------------
# Why the guard exists: masking commutes with an elementwise warp, not a coupling
# ---------------------------------------------------------------------------


def _observed_channel_error(warp, key, n_draws, n_channels=6):
    """Error on the OBSERVED channels from masking before vs after the warp.

    Marginalisation is valid only when ``P_m . G^-1 == G^-1_m . P_m``. Applying
    the warp to a masked input and comparing against the masked warp of the
    full input, restricted to the observed channels, measures exactly that.
    """
    k_y, k_m = jr.split(key)
    y = jr.normal(k_y, (n_draws, n_channels))  # (D, M)
    mask = jr.bernoulli(k_m, 0.6, (n_draws, n_channels))  # (D, M), ~40% missing

    inverse = jax.vmap(warp.inverse)
    warp_then_mask = inverse(y) * mask  # (D, M)
    mask_then_warp = inverse(y * mask) * mask  # (D, M)
    # Only the observed entries matter: the rest are marginalised out anyway.
    return jnp.abs(warp_then_mask - mask_then_warp)[mask]


def test_elementwise_warp_commutes_with_masking_exactly():
    warp = _elementwise_warp(n_channels=6)
    errors = _observed_channel_error(warp, jr.key(15), n_draws=4000)
    assert jnp.max(errors) == 0.0


def test_coupling_warp_does_not_commute_with_masking():
    """The measurement the guard exists for -- deleting the guard fails here.

    Reference measurement on a 2-layer coupling flow with M = 6 and ~40%
    missing: median error 0.49 on the observed channels, against a signal
    scale of E|z| = 0.43. The assertions below are deliberately loose in the
    exact value and tight on the conclusion: the corruption is the size of the
    signal, not a rounding artefact.
    """
    k_a, k_b, k_draw = jr.split(jr.key(16), 3)
    # A real 2-layer coupling flow: the Flip between layers is what makes every
    # channel transformed. Without it the untransformed half passes through
    # unchanged in both layers and half the entries are trivially exact.
    warp = Chain(
        [
            _coupling_warp(k_a, n_channels=6),
            Flip((6,)),
            _coupling_warp(k_b, n_channels=6),
        ]
    )
    errors = _observed_channel_error(warp, k_draw, n_draws=4000, n_channels=6)

    signal_scale = jnp.mean(
        jnp.abs(jax.vmap(warp.inverse)(jr.normal(k_draw, (4000, 6))))
    )
    assert jnp.median(errors) > 0.1 * signal_scale
    assert jnp.max(errors) > signal_scale


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_rejects_non_two_dimensional_base():
    with pytest.raises(ValueError, match=r"event shape \(T, M\)"):
        normalizing_kalman_filter(Normal(jnp.zeros(N_CHANNELS)), _elementwise_warp())


def test_rejects_scalar_warp_with_a_pointer_to_vmap():
    """The trap that bites first gets an error that names the fix."""
    with pytest.raises(ValueError, match="Vmap"):
        normalizing_kalman_filter(
            _unconditional_base(), RationalQuadraticSpline(knots=6, interval=4)
        )


def test_rejects_channel_count_mismatch():
    with pytest.raises(ValueError, match="does not match the base's channel"):
        normalizing_kalman_filter(
            _unconditional_base(), _elementwise_warp(n_channels=N_CHANNELS + 1)
        )


def test_rejects_inconsistent_n_steps():
    with pytest.raises(ValueError, match="disagrees with the base's time axis"):
        normalizing_kalman_filter(
            _unconditional_base(), _elementwise_warp(), n_steps=N_STEPS + 1
        )


def test_returns_a_transformed():
    nkf = normalizing_kalman_filter(_unconditional_base(), _elementwise_warp())
    assert isinstance(nkf, Transformed)


# ---------------------------------------------------------------------------
# Against a real gaussx state-space base
# ---------------------------------------------------------------------------


def _gaussx_or_skip():
    """Import gaussx, skipping on ANY failure.

    `pytest.importorskip` only catches ImportError, and gaussx currently fails
    here with an AttributeError instead: it needs matfree>=0.6
    (`sampler_signs`) while gauss_flows is pinned to the pre-0.6
    `sampler_rademacher`.
    """
    try:
        import gaussx
    except Exception as exc:  # gaussx import can fail in several ways here
        pytest.skip(f"gaussx is not importable here ({type(exc).__name__}: {exc})")
    return gaussx


def test_with_a_gaussx_lgssm_base():
    gaussx = _gaussx_or_skip()

    n_state = N_CHANNELS
    base = NumpyroBase(
        dist=gaussx.LGSSM(
            0.9 * jnp.eye(n_state),
            jnp.eye(N_CHANNELS),
            0.1 * jnp.eye(n_state),
            0.1 * jnp.eye(N_CHANNELS),
            jnp.zeros(n_state),
            jnp.eye(n_state),
            n_steps=N_STEPS,
        )
    )
    warp = _elementwise_warp()
    nkf = normalizing_kalman_filter(base, warp)
    assert nkf.shape == (N_STEPS, N_CHANNELS)

    y = 0.3 * jr.normal(jr.key(17), (N_STEPS, N_CHANNELS))
    z, log_det = jax.vmap(warp.inverse_and_log_det)(y)
    expected = base.log_prob(z) + log_det.sum()
    assert jnp.abs(nkf.log_prob(y) - expected) < 1e-13
