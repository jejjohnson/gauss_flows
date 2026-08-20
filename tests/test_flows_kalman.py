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
    Affine as Affine_flowjax,
    Chain,
    Coupling,
    Exp,
    Flip,
    Identity,
    Invert,
    RationalQuadraticSpline,
    Sigmoid,
    Stack,
    Vmap,
)
from flowjax.distributions import Normal, Transformed

from gauss_flows import (
    InverseGaussCDF,
    MixtureGaussianCDF,
    MixtureGaussianCDFCoupling,
    MixtureLogisticCDF,
    NumpyroBase,
    RQSplineMarginal,
    normalizing_kalman_filter,
)
from gauss_flows._src.flows.kalman import (
    _N_PROBE_POINTS,
    _mixes_channels,
    _probe_mixes_channels,
)
from gauss_flows._src.transforms.bijections.linear.rotation import (
    HouseholderRotation,
    OrthogonalRotation,
)


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

    `pytest.importorskip` only catches ImportError, and a gaussx whose own pins
    are unmet can fail partway through its import with other exception types
    (seen in practice as an AttributeError from a matfree version mismatch).
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


# ---------------------------------------------------------------------------
# Masked change of variables: the log-det counts observed channels only
# ---------------------------------------------------------------------------


def _masking_base(n_steps=N_STEPS, n_channels=N_CHANNELS):
    """A base that genuinely marginalises the unobserved entries."""
    return NumpyroBase(
        dist_factory=lambda mask: (
            ndist.Normal(0.0, 1.0).expand((n_steps, n_channels)).mask(mask).to_event(2)
        ),
        event_shape=(n_steps, n_channels),
        cond_shape=(n_steps, n_channels),
    )


def _reference_masked_log_prob(warp, y, mask):
    """The marginal likelihood, written out directly.

    Change of variables applies to the observed coordinates only, so the
    per-channel log-dets are summed under the mask.
    """
    z = jax.vmap(warp.inverse)(y)  # (T, M)
    per_entry = jax.vmap(
        lambda v: jnp.log(jnp.abs(jnp.diag(jax.jacfwd(warp.inverse)(v))))
    )(y)  # (T, M)
    logp = jax.scipy.stats.norm.logpdf(z)  # (T, M)
    return jnp.sum(jnp.where(mask, logp + per_entry, 0.0))


def test_masked_log_prob_is_the_marginal_likelihood():
    """The recommended configuration must return a marginal likelihood.

    A plain `Transformed` sums the log-det over all (T, M) entries, including
    the ones the base marginalised away. That is a data-dependent offset — it
    passed every unmasked test in this file, and was wrong by 1.62 nats on the
    first masked example tried.
    """
    # A non-volume-preserving warp, or the bug is invisible.
    warp = Chain(
        [
            Vmap(
                RationalQuadraticSpline(knots=6, interval=4),
                in_axes=None,
                axis_size=N_CHANNELS,
            ),
            Affine(loc=jnp.zeros(N_CHANNELS), scale=1.5 * jnp.ones(N_CHANNELS)),
        ]
    )
    nkf = normalizing_kalman_filter(_masking_base(), warp)

    y = 0.4 * jr.normal(jr.key(20), (N_STEPS, N_CHANNELS))
    mask = jr.bernoulli(jr.key(21), 0.6, (N_STEPS, N_CHANNELS))

    got = nkf.log_prob(y, condition=mask)
    assert jnp.abs(got - _reference_masked_log_prob(warp, y, mask)) < 1e-5


def test_masked_log_prob_ignores_placeholder_values():
    """Unobserved slots must not move the density, whatever is parked there."""
    warp = Vmap(
        RationalQuadraticSpline(knots=6, interval=4),
        in_axes=None,
        axis_size=N_CHANNELS,
    )
    nkf = normalizing_kalman_filter(_masking_base(), warp)

    y = 0.4 * jr.normal(jr.key(22), (N_STEPS, N_CHANNELS))
    mask = jr.bernoulli(jr.key(23), 0.6, (N_STEPS, N_CHANNELS))
    # Values well inside the spline interval, where its log-det is not constant.
    other = jnp.where(mask, y, 2.5)

    assert jnp.allclose(
        nkf.log_prob(y, condition=mask),
        nkf.log_prob(other, condition=mask),
        atol=1e-6,
    )


def test_all_observed_mask_matches_the_unmasked_log_det():
    """With nothing missing, the masked path must agree with the plain one."""
    warp = _elementwise_warp()
    y = 0.4 * jr.normal(jr.key(24), (N_STEPS, N_CHANNELS))
    full = jnp.ones((N_STEPS, N_CHANNELS), dtype=bool)

    masked = normalizing_kalman_filter(_masking_base(), warp)
    _, log_det = jax.vmap(warp.inverse_and_log_det)(y)
    z = jax.vmap(warp.inverse)(y)
    expected = jnp.sum(jax.scipy.stats.norm.logpdf(z)) + log_det.sum()
    assert jnp.abs(masked.log_prob(y, condition=full) - expected) < 1e-5


def test_unmasked_base_keeps_the_plain_transformed_log_det():
    """The wrapper must not alter the unconditional path."""
    warp = _elementwise_warp()
    nkf = normalizing_kalman_filter(_unconditional_base(), warp)
    y = 0.4 * jr.normal(jr.key(25), (N_STEPS, N_CHANNELS))
    z, log_det = jax.vmap(warp.inverse_and_log_det)(y)
    expected = _unconditional_base().log_prob(z) + log_det.sum()
    assert jnp.abs(nkf.log_prob(y) - expected) < 1e-13


def test_masked_log_prob_is_differentiable_and_jittable():
    warp = _elementwise_warp()
    nkf = normalizing_kalman_filter(_masking_base(), warp)
    y = 0.3 * jr.normal(jr.key(26), (N_STEPS, N_CHANNELS))
    mask = jr.bernoulli(jr.key(27), 0.6, (N_STEPS, N_CHANNELS))

    out = eqx.filter_jit(lambda m, obs, c: m.log_prob(obs, condition=c))(nkf, y, mask)
    assert jnp.isfinite(out)

    grads = eqx.filter_grad(lambda m, obs, c: -m.log_prob(obs, condition=c))(
        nkf, y, mask
    )
    leaves = [g for g in jax.tree.leaves(grads.bijection) if eqx.is_inexact_array(g)]
    assert leaves and sum(jnp.sum(g**2) for g in leaves) > 0.0


# ---------------------------------------------------------------------------
# Classification is by what a warp can represent, not by its initial Jacobian
# ---------------------------------------------------------------------------


def test_warp_that_is_diagonal_only_at_initialisation_is_rejected():
    """OrthogonalRotation probes as the identity, then trains into a rotation.

    Its Cayley parameters start at zero, so a one-time numerical probe sees the
    identity — diagonal — and would admit a warp that violates the masking
    invariant as soon as an optimiser touches it.
    """
    rotation = OrthogonalRotation(shape=(N_CHANNELS,))
    # Diagonal right now ...
    jacobian = jax.jacfwd(rotation.transform)(jnp.zeros(N_CHANNELS))
    off_diagonal = jacobian - jnp.diag(jnp.diagonal(jacobian))
    assert jnp.max(jnp.abs(off_diagonal)) == 0.0
    # ... and refused anyway.
    assert _mixes_channels(rotation)
    with pytest.raises(ValueError, match="follows from its type"):
        normalizing_kalman_filter(_conditional_base(), rotation)


def test_mixing_capable_warps_are_still_fine_over_an_unmasked_base():
    """The restriction is a property of masking, not of the warp alone."""
    for warp in (
        OrthogonalRotation(shape=(N_CHANNELS,)),
        HouseholderRotation(shape=(N_CHANNELS,), n_reflections=2),
    ):
        assert normalizing_kalman_filter(_unconditional_base(), warp).shape == (
            N_STEPS,
            N_CHANNELS,
        )


def test_repo_elementwise_bijections_are_admitted():
    """The package's own marginal bijections must work as warps.

    They are elementwise by the subpackage's documented contract, and they are
    what the docs tell users to reach for — a name-based allowlist silently
    excluded them.
    """
    warps = {
        "MixtureGaussianCDF": MixtureGaussianCDF(shape=(N_CHANNELS,), n_components=4),
        "MixtureLogisticCDF": MixtureLogisticCDF(shape=(N_CHANNELS,), n_components=4),
        "RQSplineMarginal": RQSplineMarginal(n_bins=6, shape=(N_CHANNELS,)),
        "InverseGaussCDF": InverseGaussCDF(shape=(N_CHANNELS,)),
    }
    for name, warp in warps.items():
        assert not _mixes_channels(warp), name
        nkf = normalizing_kalman_filter(_conditional_base(), warp)
        assert nkf.shape == (N_STEPS, N_CHANNELS), name


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("MixtureGaussianCDF", lambda: MixtureGaussianCDF(shape=(6,), n_components=4)),
        ("RQSplineMarginal", lambda: RQSplineMarginal(n_bins=6, shape=(6,))),
    ],
)
def test_repo_elementwise_bijections_really_are_diagonal(name, build):
    """Independent check of the module-path rule's premise."""
    warp = build()
    for seed in range(4):
        x = 0.4 * jr.normal(jr.key(seed), (6,))
        jacobian = jax.jacfwd(warp.transform)(x)
        off_diagonal = jacobian - jnp.diag(jnp.diagonal(jacobian))
        assert jnp.max(jnp.abs(off_diagonal)) == 0.0, name


def test_coupling_variant_of_a_marginal_is_not_admitted():
    """The module-path rule must not leak to the coupling namespace."""
    coupling = MixtureGaussianCDFCoupling(
        jr.key(30), shape=(N_CHANNELS,), n_components=4, nn_width=8, nn_depth=1
    )
    assert _mixes_channels(coupling)


def test_unknown_bijection_is_refused_even_without_parameters():
    """A parameter-free warp can still be input-dependent.

    Having no trainable leaves does not make the Jacobian constant in the
    input, so probe points drawn from one region say nothing about another. The
    shear below is exactly diagonal on the probe points and mixes channels past
    a threshold.
    """
    threshold = 3.0

    class _TailShear(eqx.Module):
        shape: tuple[int, ...] = (N_CHANNELS,)
        cond_shape: tuple[int, ...] | None = None

        def transform(self, x, condition=None):
            # Identity everywhere the probe looks; a shear out in the tail.
            shear = jnp.where(x[0] > threshold, x[0], 0.0)
            return x.at[1].add(shear)

        def transform_and_log_det(self, x, condition=None):
            return self.transform(x), jnp.zeros(())

        def inverse_and_log_det(self, y, condition=None):
            shear = jnp.where(y[0] > threshold, y[0], 0.0)
            return y.at[1].add(-shear), jnp.zeros(())

    warp = _TailShear()
    # Diagonal at every point the probe visits ...
    for seed in range(_N_PROBE_POINTS):
        jacobian = jax.jacfwd(warp.transform)(jr.normal(jr.key(seed), (N_CHANNELS,)))
        off_diagonal = jacobian - jnp.diag(jnp.diagonal(jacobian))
        assert jnp.max(jnp.abs(off_diagonal)) == 0.0
    assert not _probe_mixes_channels(warp)
    # ... and channel-mixing in the tail.
    tail = jnp.array([threshold + 1.0] + [0.0] * (N_CHANNELS - 1))
    tail_jacobian = jax.jacfwd(warp.transform)(tail)
    assert jnp.abs(tail_jacobian[1, 0]) > 0.0
    # So it is refused, probe or no probe.
    assert _mixes_channels(warp)
    with pytest.raises(ValueError, match="follows from its type"):
        normalizing_kalman_filter(_conditional_base(), warp)


def test_assume_elementwise_warp_opts_a_custom_warp_in():
    """The escape hatch for warps the type system cannot recognise."""

    class _ScaleByTwo(eqx.Module):
        shape: tuple[int, ...] = (N_CHANNELS,)
        cond_shape: tuple[int, ...] | None = None

        def transform(self, x, condition=None):
            return 2.0 * x

        def transform_and_log_det(self, x, condition=None):
            return 2.0 * x, jnp.log(2.0) * N_CHANNELS

        def inverse_and_log_det(self, y, condition=None):
            return y / 2.0, -jnp.log(2.0) * N_CHANNELS

    warp = _ScaleByTwo()
    assert _mixes_channels(warp)  # unrecognised type
    with pytest.raises(ValueError, match="follows from its type"):
        normalizing_kalman_filter(_conditional_base(), warp)
    nkf = normalizing_kalman_filter(
        _conditional_base(), warp, assume_elementwise_warp=True
    )
    assert nkf.shape == (N_STEPS, N_CHANNELS)


def test_assume_elementwise_warp_still_refuses_a_refutable_assertion():
    """The opt-in trusts the caller, but not past an immediate contradiction."""
    with pytest.raises(ValueError, match="the assertion is false as written"):
        normalizing_kalman_filter(
            _conditional_base(),
            _coupling_warp(jr.key(31)),
            assume_elementwise_warp=True,
        )


def test_assume_elementwise_warp_masks_the_log_det_too():
    """An opted-in warp must get the same marginal-likelihood treatment."""

    class _ScaleByThree(eqx.Module):
        shape: tuple[int, ...] = (N_CHANNELS,)
        cond_shape: tuple[int, ...] | None = None

        def transform(self, x, condition=None):
            return 3.0 * x

        def transform_and_log_det(self, x, condition=None):
            return 3.0 * x, jnp.log(3.0) * N_CHANNELS

        def inverse(self, y, condition=None):
            return y / 3.0

        def inverse_and_log_det(self, y, condition=None):
            return y / 3.0, -jnp.log(3.0) * N_CHANNELS

    warp = _ScaleByThree()
    nkf = normalizing_kalman_filter(_masking_base(), warp, assume_elementwise_warp=True)
    y = 0.4 * jr.normal(jr.key(32), (N_STEPS, N_CHANNELS))
    mask = jr.bernoulli(jr.key(33), 0.6, (N_STEPS, N_CHANNELS))

    z = y / 3.0
    expected = jnp.sum(
        jnp.where(mask, jax.scipy.stats.norm.logpdf(z) - jnp.log(3.0), 0.0)
    )
    assert jnp.abs(nkf.log_prob(y, condition=mask) - expected) < 1e-5


def test_scalar_bijection_lifted_by_vmap_is_admitted():
    """`Vmap` over a scalar bijection is elementwise whatever it contains.

    Each mapped block has event shape `()`, so the Jacobian blocks are 1x1 and
    the whole thing is diagonal by construction — no allowlist entry and no
    `assume_elementwise_warp` needed for the common "lift my custom scalar
    transform over the channels" path.
    """

    class _ScalarCube(eqx.Module):
        """An unrecognised scalar bijection: y = x**3."""

        shape: tuple[int, ...] = ()
        cond_shape: tuple[int, ...] | None = None

        def transform(self, x, condition=None):
            return x**3

        def transform_and_log_det(self, x, condition=None):
            return x**3, jnp.log(3.0 * x**2)

        def inverse(self, y, condition=None):
            return jnp.sign(y) * jnp.abs(y) ** (1.0 / 3.0)

        def inverse_and_log_det(self, y, condition=None):
            x = self.inverse(y)
            return x, -jnp.log(3.0 * x**2)

    # Rejected on its own — the type says nothing about it.
    assert _mixes_channels(_ScalarCube())
    # Admitted once lifted over the channel axis.
    lifted = Vmap(_ScalarCube(), in_axes=None, axis_size=N_CHANNELS)
    assert lifted.shape == (N_CHANNELS,)
    assert not _mixes_channels(lifted)
    assert normalizing_kalman_filter(_conditional_base(), lifted).shape == (
        N_STEPS,
        N_CHANNELS,
    )


def test_lifted_scalar_warp_gets_the_masked_log_det():
    """The Vmap-of-scalar path must still produce a marginal likelihood."""
    scale = 2.0

    class _ScalarScale(eqx.Module):
        shape: tuple[int, ...] = ()
        cond_shape: tuple[int, ...] | None = None

        def transform(self, x, condition=None):
            return scale * x

        def transform_and_log_det(self, x, condition=None):
            return scale * x, jnp.log(scale)

        def inverse(self, y, condition=None):
            return y / scale

        def inverse_and_log_det(self, y, condition=None):
            return y / scale, -jnp.log(scale)

    warp = Vmap(_ScalarScale(), in_axes=None, axis_size=N_CHANNELS)
    nkf = normalizing_kalman_filter(_masking_base(), warp)

    y = 0.4 * jr.normal(jr.key(34), (N_STEPS, N_CHANNELS))
    mask = jr.bernoulli(jr.key(35), 0.6, (N_STEPS, N_CHANNELS))
    expected = jnp.sum(
        jnp.where(mask, jax.scipy.stats.norm.logpdf(y / scale) - jnp.log(scale), 0.0)
    )
    assert jnp.abs(nkf.log_prob(y, condition=mask) - expected) < 1e-5


def test_inverted_gaussianiser_is_the_warp_that_maps_base_to_data():
    """The recommended `Invert(MixtureGaussianCDF(...))` must be admitted.

    `MixtureGaussianCDF.transform` runs data → Gaussian, so used directly as a
    warp it would model the pushforward through the Gaussianiser rather than
    the observations. `Invert` puts it the right way round, and the guard must
    accept it either way (both are elementwise).
    """
    gaussianiser = MixtureGaussianCDF(shape=(N_CHANNELS,), n_components=4)
    warp = Invert(gaussianiser)
    assert not _mixes_channels(warp)

    nkf = normalizing_kalman_filter(_conditional_base(), warp)
    assert nkf.shape == (N_STEPS, N_CHANNELS)

    # The direction claim itself: warp.transform is the Gaussianiser's inverse.
    z = 0.3 * jr.normal(jr.key(36), (N_CHANNELS,))
    assert jnp.allclose(warp.transform(z), gaussianiser.inverse(z), atol=1e-5)


def test_allowlist_matches_class_identity_not_class_name():
    """A look-alike named `Affine` must not inherit flowjax's guarantee.

    Admitting on `type(x).__name__` would route a dense map through
    `_MaskedLogDet`, whose `J @ 1 == diag(J)` identity holds only for a
    genuinely diagonal Jacobian — so the marginal likelihood would be wrong
    with no error anywhere.
    """

    class Affine(eqx.Module):
        """Dense, despite the name."""

        matrix: jnp.ndarray
        shape: tuple[int, ...] = (N_CHANNELS,)
        cond_shape: tuple[int, ...] | None = None

        def transform(self, x, condition=None):
            return self.matrix @ x

        def transform_and_log_det(self, x, condition=None):
            return self.matrix @ x, jnp.linalg.slogdet(self.matrix)[1]

        def inverse_and_log_det(self, y, condition=None):
            return jnp.linalg.solve(self.matrix, y), -jnp.linalg.slogdet(self.matrix)[1]

    impostor = Affine(matrix=jnp.eye(N_CHANNELS) + 0.3)
    assert type(impostor).__name__ == "Affine"
    assert _mixes_channels(impostor)
    with pytest.raises(ValueError, match="follows from its type"):
        normalizing_kalman_filter(_conditional_base(), impostor)

    # The real flowjax Affine is still admitted.
    assert not _mixes_channels(
        Affine_flowjax(loc=jnp.zeros(N_CHANNELS), scale=jnp.ones(N_CHANNELS))
    )


def test_masking_preserves_a_bespoke_log_det_when_nothing_is_missing():
    """An all-observed mask must reproduce the ordinary density exactly.

    `HistogramCDF(method="monotonic")` reports an inverse log-det of
    `-log(fwd'(x_rec))` while differentiating `inverse` goes through a
    separately fitted interpolator — measured ~4e-3 apart per timestep. Rebuilding
    the total from per-channel derivatives would quietly change the model and
    accumulate that gap over the time axis.
    """
    pytest.importorskip("interpax")
    from gauss_flows import HistogramCDF

    data = jr.normal(jr.key(37), (2000, N_CHANNELS))
    warp = HistogramCDF(n_bins=32, shape=(N_CHANNELS,), method="monotonic").fit(data)

    # The premise: declared and autodiff log-dets genuinely differ here.
    y_one = jnp.array([0.2, 0.5, 0.8, 0.35])[:N_CHANNELS]
    _, declared = warp.inverse_and_log_det(y_one)
    _, diagonal = jax.jvp(warp.inverse, (y_one,), (jnp.ones_like(y_one),))
    assert jnp.abs(declared - jnp.sum(jnp.log(jnp.abs(diagonal)))) > 1e-4

    masked = normalizing_kalman_filter(_masking_base(), warp)
    plain = normalizing_kalman_filter(_unconditional_base(), warp)

    y = jr.uniform(jr.key(38), (N_STEPS, N_CHANNELS), minval=0.05, maxval=0.95)
    full = jnp.ones((N_STEPS, N_CHANNELS), dtype=bool)
    assert jnp.abs(masked.log_prob(y, condition=full) - plain.log_prob(y)) < 1e-4


def test_masking_still_drops_unobserved_channels_for_a_bespoke_log_det():
    """Preserving the declared total must not stop the mask from masking."""
    pytest.importorskip("interpax")
    from gauss_flows import HistogramCDF

    data = jr.normal(jr.key(39), (2000, N_CHANNELS))
    warp = HistogramCDF(n_bins=32, shape=(N_CHANNELS,), method="monotonic").fit(data)
    nkf = normalizing_kalman_filter(_masking_base(), warp)

    y = jr.uniform(jr.key(40), (N_STEPS, N_CHANNELS), minval=0.05, maxval=0.95)
    full = jnp.ones((N_STEPS, N_CHANNELS), dtype=bool)
    partial = jr.bernoulli(jr.key(41), 0.6, (N_STEPS, N_CHANNELS))

    assert not jnp.allclose(
        nkf.log_prob(y, condition=full), nkf.log_prob(y, condition=partial)
    )
    assert jnp.isfinite(nkf.log_prob(y, condition=partial))


def test_self_consistent_warp_satisfies_both_masking_properties():
    """With a self-consistent warp there is no trade-off to make.

    `HistogramCDF(method="linear")` reports exactly the autodiff derivative, so
    the masked density both matches the unmasked path when nothing is missing
    and ignores whatever sits in the unobserved slots.
    """
    pytest.importorskip("interpax")
    from gauss_flows import HistogramCDF

    data = jr.normal(jr.key(42), (2000, N_CHANNELS))
    warp = HistogramCDF(n_bins=32, shape=(N_CHANNELS,), method="linear").fit(data)

    y_one = jnp.array([0.2, 0.5, 0.8, 0.35])[:N_CHANNELS]
    _, declared = warp.inverse_and_log_det(y_one)
    _, diagonal = jax.jvp(warp.inverse, (y_one,), (jnp.ones_like(y_one),))
    assert jnp.abs(declared - jnp.sum(jnp.log(jnp.abs(diagonal)))) < 1e-5

    masked = normalizing_kalman_filter(_masking_base(), warp)
    plain = normalizing_kalman_filter(_unconditional_base(), warp)
    y = jr.uniform(jr.key(43), (N_STEPS, N_CHANNELS), minval=0.05, maxval=0.95)
    full = jnp.ones((N_STEPS, N_CHANNELS), dtype=bool)
    partial = jr.bernoulli(jr.key(44), 0.6, (N_STEPS, N_CHANNELS))

    # Agrees with the unmasked path when nothing is missing ...
    assert jnp.abs(masked.log_prob(y, condition=full) - plain.log_prob(y)) < 1e-4
    # ... and ignores the placeholders when something is.
    other = jnp.where(partial, y, 0.5)
    assert (
        jnp.abs(
            masked.log_prob(y, condition=partial)
            - masked.log_prob(other, condition=partial)
        )
        < 1e-4
    )


# ---------------------------------------------------------------------------
# Placeholders that would break a bounded warp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "build_warp", "to_data"),
    [
        ("Exp", lambda: Vmap(Exp(), in_axes=None, axis_size=N_CHANNELS), jnp.exp),
        (
            "Sigmoid",
            lambda: Vmap(Sigmoid(), in_axes=None, axis_size=N_CHANNELS),
            jax.nn.sigmoid,
        ),
    ],
)
@pytest.mark.parametrize("placeholder", [jnp.nan, -999.0, -1.0])
def test_placeholders_outside_the_warp_support_do_not_contaminate(
    name, build_warp, to_data, placeholder
):
    """A missing entry must not take the whole timestep down with it.

    The bijection reduces its per-channel log-dets to a scalar before anything
    can be masked, so one NaN or out-of-range sentinel makes the entire
    timestep's log-determinant NaN — measured on both warps below, whose
    inverses take a logarithm of the placeholder. NaN and -999 are the two
    commonest ways to spell "missing", so this is the ordinary case, not a
    corner one.
    """
    warp = build_warp()
    nkf = normalizing_kalman_filter(_masking_base(), warp)
    latent = 0.3 * jr.normal(jr.key(45), (N_STEPS, N_CHANNELS))
    clean = to_data(latent)
    mask = jr.bernoulli(jr.key(46), 0.6, (N_STEPS, N_CHANNELS))

    reference = nkf.log_prob(clean, condition=mask)
    assert jnp.isfinite(reference), name

    contaminated = jnp.where(mask, clean, placeholder)
    assert jnp.isfinite(nkf.log_prob(contaminated, condition=mask)), name
    assert jnp.allclose(
        nkf.log_prob(contaminated, condition=mask), reference, atol=1e-5
    ), name


def test_stack_of_scalar_marginals_is_admitted():
    """`Stack` puts each scalar block on its own channel — strictly diagonal.

    Heterogeneous marginals for mixed-support coordinates are a real pattern:
    the conjugate-filter tests build exactly `Stack([Exp(), Sigmoid()])`.
    """
    warp = Stack([Exp(), Sigmoid(), Exp(), Sigmoid()][:N_CHANNELS])
    assert warp.shape == (N_CHANNELS,)
    assert not _mixes_channels(warp)

    # Independently: the Jacobian really is diagonal.
    point = 0.3 * jr.normal(jr.key(47), (N_CHANNELS,))
    jacobian = jax.jacfwd(warp.transform)(point)
    off_diagonal = jacobian - jnp.diag(jnp.diagonal(jacobian))
    assert jnp.max(jnp.abs(off_diagonal)) == 0.0

    assert normalizing_kalman_filter(_conditional_base(), warp).shape == (
        N_STEPS,
        N_CHANNELS,
    )


def test_container_subclasses_are_not_trusted_by_isinstance():
    """A `Chain` subclass may override the map its children imply.

    Inspecting only the stored children would let a dense override inherit the
    elementwise verdict of its elementwise members.
    """

    class _DenseChain(Chain):
        """Elementwise children, dense map."""

        def transform(self, x, condition=None):
            return jnp.roll(x, 1)

        def transform_and_log_det(self, x, condition=None):
            return jnp.roll(x, 1), jnp.zeros(())

        def inverse_and_log_det(self, y, condition=None):
            return jnp.roll(y, -1), jnp.zeros(())

    impostor = _DenseChain([_elementwise_warp()])
    assert all(
        not _mixes_channels(member) for member in impostor.bijections
    )  # children are fine
    assert _mixes_channels(impostor)  # the subclass is not
    with pytest.raises(ValueError, match="follows from its type"):
        normalizing_kalman_filter(_conditional_base(), impostor)

    # The real Chain of the same members is still admitted.
    assert not _mixes_channels(Chain([_elementwise_warp()]))


def test_histogram_cdf_needs_a_probit_step_to_be_a_valid_warp():
    """`Invert(HistogramCDF(...))` alone is not a Gaussian-to-data warp.

    `HistogramCDF` maps data to a *uniform* variable, so its inverse expects
    ``[0, 1]``. A Gaussian base supplies all of ℝ, and outside ``[0, 1]`` the
    map is clamped to the fitted bin edges: measured log-determinant of
    ``-inf`` and a round-trip error of 2.5 — i.e. not a bijection there, so the
    density is not normalised. Composing with `InverseGaussCDF` supplies the
    probit step `MixtureGaussianCDF` already has built in.
    """
    pytest.importorskip("interpax")
    from gauss_flows import HistogramCDF, InverseGaussCDF

    data = jr.normal(jr.key(48), (2000, N_CHANNELS))
    histogram = HistogramCDF(n_bins=32, shape=(N_CHANNELS,), method="linear").fit(data)
    outside = jnp.array([-2.5, 0.4, 3.1, -1.7])[:N_CHANNELS]

    # The premise: on [0, 1] the bare inverse is fine ...
    bare = Invert(histogram)
    inside = jnp.array([0.1, 0.2, 0.3, 0.4])[:N_CHANNELS]
    _, inside_log_det = bare.transform_and_log_det(inside)
    assert jnp.isfinite(inside_log_det)

    # ... and off it, the map degenerates.
    bare_y, bare_log_det = bare.transform_and_log_det(outside)
    assert jnp.isneginf(bare_log_det)
    assert jnp.max(jnp.abs(bare.inverse(bare_y) - outside)) > 1.0

    # The recommended composition stays a bijection across the base's range.
    warp = Invert(Chain([histogram, InverseGaussCDF(shape=(N_CHANNELS,))]))
    assert not _mixes_channels(warp)
    y, log_det = warp.transform_and_log_det(outside)
    assert jnp.isfinite(log_det)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.allclose(warp.inverse(y), outside, atol=1e-3)

    nkf = normalizing_kalman_filter(_masking_base(), warp)
    latent = 0.3 * jr.normal(jr.key(49), (N_STEPS, N_CHANNELS))
    observations = jax.vmap(warp.transform)(latent)
    mask = jr.bernoulli(jr.key(50), 0.6, (N_STEPS, N_CHANNELS))
    assert jnp.isfinite(nkf.log_prob(observations, condition=mask))


def test_sampling_under_a_partial_mask_keeps_the_imputation_draws():
    """Missing positions must carry the base's predictive draw, not a constant.

    Sampling reaches the warp with a latent the base has already drawn for
    *every* entry — including the unobserved ones, where it is the state-space
    model's imputation. Substituting a reference there (which the likelihood
    path must do, since observations carry placeholders) collapses every
    missing position onto `warp.transform(0)` with zero variance.
    """
    warp = Vmap(Exp(), in_axes=None, axis_size=N_CHANNELS)
    nkf = normalizing_kalman_filter(_masking_base(), warp)
    mask = jr.bernoulli(jr.key(51), 0.5, (N_STEPS, N_CHANNELS))
    assert not jnp.all(mask) and jnp.any(mask)  # a genuinely partial mask

    samples = nkf.sample(jr.key(52), (200,), condition=mask)  # (200, T, M)
    assert samples.shape == (200, N_STEPS, N_CHANNELS)

    spread = samples.std(axis=0)  # (T, M)
    # Unobserved entries vary as much as observed ones ...
    assert jnp.min(spread[~mask]) > 0.0
    assert spread[~mask].mean() > 0.5 * spread[mask].mean()
    # ... and are not pinned to the substitution reference.
    assert not jnp.allclose(samples[:, ~mask], warp.transform(jnp.zeros(N_CHANNELS))[0])
