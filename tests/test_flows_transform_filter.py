"""Tests for `TransformFilter` — filtering in warped coordinates.

The measured claims mirror gh-146: on a strictly positive state with
multiplicative noise, latent-coordinate filtering buys **calibration and
support** (zero credible-interval zero-crossings, zero negative ensemble
members, lower NLPD), *not* point accuracy (RMSE is a wash). Reference
values were measured on this exact problem in float64:
EKF latent RMSE 0.5407 / physical 0.5483; zero-crossings 0/80 vs 54/80;
NLPD 0.24 vs 1.35; ensemble negatives 0/8000 vs 743/8000.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import AbstractBijection, Affine, Chain, Exp

from gauss_flows import ConjugateTransformFilter, TransformFilter


# `pytest.importorskip` only catches ImportError. A gaussx whose own pins are
# unmet can fail partway through its import with other exception types (seen
# in practice as an AttributeError from a matfree version mismatch). Skip on
# any import failure so the reason is reported rather than collapsing the
# whole collection.
try:
    import gaussx
    import lineax as lx
except Exception as exc:  # gaussx import can fail in several ways here
    pytest.skip(
        f"gaussx is not importable here ({type(exc).__name__}: {exc}); "
        "install it into an environment where its pins are resolved to run "
        "the TransformFilter tests.",
        allow_module_level=True,
    )


N = 2
T = 40
CARRYING = jnp.array([1.2, 0.6])
GROWTH = 0.35
Q_LATENT = 0.35
R_LATENT = 0.60
Z_95 = 1.959964


def growth_map(x):
    """Ricker-type saturating growth: strictly positive-preserving."""
    return x * jnp.exp(GROWTH * (1.0 - x / CARRYING))


def observe(x):
    return x


def _simulate():
    """Truth in latent (log) coordinates with Gaussian noise there.

    The latent model is exactly a nonlinear-Gaussian SSM, so the latent
    filter faces its own model while the physical filter faces
    multiplicative noise approximated additively at the operating point —
    the fair-comparison setup of gh-146.
    """
    step_keys = jr.split(jr.key(0), 2)
    process_keys = jr.split(step_keys[0], T)
    obs_keys = jr.split(step_keys[1], T)
    z = jnp.log(CARRYING)
    latents, observations = [], []
    for t in range(T):
        z = jnp.log(growth_map(jnp.exp(z))) + jnp.sqrt(Q_LATENT) * jr.normal(
            process_keys[t], (N,)
        )
        y = jnp.exp(z) * jnp.exp(jnp.sqrt(R_LATENT) * jr.normal(obs_keys[t], (N,)))
        latents.append(z)
        observations.append(y)
    return jnp.exp(jnp.stack(latents)), jnp.stack(observations)  # (T, N) each


TRUTH, OBSERVATIONS = _simulate()
INIT_MEAN = jnp.log(CARRYING)
INIT_COV = 0.5 * jnp.eye(N)
# Physical-space competitor noise: matched at the operating point x = K.
Q_PHYS = jnp.diag(CARRYING**2 * Q_LATENT)
R_PHYS = jnp.diag(CARRYING**2 * R_LATENT)


def _warp():
    return Exp(shape=(N,))


def _mixing_warp():
    """A channel-mixing bijection: unmistakably non-elementwise."""

    class RollChannels(AbstractBijection):
        shape: tuple[int, ...] = (N,)
        cond_shape: None = None

        def transform_and_log_det(self, x, condition=None):
            return jnp.roll(x, 1), jnp.zeros(())

        def inverse_and_log_det(self, y, condition=None):
            return jnp.roll(y, -1), jnp.zeros(())

    return RollChannels()


@pytest.fixture(scope="module")
def latent_states():
    """Latent-frame filter runs, one per integrator."""
    filt = TransformFilter(warp_state=_warp(), warp_obs=_warp())
    return {
        name: filt.filter(
            growth_map,
            observe,
            Q_LATENT * jnp.eye(N),
            R_LATENT * jnp.eye(N),
            OBSERVATIONS,
            INIT_MEAN,
            INIT_COV,
            integrator=integrator,
        )
        for name, integrator in [
            ("ekf", gaussx.TaylorIntegrator()),
            ("ukf", gaussx.UnscentedIntegrator()),
        ]
    }


@pytest.fixture(scope="module")
def physical_states():
    """Physical-coordinate competitor runs, matched at the operating point."""
    return {
        name: gaussx.nonlinear_kalman_filter(
            growth_map,
            observe,
            Q_PHYS,
            R_PHYS,
            OBSERVATIONS,
            CARRYING,
            jnp.diag(CARRYING**2) * 0.5,
            integrator=integrator,
        )
        for name, integrator in [
            ("ekf", gaussx.TaylorIntegrator()),
            ("ukf", gaussx.UnscentedIntegrator()),
        ]
    }


# ---------------------------------------------------------------------------
# Composition and identity reduction
# ---------------------------------------------------------------------------


def test_latent_dynamics_composition_is_exact():
    """f_lat(z) equals log(f_phys(exp(z))) exactly — composition, not approx."""
    warp = _warp()
    z = jr.normal(jr.key(9), (N,))
    composed = warp.inverse(growth_map(warp.transform(z)))
    direct = jnp.log(growth_map(jnp.exp(z)))
    assert jnp.max(jnp.abs(composed - direct)) <= 1e-12  # measured 0.0


@pytest.mark.parametrize("noise_frame", ["latent", "physical"])
def test_identity_warp_reproduces_the_inner_filter_bit_for_bit(noise_frame):
    """No warps -> the wrapper IS gaussx.nonlinear_kalman_filter, both frames.

    The physical-noise loop linearises the warps per step; for the identity
    the Jacobians are exactly the identity matrix and the reduction is
    bitwise, not approximate.
    """
    direct = gaussx.nonlinear_kalman_filter(
        growth_map,
        observe,
        Q_PHYS,
        R_PHYS,
        OBSERVATIONS,
        CARRYING,
        jnp.diag(CARRYING**2) * 0.5,
    )
    wrapped = TransformFilter(noise_frame=noise_frame).filter(
        growth_map,
        observe,
        Q_PHYS,
        R_PHYS,
        OBSERVATIONS,
        CARRYING,
        jnp.diag(CARRYING**2) * 0.5,
    )
    assert jnp.array_equal(wrapped.filtered_means, direct.filtered_means)
    assert jnp.array_equal(wrapped.filtered_covs, direct.filtered_covs)
    assert jnp.array_equal(wrapped.predicted_means, direct.predicted_means)
    assert jnp.array_equal(wrapped.log_likelihood, direct.log_likelihood)


# ---------------------------------------------------------------------------
# The headline: support and calibration, not point accuracy
# ---------------------------------------------------------------------------


def _physical_moments(filt, state):
    mean, _ = filt.predictive(state.filtered_means, state.filtered_covs)
    lo, hi = filt.predictive_interval(state.filtered_means, state.filtered_covs)
    return mean, lo, hi


@pytest.mark.parametrize("name", ["ekf", "ukf"])
def test_zero_intervals_cross_zero_in_latent_coordinates(
    name, latent_states, physical_states
):
    """Latent 95% intervals respect the support; physical ones do not.

    Measured: 0/80 latent zero-crossings vs 54/80 (EKF) and 57/80 (UKF) in
    float64, 41 and 44 in float32.
    """
    filt = TransformFilter(warp_state=_warp(), warp_obs=_warp())
    _, lo, _ = _physical_moments(filt, latent_states[name])
    assert int(jnp.sum(lo <= 0.0)) == 0

    phys = physical_states[name]
    scale = jnp.sqrt(jax.vmap(jnp.diag)(phys.filtered_covs))
    physical_lo = phys.filtered_means - Z_95 * scale
    assert int(jnp.sum(physical_lo <= 0.0)) >= 35


def _nlpd_latent(state, truth):
    def one(mean, cov, x):
        z = jnp.log(x)
        residual = z - mean
        _, logdet = jnp.linalg.slogdet(cov)
        gaussian = 0.5 * (
            residual @ jnp.linalg.solve(cov, residual)
            + logdet
            + N * jnp.log(2 * jnp.pi)
        )
        return gaussian + jnp.sum(z)  # + log|dx/dz| change of variables

    return jnp.mean(jax.vmap(one)(state.filtered_means, state.filtered_covs, truth))


def _nlpd_physical(state, truth):
    def one(mean, cov, x):
        residual = x - mean
        _, logdet = jnp.linalg.slogdet(cov)
        return 0.5 * (
            residual @ jnp.linalg.solve(cov, residual)
            + logdet
            + N * jnp.log(2 * jnp.pi)
        )

    return jnp.mean(jax.vmap(one)(state.filtered_means, state.filtered_covs, truth))


@pytest.mark.parametrize("name", ["ekf", "ukf"])
def test_latent_nlpd_beats_physical(name, latent_states, physical_states):
    """Predictive density improves decisively.

    Measured: 0.24 vs 1.35 in float64, 0.91 vs 2.23 in float32 (the
    simulated trajectory itself differs between precisions).
    """
    latent = float(_nlpd_latent(latent_states[name], TRUTH))
    physical = float(_nlpd_physical(physical_states[name], TRUTH))
    assert latent < 1.2
    assert latent < physical - 0.8


@pytest.mark.parametrize("name", ["ekf", "ukf"])
def test_no_point_accuracy_win(name, latent_states, physical_states):
    """The honest negative of gh-146: RMSE is a wash, not a win.

    This feature buys calibration and support, not point accuracy — a
    future contributor seeing comparable RMSE should not "fix" it.
    Measured RMSE: latent 0.5407/0.5243 vs physical 0.5483/0.5399.
    """
    filt = TransformFilter(warp_state=_warp(), warp_obs=_warp())
    latent_mean, _, _ = _physical_moments(filt, latent_states[name])
    latent_rmse = float(jnp.sqrt(jnp.mean((latent_mean - TRUTH) ** 2)))
    physical_rmse = float(
        jnp.sqrt(jnp.mean((physical_states[name].filtered_means - TRUTH) ** 2))
    )
    assert latent_rmse > 0.85 * physical_rmse
    assert latent_rmse < 1.25 * physical_rmse


# ---------------------------------------------------------------------------
# predictive / predictive_interval
# ---------------------------------------------------------------------------


def test_predictive_returns_the_quadrature_mean_not_the_median(latent_states):
    """Trap 1 of gh-146: Gamma(E[z]) is the pushforward median, not the mean."""
    filt = TransformFilter(warp_state=_warp())
    mean = latent_states["ukf"].filtered_means[5]
    cov = latent_states["ukf"].filtered_covs[5]
    quad_mean, quad_cov = filt.predictive(mean, cov)
    median = jnp.exp(mean)

    samples = jnp.exp(jr.multivariate_normal(jr.key(7), mean, cov, (200_000,)))
    mc_mean = samples.mean(axis=0)
    assert jnp.allclose(quad_mean, mc_mean, rtol=0.01)
    assert jnp.allclose(jnp.diag(quad_cov), samples.var(axis=0), rtol=0.05)
    # The median is measurably below the mean for a lognormal pushforward.
    assert jnp.all(quad_mean > median * 1.005)


def test_predictive_handles_time_batches(latent_states):
    filt = TransformFilter(warp_state=_warp())
    state = latent_states["ekf"]
    mean, cov = filt.predictive(state.filtered_means, state.filtered_covs)
    assert mean.shape == (T, N)
    assert cov.shape == (T, N, N)
    single_mean, _ = filt.predictive(state.filtered_means[3], state.filtered_covs[3])
    assert jnp.allclose(single_mean, mean[3], rtol=1e-6)


def test_predictive_is_identity_without_a_state_warp(latent_states):
    filt = TransformFilter()
    state = latent_states["ekf"]
    mean, cov = filt.predictive(state.filtered_means, state.filtered_covs)
    assert mean is state.filtered_means
    assert cov is state.filtered_covs


def test_predictive_interval_is_positive_and_sorted(latent_states):
    filt = TransformFilter(warp_state=_warp())
    state = latent_states["ekf"]
    lo, hi = filt.predictive_interval(state.filtered_means, state.filtered_covs)
    assert lo.shape == (T, N)
    assert jnp.all(lo > 0.0)  # support respected by construction
    assert jnp.all(lo < hi)


def test_predictive_interval_sorts_a_decreasing_warp():
    """A decreasing elementwise warp flips the endpoints; they come back sorted."""
    # flowjax constrains Affine scales positive at construction; a negative
    # scale is set post-construction, per its own error message's advice.
    negate = eqx.tree_at(
        lambda affine: affine.scale,
        Affine(loc=jnp.zeros(N), scale=jnp.ones(N)),
        replace=-jnp.ones(N),
    )
    decreasing = Chain([Exp(shape=(N,)), negate])
    filt = TransformFilter(warp_state=decreasing)
    lo, hi = filt.predictive_interval(jnp.zeros(N), jnp.eye(N))
    assert jnp.all(lo < hi)
    assert jnp.all(hi < 0.0)  # image of -exp is the negative half-line


def test_predictive_interval_rejects_a_channel_mixing_warp():
    filt = TransformFilter(warp_state=_mixing_warp())
    with pytest.raises(ValueError, match="elementwise warp_state"):
        filt.predictive_interval(jnp.zeros(N), jnp.eye(N))


def test_predictive_interval_rejects_a_bad_level():
    filt = TransformFilter(warp_state=_warp())
    with pytest.raises(ValueError, match="level"):
        filt.predictive_interval(jnp.zeros(N), jnp.eye(N), level=1.5)


# ---------------------------------------------------------------------------
# noise_frame="physical"
# ---------------------------------------------------------------------------


def test_physical_noise_frame_differs_for_a_real_warp():
    """The per-step linearisation is a different (approximate) model."""
    shared = dict(
        integrator=gaussx.UnscentedIntegrator(),
    )
    latent = TransformFilter(warp_state=_warp(), warp_obs=_warp()).filter(
        growth_map,
        observe,
        Q_PHYS,
        R_PHYS,
        OBSERVATIONS,
        INIT_MEAN,
        INIT_COV,
        **shared,
    )
    physical = TransformFilter(
        warp_state=_warp(), warp_obs=_warp(), noise_frame="physical"
    ).filter(
        growth_map,
        observe,
        Q_PHYS,
        R_PHYS,
        OBSERVATIONS,
        INIT_MEAN,
        INIT_COV,
        **shared,
    )
    difference = jnp.max(jnp.abs(latent.filtered_means - physical.filtered_means))
    assert difference > 1e-3  # measured 0.070
    assert jnp.all(jnp.isfinite(physical.filtered_means))
    assert bool(jnp.isfinite(physical.log_likelihood))


def test_physical_noise_frame_accepts_a_time_varying_noise_sequence():
    q_seq = jnp.broadcast_to(Q_PHYS, (T, N, N))
    state = TransformFilter(
        warp_state=_warp(), warp_obs=_warp(), noise_frame="physical"
    ).filter(growth_map, observe, q_seq, R_PHYS, OBSERVATIONS, INIT_MEAN, INIT_COV)
    assert jnp.all(jnp.isfinite(state.filtered_means))


def test_physical_noise_frame_rejects_a_broadcastable_noise_shape():
    filt = TransformFilter(warp_state=_warp(), noise_frame="physical")
    with pytest.raises(ValueError, match="process_noise"):
        filt.filter(
            growth_map,
            observe,
            jnp.ones((1, 1)),
            R_PHYS,
            OBSERVATIONS,
            INIT_MEAN,
            INIT_COV,
        )


def test_noise_frame_is_validated_at_construction():
    with pytest.raises(ValueError, match="noise_frame"):
        TransformFilter(noise_frame="banana")


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noise_frame", ["latent", "physical"])
def test_step_mask_gates_out_garbage_observations(noise_frame):
    """A (T,) mask skips whole steps: NaN placeholders there stay inert."""
    mask = jnp.arange(T) % 3 != 0
    corrupted = jnp.where(mask[:, None], OBSERVATIONS, jnp.nan)
    state = TransformFilter(
        warp_state=_warp(), warp_obs=_warp(), noise_frame=noise_frame
    ).filter(
        growth_map,
        observe,
        Q_PHYS if noise_frame == "physical" else Q_LATENT * jnp.eye(N),
        R_PHYS if noise_frame == "physical" else R_LATENT * jnp.eye(N),
        corrupted,
        INIT_MEAN,
        INIT_COV,
        mask=mask,
    )
    assert jnp.all(jnp.isfinite(state.filtered_means))
    # Gated steps keep the prediction untouched.
    gated = ~mask
    assert jnp.allclose(state.filtered_means[gated], state.predicted_means[gated])


@pytest.mark.parametrize("noise_frame", ["latent", "physical"])
def test_channel_mask_with_an_elementwise_obs_warp_stays_finite(noise_frame):
    """A (T, M) mask with NaN placeholders survives an elementwise warp."""
    mask = jnp.ones((T, N), dtype=bool).at[::4, 1].set(False)
    corrupted = jnp.where(mask, OBSERVATIONS, jnp.nan)
    state = TransformFilter(
        warp_state=_warp(), warp_obs=_warp(), noise_frame=noise_frame
    ).filter(
        growth_map,
        observe,
        Q_PHYS if noise_frame == "physical" else Q_LATENT * jnp.eye(N),
        R_PHYS if noise_frame == "physical" else R_LATENT * jnp.eye(N),
        corrupted,
        INIT_MEAN,
        INIT_COV,
        mask=mask,
    )
    assert jnp.all(jnp.isfinite(state.filtered_means))
    assert bool(jnp.isfinite(state.log_likelihood))


def test_channel_mask_rejects_a_channel_mixing_obs_warp():
    """The warp runs before the mask, so mixing would smear placeholders."""
    filt = TransformFilter(warp_state=_warp(), warp_obs=_mixing_warp())
    mask = jnp.ones((T, N), dtype=bool)
    with pytest.raises(ValueError, match="per-channel observation mask"):
        filt.filter(
            growth_map,
            observe,
            Q_LATENT * jnp.eye(N),
            R_LATENT * jnp.eye(N),
            OBSERVATIONS,
            INIT_MEAN,
            INIT_COV,
            mask=mask,
        )


# ---------------------------------------------------------------------------
# Ensemble path
# ---------------------------------------------------------------------------

ENSEMBLE = 4000


@pytest.fixture(scope="module")
def prior_ensemble():
    return jnp.exp(
        jr.multivariate_normal(jr.key(3), INIT_MEAN, 1.2 * jnp.eye(N), (ENSEMBLE,))
    )


def test_analysis_matches_conjugate_transform_filter(prior_ensemble):
    """With one warp for both spaces, TransformFilter IS the ECTF."""
    warp = _warp()
    observation = OBSERVATIONS[0]
    noise = lx.DiagonalLinearOperator(R_LATENT * jnp.ones(N))
    perturbed = observation * jnp.exp(
        jnp.sqrt(R_LATENT) * jr.normal(jr.key(4), (ENSEMBLE, N))
    )
    ours = TransformFilter(warp_state=warp, warp_obs=warp).analysis(
        prior_ensemble,
        prior_ensemble,
        observation,
        noise,
        perturbed_obs=perturbed,
    )
    conjugate = ConjugateTransformFilter(warp=warp).analysis(
        prior_ensemble,
        prior_ensemble,
        observation,
        noise,
        perturbed_obs=perturbed,
    )
    assert jnp.array_equal(ours, conjugate)


def test_ensemble_support_zero_negative_particles(prior_ensemble):
    """Measured: 0/8000 negative members latent vs 743/8000 physical."""
    observation = jnp.array([0.30, 0.15])
    latent_posterior = TransformFilter(warp_state=_warp(), warp_obs=_warp()).analysis(
        prior_ensemble,
        prior_ensemble,
        observation,
        lx.DiagonalLinearOperator(R_LATENT * jnp.ones(N)),
        key=jr.key(11),
    )
    physical_posterior = gaussx.enkf_analysis(
        prior_ensemble,
        prior_ensemble,
        observation,
        lx.DiagonalLinearOperator(observation**2 * R_LATENT),
        key=jr.key(11),
    )
    assert int(jnp.sum(latent_posterior <= 0.0)) == 0
    assert int(jnp.sum(physical_posterior <= 0.0)) >= 400  # measured 743


def test_analysis_physical_noise_frame_reduces_for_identity_and_differs_for_exp(
    prior_ensemble,
):
    observation = OBSERVATIONS[0]
    noise = lx.MatrixLinearOperator(R_LATENT * jnp.eye(N), lx.positive_semidefinite_tag)
    perturbed = observation * jnp.exp(
        jnp.sqrt(R_LATENT) * jr.normal(jr.key(4), (ENSEMBLE, N))
    )
    shared = dict(perturbed_obs=perturbed)
    identity_latent = TransformFilter().analysis(
        prior_ensemble, prior_ensemble, observation, noise, **shared
    )
    identity_physical = TransformFilter(noise_frame="physical").analysis(
        prior_ensemble, prior_ensemble, observation, noise, **shared
    )
    assert jnp.array_equal(identity_latent, identity_physical)

    warped_latent = TransformFilter(warp_state=_warp(), warp_obs=_warp()).analysis(
        prior_ensemble, prior_ensemble, observation, noise, **shared
    )
    warped_physical = TransformFilter(
        warp_state=_warp(), warp_obs=_warp(), noise_frame="physical"
    ).analysis(prior_ensemble, prior_ensemble, observation, noise, **shared)
    assert not jnp.array_equal(warped_latent, warped_physical)


def test_analysis_validates_warp_event_shapes(prior_ensemble):
    wrong = Exp(shape=(N + 1,))
    observation = OBSERVATIONS[0]
    noise = lx.DiagonalLinearOperator(R_LATENT * jnp.ones(N))
    with pytest.raises(ValueError, match="warp_state"):
        TransformFilter(warp_state=wrong).analysis(
            prior_ensemble, prior_ensemble, observation, noise, key=jr.key(0)
        )
    with pytest.raises(ValueError, match="warp_obs"):
        TransformFilter(warp_obs=wrong).analysis(
            prior_ensemble, prior_ensemble, observation, noise, key=jr.key(0)
        )


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------


def test_filter_jits(latent_states):
    filt = TransformFilter(warp_state=_warp(), warp_obs=_warp())

    @eqx.filter_jit
    def run(observations):
        return filt.filter(
            growth_map,
            observe,
            Q_LATENT * jnp.eye(N),
            R_LATENT * jnp.eye(N),
            observations,
            INIT_MEAN,
            INIT_COV,
        )

    state = run(OBSERVATIONS)
    assert jnp.all(jnp.isfinite(state.filtered_means))
    reference = filt.filter(
        growth_map,
        observe,
        Q_LATENT * jnp.eye(N),
        R_LATENT * jnp.eye(N),
        OBSERVATIONS,
        INIT_MEAN,
        INIT_COV,
    )
    assert jnp.allclose(state.filtered_means, reference.filtered_means, rtol=1e-5)
