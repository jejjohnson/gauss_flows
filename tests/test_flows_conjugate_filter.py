"""Tests for `ConjugateTransformFilter` and `rbig_conjugate_filter`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Exp, Identity, Sigmoid, Stack

from gauss_flows import ConjugateTransformFilter, rbig_conjugate_filter


# `pytest.importorskip` only catches ImportError. gaussx currently fails to
# import in this environment with an AttributeError instead: it needs
# matfree>=0.6 (`sampler_signs`) while gauss_flows is pinned to the pre-0.6
# `sampler_rademacher`. Skip on any import failure so the reason is reported
# rather than collapsing the whole collection.
try:
    import gaussx
    import lineax as lx
except Exception as exc:  # gaussx import can fail in several ways here
    pytest.skip(
        f"gaussx is not importable here ({type(exc).__name__}: {exc}); "
        "install it into an environment where its pins are resolved to run "
        "the ConjugateTransformFilter tests.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# The reference paper's lognormal / logit-normal problem
# ---------------------------------------------------------------------------
#
# The latent state is ``zeta ~ N(mu, Sigma)``; the physical state is
# ``x = (exp(zeta_0), logistic(zeta_1))``, i.e. a strictly-positive coordinate
# and a bounded one. The first latent coordinate is observed with additive
# Gaussian noise, so the physical observation carries multiplicative lognormal
# noise -- exactly the case where a physical-space Gaussian update is applied
# in the wrong coordinates.
#
# ``Sigma`` and the observation variance are recovered from the latent
# posterior covariance the paper reports; ``mu`` and the observation are fixed
# by its reported posterior mean up to one free parameter, taken as
# ``mu[0] = 0``. The resulting exact posterior mean reproduces the paper's
# ``[0.548062, 0.353937]`` to seven figures.

PRIOR_COV = jnp.array([[0.6, 0.25], [0.25, 0.4]])
OBS_VAR = 0.05
PRIOR_MEAN = jnp.array([0.0, -23.0 / 60.0])
LATENT_OBS = jnp.array([-0.6764811])


def _warp():
    """Gamma: latent (Gaussian) -> physical. flowjax `transform` is that way."""
    return Stack([Exp(), Sigmoid()])


def _obs_warp():
    """Only the first (lognormal) coordinate is observed, so M = 1."""
    return Exp(shape=(1,))


def _filter():
    return ConjugateTransformFilter(warp=_warp(), obs_warp=_obs_warp())


def _exact_posterior_mean():
    """Exact E[x | y], by conjugacy in latent space then quadrature."""
    gain = PRIOR_COV[:, :1] / (PRIOR_COV[0, 0] + OBS_VAR)  # (2, 1)
    post_mean = PRIOR_MEAN + gain[:, 0] * (LATENT_OBS[0] - PRIOR_MEAN[0])
    post_cov = PRIOR_COV - gain @ PRIOR_COV[:1, :]

    grid = jnp.linspace(-12.0, 12.0, 200_001)
    weights = jnp.exp(-0.5 * (grid - post_mean[1]) ** 2 / post_cov[1, 1])
    weights = weights / weights.sum()
    return jnp.array(
        [
            jnp.exp(post_mean[0] + post_cov[0, 0] / 2),
            (jax.nn.sigmoid(grid) * weights).sum(),
        ]
    )


def _draw(key, n_ens):
    """Prior ensemble in physical space, plus one shared noise realisation.

    Returns the physical ensemble, the physical perturbed-observation
    ensemble, and the physical observation -- everything a filter is handed.
    """
    k_prior, k_noise = jr.split(key)
    chol = jnp.linalg.cholesky(PRIOR_COV)
    latent = PRIOR_MEAN + jr.normal(k_prior, (n_ens, 2)) @ chol.T  # (J, 2)
    perturbed_latent = LATENT_OBS[None, :] + jnp.sqrt(OBS_VAR) * jr.normal(
        k_noise, (n_ens, 1)
    )  # (J, 1)
    warp = _warp()
    physical = jax.vmap(warp.transform)(latent)  # (J, 2)
    return physical, jnp.exp(perturbed_latent), jnp.exp(LATENT_OBS)


def _noise_op():
    return lx.DiagonalLinearOperator(jnp.array([OBS_VAR]))


def _physical_enkf(physical, perturbed_physical, observation):
    """The baseline: the same update, in physical coordinates."""
    noise = lx.DiagonalLinearOperator(jnp.array([jnp.var(perturbed_physical[:, 0])]))
    return gaussx.enkf_analysis(
        physical,
        physical[:, :1],
        observation,
        noise,
        perturbed_obs=perturbed_physical,
    )


# ---------------------------------------------------------------------------
# Exactness and improvement
# ---------------------------------------------------------------------------


def test_exact_posterior_reference_matches_the_published_value():
    """Guards the fixture, so a drift in the constants fails loudly here."""
    assert jnp.allclose(
        _exact_posterior_mean(), jnp.array([0.548062, 0.353937]), atol=1e-6
    )


@pytest.mark.slow
def test_beats_the_physical_space_enkf_on_the_same_ensemble():
    """Same ensemble, same noise realisation -- only the coordinates differ."""
    exact = _exact_posterior_mean()
    physical, perturbed, observation = _draw(jr.key(0), 200_000)

    filt = _filter()
    conjugate = filt.analysis(
        physical,
        physical[:, :1],
        observation,
        _noise_op(),
        perturbed_obs=perturbed,
    )
    baseline = _physical_enkf(physical, perturbed, observation)

    conjugate_error = jnp.linalg.norm(conjugate.mean(axis=0) - exact)
    baseline_error = jnp.linalg.norm(baseline.mean(axis=0) - exact)
    assert conjugate_error < baseline_error / 50


@pytest.mark.slow
def test_conjugate_error_converges_while_the_baseline_plateaus():
    """The property that makes the method worth having.

    A mean-only test at a single ensemble size does not distinguish "better"
    from "converging": the physical-space bias is an error of coordinates, so
    it survives any ensemble size, while the conjugated update's error is pure
    sampling noise and shrinks at the Monte-Carlo rate. Errors are averaged
    over several independent draws, because a single realisation at each J is
    too noisy to read a rate off.

    Measured (4 seeds, float32):

    ==========  ==========  ==========
    J           conjugate   physical
    ==========  ==========  ==========
    2,000       0.0030      0.0239
    20,000      0.0011      0.0257
    200,000     0.0004      0.0260
    ==========  ==========  ==========
    """
    exact = _exact_posterior_mean()
    filt = _filter()
    n_seeds = 4

    conjugate_errors, baseline_errors = [], []
    for n_ens in (2_000, 20_000, 200_000):
        conjugate_at_j, baseline_at_j = [], []
        for seed in range(n_seeds):
            physical, perturbed, observation = _draw(jr.key(100 + seed), n_ens)
            conjugate = filt.analysis(
                physical,
                physical[:, :1],
                observation,
                _noise_op(),
                perturbed_obs=perturbed,
            )
            conjugate_at_j.append(jnp.linalg.norm(conjugate.mean(axis=0) - exact))
            baseline_at_j.append(
                jnp.linalg.norm(
                    _physical_enkf(physical, perturbed, observation).mean(axis=0)
                    - exact
                )
            )
        conjugate_errors.append(jnp.mean(jnp.stack(conjugate_at_j)))
        baseline_errors.append(jnp.mean(jnp.stack(baseline_at_j)))

    # Conjugated: decays with J, and ends up small in absolute terms.
    assert conjugate_errors[-1] < conjugate_errors[0] / 5
    assert conjugate_errors[1] < conjugate_errors[0]
    assert conjugate_errors[-1] < 1e-3
    # Physical-space: plateaus. A hundredfold ensemble buys nothing.
    assert baseline_errors[-1] > 0.5 * baseline_errors[0]
    assert baseline_errors[-1] > 0.01
    # And the gap is what the method is for.
    assert baseline_errors[-1] > 10 * conjugate_errors[-1]


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_identity_warp_reproduces_enkf_analysis_bit_for_bit():
    physical, perturbed, observation = _draw(jr.key(2), 512)
    filt = ConjugateTransformFilter(warp=Identity((2,)), obs_warp=Identity((1,)))
    conjugate = filt.analysis(
        physical, physical[:, :1], observation, _noise_op(), perturbed_obs=perturbed
    )
    direct = gaussx.enkf_analysis(
        physical,
        physical[:, :1],
        observation,
        _noise_op(),
        perturbed_obs=perturbed,
    )
    assert jnp.array_equal(conjugate, direct)


def test_warp_round_trips_over_the_ensemble_range():
    physical, _, _ = _draw(jr.key(3), 1024)
    warp = _warp()
    latent = jax.vmap(warp.inverse)(physical)
    assert jnp.allclose(jax.vmap(warp.transform)(latent), physical, atol=1e-6)


def test_separate_obs_warp_uses_the_right_latent_innovation():
    """State and observation spaces with different warps.

    The state is (positive, bounded) and the observation is a *bounded*
    quantity only, so it needs its own warp. Applying the state warp to the
    observation would be a shape error at best and wrong coordinates at worst;
    the correct latent innovation is the one computed under ``obs_warp``.
    """
    key = jr.key(4)
    state_warp, obs_warp = _warp(), Sigmoid(shape=(1,))
    latent_state = jr.normal(key, (2048, 2))
    physical_state = jax.vmap(state_warp.transform)(latent_state)  # (J, 2)
    latent_obs = latent_state[:, 1:]  # observe the bounded coordinate
    physical_obs = jax.vmap(obs_warp.transform)(latent_obs)  # (J, 1)
    observation = obs_warp.transform(jnp.array([0.3]))

    filt = ConjugateTransformFilter(warp=state_warp, obs_warp=obs_warp)
    got = filt.analysis(
        physical_state, physical_obs, observation, _noise_op(), key=jr.key(5)
    )

    # The same computation written out in latent coordinates.
    expected_latent = gaussx.enkf_analysis(
        latent_state, latent_obs, jnp.array([0.3]), _noise_op(), key=jr.key(5)
    )
    expected = jax.vmap(state_warp.transform)(expected_latent)
    assert jnp.allclose(got, expected, atol=1e-6)


def test_shapes_with_distinct_state_and_obs_dims():
    key = jr.key(6)
    n_ens, n_state, n_obs = 256, 5, 2
    particles = jnp.exp(jr.normal(key, (n_ens, n_state)))
    obs_particles = particles[:, :n_obs]
    filt = ConjugateTransformFilter(
        warp=Exp(shape=(n_state,)), obs_warp=Exp(shape=(n_obs,))
    )
    out = filt.analysis(
        particles,
        obs_particles,
        jnp.ones(n_obs),
        lx.DiagonalLinearOperator(0.1 * jnp.ones(n_obs)),
        key=jr.key(7),
    )
    assert out.shape == (n_ens, n_state)


def test_key_path_is_deterministic_and_differs_across_keys():
    physical, _, observation = _draw(jr.key(8), 512)
    filt = _filter()
    args = (physical, physical[:, :1], observation, _noise_op())
    a = filt.analysis(*args, key=jr.key(9))
    b = filt.analysis(*args, key=jr.key(9))
    c = filt.analysis(*args, key=jr.key(10))
    assert jnp.array_equal(a, b)
    assert not jnp.allclose(a, c)


def test_requires_exactly_one_perturbation_source():
    physical, perturbed, observation = _draw(jr.key(11), 128)
    filt = _filter()
    with pytest.raises(ValueError, match="exactly one of 'key'"):
        filt.analysis(physical, physical[:, :1], observation, _noise_op())
    with pytest.raises(ValueError, match="exactly one of 'key'"):
        filt.analysis(
            physical,
            physical[:, :1],
            observation,
            _noise_op(),
            key=jr.key(12),
            perturbed_obs=perturbed,
        )


def test_localization_is_forwarded():
    """A zero taper must leave the ensemble untouched."""
    physical, perturbed, observation = _draw(jr.key(13), 512)
    filt = _filter()
    out = filt.analysis(
        physical,
        physical[:, :1],
        observation,
        _noise_op(),
        perturbed_obs=perturbed,
        localization=jnp.zeros((2, 1)),
    )
    assert jnp.allclose(out, physical, atol=1e-10)


def test_jit():
    physical, _perturbed, observation = _draw(jr.key(14), 256)
    filt = _filter()
    jitted = eqx.filter_jit(ConjugateTransformFilter.analysis)
    out = jitted(
        filt, physical, physical[:, :1], observation, _noise_op(), key=jr.key(15)
    )
    assert out.shape == physical.shape
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# RBIG-fitted warp
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rbig_conjugate_filter_reduces_bias():
    """A fitted warp, so the tolerance is looser than for the exact one."""
    exact = _exact_posterior_mean()
    physical, perturbed, observation = _draw(jr.key(16), 20_000)

    filt = rbig_conjugate_filter(physical, n_layers=6, n_components=8)
    # RBIG fits one warp for the whole state; the observation is the first
    # coordinate, so restrict it with an explicit obs_warp.
    filt = eqx.tree_at(
        lambda f: f.obs_warp, filt, _obs_warp(), is_leaf=lambda x: x is None
    )
    fitted = filt.analysis(
        physical,
        physical[:, :1],
        observation,
        _noise_op(),
        perturbed_obs=perturbed,
    )
    baseline = _physical_enkf(physical, perturbed, observation)

    assert jnp.linalg.norm(fitted.mean(axis=0) - exact) < jnp.linalg.norm(
        baseline.mean(axis=0) - exact
    )


def test_rbig_conjugate_filter_returns_a_matching_warp():
    particles = jnp.exp(jr.normal(jr.key(17), (512, 3)))
    filt = rbig_conjugate_filter(particles, n_layers=3, n_components=4)
    assert isinstance(filt, ConjugateTransformFilter)
    assert filt.warp.shape == (3,)
    assert filt.obs_warp is None


def test_rejects_an_obs_warp_that_does_not_match_the_observation_space():
    """N != M with the default obs_warp is a confusing flowjax error otherwise."""
    physical, _, observation = _draw(jr.key(18), 128)
    filt = ConjugateTransformFilter(warp=_warp())  # 2-D warp, 1-D observations
    with pytest.raises(ValueError, match="obs_warp is None"):
        filt.analysis(
            physical, physical[:, :1], observation, _noise_op(), key=jr.key(19)
        )

    mismatched = ConjugateTransformFilter(warp=_warp(), obs_warp=Exp(shape=(2,)))
    with pytest.raises(ValueError, match="obs_warp has event shape"):
        mismatched.analysis(
            physical, physical[:, :1], observation, _noise_op(), key=jr.key(20)
        )
