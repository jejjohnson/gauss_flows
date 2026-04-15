"""Tests for simple surjections and stochastic transforms."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowjax.distributions import Normal
from scipy import stats

from gauss_flows import (
    SimpleAbsSurjection,
    SimpleMaxPoolSurjection2d,
    SimpleSortSurjection,
    StochasticPermutation,
    SurVAEFlow,
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _ZeroDecoder:
    """Decoder that returns zeros with zero log-prob — for shape checks."""

    def __init__(self, residual_shape):
        self.residual_shape = residual_shape

    def sample(self, key, *, condition):
        del key, condition
        return jnp.zeros(self.residual_shape)

    def log_prob(self, value, *, condition):
        del value, condition
        return jnp.zeros(())


class _ReplayDecoder:
    """Decoder that returns pre-stored residuals — for roundtrip tests."""

    def __init__(self):
        self.residuals = None

    def sample(self, key, *, condition):
        del key, condition
        if self.residuals is None:
            raise ValueError("Residuals must be set before sampling.")
        return self.residuals

    def log_prob(self, value, *, condition):
        del value, condition
        return jnp.zeros(())


def _assert_gaussian_ks(samples, alpha=0.001):
    """Assert that flattened samples are consistent with a standard Gaussian."""
    flat = np.asarray(samples).ravel()
    _, p = stats.kstest(flat, "norm")
    assert p > alpha, f"KS p-value {p:.3e} ≤ {alpha}; samples not ~N(0,1)"


# ---------------------------------------------------------------------------
# SimpleAbsSurjection
# ---------------------------------------------------------------------------


def test_simple_abs_single_event_forward_inverse(key):
    surj = SimpleAbsSurjection((4,))
    x = jr.normal(key, (4,))
    z, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    x_back, log_det_inv = surj.inverse_and_log_det(z, jr.fold_in(key, 2))
    expected = -4 * math.log(2.0)
    assert z.shape == (4,)
    assert x_back.shape == (4,)
    assert jnp.all(z >= 0)
    assert jnp.allclose(z, jnp.abs(x))
    assert jnp.allclose(jnp.abs(x_back), z)
    assert log_det.shape == ()
    assert log_det_inv.shape == ()
    assert jnp.allclose(log_det, expected)
    assert jnp.allclose(log_det_inv, expected)


def test_simple_abs_distribution_roundtrip(key):
    surj = SimpleAbsSurjection((8,))
    xs = jr.normal(jr.fold_in(key, 0), (4096, 8))
    keys_f = jr.split(jr.fold_in(key, 1), xs.shape[0])
    keys_i = jr.split(jr.fold_in(key, 2), xs.shape[0])
    zs = jax.vmap(lambda x, k: surj.forward_and_log_det(x, k)[0])(xs, keys_f)
    x_back = jax.vmap(lambda z, k: surj.inverse_and_log_det(z, k)[0])(zs, keys_i)
    _assert_gaussian_ks(x_back)


# ---------------------------------------------------------------------------
# SimpleSortSurjection
# ---------------------------------------------------------------------------


def test_simple_sort_1d_single_event(key):
    surj = SimpleSortSurjection((6,))
    x = jr.normal(key, (6,))
    z, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    x_back, log_det_inv = surj.inverse_and_log_det(z, jr.fold_in(key, 2))
    expected = -math.lgamma(6 + 1)
    assert z.shape == (6,)
    assert x_back.shape == (6,)
    assert jnp.all(z[:-1] <= z[1:])
    assert jnp.allclose(jnp.sort(x_back), z)
    assert log_det.shape == ()
    assert log_det_inv.shape == ()
    assert jnp.allclose(log_det, expected)
    assert jnp.allclose(log_det_inv, expected)


def test_simple_sort_multi_dim_log_det(key):
    """log_det must scale with n_slices for multi-dim events."""
    surj = SimpleSortSurjection((4, 3), axis=0)
    x = jr.normal(key, (4, 3))
    _, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    _, log_det_inv = surj.inverse_and_log_det(x, jr.fold_in(key, 2))
    expected = -3 * math.lgamma(5)  # n_slices = W = 3, axis_size = 4
    assert jnp.allclose(log_det, expected)
    assert jnp.allclose(log_det_inv, expected)


def test_simple_sort_multi_dim_independent_per_slice_perm(key):
    """Inverse draws an independent permutation per slice along ``axis``."""
    surj = SimpleSortSurjection((4, 3), axis=0)
    # Build a sorted z where each column is [0, 1, 2, 3]. For any random
    # permutation the column values are still a permutation of {0..3}, so
    # sorting the inverse along axis=0 must recover z exactly — for every key.
    z = jnp.broadcast_to(jnp.arange(4, dtype=jnp.float32)[:, None], (4, 3))
    for i in range(32):
        x, _ = surj.inverse_and_log_det(z, jr.fold_in(key, i))
        assert x.shape == (4, 3)
        assert jnp.allclose(jnp.sort(x, axis=0), z)

    # Verify independence: over many keys, the column-0 perm and column-1 perm
    # should be uncorrelated. Use argsort to recover the per-column permutation
    # and check that the Spearman-style rank correlation between columns ≈ 0.
    trials = 2048
    x_stack = jax.vmap(
        lambda k: surj.inverse_and_log_det(z, k)[0],
    )(jr.split(jr.fold_in(key, 100), trials))
    perms = jnp.argsort(x_stack, axis=1)  # (trials, 4, 3) — inverse sort
    col0 = np.asarray(perms[:, :, 0]).ravel().astype(np.float64)
    col1 = np.asarray(perms[:, :, 1]).ravel().astype(np.float64)
    corr = np.corrcoef(col0, col1)[0, 1]
    assert abs(corr) < 0.05, f"columns look correlated (r={corr:.3f})"


def test_simple_sort_multi_dim_non_zero_axis(key):
    """Sort along a non-zero axis."""
    surj = SimpleSortSurjection((3, 5), axis=1)
    x = jr.normal(key, (3, 5))
    z, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    expected_log_det = -3 * math.lgamma(6)  # n_slices=3, axis_size=5
    assert jnp.all(z[:, :-1] <= z[:, 1:])
    assert jnp.allclose(log_det, expected_log_det)
    x_back, _ = surj.inverse_and_log_det(z, jr.fold_in(key, 2))
    assert jnp.allclose(jnp.sort(x_back, axis=1), z)


def test_simple_sort_distribution_roundtrip(key):
    surj = SimpleSortSurjection((5,))
    xs = jr.normal(jr.fold_in(key, 0), (4096, 5))
    keys_f = jr.split(jr.fold_in(key, 1), xs.shape[0])
    keys_i = jr.split(jr.fold_in(key, 2), xs.shape[0])
    zs = jax.vmap(lambda x, k: surj.forward_and_log_det(x, k)[0])(xs, keys_f)
    x_back = jax.vmap(lambda z, k: surj.inverse_and_log_det(z, k)[0])(zs, keys_i)
    _assert_gaussian_ks(x_back)


# ---------------------------------------------------------------------------
# StochasticPermutation
# ---------------------------------------------------------------------------


def test_stochastic_permutation_single_event(key):
    surj = StochasticPermutation((5,))
    x = jr.normal(key, (5,))
    z, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    x_back, log_det_inv = surj.inverse_and_log_det(z, jr.fold_in(key, 2))
    assert z.shape == (5,)
    assert x_back.shape == (5,)
    assert log_det.shape == ()
    assert log_det_inv.shape == ()
    assert jnp.allclose(log_det, 0.0)
    assert jnp.allclose(log_det_inv, 0.0)
    # Must be a permutation of the input (not a different multiset).
    assert jnp.allclose(jnp.sort(z), jnp.sort(x))
    assert jnp.allclose(jnp.sort(x_back), jnp.sort(z))


def test_stochastic_permutation_multi_dim_axis(key):
    surj = StochasticPermutation((3, 4), axis=1)
    x = jr.normal(key, (3, 4))
    z, _ = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    # Each row is a permutation of the input row (same perm applied row-wise).
    assert jnp.allclose(jnp.sort(z, axis=1), jnp.sort(x, axis=1))


def test_stochastic_permutation_distribution_roundtrip(key):
    surj = StochasticPermutation((6,))
    xs = jr.normal(jr.fold_in(key, 0), (4096, 6))
    keys_f = jr.split(jr.fold_in(key, 1), xs.shape[0])
    keys_i = jr.split(jr.fold_in(key, 2), xs.shape[0])
    zs = jax.vmap(lambda x, k: surj.forward_and_log_det(x, k)[0])(xs, keys_f)
    x_back = jax.vmap(lambda z, k: surj.inverse_and_log_det(z, k)[0])(zs, keys_i)
    _assert_gaussian_ks(x_back)


# ---------------------------------------------------------------------------
# SimpleMaxPoolSurjection2d
# ---------------------------------------------------------------------------


def test_simple_maxpool_log_det_and_shape(key):
    decoder = _ZeroDecoder((2, 2, 1, 3))
    surj = SimpleMaxPoolSurjection2d((4, 4, 1), decoder)
    x = jr.normal(key, (4, 4, 1))
    z, log_det = surj.forward_and_log_det(x, jr.fold_in(key, 1))
    expected = -4 * math.log(4.0)  # N = 2*2*1 = 4 positions; pool_area = 4
    assert z.shape == (2, 2, 1)
    assert log_det.shape == ()
    assert jnp.allclose(log_det, expected)


def test_simple_maxpool_deconstruct_construct_roundtrip_4x4(key):
    decoder = _ZeroDecoder((2, 2, 1, 3))
    surj = SimpleMaxPoolSurjection2d((4, 4, 1), decoder)
    x = jr.uniform(key, (4, 4, 1))
    z, residuals, k = surj._deconstruct_x(x)
    x_rec = surj._construct_x(z, residuals, k)
    assert x_rec.shape == x.shape
    assert jnp.allclose(x_rec, x)


def test_simple_maxpool_deconstruct_construct_roundtrip_8x8(key):
    decoder = _ZeroDecoder((4, 4, 2, 3))
    surj = SimpleMaxPoolSurjection2d((8, 8, 2), decoder)
    x = jr.uniform(key, (8, 8, 2))
    z, residuals, k = surj._deconstruct_x(x)
    x_rec = surj._construct_x(z, residuals, k)
    assert x_rec.shape == x.shape
    assert jnp.allclose(x_rec, x)


def test_simple_maxpool_inverse_recovers_x_with_replay_decoder(key):
    """With stored true residuals + argmax k, inverse must recover x exactly."""
    decoder = _ReplayDecoder()
    surj = SimpleMaxPoolSurjection2d((4, 4, 1), decoder)
    x = jr.normal(key, (4, 4, 1))
    z, residuals, k_true = surj._deconstruct_x(x)
    decoder.residuals = residuals
    x_rec = surj._construct_x(z, residuals, k_true)
    assert jnp.allclose(x_rec, x)


def test_simple_maxpool_forward_jit_and_vmap_compatible(key):
    """Regression: forward must trace under jit and vmap (no boolean masking)."""
    decoder = _ZeroDecoder((2, 2, 1, 3))
    surj = SimpleMaxPoolSurjection2d((4, 4, 1), decoder)
    forward = jax.jit(lambda x, k: surj.forward_and_log_det(x, k))

    x = jr.normal(key, (4, 4, 1))
    z_jit, log_det_jit = forward(x, jr.fold_in(key, 1))
    assert z_jit.shape == (2, 2, 1)
    assert log_det_jit.shape == ()

    xs = jr.normal(jr.fold_in(key, 2), (8, 4, 4, 1))
    keys = jr.split(jr.fold_in(key, 3), 8)
    zs, log_dets = jax.vmap(forward)(xs, keys)
    assert zs.shape == (8, 2, 2, 1)
    assert log_dets.shape == (8,)


def test_simple_maxpool_rejects_bad_shape():
    decoder = _ZeroDecoder((2, 2, 1, 3))
    try:
        SimpleMaxPoolSurjection2d((5, 4, 1), decoder)  # 5 not divisible by 2
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-divisible spatial dim")


# ---------------------------------------------------------------------------
# Integration with SurVAEFlow (batching via the container, not the transform)
# ---------------------------------------------------------------------------


def test_survaeflow_integration_abs_sort_perm(key):
    """Each new transform must round-trip through SurVAEFlow without NaN."""
    dim = 4
    base = Normal(jnp.zeros(dim))
    transforms = [
        SimpleAbsSurjection((dim,)),
        SimpleSortSurjection((dim,)),
        StochasticPermutation((dim,)),
    ]
    flow = SurVAEFlow(base, transforms)
    samples = flow.sample(jr.fold_in(key, 1), sample_shape=(16,))
    log_probs = flow.log_prob(samples, jr.fold_in(key, 2))
    assert samples.shape == (16, dim)
    assert log_probs.shape == (16,)
    assert jnp.all(jnp.isfinite(log_probs))
