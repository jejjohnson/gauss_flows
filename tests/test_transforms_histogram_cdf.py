"""Equivalence of `HistogramCDF` with its interpax implementation at 81b72b0.

``tests/data/histogram_cdf_81b72b0.npz`` holds the outputs of `HistogramCDF`
when it was built on ``interpax.Interpolator1D``: fitted on 1000 standard
normal samples in two dimensions with 32 and 128 bins, read forward at the
samples, at points beyond the fitted range and at the end edges, and read
inverse at the forward outputs, at a grid of ``u`` that includes 0, 1 and
values outside ``[0, 1]``, and at the CDF knots. In float32 the 128-bin CDF
repeats values near 1 (13 repeated knots); the 32-bin one and every float64
one are strictly increasing.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gauss_flows import HistogramCDF


REFERENCE = Path(__file__).parent / "data" / "histogram_cdf_81b72b0.npz"
N_DATA = 1000

# (method, dtype) -> atol of the forward y
Y_ATOL = {
    ("linear", "float32"): 1e-6,
    ("linear", "float64"): 1e-12,
    ("monotonic", "float32"): 1e-5,
    ("monotonic", "float64"): 1e-5,
}
# (method, dtype) -> atol of log_det. PCHIP in float32 differs by up to 4e-4
# in the tails: its derivative read-out cancels values near 1 in float32, and
# both interpax and functax are off from float64 by ~1e-4 there.
LOG_DET_ATOL = {
    ("linear", "float32"): 2e-6,
    ("linear", "float64"): 1e-12,
    ("monotonic", "float32"): 5e-4,
    ("monotonic", "float64"): 1e-5,
}
# A per-dim density below ~e⁻³⁰ is a PCHIP end slope made of rounding noise
# (±1e-15 at the last knot), in either implementation; log_det there is not
# compared.
LOG_DET_SANE = 30.0

CASES = [
    (method, dtype, n_bins)
    for method in ("linear", "monotonic")
    for dtype in ("float32", "float64")
    for n_bins in (32, 128)
]


@contextlib.contextmanager
def _precision(dtype: str):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _fit(reference, method: str, dtype: str, n_bins: int) -> HistogramCDF:
    data = jnp.asarray(reference["data"], dtype)
    return HistogramCDF(n_bins, (2,), method=method).fit(data)


def _forward_points(reference, fitted: HistogramCDF, dtype: str) -> jax.Array:
    edges = np.asarray(fitted.bin_edges)
    at_edges = np.stack([edges[:, 0], edges[:, -1]], axis=0)
    points = np.concatenate([reference["data"], reference["beyond"], at_edges])
    return jnp.asarray(points, dtype)


def _inverse_points(reference, tag: str, dtype: str) -> jax.Array:
    u = np.concatenate(
        [reference[f"{tag}_y"], reference["u_grid"], reference[f"{tag}_cdf_edges"].T]
    )
    return jnp.asarray(u, dtype)


def _assert_close(
    actual, expected, atol: float, *, sane: float | None = None, rows=None
):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    # ±inf (outside the fitted support) must sit at the same entries
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    keep = np.isfinite(expected)
    if rows is not None:
        keep &= rows
    if sane is not None:
        keep &= np.abs(expected) < sane
    np.testing.assert_allclose(actual[keep], expected[keep], rtol=0.0, atol=atol)


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as archive:
        return dict(archive)


@pytest.mark.parametrize(("method", "dtype", "n_bins"), CASES)
def test_forward_matches_interpax(reference, method, dtype, n_bins):
    tag = f"{method}_{dtype}_{n_bins}"
    with _precision(dtype):
        fitted = _fit(reference, method, dtype, n_bins)
        np.testing.assert_array_equal(fitted.bin_edges, reference[f"{tag}_bin_edges"])
        np.testing.assert_array_equal(fitted.cdf_edges, reference[f"{tag}_cdf_edges"])
        points = _forward_points(reference, fitted, dtype)
        y, log_det = jax.vmap(fitted.transform_and_log_det)(points)
        assert y.dtype == jnp.dtype(dtype)
    _assert_close(y, reference[f"{tag}_y"], Y_ATOL[(method, dtype)])
    _assert_close(
        log_det,
        reference[f"{tag}_log_det"],
        LOG_DET_ATOL[(method, dtype)],
        sane=LOG_DET_SANE,
    )


@pytest.mark.parametrize("method", ["linear", "monotonic"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_outside_clamps_with_neg_inf_log_det(reference, method, dtype):
    with _precision(dtype):
        fitted = _fit(reference, method, dtype, 32)
        beyond = jnp.asarray(reference["beyond"], dtype)
        y, log_det = jax.vmap(fitted.transform_and_log_det)(beyond)
    # rows of ``beyond``: both low, mixed, mixed, both high
    np.testing.assert_array_equal(y, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    assert np.all(np.isneginf(log_det))


@pytest.mark.parametrize(
    ("method", "dtype", "n_bins"),
    [case for case in CASES if case[1:] != ("float32", 128)],
)
def test_inverse_matches_interpax(reference, method, dtype, n_bins):
    """Strictly increasing CDF knots: the inverse is unchanged."""
    tag = f"{method}_{dtype}_{n_bins}"
    with _precision(dtype):
        fitted = _fit(reference, method, dtype, n_bins)
        u = _inverse_points(reference, tag, dtype)
        x, log_det = jax.vmap(fitted.inverse_and_log_det)(u)
    _assert_close(x, reference[f"{tag}_x_inv"], Y_ATOL[(method, dtype)])
    # u = 0 or 1 reads the density at an end edge, where the PCHIP end slope
    # is rounding noise and interpax, an ulp past the edge, read its
    # ``extrap`` fill (1.0) as the density; log_det is compared inside only.
    inside = np.all((np.asarray(u) > 0.0) & (np.asarray(u) < 1.0), axis=1)
    _assert_close(
        log_det,
        reference[f"{tag}_log_det_inv"],
        LOG_DET_ATOL[(method, dtype)],
        sane=LOG_DET_SANE,
        rows=inside,
    )


@pytest.mark.parametrize("method", ["linear", "monotonic"])
def test_inverse_with_repeated_knots(reference, method):
    """float32 CDF knots that repeat near 1 still fit, and round-trip as well.

    functax's `Axis` needs strictly increasing knots, so the inverse reads
    knots whose repeats are lowered by an ulp each. ``u = 1`` maps to the
    right edge, as the docstring states (interpax's PCHIP put it on the edge
    left of a repeated last knot), and the round trip is no worse.
    """
    tag = f"{method}_float32_128"
    with _precision("float32"):
        fitted = _fit(reference, method, "float32", 128)
        assert np.any(np.diff(np.asarray(fitted.cdf_edges), axis=1) <= 0)
        x, _ = jax.vmap(fitted.inverse_and_log_det)(
            _inverse_points(reference, tag, "float32")
        )
        x_one, log_det_one = fitted.inverse_and_log_det(jnp.ones(2))
    np.testing.assert_array_equal(x_one, np.asarray(fitted.bin_edges)[:, -1])
    assert not np.isneginf(log_det_one)
    x = np.asarray(x)
    expected = reference[f"{tag}_x_inv"]
    data = reference["data"].astype(np.float32)
    if method == "linear":
        # the repeats move the inverse by at most a few ulps of x (5e-6 measured)
        np.testing.assert_allclose(x, expected, rtol=0.0, atol=1e-5)
    else:
        # PCHIP slopes next to a repeat change; the round trip gets better
        error = np.abs(x[:N_DATA] - data).max()
        error_today = np.abs(expected[:N_DATA] - data).max()
        assert error <= error_today


@pytest.mark.parametrize("method", ["linear", "monotonic"])
def test_jit_and_filter_vmap(reference, method):
    fitted = _fit(reference, method, "float32", 32)
    x = jnp.asarray(reference["data"][:16], jnp.float32)
    y_eager, log_det_eager = jax.vmap(fitted.transform_and_log_det)(x)
    y_jit, log_det_jit = jax.jit(jax.vmap(fitted.transform_and_log_det))(x)
    np.testing.assert_allclose(y_jit, y_eager, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(log_det_jit, log_det_eager, rtol=0.0, atol=1e-5)
    x_rec, _ = jax.jit(jax.vmap(fitted.inverse_and_log_det))(y_jit)
    np.testing.assert_allclose(x_rec, x, rtol=0.0, atol=1e-2)
