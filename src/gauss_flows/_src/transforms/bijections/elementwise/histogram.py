"""Empirical histogram CDF (per-dim) with linear or monotonic-cubic interpolation."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from functax.geometry import Axis
from functax.interp import Interpolant, Linear, Pchip, Policies, SeriesInterpolant
from jax import Array
from jaxtyping import ArrayLike


# The scheme is read on its own end cubic (or line) outside the knots, and the
# outside values are then written explicitly, as interpax's ``extrap=(lo, hi)``
# did: "clamp" would also work for the values but halves the autodiff
# derivative at the end knots (the gradient of ``clip`` at its bound).
_POLICIES = Policies(outside="extrapolate")


def _strictly_increasing(knots: Array) -> Array:
    """Knots made strictly increasing by lowering repeats one ulp at a time.

    The float32 CDF cannot resolve the ``1e-6`` smoothing increments of empty
    bins near 1, so it repeats values there; functax's `Axis` needs strictly
    increasing knots. Scanning from the right, each knot becomes
    ``min(c_k, nextafter(k_{k+1}, −∞))``: strictly increasing runs are left
    unchanged, the last knot keeps its value, and each repeat moves down by
    the fewest ulps that separate it from its right neighbour.
    """

    def step(right, knot):
        lowered = jnp.minimum(knot, jnp.nextafter(right, -jnp.inf))
        return lowered, lowered

    # knots: (n_knots,) -> lowered: (n_knots - 1,); nextafter has no
    # derivative, so the shift is a constant and gradients pass straight through
    fixed = jax.lax.stop_gradient(knots)
    _, lowered = jax.lax.scan(step, fixed[-1], fixed[:-1], reverse=True)
    shift = jnp.concatenate([lowered, fixed[-1:]]) - fixed
    return knots + shift


def _interpolants(
    method: str, bin_edges: Array, cdf_edges: Array
) -> tuple[Interpolant | SeriesInterpolant, Interpolant | SeriesInterpolant]:
    """The forward CDF F and its inverse, one functax interpolant per dim.

    Args:
        method: ``"linear"`` (functax `Linear`) or ``"monotonic"`` (`Pchip`).
        bin_edges: The knots x_k of every dim, ``(n_dims, n_bins + 1)``.
        cdf_edges: F(x_k), ``(n_dims, n_bins + 1)``.

    Returns:
        ``(fwd, inv)``, each one pytree with leading axis ``n_dims``: F on the
        edges, and F⁻¹ on the CDF values (made strictly increasing).
    """
    scheme = Linear() if method == "linear" else Pchip()

    def _make(knots, values):
        return scheme.fit(Axis(knots), values, policies=_POLICIES)

    def _make_inv(cdf_i, edges_i):
        return _make(_strictly_increasing(cdf_i), edges_i)

    fwd = eqx.filter_vmap(_make)(bin_edges, cdf_edges)
    inv = eqx.filter_vmap(_make_inv)(cdf_edges, bin_edges)
    return fwd, inv


def _read(interpolant: Interpolant | SeriesInterpolant, x: Array) -> Array:
    """One interpolant read at one scalar target."""
    # x: () -> targets: (1, 1) -> value: (1,) -> ()
    return interpolant(x[None, None])[0]


def _slope(interpolant: Interpolant | SeriesInterpolant, x: Array) -> Array:
    """dF/dx at one scalar target, by autodiff of the read-out in the target.

    functax 0.0.6 has a derivative read-out for `Linear` but none for the
    1-D series schemes (`Pchip`); differentiating the read-out gives the
    analytic derivative of either piecewise polynomial on the cell that
    holds ``x`` (the right cell at a knot), as interpax's ``dx=1`` did.
    """
    return jax.grad(lambda t: _read(interpolant, t))(x)


class HistogramCDF(AbstractBijection):
    """Marginal Gaussianization via a per-dimension empirical histogram CDF.

    Maps each dimension through its empirical CDF estimated from training
    data. Forward yields uniform marginals on ``[0, 1]``; out-of-training-
    range inputs are clamped to ``{0, 1}`` with ``log_det = −∞`` (signals
    "outside the support of the fitted CDF").

    The CDF is built from an equal-width histogram and routed through a
    functax 1-D interpolant per dimension (``functax.interp``). Two
    interpolation strategies are supported:

    - ``method="linear"`` (default): piecewise-linear CDF (functax
      `Linear`). Density is the piecewise-constant bin PDF; log-det is
      piecewise-constant in ``x``.
    - ``method="monotonic"``: Fritsch-Carlson monotone cubic Hermite (PCHIP,
      functax `Pchip`) interpolant through the same CDF breakpoints. Gives a
      smooth CDF and a smooth (continuous, piecewise-cubic) density via the
      analytic derivative. The inverse is a second monotonic-cubic interpolator
      on swapped axes, so roundtrip is approximate (~1e-5) rather than
      exact — use ``"linear"`` if exact roundtrip matters.

    Construction is two-phase: build an unfitted instance with
    ``HistogramCDF(n_bins, shape)``, then call ``.fit(data)`` to get a
    fitted instance with concrete bin edges, densities, CDF breakpoints,
    and the forward/inverse interpolators.

    Args:
        n_bins: Number of equal-width histogram bins per dimension.
        shape: Event shape ``(n_dims,)``. Only 1-D events are supported
            (each dim is independently histogrammed).
        method: ``"linear"`` or ``"monotonic"`` (PCHIP). Defaults to
            ``"linear"``.
        bin_edges: Optional pre-fit bin edges of shape ``(n_dims, n_bins+1)``.
            Use ``.fit(data)`` instead of passing this directly.
        bin_pdf: Optional pre-fit per-bin density of shape ``(n_dims, n_bins)``.
        cdf_edges: Optional pre-fit CDF values at bin edges of shape
            ``(n_dims, n_bins+1)``.
        fwd_interp: Optional pre-built forward-CDF interpolator.
            Use ``.fit(data)`` instead of passing this directly.
        inv_interp: Optional pre-built inverse-CDF interpolator.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)`` in ``[0, 1]``,
          scalar log_det (sum of per-dim ``log(density)``)
        - inverse_and_log_det:   ``(n_dims,)`` in ``[0, 1]`` → ``(n_dims,)``,
          scalar log_det

    Examples:
        Fit and transform on Gaussian data:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import HistogramCDF
        >>> data = jr.normal(jr.key(0), (5000, 3))
        >>> hist = HistogramCDF(n_bins=64, shape=(3,)).fit(data)
        >>> y, log_det = hist.transform_and_log_det(jnp.zeros(3))
        >>> y.shape
        (3,)
        >>> # y is roughly [0.5, 0.5, 0.5] (CDF at 0 of N(0,1) ≈ 0.5).
    """

    n_bins: int = eqx.field(static=True)
    shape: tuple[int, ...]
    method: str = eqx.field(static=True)
    cond_shape: ClassVar[None] = None
    bin_edges: Array | None
    bin_pdf: Array | None
    cdf_edges: Array | None
    fwd_interp: Interpolant | SeriesInterpolant | None
    inv_interp: Interpolant | SeriesInterpolant | None

    def __init__(
        self,
        n_bins: int,
        shape: tuple[int, ...],
        method: str = "linear",
        bin_edges: Array | None = None,
        bin_pdf: Array | None = None,
        cdf_edges: Array | None = None,
        fwd_interp: Interpolant | SeriesInterpolant | None = None,
        inv_interp: Interpolant | SeriesInterpolant | None = None,
    ):
        if len(shape) != 1:
            raise ValueError("HistogramCDF only supports 1D inputs.")
        if method not in ("linear", "monotonic"):
            raise ValueError(f"method must be 'linear' or 'monotonic', got {method!r}.")
        self.n_bins = n_bins
        self.shape = shape
        self.method = method
        self.bin_edges = bin_edges
        self.bin_pdf = bin_pdf
        self.cdf_edges = cdf_edges
        self.fwd_interp = fwd_interp
        self.inv_interp = inv_interp

    def fit(self, data: ArrayLike) -> HistogramCDF:
        """Fit the empirical CDF to ``data`` and return a new fitted instance.

        Args:
            data: Array of shape ``(n_samples, n_dims)`` from which to
                estimate the marginal CDFs.

        Returns:
            A new `HistogramCDF` with concrete ``bin_edges``,
            ``bin_pdf``, ``cdf_edges``, and per-dim forward/inverse
            interpolators populated. Idempotent: refitting with the same
            data yields the same fitted parameters.
        """
        # values: (n_samples, n_dims)
        values = jnp.asarray(data)
        if values.ndim != 2 or values.shape[1] != self.shape[0]:
            raise ValueError(
                f"Expected data shape (n_samples, {self.shape[0]}), got {values.shape}."
            )
        # mins, maxs: (n_dims,)
        mins = jnp.min(values, axis=0)
        maxs = jnp.max(values, axis=0)

        def _edges(lo, hi):
            # 1% padding so the extremes don't sit on a boundary.
            span = jnp.where(hi > lo, hi - lo, 1.0)
            start = lo - 0.01 * span
            end = hi + 0.01 * span
            return jnp.linspace(start, end, self.n_bins + 1)

        # bin_edges: (n_dims, n_bins+1)
        bin_edges = jax.vmap(_edges)(mins, maxs)

        def _hist(column, edges):
            # column: (n_samples,); edges: (n_bins+1,) -> counts: (n_bins,)
            counts, _ = jnp.histogram(column, bins=edges)
            return counts

        # counts: (n_dims, n_bins). Smooth by 1e-6 to avoid log(0) and to
        # keep cdf_edges monotone; the float32 tail can still repeat values,
        # which `_strictly_increasing` separates for the inverse's knots.
        counts = jax.vmap(_hist)(values.T, bin_edges) + 1e-6
        totals = jnp.sum(counts, axis=1, keepdims=True)  # (n_dims, 1)
        widths = jnp.diff(bin_edges, axis=1)  # (n_dims, n_bins)
        pdf = counts / (totals * widths)  # (n_dims, n_bins)
        # cdf_edges: (n_dims, n_bins+1) — CDF value at each bin edge.
        cdf_edges = jnp.concatenate(
            [jnp.zeros((self.shape[0], 1)), jnp.cumsum(pdf * widths, axis=1)],
            axis=1,
        )
        cdf_edges = jnp.minimum(cdf_edges, 1.0)

        # Per-dim interpolants batched via eqx.filter_vmap → one PyTree with
        # leading axis ``n_dims`` per leaf.
        fwd_interp, inv_interp = _interpolants(self.method, bin_edges, cdf_edges)

        return HistogramCDF(
            self.n_bins,
            self.shape,
            self.method,
            bin_edges,
            pdf,
            cdf_edges,
            fwd_interp,
            inv_interp,
        )

    def _check_fitted(self):
        if (
            self.bin_edges is None
            or self.bin_pdf is None
            or self.cdf_edges is None
            or self.fwd_interp is None
            or self.inv_interp is None
        ):
            raise ValueError("HistogramCDF requires fitting data first.")

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        """Map x → uniform via the fitted CDF; log_det = sum(log density)."""
        self._check_fitted()
        x_arr = jnp.asarray(x)
        edges = self.bin_edges  # type: ignore[assignment]

        def _one(fwd, x_i, edges_i):
            # fwd: per-dim interpolant; x_i, edges_i: scalar / (n_bins+1,)
            y_i = _read(fwd, x_i)
            # Analytic derivative: for "linear" this is the piecewise-constant
            # bin PDF; for "monotonic" it is the smooth Hermite derivative.
            density = _slope(fwd, x_i)
            inside = (x_i >= edges_i[0]) & (x_i <= edges_i[-1])
            # Outside the fitted range F is 0 below and 1 above.
            y_i = jnp.where(x_i < edges_i[0], 0.0, y_i)
            y_i = jnp.where(x_i > edges_i[-1], 1.0, y_i)
            # Redundant inside the range, but guards against monotonic-cubic
            # overshoot (PCHIP does not overshoot; cheap).
            y_i = jnp.clip(y_i, 0.0, 1.0)
            log_det_i = jnp.where(
                inside, jnp.log(jnp.clip(density, min=1e-38)), -jnp.inf
            )
            return y_i, log_det_i

        y, log_det = eqx.filter_vmap(_one)(self.fwd_interp, x_arr, edges)
        return y, jnp.sum(log_det)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        """Map uniform y → x via the inverse CDF; log_det = -sum(log density).

        Log-det uses the forward density at the recovered ``x`` (i.e.
        ``-log(fwd'(x_rec))``) rather than the inverse-interpolator's own
        derivative. For ``method="linear"`` the two are exact reciprocals
        (roundtrip is exact). For ``method="monotonic"`` the two
        interpolators are independent monotone-cubic fits on swapped axes,
        so the sum ``log_det_f + log_det_i`` is approximately zero rather
        than exactly zero (typically < 1e-3 for well-fit histograms).
        """
        self._check_fitted()
        y_arr = jnp.asarray(y)
        edges = self.bin_edges  # type: ignore[assignment]

        def _one(inv, fwd, y_i, edges_i):
            # Exact boundaries y == 0 / y == 1 are treated as inside (they
            # map to edges[0] / edges[-1] and carry a finite log_det),
            # consistent with forward's treatment of x == edges[0/-1].
            y_clipped = jnp.clip(y_i, 0.0, 1.0)
            x_i = _read(inv, y_clipped)
            # Above the last CDF knot (float32 can leave it just below 1), F⁻¹
            # is the right edge.
            x_i = jnp.where(y_clipped > inv.source.values[-1], edges_i[-1], x_i)
            density = _slope(fwd, x_i)
            outside = (y_i < 0.0) | (y_i > 1.0)
            x_i = jnp.where(y_i < 0.0, edges_i[0], x_i)
            x_i = jnp.where(y_i > 1.0, edges_i[-1], x_i)
            log_det_i = jnp.where(
                outside, -jnp.inf, -jnp.log(jnp.clip(density, min=1e-38))
            )
            return x_i, log_det_i

        x, log_det = eqx.filter_vmap(_one)(
            self.inv_interp, self.fwd_interp, y_arr, edges
        )
        return x, jnp.sum(log_det)


__all__ = ["HistogramCDF"]
