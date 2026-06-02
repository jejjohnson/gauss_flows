"""Empirical histogram CDF (per-dim) with linear or monotonic-cubic interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike


if TYPE_CHECKING:
    import interpax


class HistogramCDF(AbstractBijection):
    """Marginal Gaussianization via a per-dimension empirical histogram CDF.

    Maps each dimension through its empirical CDF estimated from training
    data. Forward yields uniform marginals on ``[0, 1]``; out-of-training-
    range inputs are clamped to ``{0, 1}`` with ``log_det = −∞`` (signals
    "outside the support of the fitted CDF").

    The CDF is built from an equal-width histogram and routed through an
    :class:`interpax.Interpolator1D` per dimension. Two interpolation
    strategies are supported:

    - ``method="linear"`` (default): piecewise-linear CDF. Bit-identical to
      the previous ``jnp.searchsorted`` implementation. Density is the
      piecewise-constant bin PDF; log-det is piecewise-constant in ``x``.
    - ``method="monotonic"``: Fritsch-Carlson monotone cubic Hermite (PCHIP)
      interpolant through the same CDF breakpoints. Gives a smooth CDF and
      a smooth (continuous, piecewise-cubic) density via the analytic
      derivative. The inverse is a second monotonic-cubic interpolator
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

    Example:
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
    fwd_interp: interpax.Interpolator1D | None
    inv_interp: interpax.Interpolator1D | None

    def __init__(
        self,
        n_bins: int,
        shape: tuple[int, ...],
        method: str = "linear",
        bin_edges: Array | None = None,
        bin_pdf: Array | None = None,
        cdf_edges: Array | None = None,
        fwd_interp: interpax.Interpolator1D | None = None,
        inv_interp: interpax.Interpolator1D | None = None,
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
            A new :class:`HistogramCDF` with concrete ``bin_edges``,
            ``bin_pdf``, ``cdf_edges``, and per-dim forward/inverse
            interpolators populated. Idempotent: refitting with the same
            data yields the same fitted parameters.

        Raises:
            ImportError: If ``interpax`` is not installed. Install it with
                ``pip install gauss-flows[interp]``.
        """
        try:
            import interpax
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "HistogramCDF.fit() requires interpax. "
                "Install it with: pip install gauss-flows[interp]"
            ) from e
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
        # keep cdf_edges strictly monotone (required by interpax).
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

        # Per-dim interpolators batched via eqx.filter_vmap → one PyTree with
        # leading axis ``n_dims`` per leaf. ``extrap=(lo, hi)`` clamps queries
        # outside the fitted range to the boundary values.
        method = self.method

        def _make_fwd(edges_i, cdf_i):
            return interpax.Interpolator1D(
                edges_i, cdf_i, method=method, extrap=(0.0, 1.0)
            )

        def _make_inv(cdf_i, edges_i):
            # extrap fills per-dim with that dim's own left/right edge.
            return interpax.Interpolator1D(
                cdf_i, edges_i, method=method, extrap=(edges_i[0], edges_i[-1])
            )

        fwd_interp = eqx.filter_vmap(_make_fwd)(bin_edges, cdf_edges)
        inv_interp = eqx.filter_vmap(_make_inv)(cdf_edges, bin_edges)

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
            # fwd: per-dim Interpolator1D; x_i, edges_i: scalar / (n_bins+1,)
            y_i = fwd(x_i)
            # Analytic derivative: for "linear" this is the piecewise-constant
            # bin PDF; for "monotonic" it is the smooth Hermite derivative.
            density = fwd(x_i, dx=1)
            inside = (x_i >= edges_i[0]) & (x_i <= edges_i[-1])
            # Clamp y to the extrap boundaries we passed to interpax. This is
            # redundant inside the range but guards against monotonic-cubic
            # overshoot at the extrapolation seam (shouldn't happen with
            # extrap=(0, 1) but cheap).
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
            x_i = inv(y_clipped)
            density = fwd(x_i, dx=1)
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
