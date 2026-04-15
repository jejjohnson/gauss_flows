"""Marginal transforms for Gaussianization flows.

These bijections operate element-wise along the marginal dimensions, providing
various ways to Gaussianize univariate marginal distributions.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from flowjax.bijections import AbstractBijection, RationalQuadraticSpline, Vmap
from jax import Array
from jax.nn import softmax, softplus
from jaxtyping import ArrayLike


class MixtureGaussianCDF(AbstractBijection):
    """Marginal Gaussianization via a mixture of Gaussians CDF.

    Applies the CDF of a Gaussian mixture model to each dimension independently,
    mapping the data to uniform, then applies the Gaussian inverse CDF (probit).

    Args:
        n_components: Number of mixture components per dimension.
        shape: Shape of the input (n_dims,).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    means: Array
    log_scales: Array
    log_weights: Array

    def __init__(self, n_components: int, shape: tuple[int, ...]):
        if len(shape) != 1:
            raise ValueError("MixtureGaussianCDF only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        self.means = jnp.zeros((n_dims, n_components))
        self.log_scales = jnp.zeros((n_dims, n_components))
        self.log_weights = jnp.zeros((n_dims, n_components))

    def _gmm_cdf(self, x: Array, means: Array, scales: Array, weights: Array) -> Array:
        """CDF of a 1D Gaussian mixture evaluated at x."""
        cdfs = jstats.norm.cdf(x[:, None], loc=means, scale=scales)
        return jnp.sum(weights * cdfs, axis=-1)

    def _gmm_logpdf(
        self, x: Array, means: Array, scales: Array, weights: Array
    ) -> Array:
        """Log PDF of a 1D Gaussian mixture evaluated at x."""
        log_pdfs = jstats.norm.logpdf(x[:, None], loc=means, scale=scales)
        return jnp.log(jnp.sum(weights * jnp.exp(log_pdfs), axis=-1) + 1e-38)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        scales = softplus(self.log_scales) + 1e-5
        weights = softmax(self.log_weights, axis=-1)

        # GMM CDF -> uniform -> probit (inverse normal CDF)
        u = self._gmm_cdf(x, self.means, scales, weights)
        u = jnp.clip(u, 1e-6, 1 - 1e-6)
        y = jax.scipy.special.ndtri(u)

        # Log det: log |dy/dx| = log |phi^{-1}'(u) * gmm_pdf(x)|
        # = log_gmm_pdf(x) - log_norm_pdf(y)
        log_pdf_x = self._gmm_logpdf(x, self.means, scales, weights)
        log_pdf_y = jstats.norm.logpdf(y)
        log_det = jnp.sum(log_pdf_x - log_pdf_y)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        from gauss_flows._src.utils import bisection_inverse

        scales = softplus(self.log_scales) + 1e-5
        weights = softmax(self.log_weights, axis=-1)

        # Probit -> uniform CDF
        u = jax.scipy.special.ndtr(y)

        # Invert GMM CDF: find x such that GMM_CDF(x) = u
        def _cdf_i(u_i, means_i, scales_i, weights_i):
            def _fn(xi):
                return self._gmm_cdf(
                    xi[None], means_i[None], scales_i[None], weights_i[None]
                )[0]

            return bisection_inverse(_fn, u_i)

        x = jax.vmap(_cdf_i)(u, self.means, scales, weights)

        # Log det of inverse = -log_det of forward
        log_pdf_x = self._gmm_logpdf(x, self.means, scales, weights)
        log_pdf_y = jstats.norm.logpdf(y)
        log_det = -jnp.sum(log_pdf_x - log_pdf_y)
        return x, log_det


class MixtureLogisticCDF(AbstractBijection):
    """Marginal Gaussianization via a mixture of logistics CDF.

    Similar to MixtureGaussianCDF but uses a logistic mixture.

    Args:
        n_components: Number of mixture components per dimension.
        shape: Shape of the input (n_dims,).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    means: Array
    log_scales: Array
    log_weights: Array

    def __init__(self, n_components: int, shape: tuple[int, ...]):
        if len(shape) != 1:
            raise ValueError("MixtureLogisticCDF only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        self.means = jnp.zeros((n_dims, n_components))
        self.log_scales = jnp.zeros((n_dims, n_components))
        self.log_weights = jnp.zeros((n_dims, n_components))

    def _mixture_cdf(
        self, x: Array, means: Array, scales: Array, weights: Array
    ) -> Array:
        """CDF of a 1D logistic mixture evaluated at x."""
        cdfs = jax.scipy.special.expit((x[:, None] - means) / scales)
        return jnp.sum(weights * cdfs, axis=-1)

    def _mixture_logpdf(
        self, x: Array, means: Array, scales: Array, weights: Array
    ) -> Array:
        """Log PDF of a 1D logistic mixture evaluated at x."""
        z = (x[:, None] - means) / scales
        log_pdfs = -jnp.log(scales) + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z)
        return jnp.log(jnp.sum(weights * jnp.exp(log_pdfs), axis=-1) + 1e-38)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        scales = softplus(self.log_scales) + 1e-5
        weights = softmax(self.log_weights, axis=-1)

        u = self._mixture_cdf(x, self.means, scales, weights)
        u = jnp.clip(u, 1e-6, 1 - 1e-6)
        y = jax.scipy.special.ndtri(u)

        log_pdf_x = self._mixture_logpdf(x, self.means, scales, weights)
        log_pdf_y = jstats.norm.logpdf(y)
        log_det = jnp.sum(log_pdf_x - log_pdf_y)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        from gauss_flows._src.utils import bisection_inverse

        scales = softplus(self.log_scales) + 1e-5
        weights = softmax(self.log_weights, axis=-1)

        u = jax.scipy.special.ndtr(y)

        def _cdf_i(u_i, means_i, scales_i, weights_i):
            def _fn(xi):
                return self._mixture_cdf(
                    xi[None], means_i[None], scales_i[None], weights_i[None]
                )[0]

            return bisection_inverse(_fn, u_i)

        x = jax.vmap(_cdf_i)(u, self.means, scales, weights)

        log_pdf_x = self._mixture_logpdf(x, self.means, scales, weights)
        log_pdf_y = jstats.norm.logpdf(y)
        log_det = -jnp.sum(log_pdf_x - log_pdf_y)
        return x, log_det


class RQSplineMarginal(AbstractBijection):
    """Marginal Gaussianization via rational quadratic splines.

    Applies a rational quadratic spline independently to each dimension.

    Args:
        n_bins: Number of spline bins.
        shape: Shape of the input (n_dims,).
        interval: Interval for the spline. Defaults to 5.0.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    spline: AbstractBijection

    def __init__(self, n_bins: int, shape: tuple[int, ...], interval: float = 5.0):
        if len(shape) != 1:
            raise ValueError("RQSplineMarginal only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        spline = eqx.filter_vmap(
            lambda: RationalQuadraticSpline(knots=n_bins, interval=interval),
            axis_size=n_dims,
        )()
        self.spline = Vmap(spline, in_axes=eqx.if_array(0))

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        return self.spline.transform_and_log_det(x, condition)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        return self.spline.inverse_and_log_det(y, condition)


class HistogramCDF(AbstractBijection):
    """Marginal Gaussianization via a per-dimension empirical histogram CDF.

    Maps each dimension through its empirical CDF estimated from training
    data. The CDF is piecewise-linear: equal-width histogram bins with linear
    interpolation between bin edges. Forward yields uniform marginals on
    `[0, 1]`; out-of-training-range inputs are clamped to `{0, 1}` with
    `log_det = −∞` (signals "outside the support of the fitted CDF").

    Construction is two-phase: build an unfitted instance with
    ``HistogramCDF(n_bins, shape)``, then call ``.fit(data)`` to get a
    fitted instance with concrete bin edges, densities, and CDF breakpoints.
    The fitted instance is itself a fully-formed transform — pass it to
    flows or call ``transform_and_log_det`` directly.

    Args:
        n_bins: Number of equal-width histogram bins per dimension.
        shape: Event shape ``(n_dims,)``. Only 1-D events are supported
            (each dim is independently histogrammed).
        bin_edges: Optional pre-fit bin edges of shape ``(n_dims, n_bins+1)``.
            Use ``.fit(data)`` instead of passing this directly.
        bin_pdf: Optional pre-fit per-bin density of shape ``(n_dims, n_bins)``.
        cdf_edges: Optional pre-fit CDF values at bin edges of shape
            ``(n_dims, n_bins+1)``.

    Shape:
        - Input  ``x``:  ``(n_dims,)``
        - Output ``y``:  ``(n_dims,)`` in ``[0, 1]``
        - ``log_det``:   scalar (sum of per-dim ``log(density)``)

    Example:
        Fit and transform on Gaussian data:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import HistogramCDF
        >>>
        >>> data = jr.normal(jr.key(0), (5000, 3))
        >>> hist = HistogramCDF(n_bins=64, shape=(3,)).fit(data)
        >>> y, log_det = hist.transform_and_log_det(jnp.zeros(3))
        >>> # y is roughly [0.5, 0.5, 0.5] (CDF at 0 of N(0,1) ≈ 0.5).
    """

    n_bins: int = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    bin_edges: Array | None
    bin_pdf: Array | None
    cdf_edges: Array | None

    def __init__(
        self,
        n_bins: int,
        shape: tuple[int, ...],
        bin_edges: Array | None = None,
        bin_pdf: Array | None = None,
        cdf_edges: Array | None = None,
    ):
        if len(shape) != 1:
            raise ValueError("HistogramCDF only supports 1D inputs.")
        self.n_bins = n_bins
        self.shape = shape
        self.bin_edges = bin_edges
        self.bin_pdf = bin_pdf
        self.cdf_edges = cdf_edges

    def fit(self, data: ArrayLike) -> HistogramCDF:
        """Fit the empirical CDF to ``data`` and return a new fitted instance.

        Args:
            data: Array of shape ``(n_samples, n_dims)`` from which to
                estimate the marginal CDFs.

        Returns:
            A new :class:`HistogramCDF` with concrete ``bin_edges``,
            ``bin_pdf``, and ``cdf_edges`` populated. Idempotent: refitting
            with the same data yields the same fitted parameters.
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

        # counts: (n_dims, n_bins). Smooth by 1e-6 to avoid log(0).
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
        return HistogramCDF(self.n_bins, self.shape, bin_edges, pdf, cdf_edges)

    def _check_fitted(self):
        if self.bin_edges is None or self.bin_pdf is None or self.cdf_edges is None:
            raise ValueError("HistogramCDF requires fitting data first.")

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        """Map x → uniform via the fitted CDF; log_det = sum(log density)."""
        self._check_fitted()
        x_arr = jnp.asarray(x)
        # edges: (n_dims, n_bins+1); pdf: (n_dims, n_bins); cdf: (n_dims, n_bins+1)
        edges = self.bin_edges  # type: ignore[assignment]
        pdf = self.bin_pdf  # type: ignore[assignment]
        cdf_edges = self.cdf_edges  # type: ignore[assignment]

        def _transform(x_i, edges_i, pdf_i, cdf_i):
            # x_i: scalar; edges_i: (n_bins+1,); pdf_i, cdf_i: (n_bins+1,)/(n_bins,)
            idx = jnp.clip(
                jnp.searchsorted(edges_i, x_i, side="right") - 1, 0, self.n_bins - 1
            )
            left = edges_i[idx]
            density = pdf_i[idx]
            cdf_left = cdf_i[idx]
            inside = (x_i >= edges_i[0]) & (x_i <= edges_i[-1])
            # Linear interpolation within bin: y = cdf_left + density * (x - left)
            y_i = cdf_left + density * (x_i - left)
            y_i = jnp.where(x_i < edges_i[0], 0.0, y_i)
            y_i = jnp.where(x_i > edges_i[-1], 1.0, y_i)
            log_det = jnp.where(inside, jnp.log(density), -jnp.inf)
            return y_i, log_det

        # vmap over n_dims: y: (n_dims,), log_det: (n_dims,) -> sum to scalar
        y, log_det = jax.vmap(_transform)(x_arr, edges, pdf, cdf_edges)
        return y, jnp.sum(log_det)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        """Map uniform y → x via the inverse CDF; log_det = -sum(log density)."""
        self._check_fitted()
        y_arr = jnp.asarray(y)
        edges = self.bin_edges  # type: ignore[assignment]
        pdf = self.bin_pdf  # type: ignore[assignment]
        cdf_edges = self.cdf_edges  # type: ignore[assignment]

        def _inverse(y_i, edges_i, pdf_i, cdf_i):
            # y_i: scalar in [0, 1] (clipped); searchsorted on the CDF
            # gives the bin containing y, then we invert the linear interp.
            # Strict `<` / `>` for the outside check: exact boundaries
            # ``y == 0`` / ``y == 1`` map to ``edges[0]`` / ``edges[-1]`` and
            # carry a finite log_det, consistent with forward_and_log_det
            # which treats ``x == edges[0]`` / ``x == edges[-1]`` as inside.
            y_clipped = jnp.clip(y_i, 0.0, 1.0)
            idx = jnp.clip(
                jnp.searchsorted(cdf_i, y_clipped, side="right") - 1,
                0,
                self.n_bins - 1,
            )
            cdf_left = cdf_i[idx]
            density = pdf_i[idx]
            left = edges_i[idx]
            x_i = left + (y_clipped - cdf_left) / density
            outside = (y_i < 0.0) | (y_i > 1.0)
            x_i = jnp.where(y_i < 0.0, edges_i[0], x_i)
            x_i = jnp.where(y_i > 1.0, edges_i[-1], x_i)
            log_det = jnp.where(outside, -jnp.inf, -jnp.log(density))
            return x_i, log_det

        # vmap over n_dims: x: (n_dims,), log_det: (n_dims,) -> sum to scalar
        x, log_det = jax.vmap(_inverse)(y_arr, edges, pdf, cdf_edges)
        return x, jnp.sum(log_det)


class InverseGaussCDF(AbstractBijection):
    """Apply the inverse Gaussian CDF (probit function) element-wise.

    This maps uniform marginals to Gaussian marginals using the probit function.
    It is typically used after a CDF transform.

    Args:
        shape: Shape of the input.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x = jnp.asarray(x)
        x_clipped = jnp.clip(x, 1e-6, 1 - 1e-6)
        y = jax.scipy.special.ndtri(x_clipped)
        log_det = jnp.sum(-jstats.norm.logpdf(y))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        y = jnp.asarray(y)
        x = jax.scipy.special.ndtr(y)
        log_det = jnp.sum(jstats.norm.logpdf(y))
        return x, log_det


__all__ = [
    "HistogramCDF",
    "InverseGaussCDF",
    "MixtureGaussianCDF",
    "MixtureLogisticCDF",
    "RQSplineMarginal",
]
