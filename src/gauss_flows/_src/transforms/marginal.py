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
    """Marginal Gaussianization via histogram-based empirical CDF.

    Fits a piecewise-linear CDF per dimension from training data, using
    equal-width histogram bins. The fitted bin edges, densities, and CDF
    breakpoints are stored as static arrays.

    Args:
        n_bins: Number of histogram bins.
        shape: Shape of the input (n_dims,).
    """

    n_bins: int = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    bin_edges: Array | None = eqx.field(default=None, static=True)
    bin_pdf: Array | None = eqx.field(default=None, static=True)
    cdf_edges: Array | None = eqx.field(default=None, static=True)

    def __init__(self, n_bins: int, shape: tuple[int, ...]):
        if len(shape) != 1:
            raise ValueError("HistogramCDF only supports 1D inputs.")
        self.n_bins = n_bins
        self.shape = shape

    def fit(self, data: ArrayLike) -> HistogramCDF:
        """Fit empirical CDF to data.

        Returns a new :class:`HistogramCDF` instance with static histogram
        parameters. Repeated calls with the same data are idempotent.
        """

        values = jnp.asarray(data)
        if values.ndim != 2 or values.shape[1] != self.shape[0]:
            raise ValueError(
                f"Expected data shape (n_samples, {self.shape[0]}), got {values.shape}."
            )
        mins = jnp.min(values, axis=0)
        maxs = jnp.max(values, axis=0)

        def _edges(lo, hi):
            span = jnp.where(hi > lo, hi - lo, 1.0)
            start = lo - 0.01 * span
            end = hi + 0.01 * span
            return jnp.linspace(start, end, self.n_bins + 1)

        bin_edges = jax.vmap(_edges)(mins, maxs)

        def _hist(column, edges):
            counts, _ = jnp.histogram(column, bins=edges)
            return counts

        counts = jax.vmap(_hist)(values.T, bin_edges)
        counts = counts + 1e-6  # Smooth to avoid zero-density bins
        totals = jnp.sum(counts, axis=1, keepdims=True)
        widths = jnp.diff(bin_edges, axis=1)
        pdf = counts / (totals * widths)
        cdf_edges = jnp.concatenate(
            [jnp.zeros((self.shape[0], 1)), jnp.cumsum(pdf * widths, axis=1)],
            axis=1,
        )
        cdf_edges = jnp.minimum(cdf_edges, 1.0)
        fitted = HistogramCDF(self.n_bins, self.shape)
        object.__setattr__(fitted, "bin_edges", bin_edges)
        object.__setattr__(fitted, "bin_pdf", pdf)
        object.__setattr__(fitted, "cdf_edges", cdf_edges)
        return fitted

    def _check_fitted(self):
        if self.bin_edges is None or self.bin_pdf is None or self.cdf_edges is None:
            raise ValueError("HistogramCDF requires fitting data first.")

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        self._check_fitted()
        x_arr = jnp.asarray(x)
        edges = self.bin_edges  # type: ignore[assignment]
        pdf = self.bin_pdf  # type: ignore[assignment]
        cdf_edges = self.cdf_edges  # type: ignore[assignment]

        def _transform(x_i, edges_i, pdf_i, cdf_i):
            idx = jnp.clip(
                jnp.searchsorted(edges_i, x_i, side="right") - 1, 0, self.n_bins - 1
            )
            left = edges_i[idx]
            density = pdf_i[idx]
            cdf_left = cdf_i[idx]
            inside = (x_i >= edges_i[0]) & (x_i <= edges_i[-1])
            y_i = cdf_left + density * (x_i - left)
            y_i = jnp.where(x_i < edges_i[0], 0.0, y_i)
            y_i = jnp.where(x_i > edges_i[-1], 1.0, y_i)
            log_det = jnp.where(inside, jnp.log(density), -jnp.inf)
            return y_i, log_det

        y, log_det = jax.vmap(_transform)(x_arr, edges, pdf, cdf_edges)
        return y, jnp.sum(log_det)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        self._check_fitted()
        y_arr = jnp.asarray(y)
        edges = self.bin_edges  # type: ignore[assignment]
        pdf = self.bin_pdf  # type: ignore[assignment]
        cdf_edges = self.cdf_edges  # type: ignore[assignment]

        def _inverse(y_i, edges_i, pdf_i, cdf_i):
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
            outside = (y_i <= 0.0) | (y_i >= 1.0)
            x_i = jnp.where(y_i <= 0.0, edges_i[0], x_i)
            x_i = jnp.where(y_i >= 1.0, edges_i[-1], x_i)
            log_det = jnp.where(outside, -jnp.inf, -jnp.log(density))
            return x_i, log_det

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
