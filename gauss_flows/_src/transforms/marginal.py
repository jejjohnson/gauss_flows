"""Marginal transforms for Gaussianization flows.

These bijections operate element-wise along the marginal dimensions, providing
various ways to Gaussianize univariate marginal distributions.
"""

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
    """Marginal Gaussianization via histogram CDF.

    Fits a piecewise-linear CDF using a histogram approximation to each
    marginal dimension, then applies the probit (Gaussian inverse CDF).

    Args:
        n_bins: Number of histogram bins.
        shape: Shape of the input (n_dims,).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    log_bin_heights: Array

    def __init__(self, n_bins: int, shape: tuple[int, ...]):
        if len(shape) != 1:
            raise ValueError("HistogramCDF only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        self.log_bin_heights = jnp.zeros((n_dims, n_bins))

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        raise NotImplementedError("HistogramCDF requires fitting data first.")

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        raise NotImplementedError("HistogramCDF requires fitting data first.")


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
    "InverseGaussCDF",
    "MixtureGaussianCDF",
    "MixtureLogisticCDF",
    "RQSplineMarginal",
]
