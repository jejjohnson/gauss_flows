"""Mixture-CDF marginal bijections: Gaussian and logistic mixtures.

Both classes map each input dim independently through its mixture CDF
(→ uniform on [0, 1]) then through the inverse normal CDF (probit), giving
Gaussianised marginals. Inversion uses a bisection solver on each dim.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from flowjax.bijections import AbstractBijection
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


__all__ = ["MixtureGaussianCDF", "MixtureLogisticCDF"]
