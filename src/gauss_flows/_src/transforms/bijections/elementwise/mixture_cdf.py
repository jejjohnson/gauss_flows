"""Mixture-CDF marginal bijections: Gaussian and logistic mixtures.

Both classes map each input dim independently through its mixture CDF
(→ uniform on [0, 1]) then through the inverse normal CDF (probit), giving
Gaussianised marginals. Inversion uses a bisection solver on each dim.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import jax.scipy.stats as jstats
from flowjax.bijections import AbstractBijection
from jax import Array
from jax.nn import softmax, softplus
from jaxtyping import ArrayLike


def _unit_interval_eps(x: Array) -> Array:
    """Smallest safe distance from 0/1 for the given floating dtype."""
    return jnp.asarray(jnp.finfo(x.dtype).eps, dtype=x.dtype)


def _ndtri_exp(log_p: Array, n_iter: int = 8) -> Array:
    """Probit ``ndtri(exp(log_p))`` evaluated from the log-probability directly.

    Computing ``ndtri(jnp.exp(log_p))`` discards all tail precision the moment
    ``exp(log_p)`` underflows: in float32 anything below ``finfo.tiny`` (~1e-38)
    collapses to the smallest normal, capping the probit at ~±12.9 and mapping
    every deep-tail sample to the same value. Instead solve ``log_ndtr(y) =
    log_p`` with Newton's method on the tail-stable ``log_ndtr``; the derivative
    is the inverse Mills ratio ``phi(y) / Phi(y) = exp(logpdf(y) - log_ndtr(y))``.

    ``log_p`` must be ``<= log(0.5)`` (i.e. the smaller of log-CDF/log-SF) so the
    initial probit guess stays finite.
    """
    dtype = log_p.dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    y = jax.scipy.special.ndtri(jnp.clip(jnp.exp(log_p), tiny, 1.0))

    def _step(_, y):
        log_ndtr = jax.scipy.special.log_ndtr(y)
        f = log_ndtr - log_p
        df = jnp.exp(jstats.norm.logpdf(y) - log_ndtr)
        return y - f / df

    return jax.lax.fori_loop(0, n_iter, _step, y)


class MixtureGaussianCDF(AbstractBijection):
    """Marginal Gaussianization via a mixture-of-Gaussians CDF.

    Maps each dimension independently through its Gaussian-mixture CDF
    (→ uniform on ``[0, 1]``) then through the inverse normal CDF (probit),
    giving Gaussianised marginals. Each dim has its own ``n_components`` means,
    log-scales, and log-weights; scales use a soft floor
    ``σ = softplus(log_scale) + 5e−3`` for training stability and weights are a
    softmax over ``log_weights``. The inverse runs a per-dim bisection solver to
    invert the (monotone) mixture CDF.

    Use as the marginal block of a Gaussianization / RBIG flow. Construct
    parameters at zero (uniform components) with ``MixtureGaussianCDF(...)``, or
    data-adapt the means/scales to per-dim quantiles via `from_data`.

    Args:
        n_components: Number of mixture components ``K`` per dimension.
        shape: Event shape ``(n_dims,)``. Only 1-D events are supported.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)``, scalar log_det
        - inverse_and_log_det:   ``(n_dims,)`` → ``(n_dims,)``, scalar log_det

    Examples:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import MixtureGaussianCDF
        >>> t = MixtureGaussianCDF(n_components=4, shape=(3,))
        >>> x = jnp.zeros(3)
        >>> y, log_det = t.transform_and_log_det(x)
        >>> y.shape
        (3,)
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

    def _scales(self) -> Array:
        """σ = softplus(log_scales) + 5e-3 (soft floor for training stability)."""
        return softplus(self.log_scales) + 5e-3

    @classmethod
    def from_data(
        cls,
        x: ArrayLike,
        n_components: int = 8,
    ) -> MixtureGaussianCDF:
        """Build a marginal layer with means at per-dim quantiles of ``x``.

        The per-dim component means are placed at evenly spaced quantiles
        of ``x[:, i]``, and the log-scales are set so ``softplus(log_scale)
        + 5e-3`` matches the inter-quantile spacing × per-dim std — each
        dim gets its *own* target scale based on its own standard deviation.
        Weights remain uniform. This gives a meaningful first forward pass
        without running the full RBIG fit.

        Args:
            x: Training data of shape ``(n, d)``.
            n_components: Mixture components ``K``. Defaults to 8.

        Returns:
            A `MixtureGaussianCDF` with data-adapted means/log-scales.
        """
        import numpy as np
        from paramax.utils import inv_softplus

        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (n, d); got shape {x.shape}")
        n_dims = x.shape[-1]
        qs = np.linspace(0.5 / n_components, 1.0 - 0.5 / n_components, n_components)
        means = np.stack(
            [np.quantile(x[:, i], qs) for i in range(n_dims)], axis=0
        ).astype("float32")
        data_std_per_dim = np.asarray(x.std(axis=0), dtype=np.float32)  # (d,)
        if n_components > 1:
            target_scale = np.maximum(
                float(np.mean(np.diff(qs))) * data_std_per_dim, 0.1
            )
        else:
            target_scale = np.maximum(data_std_per_dim, 0.1)
        # softplus(raw) + 5e-3 ≈ target_scale → raw = inv_softplus(target_scale - 5e-3)
        raw_log_scale_per_dim = np.asarray(
            inv_softplus(jnp.asarray(np.maximum(target_scale - 5e-3, 5e-3)))
        )  # (d,)
        log_scales = jnp.broadcast_to(
            jnp.asarray(raw_log_scale_per_dim)[:, None], (n_dims, n_components)
        )

        obj = cls(n_components=n_components, shape=(n_dims,))
        obj = eqx.tree_at(lambda m: m.means, obj, jnp.asarray(means))
        obj = eqx.tree_at(lambda m: m.log_scales, obj, log_scales)
        return obj

    def _gmm_logcdf(
        self, x: Array, means: Array, scales: Array, weights: Array
    ) -> Array:
        """Log CDF of a 1D Gaussian mixture evaluated at x."""
        log_cdfs = jstats.norm.logcdf(x[:, None], loc=means, scale=scales)
        return jsp_special.logsumexp(jnp.log(weights) + log_cdfs, axis=-1)

    def _gmm_logsf(
        self, x: Array, means: Array, scales: Array, weights: Array
    ) -> Array:
        """Log survival function of a 1D Gaussian mixture evaluated at x."""
        log_sfs = jstats.norm.logsf(x[:, None], loc=means, scale=scales)
        return jsp_special.logsumexp(jnp.log(weights) + log_sfs, axis=-1)

    def _gmm_logpdf(
        self, x: Array, means: Array, scales: Array, weights: Array
    ) -> Array:
        """Log PDF of a 1D Gaussian mixture evaluated at x."""
        log_pdfs = jstats.norm.logpdf(x[:, None], loc=means, scale=scales)
        return jnp.log(jnp.sum(weights * jnp.exp(log_pdfs), axis=-1) + 1e-38)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x = jnp.asarray(x)
        scales = self._scales()
        weights = softmax(self.log_weights, axis=-1)

        # GMM CDF -> uniform -> probit, done entirely in log-space. Take the
        # probit of whichever tail is smaller (log-CDF in the lower tail,
        # log-SF in the upper) and feed the *log*-probability straight into the
        # probit via `_ndtri_exp`. Exponentiating first underflows in the tails
        # (float32 caps the probit near ±12.9), collapsing distinct tail samples
        # onto the same y and destroying invertibility.
        log_cdf = self._gmm_logcdf(x, self.means, scales, weights)
        log_sf = self._gmm_logsf(x, self.means, scales, weights)
        lower = log_cdf <= log_sf
        mag = _ndtri_exp(jnp.where(lower, log_cdf, log_sf))
        y = jnp.where(lower, mag, -mag)

        # Log det: log |dy/dx| = log |phi^{-1}'(u) * gmm_pdf(x)|
        # = log_gmm_pdf(x) - log_norm_pdf(y)
        log_pdf_x = self._gmm_logpdf(x, self.means, scales, weights)
        log_pdf_y = jstats.norm.logpdf(y)
        log_det = jnp.sum(log_pdf_x - log_pdf_y)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        from gauss_flows._src.utils import bisection_inverse

        y = jnp.asarray(y)
        scales = self._scales()
        weights = softmax(self.log_weights, axis=-1)

        # Invert in log-space to mirror the forward pass: solve
        # log-CDF(x) = log Φ(y) in the lower tail and log-SF(x) = log Φ(-y) in
        # the upper tail. Bracketing on the linear CDF/SF target would underflow
        # for |y| ≳ 5.5 in float32 and collapse the bisection onto the bounds.
        def _invert_lower(ops):
            y_i, means_i, scales_i, weights_i = ops
            target = jax.scipy.special.log_ndtr(y_i)

            def _fn(xi):
                return self._gmm_logcdf(
                    xi[None], means_i[None], scales_i[None], weights_i[None]
                )[0]

            return bisection_inverse(_fn, target)

        def _invert_upper(ops):
            y_i, means_i, scales_i, weights_i = ops
            target = jax.scipy.special.log_ndtr(-y_i)

            def _fn(xi):
                return -self._gmm_logsf(
                    xi[None], means_i[None], scales_i[None], weights_i[None]
                )[0]

            return bisection_inverse(_fn, -target)

        def _invert_i(y_i, means_i, scales_i, weights_i):
            return jax.lax.cond(
                y_i < 0.0,
                _invert_lower,
                _invert_upper,
                (y_i, means_i, scales_i, weights_i),
            )

        x = jax.vmap(_invert_i)(y, self.means, scales, weights)

        # Log det of inverse = -log_det of forward
        log_pdf_x = self._gmm_logpdf(x, self.means, scales, weights)
        log_pdf_y = jstats.norm.logpdf(y)
        log_det = -jnp.sum(log_pdf_x - log_pdf_y)
        return x, log_det


class MixtureLogisticCDF(AbstractBijection):
    """Marginal Gaussianization via a mixture-of-logistics CDF.

    Identical in structure to `MixtureGaussianCDF` but each per-dim
    mixture component is logistic rather than Gaussian: the component CDF is the
    sigmoid ``σ((x − μ) / s)``. Maps each dimension through its logistic-mixture
    CDF (→ uniform on ``[0, 1]``) then through the inverse normal CDF (probit).
    Scales use the same soft floor ``s = softplus(log_scale) + 5e−3`` and
    weights are a softmax over ``log_weights``; the inverse uses a per-dim
    bisection solver.

    Args:
        n_components: Number of mixture components ``K`` per dimension.
        shape: Event shape ``(n_dims,)``. Only 1-D events are supported.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)``, scalar log_det
        - inverse_and_log_det:   ``(n_dims,)`` → ``(n_dims,)``, scalar log_det

    Examples:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import MixtureLogisticCDF
        >>> t = MixtureLogisticCDF(n_components=4, shape=(3,))
        >>> y, log_det = t.transform_and_log_det(jnp.zeros(3))
        >>> y.shape
        (3,)
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

    def _scales(self) -> Array:
        """σ = softplus(log_scales) + 5e-3."""
        return softplus(self.log_scales) + 5e-3

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
        x = jnp.asarray(x)
        scales = self._scales()
        weights = softmax(self.log_weights, axis=-1)
        eps = _unit_interval_eps(x)

        u = self._mixture_cdf(x, self.means, scales, weights)
        u = jnp.clip(u, eps, 1 - eps)
        y = jax.scipy.special.ndtri(u)

        log_pdf_x = self._mixture_logpdf(x, self.means, scales, weights)
        log_pdf_y = jstats.norm.logpdf(y)
        log_det = jnp.sum(log_pdf_x - log_pdf_y)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        from gauss_flows._src.utils import bisection_inverse

        y = jnp.asarray(y)
        scales = self._scales()
        weights = softmax(self.log_weights, axis=-1)
        eps = _unit_interval_eps(y)

        u = jnp.clip(jax.scipy.special.ndtr(y), eps, 1.0 - eps)

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
