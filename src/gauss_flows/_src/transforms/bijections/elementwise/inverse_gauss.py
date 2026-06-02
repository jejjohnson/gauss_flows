"""Elementwise inverse Gaussian CDF (probit)."""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from flowjax.bijections import AbstractBijection
from jaxtyping import ArrayLike


class InverseGaussCDF(AbstractBijection):
    """Apply the inverse Gaussian CDF (probit) element-wise.

    Maps uniform marginals on ``[0, 1]`` to standard-Gaussian marginals via the
    probit function ``Φ⁻¹``. This is the standard final step of a marginal
    Gaussianization layer: a CDF transform (histogram, mixture, …) sends data to
    uniform, then this layer sends uniform to ``N(0, 1)``. The forward map is
    ``y = Φ⁻¹(x)``; inputs are clipped to ``[1e−6, 1 − 1e−6]`` to keep the probit
    finite at the boundaries. The inverse is the Gaussian CDF ``x = Φ(y)``.

    Args:
        shape: Event shape ``(n_dims,)``. The transform is applied
            independently to each entry.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)``, scalar log_det
        - inverse_and_log_det:   ``(n_dims,)`` → ``(n_dims,)``, scalar log_det

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import InverseGaussCDF
        >>> t = InverseGaussCDF(shape=(3,))
        >>> x = jnp.array([0.2, 0.5, 0.8])
        >>> y, log_det = t.transform_and_log_det(x)
        >>> y.shape
        (3,)
        >>> bool(jnp.allclose(y[1], 0.0))  # Φ⁻¹(0.5) == 0
        True
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


__all__ = ["InverseGaussCDF"]
