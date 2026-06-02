"""Trainable Gaussian mixture base distribution.

Natural companion to `MixtureGaussianCDF` (the marginal transform):
provides a joint multi-modal base for Gaussianization pipelines on targets
that are multi-modal or fat-tailed.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
from flowjax.distributions import AbstractDistribution
from jax import Array
from jax.scipy.special import logsumexp
from jaxtyping import PRNGKeyArray


class GaussianMixture(AbstractDistribution):
    """Trainable multivariate Gaussian-mixture base distribution.

    Parameterises a ``K``-component mixture of ``D``-dimensional Gaussians
    with trainable component means, scales, and weights. The density is
    ``p(x) = ∑ₖ wₖ · N(x; locₖ, Σₖ)`` with mixing weights ``w = softmax(
    log_weights)``. Two covariance parameterisations are supported: a
    diagonal one (one scale per component per dim) and a full-covariance
    one (a per-component lower-triangular Cholesky factor).

    Use as a flow base when the Gaussianized target is multi-modal or
    fat-tailed — the natural companion to `MixtureGaussianCDF`,
    the per-dimension marginal transform.

    Args:
        key: JAX random key used to initialise the component means.
        n_components: Number of mixture components ``K``.
        event_shape: Shape of a single sample ``(D,)``. 1-D only.
        diagonal: If ``True`` (default), diagonal covariance via
            ``log_scale``. If ``False``, full covariance via a per-component
            lower-triangular Cholesky factor. Defaults to ``True``.
        loc_init_scale: Std of the Gaussian used to initialise component
            means. Defaults to ``1.0``.
        log_scale_init: Initial value of ``log_scale`` (resp. Cholesky
            log-diagonal for ``diagonal=False``). Defaults to ``0.0``
            (unit scale at init).

    Note:
        The event shape is stored as ``self.shape`` per the flowjax
        distribution contract; mixing weights are uniform at init.

    Shape:
        - log_prob: ``(D,)`` → scalar    (batch via vmap / leading axes)
        - sample:   ``(key, sample_shape)`` → ``(*sample_shape, D)``

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import GaussianMixture
        >>> dist = GaussianMixture(jr.key(0), n_components=3, event_shape=(2,))
        >>> dist.sample(jr.key(1), (8,)).shape
        (8, 2)
        >>> dist.log_prob(jnp.zeros(2)).shape
        ()
    """

    n_components: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    log_scale: Array
    strict_tril: Array
    log_weights: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        n_components: int,
        event_shape: tuple[int, ...],
        *,
        diagonal: bool = True,
        loc_init_scale: float = 1.0,
        log_scale_init: float = 0.0,
    ):
        if n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}.")
        if len(event_shape) != 1:
            raise ValueError("GaussianMixture only supports 1D event shapes.")
        d = event_shape[0]
        k = n_components
        self.n_components = k
        self.diagonal = diagonal
        self.shape = event_shape
        self.loc = jr.normal(key, (k, d)) * loc_init_scale
        self.log_scale = jnp.full((k, d), log_scale_init)
        n_strict = d * (d - 1) // 2
        self.strict_tril = (
            jnp.zeros((k, n_strict)) if not diagonal else jnp.zeros((k, 0))
        )
        self.log_weights = jnp.zeros((k,))

    def _build_L(self) -> Array:
        """Build component Cholesky factors ``L`` of shape ``(K, D, D)``.

        For ``diagonal=True`` this is simply ``diag(exp(log_scale))``.
        For ``diagonal=False``, the positive diagonal comes from
        ``exp(log_scale)`` and the strict lower triangle from
        ``strict_tril``.
        """
        d = self.log_scale.shape[1]
        diag_scale = jnp.exp(self.log_scale)
        L_diag = jax.vmap(jnp.diag)(diag_scale)
        if self.diagonal:
            return L_diag
        # Scatter strict_tril into the strict lower triangle of each (D, D) block.
        idx = jnp.tril_indices(d, k=-1)

        def _fill(block, strict):
            return block.at[idx].set(strict)

        return jax.vmap(_fill)(L_diag, self.strict_tril)

    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        log_weights = jax.nn.log_softmax(self.log_weights)
        if self.diagonal:
            scales = jnp.exp(self.log_scale)
            # jstats.norm.logpdf broadcasts (D,) vs (K, D) -> (K, D); sum last axis.
            log_pdfs = jnp.sum(
                jstats.norm.logpdf(x[None, :], self.loc, scales), axis=-1
            )
        else:
            L = self._build_L()

            def _component_logpdf(loc_k: Array, L_k: Array) -> Array:
                diff = x - loc_k
                z = jax.scipy.linalg.solve_triangular(L_k, diff, lower=True)
                d = diff.shape[0]
                log_det = jnp.sum(jnp.log(jnp.diag(L_k)))
                return -0.5 * d * jnp.log(2.0 * jnp.pi) - log_det - 0.5 * jnp.sum(z**2)

            log_pdfs = jax.vmap(_component_logpdf)(self.loc, L)
        return logsumexp(log_weights + log_pdfs)

    def _sample(self, key: PRNGKeyArray, condition: Array | None = None) -> Array:
        k_comp, k_normal = jr.split(key)
        weights = jax.nn.softmax(self.log_weights)
        component = jr.choice(k_comp, self.n_components, p=weights)
        z = jr.normal(k_normal, self.shape)
        if self.diagonal:
            scale = jnp.exp(self.log_scale[component])
            return self.loc[component] + scale * z
        L = self._build_L()[component]
        return self.loc[component] + L @ z
