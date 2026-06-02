"""Probabilistic-PCA base distribution (low-rank + diagonal Gaussian).

Parameterises ``Sigma = W W^T + sigma^2 I`` with ``W in R^{D x K}`` via
:class:`GaussianPCA`. ``O(DK)`` parameters and ``O(DK)`` sampling via the
reparametrisation ``x = loc + W z + sigma eps``. Log-prob uses the
Woodbury identity / matrix-determinant lemma so inversion and
log-determinant stay ``O(K^3)`` + ``O(DK)`` rather than ``O(D^3)``.

Reference: Tipping & Bishop 1999, *Probabilistic Principal Component
Analysis*.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import AbstractDistribution
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


class GaussianPCA(AbstractDistribution):
    """Probabilistic-PCA Gaussian base distribution (low-rank + diagonal Σ).

    ``x ~ N(loc, Σ)`` with low-rank-plus-diagonal covariance
    ``Σ = W Wᵀ + σ² I`` and ``W ∈ ℝ^{D × K}``. This costs only ``O(DK)``
    parameters; sampling uses the reparametrisation ``x = loc + W z + σ ε``
    (``z ∈ ℝ^K``, ``ε ∈ ℝ^D``), and ``log_prob`` applies the Woodbury
    identity / matrix-determinant lemma so the inverse and log-determinant
    cost ``O(K³) + O(DK)`` rather than ``O(D³)``.

    Use as a flow base when the Gaussianized target has approximately
    low-rank correlation structure in ``D`` dimensions.

    Args:
        key: JAX random key used to initialise ``W``.
        event_shape: Shape of a single sample ``(D,)``. 1-D only.
        latent_dim: Rank ``K`` of the factor loading ``W`` (``1 ≤ K ≤ D``).
        loc: Mean vector. Scalar is broadcast to shape ``(D,)``.
            Defaults to ``0.0``.
        log_sigma_init: Initial value of ``log σ``. Defaults to ``0.0``.
        W_init_scale: Std of the Gaussian used to initialise ``W``.
            Defaults to ``0.1``.
        eps: Numerical floor added to ``σ²`` so the Cholesky of the
            ``K × K`` capacitance matrix stays well-defined and the
            Woodbury quadratic stays finite even if ``log σ`` drifts very
            negative during training. Defaults to ``1e-12`` — far below any
            reasonable target variance, with no effect on gradients in the
            well-conditioned regime.

    Shape:
        - log_prob: ``(D,)`` → scalar    (batch via vmap / leading axes)
        - sample:   ``(key, sample_shape)`` → ``(*sample_shape, D)``

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import GaussianPCA
        >>> dist = GaussianPCA(jr.key(0), event_shape=(5,), latent_dim=2)
        >>> dist.sample(jr.key(1), (8,)).shape
        (8, 5)
        >>> dist.log_prob(jnp.zeros(5)).shape
        ()

    References:
        Tipping & Bishop 1999, *Probabilistic Principal Component Analysis*.
    """

    latent_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    W: Array
    log_sigma: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        event_shape: tuple[int, ...],
        latent_dim: int,
        *,
        loc: ArrayLike = 0.0,
        log_sigma_init: float = 0.0,
        W_init_scale: float = 0.1,
        eps: float = 1e-12,
    ):
        if len(event_shape) != 1:
            raise ValueError("GaussianPCA only supports 1D event shapes.")
        d = event_shape[0]
        if not 1 <= latent_dim <= d:
            raise ValueError(f"latent_dim must be in [1, {d}], got {latent_dim}.")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}.")
        self.latent_dim = latent_dim
        self.eps = eps
        self.shape = event_shape
        self.loc = jnp.broadcast_to(jnp.asarray(loc, dtype=float), event_shape)
        self.W = jr.normal(key, (d, latent_dim)) * W_init_scale
        self.log_sigma = jnp.asarray(log_sigma_init, dtype=float)

    def _sigma2(self) -> Array:
        """Variance with a small floor so it cannot underflow to zero.

        Adding ``eps`` to ``exp(2*log_sigma)`` keeps the Woodbury quadratic
        finite and the capacitance matrix ``M`` strictly PD even if
        ``log_sigma`` drifts very negative during training.
        """
        return jnp.exp(2.0 * self.log_sigma) + self.eps

    def covariance(self) -> Array:
        """Return the explicit ``D x D`` covariance ``W W^T + sigma^2 I``."""
        return self.W @ self.W.T + self._sigma2() * jnp.eye(self.shape[0])

    def _sample(self, key: PRNGKeyArray, condition: Array | None = None) -> Array:
        k_z, k_eps = jr.split(key)
        z = jr.normal(k_z, (self.latent_dim,))
        eps_noise = jr.normal(k_eps, self.shape)
        sigma = jnp.sqrt(self._sigma2())
        return self.loc + self.W @ z + sigma * eps_noise

    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        d = self.shape[0]
        k = self.latent_dim
        sigma2 = self._sigma2()
        diff = x - self.loc
        # Small K x K capacitance matrix from the Woodbury identity.
        M = sigma2 * jnp.eye(k) + self.W.T @ self.W
        L = jnp.linalg.cholesky(M)
        Wt_diff = self.W.T @ diff
        y = jax.scipy.linalg.cho_solve((L, True), Wt_diff)
        # Sigma^-1 (x - loc) via Woodbury:
        #   quadratic = sigma^-2 * ||diff||^2 - sigma^-4 * (W^T diff)^T M^-1 (W^T diff).
        quad = jnp.sum(diff**2) / sigma2 - (Wt_diff @ y) / (sigma2**2)
        # Matrix determinant lemma:
        #   log|Sigma| = (D-K) log(sigma^2) + log|M|.
        log_det_M = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        log_det = (d - k) * jnp.log(sigma2) + log_det_M
        return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + quad)
