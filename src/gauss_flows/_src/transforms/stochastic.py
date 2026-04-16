"""Stochastic transforms: both directions sample from a non-degenerate kernel.

A "stochastic transform" in the SurVAE hierarchy is one where ``forward`` and
``inverse`` are both random — neither direction is a deterministic function of
its input. The canonical example is :class:`StochasticPermutation`: the
forward map is a uniform draw from the symmetric group, and so is the inverse.

These transforms always contribute a lower-bound term to ``log_prob`` (see
``AbstractStochastic.lower_bound = True``), but for simple cases like a uniform
permutation the contribution is exactly zero.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src._protocols import ConditionalDistribution
from gauss_flows._src.transforms.base import AbstractStochastic


class VAE(AbstractStochastic):
    """Variational autoencoder as a SurVAE stochastic transform.

    The forward direction encodes data into a latent sample and returns the
    per-event ELBO term ``log p(x | z) − log q(z | x)``. The inverse samples
    from the decoder for generation; its log-det is zero (the inverse is not
    used inside ``log_prob``).

    Args:
        encoder: Conditional distribution ``q(z | x)`` supplying ``sample`` and
            ``log_prob`` methods that accept the conditioning variable via the
            ``condition`` keyword.
        decoder: Conditional distribution ``p(x | z)`` with the same protocol.

    Shape:
        - Input ``x``: ``encoder`` condition shape
        - Output ``z``: ``encoder`` sample shape
        - ``log_det``: scalar (shape ``()``)

    Example:
        Encode and decode a single event, then use inside a :class:`SurVAEFlow`:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from flowjax.distributions import Normal
        >>> from gauss_flows import SurVAEFlow, VAE
        >>>
        >>> encoder = Normal(jnp.zeros(2))
        >>> decoder = Normal(jnp.zeros(2))
        >>> vae = VAE(encoder, decoder)
        >>> x = jnp.array([0.5, -1.2])
        >>> z, log_det = vae.forward_and_log_det(x, jr.key(0))
        >>> x_sample, _ = vae.inverse_and_log_det(z, jr.key(1))
        >>>
        >>> base = Normal(jnp.zeros(2))
        >>> flow = SurVAEFlow(base, [vae])
    """

    encoder: ConditionalDistribution
    decoder: ConditionalDistribution

    def __init__(
        self,
        encoder: ConditionalDistribution,
        decoder: ConditionalDistribution,
    ):
        self.encoder = encoder
        self.decoder = decoder

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample ``z ~ q(z | x)``; return ``log p(x | z) − log q(z | x)``."""
        del cond
        x_arr = jnp.asarray(x)
        z = self.encoder.sample(key, condition=x_arr)
        log_p = self.decoder.log_prob(x_arr, condition=jnp.asarray(z))
        log_q = self.encoder.log_prob(z, condition=x_arr)
        log_det = log_p - log_q
        return jnp.asarray(z), log_det

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample ``x ~ p(x | z)``; inverse log-det is zero."""
        del cond
        x = self.decoder.sample(key, condition=jnp.asarray(z))
        return x, jnp.zeros(())


class StochasticPermutation(AbstractStochastic):
    """Uniform random permutation along one axis (zero log-det).

    Both directions draw a fresh permutation σ of ``{0, …, n−1}`` (with
    ``n = shape[axis]``) and apply it along ``axis``. The forward and inverse
    sampling distributions are identical:

        q_forward(z | x) = q_inverse(x | z) = 1 / n!  for each σ ∈ S_n

    so the entropy contribution to ``log_prob`` cancels and ``log_det = 0`` in
    both directions. The transform is volume-preserving in the discrete sense:
    every output is a permutation of the input.

    Args:
        shape: Event shape of a single input (no leading batch dim — callers
            vmap externally; see the project-wide single-event convention in
            ``CLAUDE.md``).
        axis: Axis along which to permute (default ``0``). Wrapped modulo
            ``len(shape)``.

    Shape:
        - Input ``x``: ``shape``
        - Output ``z``: ``shape`` (a permutation of ``x`` along ``axis``)
        - ``log_det``: scalar (shape ``()``)

    Example:
        Permute a 1D event:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import StochasticPermutation
        >>>
        >>> surj = StochasticPermutation((4,))
        >>> z, log_det = surj.forward_and_log_det(jnp.arange(4.0), jr.key(0))
        >>> # z is a permutation of [0., 1., 2., 3.]; log_det == 0.

        Permute the columns of a 2D event:

        >>> surj2 = StochasticPermutation((3, 4), axis=1)
        >>> z, _ = surj2.forward_and_log_det(jnp.zeros((3, 4)), jr.key(0))
        >>> # z.shape == (3, 4); same permutation applied to every row.

        Use inside a SurVAEFlow (the container vmaps over leading sample dims):

        >>> from gauss_flows import SurVAEFlow
        >>> from flowjax.distributions import Normal
        >>> base = Normal(jnp.zeros(4))
        >>> flow = SurVAEFlow(base, [StochasticPermutation((4,))])
    """

    shape: tuple[int, ...]
    axis: int
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...], axis: int = 0):
        if not shape:
            raise ValueError("Shape must be non-empty for StochasticPermutation.")
        self.shape = shape
        self.axis = axis % len(shape)

    def _permute(self, key: PRNGKeyArray, x: Array) -> Array:
        # x: shape; perm: (shape[axis],); out: shape
        perm = jr.permutation(key, self.shape[self.axis])
        return jnp.take(x, perm, axis=self.axis)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample z = σ_forward(x) with σ_forward ~ Uniform(S_n)."""
        del cond
        z = self._permute(key, jnp.asarray(x))
        return z, jnp.zeros(())

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample x = σ_inverse(z) with σ_inverse ~ Uniform(S_n)."""
        del cond
        x = self._permute(key, jnp.asarray(z))
        return x, jnp.zeros(())


__all__ = ["VAE", "StochasticPermutation"]
