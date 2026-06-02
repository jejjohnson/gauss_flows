"""Variational autoencoder as a SurVAE stochastic transform."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src._protocols import ConditionalDistribution
from gauss_flows._src.transforms.base import AbstractStochastic


class VAE(AbstractStochastic):
    r"""Variational autoencoder as a SurVAE stochastic transform.

    Wraps an encoder ``q(z | x)`` and decoder ``p(x | z)`` so that a VAE
    becomes a single link in a `SurVAEFlow` chain. Stacking ``N`` of
    these yields an ``N``-level HVAE with no bespoke container code — the
    chain machinery already handles key splitting and ELBO accumulation.

    Forward (data → latent, used inside ``log_prob``):
    ```python
    z ~ q(z | x)
    log_det = log p(x | z) − log q(z | x)
    ```

    The forward ``log_det`` is exactly the per-event ELBO contribution of a
    single stochastic lift (Nielsen et al. 2020, eq. 7): together with the
    base-distribution term ``log p(z)`` it yields the standard ELBO
    ``E_q[log p(x|z) + log p(z) − log q(z|x)]``.

    Inverse (latent → data, used for generation):
    ```python
    x ~ p(x | z)
    log_det = 0   (inverse is not consumed by ``log_prob``)
    ```

    Args:
        encoder: Conditional distribution ``q(z | x)``. Must expose
            ``sample(key, *, condition)`` and
            ``log_prob(value, *, condition)`` — the
            `ConditionalDistribution` protocol. **``sample`` must
            be reparameterized** (e.g. location–scale Gaussian with
            ``mean + scale * eps``); otherwise gradients through the ELBO
            are biased.
        decoder: Conditional distribution ``p(x | z)`` with the same
            protocol. Reparameterization is **not** required for the
            decoder — its ``log_prob`` is the only thing differentiated.

    Shape:
        - Input ``x``: ``encoder`` condition shape
        - Output ``z``: ``encoder`` sample shape
        - ``log_det``: scalar (shape ``()``)

    Note:
        Unconditional distributions such as ``flowjax.distributions.Normal``
        do **not** satisfy the `ConditionalDistribution` protocol
        (they have no ``condition=`` kwarg). A small wrapper that amortises
        the parameters on the conditioning variable is required — see the
        ``DiagGaussianConditional`` helper used in
        ``tests/test_transforms_stochastic.py`` for a minimal reference.

    Examples:
        Compose with a standard Normal prior to form a one-level VAE flow:
        ```python
        import jax.numpy as jnp
        from flowjax.distributions import Normal
        from gauss_flows import SurVAEFlow, VAE

        encoder = MyConditionalGaussian(...)   # q(z|x), reparameterized
        decoder = MyConditionalGaussian(...)   # p(x|z)
        vae = VAE(encoder, decoder)
        flow = SurVAEFlow(Normal(jnp.zeros(latent_dim)), [vae])
        ```

    References:
        Nielsen, Jaini, Hoogeboom, Winther, Welling.
        *SurVAE Flows: Surjections to Bridge the Gap between VAEs and
        Flows*, NeurIPS 2020.
    """

    encoder: ConditionalDistribution
    decoder: ConditionalDistribution
    cond_shape: ClassVar[None] = None

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
        r"""Sample ``z ~ q(z | x)`` and return ``log p(x | z) − log q(z | x)``.

        Uses a single reparameterized sample for both the returned ``z`` and
        the ELBO estimate — the caller's vmap over a batch axis amortizes
        the Monte Carlo variance.
        """
        del cond
        x_arr = jnp.asarray(x)
        z = self.encoder.sample(key, condition=x_arr)
        log_p = self.decoder.log_prob(x_arr, condition=z)
        log_q = self.encoder.log_prob(z, condition=x_arr)
        return z, log_p - log_q

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample ``x ~ p(x | z)``; inverse log-det is zero by convention."""
        del cond
        x = self.decoder.sample(key, condition=jnp.asarray(z))
        return x, jnp.zeros(())


__all__ = ["VAE"]
