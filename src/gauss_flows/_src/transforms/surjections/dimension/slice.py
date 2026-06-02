"""Inference surjection that drops the trailing dimensions of a 1D event."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src._protocols import ConditionalDistribution
from gauss_flows._src.transforms.base import AbstractSurjection


class Slice(AbstractSurjection):
    """Inference surjection that drops the trailing dimensions of a 1D event.

    Forward keeps the first ``keep_dims`` entries of ``x`` and scores the dropped
    tail under ``decoder`` conditioned on the kept prefix:

        forward:  z = x[:keep_dims]                  (deterministic)
                  log_det = log q(x[keep_dims:] | z)

        inverse:  dropped ~ q(· | z)                 (decoder sample)
                  x = concat([z, dropped])

    The forward log-det contribution is exact when the decoder ``q`` matches the
    true conditional density of the dropped dims given the kept dims; otherwise
    it is a (typically loose) lower bound on ``log p(x)``.

    Args:
        keep_dims: Number of leading entries to keep. Must be positive.
        decoder: Object satisfying `ConditionalDistribution` —
            ``sample(key, *, condition=z)`` must produce arrays of shape
            ``(D − keep_dims,)`` for the dropped tail and
            ``log_prob(value, *, condition=z)`` must return a scalar.

    Shape:
        - Input  ``x``:  ``(D,)`` (with ``D > keep_dims``)
        - Output ``z``:  ``(keep_dims,)``
        - ``log_det``:   scalar (shape ``()``)

    Note:
        Unlike most transforms, ``Slice`` does **not** expose a ``shape``
        attribute. Its forward-input dim ``D`` is determined by whatever the
        upstream transform produces — it is a chain-context property, not a
        constructor parameter, so there is no fixed ``shape`` to declare. The
        full data shape lives on `SurVAEFlow.data_shape` instead.

    Examples:
        Score the dropped tail under a unit Gaussian decoder:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from flowjax.distributions import Normal
        >>> from gauss_flows import Slice
        >>>
        >>> dec = Normal(jnp.zeros(2))   # decoder for the dropped 2 dims
        >>> surj = Slice(keep_dims=3, decoder=dec)
        >>> x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> z, log_det = surj.forward_and_log_det(x, jr.key(0))
        >>> # z == [1., 2., 3.]; log_det == sum of Normal(0,1).log_prob([4., 5.])

        Inside a SurVAEFlow that maps (4,) data → (2,) latent — note the
        explicit ``data_shape`` since the chain changes dimensionality:

        >>> from gauss_flows import SurVAEFlow
        >>> base = Normal(jnp.zeros(2))
        >>> flow = SurVAEFlow(
        ...     base,
        ...     [Slice(keep_dims=2, decoder=Normal(jnp.zeros(2)))],
        ...     data_shape=(4,),
        ... )
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    keep_dims: int
    decoder: ConditionalDistribution

    def __init__(self, keep_dims: int, decoder: ConditionalDistribution):
        if keep_dims <= 0:
            raise ValueError("keep_dims must be positive.")
        self.keep_dims = keep_dims
        self.decoder = decoder

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Drop trailing dims; return ``(kept, log q(dropped | kept))``."""
        del key, cond
        x_arr = jnp.asarray(x)
        # x: (D,) -> kept: (keep_dims,), dropped: (D - keep_dims,)
        kept = x_arr[: self.keep_dims]
        dropped = x_arr[self.keep_dims :]
        log_det = self.decoder.log_prob(dropped, condition=kept)
        return kept, log_det

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample the missing tail from the decoder; concat back.

        Returns ``log q(dropped_sampled | z)`` to mirror
        ``forward_and_log_det`` and match `SimpleMaxPoolSurjection2d`'s
        inverse convention. ``SurVAEFlow.log_prob`` doesn't consume this value
        (it only uses ``forward_and_log_det``); the consistent return shape is
        for callers using the inverse standalone (e.g. ELBO computations).
        """
        del cond
        z_arr = jnp.asarray(z)
        # dropped: (D - keep_dims,) sampled from decoder; x: (D,)
        dropped = self.decoder.sample(key, condition=z_arr)
        x = jnp.concatenate([z_arr, dropped])
        log_det = self.decoder.log_prob(dropped, condition=z_arr)
        return x, log_det


__all__ = ["Slice"]
