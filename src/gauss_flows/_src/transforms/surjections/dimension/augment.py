"""Generative surjection that appends conditionally sampled dimensions."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src._protocols import ConditionalDistribution
from gauss_flows._src.transforms.base import AbstractSurjection


class Augment(AbstractSurjection):
    """Generative surjection that appends conditionally sampled dimensions.

    Forward draws ``augment_size`` extra dims from ``encoder`` conditioned on
    the input and concatenates them; inverse drops the augmented tail:

        forward:  z_aug ~ q(· | x)                   (encoder sample)
                  z       = concat([x, z_aug])
                  log_det = − log q(z_aug | x)

        inverse:  x = z[:x_size]                     (deterministic drop)

    Because the forward direction is stochastic, the contribution to
    ``log_prob`` is an ELBO over the encoder's samples, not an exact density.
    Use Augment to give a flow extra latent dimensions to model multi-modal
    structure that would not fit in the data dim alone.

    Args:
        encoder: Object satisfying `ConditionalDistribution` —
            ``sample(key, *, condition=x)`` must produce arrays of shape
            ``(augment_size,)`` and ``log_prob(value, *, condition=x)`` must
            return a scalar.
        x_size: Size of the data input event. Determines ``self.shape``.
        augment_size: Number of latent dims to append.

    Attributes:
        shape: ``(x_size,)`` — the forward-input event shape, matching the
            project-wide ``x.shape == self.shape`` convention. ``forward``
            runtime-validates this.

    Shape:
        - Input  ``x``:  ``(x_size,)``
        - Output ``z``:  ``(x_size + augment_size,)``
        - ``log_det``:   scalar (shape ``()``)

    Examples:
        Augment a 2-D data event with a 2-D learned latent:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from flowjax.distributions import Normal
        >>> from gauss_flows import Augment
        >>>
        >>> enc = Normal(jnp.zeros(2))
        >>> surj = Augment(encoder=enc, x_size=2, augment_size=2)
        >>> x = jnp.array([0.5, -0.3])
        >>> z, log_det = surj.forward_and_log_det(x, jr.key(0))
        >>> # z.shape == (4,); log_det == -Normal(0,1).log_prob(z[2:]).sum()

        Inside a SurVAEFlow that maps (2,) data → (4,) latent — supply the
        explicit ``data_shape`` since the chain changes dimensionality:

        >>> from gauss_flows import SurVAEFlow
        >>> base = Normal(jnp.zeros(4))
        >>> flow = SurVAEFlow(
        ...     base,
        ...     [Augment(encoder=Normal(jnp.zeros(2)), x_size=2, augment_size=2)],
        ...     data_shape=(2,),
        ... )
    """

    stochastic_forward: ClassVar[bool] = True
    stochastic_inverse: ClassVar[bool] = False
    lower_bound: ClassVar[bool] = True
    shape: tuple[int, ...]
    encoder: ConditionalDistribution
    x_size: int
    augment_size: int

    def __init__(
        self,
        encoder: ConditionalDistribution,
        x_size: int,
        augment_size: int,
    ):
        if x_size <= 0:
            raise ValueError("x_size must be positive.")
        if augment_size <= 0:
            raise ValueError("augment_size must be positive.")
        self.shape = (x_size,)
        self.encoder = encoder
        self.x_size = x_size
        self.augment_size = augment_size

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample ``z_aug`` from encoder; ``z = concat([x, z_aug])``."""
        del cond
        x_arr = jnp.asarray(x)
        if x_arr.shape != self.shape:
            raise ValueError(
                f"Augment.forward expected x.shape == {self.shape}; got {x_arr.shape}."
            )
        # z_aug: (augment_size,); z: (x_size + augment_size,)
        z_aug = self.encoder.sample(key, condition=x_arr)
        log_det = -self.encoder.log_prob(z_aug, condition=x_arr)
        z = jnp.concatenate([x_arr, z_aug])
        return z, log_det

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Drop the augmented tail; deterministic, log_det = 0."""
        del key, cond
        z_arr = jnp.asarray(z)
        # x: (x_size,)
        x = z_arr[: self.x_size]
        return x, jnp.zeros(())


__all__ = ["Augment"]
