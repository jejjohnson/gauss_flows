"""Class-conditional flow convenience wrapper."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.distributions.class_cond_diag_gaussian import (
    ClassCondDiagGaussian,
)
from gauss_flows._src.flows.survae import SurVAEFlow
from gauss_flows._src.transforms.base import AbstractSurjection


class ClassCondFlow(eqx.Module):
    """Class-conditional flow built on a :class:`ClassCondDiagGaussian` base.

    A thin convenience wrapper around :class:`SurVAEFlow` that:

    - constructs a :class:`ClassCondDiagGaussian` base with the given
      ``n_classes`` and ``event_shape``;
    - exposes a ``label`` keyword in its public API (instead of the more
      generic ``condition``) for ergonomics in the class-conditional case;
    - carries ``n_classes`` as static metadata for downstream tooling.

    The transform chain may be entirely unconditional (couplings without
    ``cond_dim``), entirely conditional (couplings with ``cond_dim ==
    label_dim``), or any mix — the class label is broadcast to every layer
    that opts in (see :class:`SurVAEFlow` "Conditioning model").

    Args:
        key: PRNG key used to initialise the base distribution.
        n_classes: Number of classes ``C`` (must be positive).
        event_shape: Single-event shape ``(D,)``.
        transforms: Sequence of bijections / surjections, applied
            left-to-right in the base→data direction.
        learn_mean: Forwarded to :class:`ClassCondDiagGaussian`.
        logscale_factor: Forwarded to :class:`ClassCondDiagGaussian`.

    Shape:
        - ``label``: scalar int (``cond_shape == ()``)
        - ``x`` / ``y``: ``event_shape``

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import ClassCondFlow, AffineCoupling
        >>> coupling = AffineCoupling(jr.key(0), shape=(4,))   # unconditional layer
        >>> flow = ClassCondFlow(
        ...     jr.key(1), n_classes=3, event_shape=(4,),
        ...     transforms=[coupling],
        ... )
        >>> label = jnp.int32(1)
        >>> x = flow.sample(jr.key(2), label=label)
        >>> log_p = flow.log_prob(x, jr.key(3), label=label)
        >>> assert x.shape == (4,) and log_p.shape == ()
    """

    flow: SurVAEFlow
    n_classes: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        n_classes: int,
        event_shape: tuple[int, ...],
        transforms: Sequence[AbstractBijection | AbstractSurjection],
        learn_mean: bool = True,
        logscale_factor: float = 1.0,
    ):
        base = ClassCondDiagGaussian(
            key,
            n_classes=n_classes,
            event_shape=event_shape,
            learn_mean=learn_mean,
            logscale_factor=logscale_factor,
        )
        self.flow = SurVAEFlow(base, transforms)
        self.n_classes = int(n_classes)

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
        *,
        label: ArrayLike,
    ) -> Array:
        """Sample by drawing from the class-conditional base and pushing through."""
        return self.flow.sample(
            key, sample_shape, condition=jnp.asarray(label, dtype=jnp.int32)
        )

    def log_prob(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        *,
        label: ArrayLike,
    ) -> Array:
        """Evaluate ``log p(x | label)``."""
        return self.flow.log_prob(x, key, condition=jnp.asarray(label, dtype=jnp.int32))


__all__ = ["ClassCondFlow"]
