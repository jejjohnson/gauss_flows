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
    """Class-conditional flow built on a `ClassCondDiagGaussian` base.

    A thin convenience wrapper around `SurVAEFlow` that:

    - constructs a `ClassCondDiagGaussian` base with the given
      ``n_classes`` and ``event_shape``;
    - exposes a ``label`` keyword in its public API (instead of the more
      generic ``condition``) for ergonomics in the class-conditional case;
    - carries ``n_classes`` as static metadata for downstream tooling.

    Only the **base** is class-conditioned. The transform chain must be
    unconditional — every transform's ``cond_shape`` must be ``None``. This
    is because the class label is broadcast as a scalar ``int`` (a shape-``()``
    condition that the base requires for index lookup), which is incompatible
    with the ``(cond_dim,)``-shaped condition that conditional couplings
    expect. If you want conditional transforms in addition to a class-
    conditional base, build a `SurVAEFlow` directly and pack a
    label-derived feature vector (e.g. one-hot) into a custom condition;
    this wrapper deliberately does not support that to keep the contract
    simple.

    Args:
        key: PRNG key used to initialise the base distribution.
        n_classes: Number of classes ``C`` (must be positive).
        event_shape: Single-event shape ``(D,)``.
        transforms: Sequence of unconditional bijections / surjections,
            applied left-to-right in the base→data direction. Conditional
            transforms (``cond_shape != None``) are rejected at construction.
        learn_mean: Forwarded to `ClassCondDiagGaussian`.
        logscale_factor: Forwarded to `ClassCondDiagGaussian`.

    Shape:
        - ``label``: scalar int (``cond_shape == ()``)
        - ``x`` / ``y``: ``event_shape``

    Examples:
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
        # The class label is a scalar int; conditional transforms expect a
        # vector condition (cond_dim,). There is no single condition shape
        # that satisfies both base (()-shape) and a conditional transform,
        # so we reject the latter at construction with a clear message.
        for i, t in enumerate(transforms):
            t_cond = getattr(t, "cond_shape", None)
            if t_cond is not None:
                raise ValueError(
                    f"ClassCondFlow only conditions the base distribution; "
                    f"transforms[{i}] has cond_shape={t_cond!r}, but the "
                    "label is forwarded as a scalar int. Either drop "
                    "cond_dim from this transform, or build a SurVAEFlow "
                    "directly and pack a label-derived feature vector into "
                    "a custom condition."
                )
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
