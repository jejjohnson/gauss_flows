"""Wrap an arbitrary numpyro distribution as a flowjax base distribution."""

from __future__ import annotations

from collections.abc import Callable

import numpyro.distributions as ndist
from flowjax.distributions import AbstractDistribution
from jax import Array
from jaxtyping import PRNGKeyArray


class NumpyroBase(AbstractDistribution):
    """Wrap any `numpyro.distributions.Distribution` as a flowjax base.

    Two construction modes:

    - **Unconditional** (``dist=...``): a fixed numpyro distribution. The
      ``event_shape`` is read from ``dist.event_shape`` and ``cond_shape`` is
      ``None``. The numpyro distribution must have ``batch_shape == ()``; if
      you need to broadcast a scalar dist to a vector event, use
      ``.expand(shape).to_event(rank)`` so the broadcasted dims are
      reinterpreted as event dims (otherwise ``batch_shape`` will be
      non-trivial and this wrapper will reject it).
    - **Conditional** (``dist_factory=...``): a callable
      ``condition -> numpyro.distributions.Distribution`` invoked per call.
      The user must supply ``event_shape`` and ``cond_shape`` because the
      factory is not run at construction time. Each invocation is
      responsible for returning a distribution with matching ``event_shape``
      and ``batch_shape == ()``.

    Vector events: numpyro returns one log-prob per *event-shape* position
    by default. Wrap your distribution in ``.to_event(rank)`` so that
    ``log_prob`` returns a scalar; alternatively, use a multivariate
    distribution like `numpyro.distributions.MultivariateNormal` whose
    ``event_shape`` already encodes the rank.

    Args:
        dist: A numpyro distribution. Mutually exclusive with ``dist_factory``.
        dist_factory: A callable mapping a condition array to a numpyro
            distribution. Mutually exclusive with ``dist``.
        event_shape: Required when using ``dist_factory``. Ignored when
            using ``dist`` (read from ``dist.event_shape`` instead).
        cond_shape: Required when using ``dist_factory``.

    Shape:
        - Condition: ``cond_shape`` (``None`` if unconditional)
        - Sample / log_prob input: ``event_shape``
        - log_prob output: scalar ``()``

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import numpyro.distributions as ndist
        >>> from gauss_flows import NumpyroBase
        >>> # Unconditional 3D Gaussian
        >>> base = NumpyroBase(dist=ndist.Normal(0.0, 1.0).expand((3,)).to_event(1))
        >>> base.shape, base.cond_shape
        ((3,), None)
        >>> base.log_prob(jnp.zeros(3)).shape
        ()
        >>> # Conditional: loc depends on context
        >>> def factory(c):
        ...     return ndist.Normal(c, 1.0).to_event(1)
        >>> cbase = NumpyroBase(
        ...     dist_factory=factory, event_shape=(3,), cond_shape=(3,)
        ... )
        >>> cbase.log_prob(jnp.zeros(3), condition=jnp.ones(3)).shape
        ()
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    _dist: ndist.Distribution | None
    # ``_factory`` is a regular (non-static) field so that an ``eqx.Module``
    # factory carrying learnable parameters remains visible to
    # ``eqx.filter_grad`` and to optimizers. Plain Python callables (lambdas /
    # closures) are still accepted — equinox treats them as opaque PyTree
    # leaves that ``eqx.is_array`` filters skip.
    _factory: Callable[[Array], ndist.Distribution] | None

    def __init__(
        self,
        *,
        dist: ndist.Distribution | None = None,
        dist_factory: Callable[[Array], ndist.Distribution] | None = None,
        event_shape: tuple[int, ...] | None = None,
        cond_shape: tuple[int, ...] | None = None,
    ):
        if (dist is None) == (dist_factory is None):
            raise ValueError(
                "Pass exactly one of 'dist' (unconditional) or 'dist_factory' "
                "(conditional)."
            )

        if dist is not None:
            if dist.batch_shape != ():
                raise ValueError(
                    "NumpyroBase requires a single-event distribution with "
                    f"batch_shape == (); got {dist.batch_shape}. Wrap with "
                    ".expand(...) only if necessary, then .to_event(...) the "
                    "broadcasted dims."
                )
            inferred = tuple(dist.event_shape)
            if event_shape is not None and tuple(event_shape) != inferred:
                raise ValueError(
                    f"event_shape={event_shape} does not match the wrapped "
                    f"distribution's event_shape={inferred}."
                )
            if cond_shape is not None:
                raise ValueError(
                    "cond_shape must be None when wrapping a fixed 'dist' "
                    "(unconditional mode)."
                )
            self.shape = inferred
            self.cond_shape = None
            self._dist = dist
            self._factory = None
        else:
            if event_shape is None or cond_shape is None:
                raise ValueError(
                    "When using 'dist_factory', both 'event_shape' and "
                    "'cond_shape' are required."
                )
            self.shape = tuple(event_shape)
            self.cond_shape = tuple(cond_shape)
            self._dist = None
            self._factory = dist_factory

    def _resolve(self, condition: Array | None) -> ndist.Distribution:
        if self._factory is None:
            return self._dist  # type: ignore[return-value]
        return self._factory(condition)

    def _sample(self, key: PRNGKeyArray, condition: Array | None = None) -> Array:
        return self._resolve(condition).sample(key)

    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        return self._resolve(condition).log_prob(x)


__all__ = ["NumpyroBase"]
