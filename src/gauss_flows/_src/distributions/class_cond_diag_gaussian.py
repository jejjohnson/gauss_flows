"""Class-conditional diagonal Gaussian base distribution."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import AbstractDistribution
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


_LOG_2PI = jnp.log(2.0 * jnp.pi)


class ClassCondDiagGaussian(AbstractDistribution):
    """Class-conditional diagonal Gaussian with learned per-class parameters.

    Stores a ``(n_classes, *event_shape)`` lookup table for the means
    (when ``learn_mean=True``) and another for the raw log-scales. The
    effective log-scale is ``log_scale_raw[idx] * logscale_factor``; the
    multiplicative factor matches the Glow stabilisation trick (Kingma &
    Dhariwal 2018) where small stored values keep the effective scales
    well-conditioned at init.

    The condition is a **scalar** integer class label (``cond_shape = ()``).
    Out-of-range indices follow JAX gather semantics — the caller is
    responsible for passing in-range labels.

    Args:
        key: PRNG key. Currently unused (parameters are deterministically
            initialised); accepted to match the constructor convention used
            by other learnable bases in this package.
        n_classes: Number of classes ``C`` (must be positive).
        event_shape: Single-event shape (any rank).
        learn_mean: If ``False``, the per-class mean is frozen at zero.
            (Glow uses ``False`` in some configurations.)
        logscale_factor: Multiplicative factor applied to ``log_scale_raw``
            before exponentiation. Default ``1.0`` (no-op). Use ``3.0`` to
            recover the canonical Glow base.

    Shape:
        - Condition: ``()`` (scalar int)
        - Sample / log_prob input: ``event_shape``
        - log_prob output: scalar ``()``

    Glow recipe:
        ``ClassCondDiagGaussian(key, n_classes=10, event_shape=(D,),
        learn_mean=False, logscale_factor=3.0)``

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import ClassCondDiagGaussian
        >>> base = ClassCondDiagGaussian(jr.key(0), n_classes=4, event_shape=(3,))
        >>> label = jnp.int32(2)
        >>> x = base.sample(jr.key(1), condition=label)
        >>> log_p = base.log_prob(x, condition=label)
        >>> assert x.shape == (3,) and log_p.shape == ()
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] = eqx.field(static=True)
    n_classes: int = eqx.field(static=True)
    learn_mean: bool = eqx.field(static=True)
    logscale_factor: float = eqx.field(static=True)
    loc: Array
    log_scale_raw: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        n_classes: int,
        event_shape: tuple[int, ...],
        learn_mean: bool = True,
        logscale_factor: float = 1.0,
    ):
        del key  # parameters are deterministically initialised
        if n_classes < 1:
            raise ValueError(f"n_classes must be positive; got {n_classes}.")
        if any(d < 1 for d in event_shape):
            raise ValueError(
                f"event_shape entries must be positive; got {event_shape}."
            )
        if logscale_factor <= 0.0:
            raise ValueError(
                f"logscale_factor must be positive; got {logscale_factor}."
            )

        self.shape = tuple(event_shape)
        self.cond_shape = ()
        self.n_classes = int(n_classes)
        self.learn_mean = bool(learn_mean)
        self.logscale_factor = float(logscale_factor)
        self.loc = jnp.zeros((self.n_classes, *self.shape))
        self.log_scale_raw = jnp.zeros((self.n_classes, *self.shape))

    def _resolve_params(self, condition: ArrayLike) -> tuple[Array, Array]:
        idx = jnp.asarray(condition, dtype=jnp.int32)
        # flowjax's AbstractDistribution decorator does not currently enforce
        # ``condition.shape == ()`` for scalar cond_shapes, so a non-scalar
        # condition here would silently gather multiple rows and produce a
        # batched (loc, log_scale) — breaking the single-event contract.
        if idx.shape != ():
            raise ValueError(
                "ClassCondDiagGaussian expects a scalar integer class label "
                f"(condition.shape == ()); got shape {idx.shape}. For batched "
                "labels wrap the call in jax.vmap."
            )
        log_scale = self.log_scale_raw[idx] * self.logscale_factor
        if self.learn_mean:
            loc = self.loc[idx]
        else:
            loc = jnp.zeros(self.shape, dtype=log_scale.dtype)
        return loc, log_scale

    def _sample(self, key: PRNGKeyArray, condition: ArrayLike) -> Array:
        loc, log_scale = self._resolve_params(condition)
        z = jr.normal(key, self.shape, dtype=loc.dtype)
        return loc + jnp.exp(log_scale) * z

    def _log_prob(self, x: Array, condition: ArrayLike) -> Array:
        loc, log_scale = self._resolve_params(condition)
        z = (x - loc) * jnp.exp(-log_scale)
        return -0.5 * jnp.sum(z * z + 2.0 * log_scale + _LOG_2PI)


__all__ = ["ClassCondDiagGaussian"]
