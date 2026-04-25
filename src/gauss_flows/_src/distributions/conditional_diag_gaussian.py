"""Diagonal Gaussian base distribution amortised by a single context net."""

from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import AbstractDistribution
from jax import Array
from jaxtyping import PRNGKeyArray


_LOG_2PI = jnp.log(2.0 * jnp.pi)


class ConditionalDiagGaussian(AbstractDistribution):
    """Diagonal Gaussian whose ``(loc, log_scale)`` are produced by a single net.

    The parameter network maps a 1-D context vector
    ``condition: (cond_dim,)`` to ``(2 * event_dim,)`` outputs which are then
    split (along the last axis) into ``loc`` and ``log_scale``. This mirrors
    the coupling-net pattern used by :class:`AffineCoupling` and avoids the
    duplication of two parallel sub-networks.

    Args:
        key: PRNG key used to initialise the default :class:`equinox.nn.MLP`.
            Ignored when a pre-built ``param_net`` is supplied.
        event_shape: Single-event shape ``(D,)``. 1-D only.
        cond_shape: Condition shape ``(cond_dim,)``. 1-D only.
        param_net: Optional pre-built callable ``(cond_dim,) -> (2 * D,)``.
            If omitted, a default :class:`equinox.nn.MLP` is built using
            ``nn_width`` / ``nn_depth`` / ``activation``.
        nn_width: Width of the default MLP.
        nn_depth: Depth of the default MLP.
        activation: Activation between hidden layers of the default MLP.

    Shape:
        - Condition: ``(cond_dim,)``
        - Sample / log_prob input: ``(D,)``
        - log_prob output: scalar ``()``

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import ConditionalDiagGaussian
        >>> base = ConditionalDiagGaussian(
        ...     jr.key(0), event_shape=(3,), cond_shape=(2,)
        ... )
        >>> cond = jnp.array([0.5, -0.5])
        >>> x = base.sample(jr.key(1), condition=cond)
        >>> log_p = base.log_prob(x, condition=cond)
        >>> assert x.shape == (3,)
        >>> assert log_p.shape == ()
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    param_net: Callable[[Array], Array]

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        event_shape: tuple[int, ...],
        cond_shape: tuple[int, ...],
        param_net: Callable[[Array], Array] | None = None,
        nn_width: int = 64,
        nn_depth: int = 2,
        activation: Callable[[Array], Array] = jnn.tanh,
    ):
        if len(event_shape) != 1:
            raise ValueError(
                "ConditionalDiagGaussian only supports 1D event shapes; "
                f"got {event_shape}."
            )
        if event_shape[0] < 1:
            raise ValueError(
                f"event_shape must have a positive dimension; got {event_shape}."
            )
        if len(cond_shape) != 1:
            raise ValueError(
                "ConditionalDiagGaussian only supports 1D condition shapes; "
                f"got {cond_shape}. Flatten higher-rank context before passing."
            )
        if cond_shape[0] < 1:
            raise ValueError(
                f"cond_shape must have a positive dimension; got {cond_shape}."
            )

        self.shape = event_shape
        self.cond_shape = cond_shape
        d = event_shape[0]
        cond_dim = cond_shape[0]

        if param_net is None:
            param_net = eqx.nn.MLP(
                in_size=cond_dim,
                out_size=2 * d,
                width_size=nn_width,
                depth=nn_depth,
                activation=activation,
                key=key,
            )
        else:
            # Validate the user-supplied net's output shape on a zero context.
            test_out = jnp.asarray(param_net(jnp.zeros(cond_shape)))
            if test_out.shape != (2 * d,):
                raise ValueError(
                    "param_net output must have shape (2 * event_dim,) = "
                    f"({2 * d},); got {test_out.shape}."
                )

        self.param_net = param_net

    def _resolve_params(self, condition: Array) -> tuple[Array, Array]:
        out = self.param_net(condition)
        loc, log_scale = jnp.split(out, 2, axis=-1)
        return loc, log_scale

    def _sample(self, key: PRNGKeyArray, condition: Array) -> Array:
        loc, log_scale = self._resolve_params(condition)
        z = jr.normal(key, self.shape, dtype=loc.dtype)
        return loc + jnp.exp(log_scale) * z

    def _log_prob(self, x: Array, condition: Array) -> Array:
        loc, log_scale = self._resolve_params(condition)
        z = (x - loc) * jnp.exp(-log_scale)
        return -0.5 * jnp.sum(z * z + 2.0 * log_scale + _LOG_2PI)


__all__ = ["ConditionalDiagGaussian"]
