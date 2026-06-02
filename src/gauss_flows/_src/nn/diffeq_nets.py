"""Vector-field networks for continuous normalizing flows."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.conditioning import unpack_time_control
from gauss_flows._src.nn.time_net import TimeFourier


_TimeNet = Callable[[ArrayLike], Array]


def _as_scalar_time(t: ArrayLike) -> Array:
    t_array = jnp.asarray(t)
    if t_array.shape == ():
        return t_array
    if t_array.shape == (1,):
        return t_array[0]
    raise ValueError(f"time must have shape () or (1,); got {t_array.shape}.")


def _hidden_sizes(hidden: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden, int):
        if hidden < 1:
            raise ValueError(f"hidden must be positive; got {hidden}.")
        return (hidden,)
    hidden_sizes = tuple(hidden)
    if not hidden_sizes or any(width < 1 for width in hidden_sizes):
        raise ValueError("hidden must contain at least one positive width.")
    return hidden_sizes


def _maybe_condition_features(
    condition: ArrayLike | None,
    control_dim: int,
    *,
    dtype,
) -> Array:
    if control_dim == 0:
        return jnp.empty((0,), dtype=dtype)
    if condition is None:
        raise ValueError(
            "Conditional diffeq nets require a packed condition. "
            "Use pack_time_control(t, control)."
        )
    query_t, control = unpack_time_control(
        jnp.asarray(condition, dtype=dtype), control_dim
    )
    # [t_query, c]: (1 + control_dim,)
    return jnp.concatenate((query_t, control))


class _MLPHead(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        input_dim: int,
        output_dim: int,
        hidden: tuple[int, ...],
        activation: Callable[[Array], Array],
    ):
        widths = (input_dim, *hidden, output_dim)
        keys = jr.split(key, len(widths) - 1)
        self.layers = tuple(
            eqx.nn.Linear(in_features, out_features, key=layer_key)
            for in_features, out_features, layer_key in zip(
                widths[:-1],
                widths[1:],
                keys,
                strict=True,
            )
        )
        self.activation = activation

    def __call__(self, x: ArrayLike) -> Array:
        x_array = jnp.asarray(x)
        for layer in self.layers[:-1]:
            x_array = self.activation(layer(x_array))
        return self.layers[-1](x_array)


class DiffeqMLP(eqx.Module):
    """MLP vector field ``f(ode_time, x, condition)`` via concatenation.

    For conditional models this concatenates the ODE time, the packed
    conditioning time, the state, and the control vector. For unconditional
    models it concatenates only the ODE time and state.

    Args:
        key: PRNG key for network initialisation.
        in_dim: Event dimension ``dim``.
        control_dim: Size of the optional control vector ``c``.
        hidden: Hidden-layer widths.
        activation: Non-linearity applied after each hidden layer.

    Shape:
        - Input ``ode_time``: ``()`` or ``(1,)``
        - Input ``x``: ``(dim,)``
        - Input ``condition``: ``(1 + control_dim,)`` or ``None`` when
          ``control_dim == 0``
        - Output: ``(dim,)``

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import DiffeqMLP, pack_time_control
        >>> net = DiffeqMLP(jr.key(0), in_dim=3, control_dim=2, hidden=(16, 16))
        >>> x = jnp.ones((3,))
        >>> cond = pack_time_control(0.5, jnp.array([1.0, -1.0]))
        >>> net(0.25, x, cond).shape
        (3,)
    """

    in_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    mlp: _MLPHead

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        in_dim: int,
        control_dim: int = 0,
        hidden: int | Sequence[int] = (64, 64),
        activation: Callable[[Array], Array] = jnn.tanh,
    ):
        if in_dim < 1:
            raise ValueError(f"in_dim must be positive; got {in_dim}.")
        if control_dim < 0:
            raise ValueError(f"control_dim must be non-negative; got {control_dim}.")

        hidden_sizes = _hidden_sizes(hidden)
        # unconditional: [ode_time, x] -> (1 + dim)
        # conditional: [ode_time, x, t_query, c] -> (2 + dim + control_dim)
        input_dim = 1 + in_dim if control_dim == 0 else 2 + in_dim + control_dim
        self.in_dim = in_dim
        self.control_dim = control_dim
        self.mlp = _MLPHead(
            key,
            input_dim=input_dim,
            output_dim=in_dim,
            hidden=hidden_sizes,
            activation=activation,
        )

    def __call__(
        self,
        ode_time: ArrayLike,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> Array:
        x_array = jnp.asarray(x)
        ode_time_scalar = jnp.reshape(_as_scalar_time(ode_time), (1,))
        condition_features = _maybe_condition_features(
            condition,
            self.control_dim,
            dtype=x_array.dtype,
        )
        # [ode_time, x, t_query, c?]: (1 + dim + maybe(1 + control_dim),)
        features = jnp.concatenate((ode_time_scalar, x_array, condition_features))
        return self.mlp(features)


class DiffeqConcat(eqx.Module):
    """MLP vector field with a learnable scalar time embedding.

    This variant replaces the raw ODE time with ``time_net(ode_time)`` before
    concatenation, while preserving the same packed-condition API as
    :class:`DiffeqMLP`.

    Args:
        key: PRNG key for the MLP and default time net.
        in_dim: Event dimension ``dim``.
        control_dim: Size of the optional control vector ``c``.
        hidden: Hidden-layer widths.
        time_net: Optional scalar time embedding. When omitted, a
            :class:`gauss_flows.TimeFourier` module is created.
        activation: Non-linearity applied after each hidden layer.

    Shape:
        - Input ``ode_time``: ``()`` or ``(1,)``
        - Input ``x``: ``(dim,)``
        - Input ``condition``: ``(1 + control_dim,)`` or ``None`` when
          ``control_dim == 0``
        - Output: ``(dim,)``

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import DiffeqConcat, pack_time_control
        >>> net = DiffeqConcat(jr.key(0), in_dim=3, control_dim=2, hidden=(16, 16))
        >>> x = jnp.ones((3,))
        >>> cond = pack_time_control(0.5, jnp.array([1.0, -1.0]))
        >>> net(0.25, x, cond).shape
        (3,)
    """

    in_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    time_net: _TimeNet
    mlp: _MLPHead

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        in_dim: int,
        control_dim: int = 0,
        hidden: int | Sequence[int] = (64, 64),
        time_net: _TimeNet | None = None,
        activation: Callable[[Array], Array] = jnn.tanh,
    ):
        if in_dim < 1:
            raise ValueError(f"in_dim must be positive; got {in_dim}.")
        if control_dim < 0:
            raise ValueError(f"control_dim must be non-negative; got {control_dim}.")

        key_mlp, key_time = jr.split(key)
        hidden_sizes = _hidden_sizes(hidden)
        # unconditional: [time_net(ode_time), x] -> (1 + dim)
        # conditional: [time_net(ode_time), x, t_query, c] -> (2 + dim + control_dim)
        input_dim = 1 + in_dim if control_dim == 0 else 2 + in_dim + control_dim
        self.in_dim = in_dim
        self.control_dim = control_dim
        self.time_net = TimeFourier(key_time) if time_net is None else time_net
        self.mlp = _MLPHead(
            key_mlp,
            input_dim=input_dim,
            output_dim=in_dim,
            hidden=hidden_sizes,
            activation=activation,
        )

    def __call__(
        self,
        ode_time: ArrayLike,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> Array:
        x_array = jnp.asarray(x)
        embedded_time = jnp.reshape(
            jnp.asarray(self.time_net(_as_scalar_time(ode_time)), dtype=x_array.dtype),
            (1,),
        )
        condition_features = _maybe_condition_features(
            condition,
            self.control_dim,
            dtype=x_array.dtype,
        )
        # [time_net(ode_time), x, t_query, c?]: (1 + dim + maybe(1 + control_dim),)
        features = jnp.concatenate((embedded_time, x_array, condition_features))
        return self.mlp(features)


__all__ = ["DiffeqConcat", "DiffeqMLP"]
