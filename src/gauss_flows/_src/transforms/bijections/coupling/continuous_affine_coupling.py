"""Continuous-time affine coupling layer."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.conditioning import time_control_cond_shape, unpack_time_control
from gauss_flows._src.nn import TimeFourier


_TimeGate = Callable[[ArrayLike], Array]


class ContinuousAffineCoupling(AbstractBijection):
    """Time-gated affine coupling with optional control conditioning.

    For an input split into passive and active halves, this layer computes

    ``y_active = x_active · exp(g(t) · log_scale(x_passive, c))``
    ``            + g(t) · shift(x_passive, c)``

    where ``g(0) = 0`` by construction, so the whole bijection is exactly the
    identity at ``t = 0``.

    A single layer leaves the passive half ``x[:dim // 2]`` untouched. To
    transform every dimension, stack multiple layers with alternating masks
    (e.g. via ``flowjax.bijections.Chain`` combined with ``Flip`` / ``Permute``).

    Args:
        key: PRNG key for the conditioner MLP and the default time gate.
        shape: Event shape ``(dim,)``.
        control_dim: Size of the optional control vector ``c``.
        n_hidden: Hidden width of the conditioner MLP.
        n_layers: Depth of the conditioner MLP.
        time_net: Optional pre-built gate ``g`` with ``g(0) = 0``. Any
            ``equinox.Module`` with a scalar-in / scalar-out ``__call__`` works
            — see :class:`gauss_flows.TimeFourier`, :class:`gauss_flows.TimeTanh`,
            :class:`gauss_flows.TimeIdentity`. When ``None``, a
            :class:`gauss_flows.TimeFourier` gate is built with
            ``embedding_dim=time_embedding_dim``.
        time_embedding_dim: Number of Fourier features used by the default
            :class:`gauss_flows.TimeFourier` gate. Ignored when ``time_net`` is
            provided.

    Shape:
        - Input ``x``: ``(dim,)``
        - Condition: ``(1 + control_dim,)`` packed as ``[t, c]``
        - Output ``y``: ``(dim,)``
        - ``log_det``: scalar ``()``

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import ContinuousAffineCoupling, pack_time_control
        >>> bij = ContinuousAffineCoupling(jr.key(0), shape=(4,), control_dim=2)
        >>> x = jr.normal(jr.key(1), (4,))
        >>> condition = pack_time_control(0.5, jnp.array([1.0, -1.0]))
        >>> y, log_det = bij.transform_and_log_det(x, condition)
        >>> x_rec, log_det_inv = bij.inverse_and_log_det(y, condition)
        >>> assert jnp.allclose(x, x_rec)
        >>> assert jnp.allclose(log_det + log_det_inv, 0.0)
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    control_dim: int = eqx.field(static=True)
    untransformed_dim: int = eqx.field(static=True)
    nn: eqx.nn.MLP
    time_net: _TimeGate

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        *,
        control_dim: int = 0,
        n_hidden: int = 64,
        n_layers: int = 2,
        time_net: _TimeGate | None = None,
        time_embedding_dim: int = 16,
    ):
        if len(shape) != 1:
            raise ValueError("ContinuousAffineCoupling only supports 1D inputs.")
        if shape[0] < 2:
            raise ValueError("ContinuousAffineCoupling requires at least 2 dimensions.")
        if control_dim < 0:
            raise ValueError(f"control_dim must be non-negative; got {control_dim}.")

        dim = shape[0]
        transformed_dim = dim - dim // 2
        self.shape = shape
        self.cond_shape = time_control_cond_shape(control_dim)
        self.control_dim = control_dim
        self.untransformed_dim = dim // 2

        key_nn, key_time = jr.split(key)
        nn = eqx.nn.MLP(
            in_size=self.untransformed_dim + control_dim,
            out_size=2 * transformed_dim,
            width_size=n_hidden,
            depth=n_layers,
            activation=jnn.relu,
            key=key_nn,
        )
        final_layer = nn.layers[-1]
        nn = eqx.tree_at(
            lambda model: (model.layers[-1].weight, model.layers[-1].bias),
            nn,
            (final_layer.weight * 0.01, final_layer.bias * 0.01),
        )
        self.nn = nn
        self.time_net = (
            TimeFourier(key_time, embedding_dim=time_embedding_dim)
            if time_net is None
            else time_net
        )

    def _gated_active_params(
        self,
        passive: Array,
        condition: ArrayLike | None,
        *,
        dtype,
    ) -> tuple[Array, Array]:
        if condition is None:
            raise ValueError(
                "ContinuousAffineCoupling requires a packed condition. "
                "Use pack_time_control(t, control)."
            )

        condition = jnp.asarray(condition, dtype=dtype)
        t, control = unpack_time_control(condition, self.control_dim)
        gate = jnp.asarray(self.time_net(t), dtype=dtype)

        nn_input = passive if control is None else jnp.concatenate((passive, control))
        shift_active, log_scale_active = jnp.split(self.nn(nn_input), 2)
        return gate * shift_active, gate * log_scale_active

    def transform_and_log_det(
        self,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        x = jnp.asarray(x)
        passive = x[: self.untransformed_dim]
        active = x[self.untransformed_dim :]
        shift, log_scale = self._gated_active_params(passive, condition, dtype=x.dtype)
        active_y = active * jnp.exp(log_scale) + shift
        y = jnp.concatenate((passive, active_y))
        log_det = jnp.sum(log_scale)
        return y, log_det

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        y = jnp.asarray(y)
        passive = y[: self.untransformed_dim]
        active_y = y[self.untransformed_dim :]
        shift, log_scale = self._gated_active_params(passive, condition, dtype=y.dtype)
        active_x = (active_y - shift) * jnp.exp(-log_scale)
        x = jnp.concatenate((passive, active_x))
        log_det = -jnp.sum(log_scale)
        return x, log_det


__all__ = ["ContinuousAffineCoupling"]
