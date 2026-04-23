"""Closed-form linear neural flow via a matrix exponential."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array
from jax.scipy import linalg as jsp_linalg
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.conditioning import unpack_time_control
from gauss_flows._src.nn import TimeIdentity


_TimeGate = Callable[[ArrayLike], Array]


class MatrixExponential(AbstractBijection):
    """Closed-form linear neural flow parameterised by ``exp(W · t)``.

    Computes

    ``y = exp(W · t) · x + h(t) · b``

    with exact log-determinant

    ``log |det ∂y/∂x| = t · tr(W)``.

    The map is exactly the identity at ``t = 0`` because ``exp(0) = I`` and the
    default time gate satisfies ``h(0) = 0``.

    Args:
        key: PRNG key for parameter initialisation.
        shape: Event shape ``(dim,)``.
        w_init_scale: Standard deviation used to initialise ``W`` and ``b``.
        use_bias: If ``False``, drop the ``h(t) · b`` term.
        time_bias_net: Optional scalar time gate ``h`` with ``h(0) = 0``.
            Defaults to :class:`gauss_flows.TimeIdentity` when ``use_bias=True``.

    Shape:
        - Input ``x``: ``(dim,)``
        - Condition: packed time ``(1,)`` from :func:`gauss_flows.pack_time_control`
        - Output ``y``: ``(dim,)``
        - ``log_det``: scalar ``()``

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import MatrixExponential, pack_time_control
        >>> bij = MatrixExponential(jr.key(0), shape=(3,))
        >>> x = jnp.array([1.0, -1.0, 0.5])
        >>> y, log_det = bij.transform_and_log_det(x, pack_time_control(0.25))
        >>> x_rec, log_det_inv = bij.inverse_and_log_det(y, pack_time_control(0.25))
        >>> assert jnp.allclose(x, x_rec)
        >>> assert jnp.allclose(log_det + log_det_inv, 0.0)
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[tuple[int, ...]] = (1,)
    W: Array
    b: Array | None
    time_bias_net: _TimeGate | None

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        *,
        w_init_scale: float = 1e-2,
        use_bias: bool = True,
        time_bias_net: _TimeGate | None = None,
    ):
        if len(shape) != 1:
            raise ValueError("MatrixExponential only supports 1D inputs.")

        n_dims = shape[0]
        key_w, key_b = jr.split(key)
        self.shape = shape
        self.W = jr.normal(key_w, (n_dims, n_dims)) * w_init_scale
        self.b = jr.normal(key_b, (n_dims,)) * w_init_scale if use_bias else None
        self.time_bias_net = (
            TimeIdentity() if use_bias and time_bias_net is None else time_bias_net
        )

    def transform_and_log_det(
        self,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        if condition is None:
            raise ValueError(
                "MatrixExponential requires a packed time condition. "
                "Use pack_time_control(t)."
            )

        x = jnp.asarray(x)
        t, _ = unpack_time_control(jnp.asarray(condition, dtype=x.dtype), control_dim=0)
        t_scalar = t[0]
        M = jsp_linalg.expm(self.W * t_scalar)
        y = M @ x
        if self.b is not None and self.time_bias_net is not None:
            y = y + jnp.asarray(self.time_bias_net(t), dtype=x.dtype) * self.b
        log_det = t_scalar * jnp.trace(self.W)
        return y, log_det

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        if condition is None:
            raise ValueError(
                "MatrixExponential requires a packed time condition. "
                "Use pack_time_control(t)."
            )

        y = jnp.asarray(y)
        t, _ = unpack_time_control(jnp.asarray(condition, dtype=y.dtype), control_dim=0)
        t_scalar = t[0]
        y_shifted = y
        if self.b is not None and self.time_bias_net is not None:
            gate = jnp.asarray(self.time_bias_net(t), dtype=y.dtype)
            y_shifted = y_shifted - gate * self.b
        M_inv = jsp_linalg.expm(-self.W * t_scalar)
        x = M_inv @ y_shifted
        log_det = -t_scalar * jnp.trace(self.W)
        return x, log_det


__all__ = ["MatrixExponential"]
