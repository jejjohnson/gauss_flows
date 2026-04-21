"""Helpers for packing ``(time, control)`` into FlowJax's single-condition API."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike


def _validate_control_dim(control_dim: int) -> None:
    if control_dim < 0:
        raise ValueError(f"control_dim must be non-negative; got {control_dim}.")


def _as_time_column(t: ArrayLike) -> Array:
    t_array = jnp.asarray(t, dtype=float)
    if t_array.ndim == 0:
        return jnp.reshape(t_array, (1,))
    if t_array.shape[-1] == 1:
        return t_array
    return t_array[..., None]


def pack_time_control(
    t: ArrayLike,
    control: ArrayLike | None = None,
) -> Array:
    """Pack time and optional control vector into one condition array.

    Args:
        t: Scalar time value or an array with shape ``(...,)`` or ``(..., 1)``.
        control: Optional control vector with shape ``(..., C)``.

    Returns:
        Packed condition with shape ``(..., 1 + C)`` — or ``(..., 1)`` when
        ``control`` is ``None``.

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows._src.conditioning import pack_time_control
        >>> pack_time_control(0.5).shape
        (1,)
        >>> pack_time_control(0.5, jnp.array([1.0, 2.0])).shape
        (3,)
        >>> pack_time_control(jnp.array([0.0, 1.0]), jnp.ones((2, 2))).shape
        (2, 3)
    """
    t_array = _as_time_column(t)
    if control is None:
        return t_array

    control_array = jnp.atleast_1d(jnp.asarray(control, dtype=float))
    if t_array.shape[:-1] != control_array.shape[:-1]:
        raise ValueError(
            "t and control must share the same leading batch shape; "
            f"got {t_array.shape[:-1]} and {control_array.shape[:-1]}."
        )
    return jnp.concatenate((t_array, control_array), axis=-1)


def unpack_time_control(
    condition: Array,
    control_dim: int = 0,
) -> tuple[Array, Array | None]:
    """Unpack a packed ``(time, control)`` condition.

    Args:
        condition: Packed condition with shape ``(..., 1 + control_dim)``.
        control_dim: Size of the control vector. Use ``0`` for time-only
            conditioning.

    Returns:
        A tuple ``(t, control)`` where ``t`` has shape ``(..., 1)`` and
        ``control`` has shape ``(..., control_dim)`` or is ``None``.

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows._src.conditioning import unpack_time_control
        >>> t, control = unpack_time_control(jnp.array([0.5, 1.0, 2.0]), control_dim=2)
        >>> t.shape, control.shape
        ((1,), (2,))
    """
    _validate_control_dim(control_dim)
    condition = jnp.asarray(condition, dtype=float)
    if condition.ndim == 0:
        raise ValueError("condition must have at least one dimension.")

    expected = 1 + control_dim
    if condition.shape[-1] != expected:
        raise ValueError(
            "condition has incompatible trailing dimension; "
            f"expected {expected}, got {condition.shape[-1]}."
        )

    t = condition[..., :1]
    if control_dim == 0:
        return t, None
    return t, condition[..., 1:]


def time_control_cond_shape(control_dim: int = 0) -> tuple[int, ...]:
    """Return the FlowJax ``cond_shape`` for packed ``(time, control)`` inputs.

    Args:
        control_dim: Size of the control vector. Use ``0`` for time-only
            conditioning.

    Returns:
        A one-tuple ``(1 + control_dim,)`` suitable for ``AbstractBijection``.

    Example:
        >>> from gauss_flows._src.conditioning import time_control_cond_shape
        >>> time_control_cond_shape()
        (1,)
        >>> time_control_cond_shape(3)
        (4,)
    """
    _validate_control_dim(control_dim)
    return (1 + control_dim,)


__all__ = [
    "pack_time_control",
    "time_control_cond_shape",
    "unpack_time_control",
]
