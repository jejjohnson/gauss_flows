"""Time-gating modules for continuous-time conditional flows."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


def _scalar_time(t: ArrayLike) -> Array:
    t_array = jnp.asarray(t)
    if t_array.shape == ():
        return t_array
    if t_array.shape == (1,):
        return t_array[0]
    raise ValueError(
        "Time nets expect a single time value with shape () or (1,); "
        f"got {t_array.shape}."
    )


class TimeFourier(eqx.Module):
    """Fourier-feature time gate with ``g(0) = 0`` by construction.

    The gate is

    ``g(t) = Σᵢ wᵢ sin(ωᵢ t)``

    so every feature vanishes exactly at ``t = 0``.

    Args:
        key: PRNG key for parameter initialisation.
        embedding_dim: Number of sinusoidal features.

    Shape:
        - Input ``t``: ``()`` or ``(1,)``
        - Output: scalar ``()``

    Examples:
        >>> import jax.random as jr
        >>> from gauss_flows import TimeFourier
        >>> gate = TimeFourier(jr.key(0), embedding_dim=8)
        >>> float(gate(0.0))
        0.0
    """

    frequencies: Array
    weights: Array

    def __init__(self, key: PRNGKeyArray, embedding_dim: int = 16):
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive; got {embedding_dim}.")
        key_freq, key_weight = jr.split(key)
        self.frequencies = jr.normal(key_freq, (embedding_dim,))
        self.weights = jr.normal(key_weight, (embedding_dim,)) * 0.1

    def __call__(self, t: ArrayLike) -> Array:
        t_scalar = _scalar_time(t)
        return jnp.dot(self.weights, jnp.sin(self.frequencies * t_scalar))


class TimeTanh(eqx.Module):
    """Tanh time gate with ``g(0) = 0`` by construction.

    The gate is

    ``g(t) = Σᵢ wᵢ tanh(aᵢ t)``

    so every feature vanishes exactly at ``t = 0``.

    Args:
        key: PRNG key for parameter initialisation.
        embedding_dim: Number of tanh features.

    Shape:
        - Input ``t``: ``()`` or ``(1,)``
        - Output: scalar ``()``

    Examples:
        >>> import jax.random as jr
        >>> from gauss_flows import TimeTanh
        >>> gate = TimeTanh(jr.key(0), embedding_dim=8)
        >>> float(gate(0.0))
        0.0
    """

    slopes: Array
    weights: Array

    def __init__(self, key: PRNGKeyArray, embedding_dim: int = 16):
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive; got {embedding_dim}.")
        key_slope, key_weight = jr.split(key)
        self.slopes = jr.normal(key_slope, (embedding_dim,))
        self.weights = jr.normal(key_weight, (embedding_dim,)) * 0.1

    def __call__(self, t: ArrayLike) -> Array:
        t_scalar = _scalar_time(t)
        return jnp.dot(self.weights, jnp.tanh(self.slopes * t_scalar))


class TimeIdentity(eqx.Module):
    """Identity time gate ``g(t) = t``.

    Shape:
        - Input ``t``: ``()`` or ``(1,)``
        - Output: scalar ``()``

    Examples:
        >>> from gauss_flows import TimeIdentity
        >>> gate = TimeIdentity()
        >>> float(gate(0.25))
        0.25
    """

    def __call__(self, t: ArrayLike) -> Array:
        return _scalar_time(t)


__all__ = [
    "TimeFourier",
    "TimeIdentity",
    "TimeTanh",
]
