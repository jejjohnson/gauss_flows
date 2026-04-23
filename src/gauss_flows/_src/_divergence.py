"""Divergence helpers for continuous normalizing flows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray
from matfree import stochtrace


_ProbeDistribution = Literal["rademacher", "normal"]


def exact_divergence(
    vector_field: Callable[[Array, Array, Array | None], Array],
    t: ArrayLike,
    x: ArrayLike,
    condition: Array | None,
) -> Array:
    """Compute ``tr(∂f/∂x)`` exactly with ``jax.jacfwd``.

    Args:
        vector_field: Callable with signature ``f(t, x, condition)``.
        t: ODE integration time.
        x: Single event with shape ``(dim,)``.
        condition: Optional packed condition.

    Returns:
        Scalar divergence ``tr(∂f/∂x)``.
    """
    x_array = jnp.asarray(x)
    jacobian = jax.jacfwd(lambda x_: vector_field(jnp.asarray(t), x_, condition))(
        x_array
    )
    return jnp.trace(jacobian)


def hutchinson_divergence(
    vector_field: Callable[[Array, Array, Array | None], Array],
    t: ArrayLike,
    x: ArrayLike,
    condition: Array | None,
    *,
    key: PRNGKeyArray,
    n_samples: int = 1,
    distribution: _ProbeDistribution = "rademacher",
) -> Array:
    """Estimate ``tr(∂f/∂x)`` with Hutchinson's identity.

    Args:
        vector_field: Callable with signature ``f(t, x, condition)``.
        t: ODE integration time.
        x: Single event with shape ``(dim,)``.
        condition: Optional packed condition.
        key: PRNG key for probe sampling.
        n_samples: Number of probe vectors.
        distribution: Probe distribution, either ``"rademacher"`` or
            ``"normal"``.

    Returns:
        Scalar Hutchinson estimate of ``tr(∂f/∂x)``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive; got {n_samples}.")

    x_array = jnp.asarray(x)
    sampler = _make_sampler(x_array, n_samples=n_samples, distribution=distribution)
    estimator = stochtrace.estimator(stochtrace.integrand_trace(), sampler)

    def jacobian_matvec(v: Array) -> Array:
        # v: (dim,) -> J_f(x) @ v: (dim,)
        _, tangent = jax.jvp(
            lambda x_: vector_field(jnp.asarray(t), x_, condition),
            (x_array,),
            (v,),
        )
        return tangent

    return jnp.asarray(estimator(jacobian_matvec, key))


def _make_sampler(
    x: Array,
    *,
    n_samples: int,
    distribution: _ProbeDistribution,
):
    if distribution == "rademacher":
        return stochtrace.sampler_rademacher(jnp.zeros_like(x), num=n_samples)
    if distribution == "normal":
        return stochtrace.sampler_normal(jnp.zeros_like(x), num=n_samples)
    raise ValueError(
        f"distribution must be 'rademacher' or 'normal'; got {distribution!r}."
    )


__all__ = ["exact_divergence", "hutchinson_divergence"]
