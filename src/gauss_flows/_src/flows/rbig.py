"""Iterative RBIG (Rotation-Based Iterative Gaussianization) flow."""

from __future__ import annotations

from flowjax.distributions import Transformed
from jaxtyping import PRNGKeyArray

from gauss_flows._src.flows.gaussianization import gaussianization_flow


def iterative_rbig(
    key: PRNGKeyArray,
    *,
    n_dims: int,
    n_layers: int = 100,
    n_components: int = 8,
    rotation: str = "householder",
    n_reflections: int | None = None,
) -> Transformed:
    """Construct an iterative RBIG (Rotation-Based Iterative Gaussianization) flow.

    Implements the classic RBIG algorithm as a normalizing flow: each of
    ``n_layers`` iterations alternates a marginal Gaussianization step G
    with a rotation R (the G∘R block). This is a thin wrapper around
    `gauss_flows.gaussianization_flow` with a large default
    ``n_layers`` reflecting the many shallow iterations RBIG typically
    uses; all arguments are forwarded unchanged.

    Args:
        key: JAX random key.
        n_dims: Dimensionality of the data.
        n_layers: Number of RBIG iterations. Defaults to 100.
        n_components: Number of mixture components for marginal Gaussianization.
            Defaults to 8.
        rotation: Type of rotation, either ``"householder"`` or
            ``"orthogonal"``. Defaults to ``"householder"``.
        n_reflections: Number of Householder reflections. Defaults to ``n_dims``.

    Returns:
        A flowjax ``Transformed`` distribution with ``log_prob`` and ``sample``.

    Examples:
        >>> import jax.random as jr
        >>> from gauss_flows import iterative_rbig
        >>> flow = iterative_rbig(jr.key(0), n_dims=4, n_layers=3, n_components=4)
        >>> flow.sample(jr.key(1), (5,)).shape
        (5, 4)
    """
    return gaussianization_flow(
        key,
        n_dims=n_dims,
        n_layers=n_layers,
        n_components=n_components,
        rotation=rotation,
        n_reflections=n_reflections,
    )


__all__ = ["iterative_rbig"]
