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

    This implements the classic RBIG algorithm as a normalizing flow, alternating
    between marginal Gaussianization and rotation at each iteration.

    Args:
        key: JAX random key.
        n_dims: Dimensionality of the data.
        n_layers: Number of RBIG iterations. Defaults to 100.
        n_components: Number of mixture components for marginal Gaussianization.
            Defaults to 8.
        rotation: Type of rotation, either "householder" or "orthogonal".
            Defaults to "householder".
        n_reflections: Number of Householder reflections. Defaults to n_dims.

    Returns:
        A flowjax Transformed distribution.
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
