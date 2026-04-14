"""Flow constructors for Gaussianization flows.

This module provides factory functions for constructing various normalizing
flow architectures based on RBIG (Rotation-Based Iterative Gaussianization)
and related methods.
"""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import Chain, Flip, Invert, Permute, Scan
from flowjax.distributions import AbstractDistribution, Normal, Transformed
from jaxtyping import PRNGKeyArray

from gauss_flows._src.transforms.coupling import RQSplineCoupling
from gauss_flows._src.transforms.marginal import MixtureGaussianCDF
from gauss_flows._src.transforms.rotation import HouseholderRotation, OrthogonalRotation


def gaussianization_flow(
    key: PRNGKeyArray,
    *,
    n_dims: int,
    n_layers: int = 8,
    n_components: int = 8,
    rotation: str = "householder",
    n_reflections: int | None = None,
    base_dist: AbstractDistribution | None = None,
) -> Transformed:
    """Construct a Gaussianization flow.

    Each layer consists of a marginal Gaussianization step (using a mixture of
    Gaussians CDF) followed by a rotation (Householder or orthogonal).

    Args:
        key: JAX random key.
        n_dims: Dimensionality of the data.
        n_layers: Number of flow layers. Defaults to 8.
        n_components: Number of mixture components for marginal Gaussianization.
            Defaults to 8.
        rotation: Type of rotation, either "householder" or "orthogonal".
            Defaults to "householder".
        n_reflections: Number of Householder reflections (only for "householder").
            Defaults to n_dims.
        base_dist: Optional base distribution override (must have event
            shape ``(n_dims,)``). Defaults to a standard ``Normal``.

    Returns:
        A flowjax Transformed distribution.
    """
    if n_reflections is None:
        n_reflections = n_dims

    shape = (n_dims,)
    if base_dist is None:
        base_dist = Normal(jnp.zeros(n_dims))
    else:
        if base_dist.shape != shape:
            raise ValueError(
                f"base_dist.shape {base_dist.shape} does not match (n_dims,) = {shape}."
            )
        if base_dist.cond_shape is not None:
            raise ValueError(
                "Conditional base distributions are not supported by "
                "gaussianization_flow (no condition is threaded through). "
                f"Got base_dist with cond_shape={base_dist.cond_shape}."
            )

    def make_layer(key):
        marginal = MixtureGaussianCDF(n_components=n_components, shape=shape)
        if rotation == "householder":
            rot = HouseholderRotation(n_reflections=n_reflections, shape=shape)
        elif rotation == "orthogonal":
            rot = OrthogonalRotation(shape=shape)
        else:
            raise ValueError(f"Unknown rotation type: {rotation!r}")
        return Chain([marginal, rot])

    keys = jr.split(key, n_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers))
    return Transformed(base_dist, bijection)


def coupling_gaussianization_flow(
    key: PRNGKeyArray,
    *,
    n_dims: int,
    n_layers: int = 8,
    n_bins: int = 8,
    nn_width: int = 64,
    nn_depth: int = 2,
    interval: float = 5.0,
    invert: bool = True,
) -> Transformed:
    """Construct a coupling-based Gaussianization flow.

    Uses rational quadratic spline coupling layers interleaved with
    permutations. Each layer can represent complex non-linear transformations.

    Args:
        key: JAX random key.
        n_dims: Dimensionality of the data.
        n_layers: Number of coupling layers. Defaults to 8.
        n_bins: Number of spline bins in each coupling layer. Defaults to 8.
        nn_width: Hidden layer width for coupling conditioner. Defaults to 64.
        nn_depth: Depth of the coupling conditioner MLP. Defaults to 2.
        interval: Interval for the rational quadratic spline. Defaults to 5.0.
        invert: Whether to invert the bijection for faster log_prob. Defaults to True.

    Returns:
        A flowjax Transformed distribution.
    """
    shape = (n_dims,)
    base_dist = Normal(jnp.zeros(n_dims))

    def make_layer(key):
        bij_key, perm_key = jr.split(key)
        coupling = RQSplineCoupling(
            key=bij_key,
            shape=shape,
            n_bins=n_bins,
            interval=interval,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )
        if n_dims == 1:
            return coupling
        elif n_dims == 2:
            return Chain([coupling, Flip(shape)]).merge_chains()
        perm = Permute(jr.permutation(perm_key, jnp.arange(n_dims)))
        return Chain([coupling, perm]).merge_chains()

    keys = jr.split(key, n_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


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


__all__ = [
    "coupling_gaussianization_flow",
    "gaussianization_flow",
    "iterative_rbig",
]
