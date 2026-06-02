"""RBIG-style Gaussianization flow constructors."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import Chain, Flip, Invert, Permute, Scan
from flowjax.distributions import AbstractDistribution, Normal, Transformed
from jaxtyping import PRNGKeyArray

from gauss_flows._src.transforms.bijections.coupling.spline import RQSplineCoupling
from gauss_flows._src.transforms.bijections.elementwise.mixture_cdf import (
    MixtureGaussianCDF,
)
from gauss_flows._src.transforms.bijections.linear.lu import LULinearPermute
from gauss_flows._src.transforms.bijections.linear.rotation import (
    HouseholderRotation,
    OrthogonalRotation,
)


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

    Stacks ``n_layers`` alternating G∘R blocks, where each block is a
    marginal Gaussianization step G (a mixture-of-Gaussians CDF applied
    per dimension) followed by a rotation R (Householder, orthogonal, or
    LU-parametrised linear). Mixture means and Householder vectors are
    randomly initialised to break symmetry so gradient training has a
    signal to separate the components. The block stack is composed with a
    memory-efficient `flowjax.bijections.Scan` and wrapped in a
    base `flowjax.distributions.Normal`.

    Args:
        key: JAX random key.
        n_dims: Dimensionality of the data.
        n_layers: Number of flow layers. Defaults to 8.
        n_components: Number of mixture components for marginal Gaussianization.
            Defaults to 8.
        rotation: Type of rotation, one of ``"householder"``, ``"orthogonal"``,
            or ``"lu"`` (LU-parametrised linear + reverse permutation).
            Defaults to ``"householder"``.
        n_reflections: Number of Householder reflections (only for
            ``"householder"``). Defaults to ``n_dims``.
        base_dist: Optional base distribution override (must have event
            shape ``(n_dims,)`` and be unconditional). Defaults to a standard
            ``Normal``.

    Returns:
        A flowjax ``Transformed`` distribution with ``log_prob`` and ``sample``.

    Raises:
        ValueError: If ``base_dist`` has the wrong event shape, is
            conditional, or ``rotation`` is not a recognised name.

    Examples:
        >>> import jax.random as jr
        >>> from gauss_flows import gaussianization_flow
        >>> flow = gaussianization_flow(jr.key(0), n_dims=4, n_layers=4, n_components=8)
        >>> flow.sample(jr.key(1), (8,)).shape
        (8, 4)
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
        # Break symmetry at init: a plain MixtureGaussianCDF has means=0 and
        # a plain HouseholderRotation has deterministic params, so all
        # mixture components collapse to the same Gaussian and training
        # sees no gradient to push them apart. The Keras reference
        # (research_notebook/projects/gaussianization) uses Uniform(-3, 3)
        # means and RandomNormal Householder vectors for exactly this
        # reason.
        marginal_key, rot_key = jr.split(key)
        marginal = MixtureGaussianCDF(n_components=n_components, shape=shape)
        init_means = jr.uniform(
            marginal_key, marginal.means.shape, minval=-3.0, maxval=3.0
        )
        marginal = eqx.tree_at(lambda m: m.means, marginal, init_means)

        if rotation == "householder":
            rot = HouseholderRotation(n_reflections=n_reflections, shape=shape)
            init_params = jr.normal(rot_key, rot.params.shape)
            rot = eqx.tree_at(lambda r: r.params, rot, init_params)
        elif rotation == "orthogonal":
            rot = OrthogonalRotation(shape=shape)
        elif rotation == "lu":
            rot = LULinearPermute(shape=shape)
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

    Stacks ``n_layers`` alternating C∘P blocks, where each block is a
    rational-quadratic-spline coupling layer C followed by a permutation P
    (a `flowjax.bijections.Flip` for ``n_dims == 2``, a random
    `flowjax.bijections.Permute` otherwise; no permutation for
    ``n_dims == 1``). Unlike `gaussianization_flow`, the coupling
    conditioner lets each layer represent non-linear, cross-coordinate
    transformations. The block stack is composed with
    `flowjax.bijections.Scan` and wrapped in a base
    `flowjax.distributions.Normal`.

    Args:
        key: JAX random key.
        n_dims: Dimensionality of the data.
        n_layers: Number of coupling layers. Defaults to 8.
        n_bins: Number of spline bins in each coupling layer. Defaults to 8.
        nn_width: Hidden layer width for coupling conditioner. Defaults to 64.
        nn_depth: Depth of the coupling conditioner MLP. Defaults to 2.
        interval: Interval for the rational quadratic spline. Defaults to 5.0.
        invert: Whether to invert the bijection so ``log_prob`` runs in the
            fast direction. Defaults to True.

    Returns:
        A flowjax ``Transformed`` distribution with ``log_prob`` and ``sample``.

    Examples:
        >>> import jax.random as jr
        >>> from gauss_flows import coupling_gaussianization_flow
        >>> flow = coupling_gaussianization_flow(jr.key(0), n_dims=4, n_layers=2)
        >>> flow.sample(jr.key(1), (8,)).shape
        (8, 4)
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


__all__ = ["coupling_gaussianization_flow", "gaussianization_flow"]
