"""Flow constructors for Gaussianization flows.

This module provides factory functions for constructing various normalizing
flow architectures based on RBIG (Rotation-Based Iterative Gaussianization)
and related methods, plus the :class:`SurVAEFlow` container that mixes
ordinary FlowJax bijections with the keyed surjection / stochastic
transforms defined in :mod:`gauss_flows._src.transforms.base`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection, Chain, Flip, Invert, Permute, Scan
from flowjax.distributions import AbstractDistribution, Normal, Transformed
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms.base import AbstractSurjection
from gauss_flows._src.transforms.coupling import RQSplineCoupling
from gauss_flows._src.transforms.marginal import MixtureGaussianCDF
from gauss_flows._src.transforms.rotation import (
    HouseholderRotation,
    LULinearPermute,
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

    Each layer consists of a marginal Gaussianization step (using a mixture of
    Gaussians CDF) followed by a rotation (Householder or orthogonal).

    Args:
        key: JAX random key.
        n_dims: Dimensionality of the data.
        n_layers: Number of flow layers. Defaults to 8.
        n_components: Number of mixture components for marginal Gaussianization.
            Defaults to 8.
        rotation: Type of rotation, one of ``"householder"``, ``"orthogonal"``,
            or ``"lu"`` (LU-parametrised linear + reverse permutation).
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


def _to_data(
    transform: AbstractBijection | AbstractSurjection,
    x: Array,
    key: PRNGKeyArray,
    cond: Array | None,
) -> tuple[Array, Array]:
    """Apply one chain element in the base -> data direction (used by ``sample``).

    Note the directional mismatch between the two ABCs: FlowJax's
    ``AbstractBijection.transform_and_log_det`` is the base -> data
    direction (its "sample" side), while SurVAE's
    ``AbstractSurjection.inverse_and_log_det`` is the latent -> data
    direction (latent ≡ base). The isinstance dispatch hides this so
    :class:`SurVAEFlow` keeps a uniform call site without wrapping
    bijections in adapters.
    """
    if isinstance(transform, AbstractBijection):
        return transform.transform_and_log_det(x, cond)
    return transform.inverse_and_log_det(x, key, cond)


def _to_base(
    transform: AbstractBijection | AbstractSurjection,
    z: Array,
    key: PRNGKeyArray,
    cond: Array | None,
) -> tuple[Array, Array]:
    """Apply one chain element in the data -> base direction (used by ``log_prob``).

    Mirror of :func:`_to_data`: bijections use ``inverse_and_log_det``
    (FlowJax's "log_prob" side), surjections use ``forward_and_log_det``
    (SurVAE's data -> latent side).
    """
    if isinstance(transform, AbstractBijection):
        return transform.inverse_and_log_det(z, cond)
    return transform.forward_and_log_det(z, key, cond)


class SurVAEFlow(eqx.Module):
    """Container that threads PRNG keys through a mixed bijection / surjection chain.

    Composes any sequence of :class:`flowjax.bijections.AbstractBijection` and
    :class:`AbstractSurjection` transforms. Forward direction is base -> data
    (used by ``sample``); inverse direction is data -> base (used by
    ``log_prob``). Each transform receives its own independent split of the
    user-supplied key, regardless of whether it actually consumes it — this
    keeps the call shape uniform and JIT-friendly.

    For a chain consisting solely of :class:`AbstractBijection` instances,
    ``log_prob`` is identical (up to numerical noise) to that of the
    equivalent ``flowjax.distributions.Transformed(base, Chain(transforms))``,
    so users can migrate without changing semantics.

    Attributes:
        base_dist: A :class:`flowjax.distributions.AbstractDistribution`
            whose event shape matches the start of the forward chain.
        transforms: Tuple of bijections / surjections, applied left-to-right
            in the forward direction.
        data_shape: Event shape of the data (output of the forward chain /
            input to ``log_prob``). Defaults to ``base_dist.shape`` for
            shape-preserving chains; **must** be supplied explicitly when
            the chain contains a surjection that changes dimensionality
            (e.g. :class:`Slice`, :class:`Augment`,
            :class:`SimpleMaxPoolSurjection2d`).

    Properties:
        lower_bound: ``True`` iff at least one surjection in the chain
            contributes a lower bound to ``log_prob`` (i.e. has its
            class-level ``lower_bound`` flag set).
    """

    base_dist: AbstractDistribution
    transforms: tuple[AbstractBijection | AbstractSurjection, ...]
    data_shape: tuple[int, ...]

    def __init__(
        self,
        base_dist: AbstractDistribution,
        transforms: tuple[AbstractBijection | AbstractSurjection, ...]
        | list[AbstractBijection | AbstractSurjection],
        data_shape: tuple[int, ...] | None = None,
    ):
        self.base_dist = base_dist
        self.transforms = tuple(transforms)
        self.data_shape = data_shape if data_shape is not None else base_dist.shape

    @property
    def lower_bound(self) -> bool:
        return any(
            isinstance(t, AbstractSurjection) and t.lower_bound for t in self.transforms
        )

    def _split_keys(self, key: PRNGKeyArray) -> tuple[PRNGKeyArray, list[PRNGKeyArray]]:
        """Split ``key`` into one base-distribution key plus one per transform."""
        all_keys = jr.split(key, len(self.transforms) + 1)
        return all_keys[0], list(all_keys[1:])

    def _single_sample(self, key: PRNGKeyArray, cond: Array | None) -> Array:
        k_base, t_keys = self._split_keys(key)
        x = self.base_dist._sample(k_base, cond)
        for t, k_t in zip(self.transforms, t_keys, strict=True):
            x, _ = _to_data(t, x, k_t, cond)
        return x

    def _single_log_prob(
        self, x: Array, key: PRNGKeyArray, cond: Array | None
    ) -> Array:
        _k_base, t_keys = self._split_keys(key)
        z = x
        log_det = jnp.zeros(())
        for t, k_t in zip(reversed(self.transforms), reversed(t_keys), strict=True):
            z, ld = _to_base(t, z, k_t, cond)
            log_det = log_det + ld
        return self.base_dist._log_prob(z, cond) + log_det

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        """Sample by drawing from ``base_dist`` and pushing through to data.

        Each draw uses an independent key derived from ``key``; ``sample_shape``
        is realised via :func:`jax.vmap` over a leading batch of keys, mirroring
        the FlowJax convention.
        """
        cond = None if condition is None else jnp.asarray(condition)

        if sample_shape == ():
            return self._single_sample(key, cond)
        n = 1
        for s in sample_shape:
            n *= s
        keys = jr.split(key, n)
        flat = jax.vmap(lambda k: self._single_sample(k, cond))(keys)
        return flat.reshape(sample_shape + flat.shape[1:])

    def log_prob(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        condition: ArrayLike | None = None,
    ) -> Array:
        """Evaluate ``log p(x)`` (or its lower bound if any surjection is lower-bound).

        Accepts batched inputs of shape ``sample_shape + data_shape``; the
        leading ``sample_shape`` dimensions are vmapped over and each sample
        receives its own key derived from ``key``. For deterministic chains
        (bijections + non-stochastic surjections) the key is unused but still
        consumed by the dispatch.
        """
        cond = None if condition is None else jnp.asarray(condition)
        x = jnp.asarray(x)

        n_data = len(self.data_shape)
        # x: (*sample_shape, *data_shape) -> sample_shape: (*sample_shape,)
        if n_data > 0 and x.shape[-n_data:] != self.data_shape:
            raise ValueError(
                f"x trailing shape {x.shape[-n_data:]} does not match "
                f"data_shape {self.data_shape}."
            )
        sample_shape = x.shape if n_data == 0 else x.shape[:-n_data]

        if sample_shape == ():
            return self._single_log_prob(x, key, cond)

        n = 1
        for s in sample_shape:
            n *= s
        # flat: (n, *data_shape) for the vmap.
        flat = x.reshape((n, *self.data_shape))
        keys = jr.split(key, n)
        out = jax.vmap(lambda xi, ki: self._single_log_prob(xi, ki, cond))(flat, keys)
        return out.reshape(sample_shape)


__all__ = [
    "SurVAEFlow",
    "coupling_gaussianization_flow",
    "gaussianization_flow",
    "iterative_rbig",
]
