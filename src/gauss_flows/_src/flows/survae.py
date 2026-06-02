"""SurVAEFlow container: mixes FlowJax bijections with keyed surjections."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from flowjax.distributions import AbstractDistribution
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms.base import AbstractSurjection


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
    `SurVAEFlow` keeps a uniform call site without wrapping
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

    Mirror of `_to_data`: bijections use ``inverse_and_log_det``
    (FlowJax's "log_prob" side), surjections use ``forward_and_log_det``
    (SurVAE's data -> latent side).
    """
    if isinstance(transform, AbstractBijection):
        return transform.inverse_and_log_det(z, cond)
    return transform.forward_and_log_det(z, key, cond)


class SurVAEFlow(eqx.Module):
    """Container that threads PRNG keys through a mixed bijection / surjection chain.

    Composes any sequence of `flowjax.bijections.AbstractBijection` and
    `AbstractSurjection` transforms. Forward direction is base -> data
    (used by ``sample``); inverse direction is data -> base (used by
    ``log_prob``). Each transform receives its own independent split of the
    user-supplied key, regardless of whether it actually consumes it — this
    keeps the call shape uniform and JIT-friendly.

    For a chain consisting solely of `AbstractBijection` instances,
    ``log_prob`` is identical (up to numerical noise) to that of the
    equivalent ``flowjax.distributions.Transformed(base, Chain(transforms))``,
    so users can migrate without changing semantics.

    Attributes:
        base_dist: A `flowjax.distributions.AbstractDistribution`
            whose event shape matches the start of the forward chain.
        transforms: Tuple of bijections / surjections, applied left-to-right
            in the forward direction.
        data_shape: Event shape of the data (output of the forward chain /
            input to ``log_prob``). Defaults to ``base_dist.shape`` for
            shape-preserving chains; **must** be supplied explicitly when
            the chain contains a surjection that changes dimensionality
            (e.g. `Slice`, `Augment`,
            `SimpleMaxPoolSurjection2d`).

    Properties:
        lower_bound: ``True`` iff at least one surjection in the chain
            contributes a lower bound to ``log_prob`` (i.e. has its
            class-level ``lower_bound`` flag set).

    Conditioning model:
        ``sample`` and ``log_prob`` accept a single optional ``condition``
        kwarg that is broadcast to **every** layer in the chain *and* to the
        base distribution. A layer that opts in (``cond_shape is not None``)
        consumes the condition; a layer that does not (``cond_shape is
        None``) silently ignores it. So a single global context vector can
        drive any subset of {base, couplings, FiLM-wrapped layers} —
        including:

        - **base only**: pair a `ConditionalDiagGaussian` /
          `ClassCondDiagGaussian` / `NumpyroBase` (factory
          mode) base with otherwise unconditional transforms.
        - **transforms only**: pair an unconditional base
          (`flowjax.distributions.Normal`, …) with one or more
          coupling layers built with ``cond_dim > 0`` (or any transform
          wrapped in `Conditioner`).
        - **both**: a conditional base **and** condition-aware transforms;
          the same ``condition`` reaches both.

        Inside a coupling layer there are two distinct conditioning streams:
        the *data-dependent* untransformed half of ``x`` (always present —
        the whole point of coupling) and the *external* ``condition``
        (present only when ``cond_dim > 0``). The two are concatenated
        inside the inner MLP; users do not slice or merge them manually.

    Shape:
        Methods follow the single-event convention but accept a leading
        ``sample_shape`` that is vmapped over internally:

        - ``sample(key, sample_shape, condition)`` → ``sample_shape + data_shape``
        - ``log_prob(x, key, condition)``: ``x`` of shape
          ``sample_shape + data_shape`` → ``sample_shape``

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from flowjax.distributions import Normal
        >>> from gauss_flows import SurVAEFlow, AffineCoupling
        >>> base = Normal(jnp.zeros(4))
        >>> flow = SurVAEFlow(base, [AffineCoupling(jr.key(0), shape=(4,))])
        >>> x = flow.sample(jr.key(1), (5,))
        >>> x.shape
        (5, 4)
        >>> flow.log_prob(x, jr.key(2)).shape
        (5,)
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
        is realised via `jax.vmap` over a leading batch of keys, mirroring
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


__all__ = ["SurVAEFlow"]
