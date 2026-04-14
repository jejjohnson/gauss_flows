"""Foundation types for SurVAE-style flows: surjections and stochastic transforms.

Surjections (Nielsen et al. 2020) are deterministic in one direction and
stochastic in the other. Stochastic transforms (e.g. VAE encoders/decoders)
are stochastic in both directions. Together with ordinary FlowJax
bijections they form the SurVAE hierarchy:

    Transform (base)
    |- Bijection            <- flowjax.bijections.AbstractBijection
    |- Surjection           <- AbstractSurjection (this module)
    |   |- Inference        deterministic forward, stochastic inverse
    |   `- Generative       stochastic forward, deterministic inverse
    `- Stochastic           <- AbstractStochastic (both directions stochastic)

This is a *parallel* hierarchy: surjections do **not** inherit from
``flowjax.bijections.AbstractBijection``. They implement keyed
``forward_and_log_det(x, key, cond=None)`` /
``inverse_and_log_det(z, key, cond=None)`` methods so that PRNG keys can be
threaded through the chain. ``SurVAEFlow`` is the container that mixes
bijections with surjections.

References:
    Nielsen, Jaini, Hoogeboom, Winther, Welling.
    *SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows.*
    NeurIPS 2020.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


class AbstractSurjection(eqx.Module):
    """Base class for SurVAE surjective transforms.

    Subclasses set the three class-level flags so :class:`SurVAEFlow` (and
    other containers) can decide whether ``log_prob`` is exact or only an
    evidence lower bound:

    Attributes:
        stochastic_forward: ``True`` iff ``forward_and_log_det`` consumes
            ``key`` and the result is a sample from a non-degenerate
            distribution. ``False`` for deterministic forward (e.g.
            inference surjections like ``Slice``).
        stochastic_inverse: ``True`` iff ``inverse_and_log_det`` consumes
            ``key`` and is a sample from a non-degenerate distribution.
            ``False`` for deterministic inverse (e.g. generative
            surjections like ``Augment``).
        lower_bound: ``True`` iff the surjection's contribution to
            ``log_prob`` is a lower bound rather than the exact value.
            For Nielsen-style surjections this is equivalent to
            ``stochastic_forward`` (the forward direction is the one used
            inside ``log_prob``).
    """

    stochastic_forward: ClassVar[bool]
    stochastic_inverse: ClassVar[bool]
    lower_bound: ClassVar[bool]

    @abstractmethod
    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Compute ``z = f(x)`` and ``log |det J_f(x)|`` (or its bound)."""

    @abstractmethod
    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Compute ``x = f^{-1}(z)`` and ``log |det J_{f^{-1}}(z)|`` (or its bound)."""


class AbstractStochastic(AbstractSurjection):
    """Base class for transforms that are stochastic in **both** directions.

    Canonical example: a VAE layer where ``forward`` samples from
    ``q(z | x)`` and ``inverse`` samples from ``p(x | z)``. ``log_prob``
    returns an ELBO contribution.
    """

    stochastic_forward: ClassVar[bool] = True
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = True


class _IdentitySurjection(AbstractSurjection):
    """Concrete identity surjection used as a smoke test for the ABC.

    Both directions are deterministic and the log-determinant is zero —
    behaves exactly like a unit-Jacobian bijection but exposes the
    keyed surjection signature.
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = False
    lower_bound: ClassVar[bool] = False

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del key, cond
        return jnp.asarray(x), jnp.zeros(())

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        del key, cond
        return jnp.asarray(z), jnp.zeros(())


__all__ = [
    "AbstractStochastic",
    "AbstractSurjection",
    "_IdentitySurjection",
]
