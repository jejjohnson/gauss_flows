"""Structural protocols used to plug external distribution objects into gauss_flows.

These ``Protocol`` definitions describe the minimal duck-typed interfaces that
SurVAE-style transforms (and their tests) need from a "conditional
distribution" object — without forcing inheritance from any concrete base
class. Concrete implementations can come from `numpyro.distributions`,
`flowjax.distributions` (via thin wrappers), or user code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array
from jaxtyping import PRNGKeyArray


@runtime_checkable
class ConditionalDistribution(Protocol):
    """Minimal ``p(x | cond)`` interface.

    Implementations must accept the conditioning variable as a keyword
    argument (``condition=...``) — chosen so that the same call site works
    whether the underlying distribution is conditional or unconditional
    (in the latter case, the impl can simply ignore ``condition``).
    """

    def sample(self, key: PRNGKeyArray, *, condition: Array) -> Array: ...

    def log_prob(self, value: Array, *, condition: Array) -> Array: ...


__all__ = ["ConditionalDistribution"]
