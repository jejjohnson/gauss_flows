"""Stochastic uniform permutation along one axis (zero log-det)."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms.base import AbstractStochastic


class StochasticPermutation(AbstractStochastic):
    """Uniform random permutation along one axis (zero log-det).

    Both directions draw a fresh permutation σ of ``{0, …, n−1}`` (with
    ``n = shape[axis]``) and apply it along ``axis``. The forward and inverse
    sampling distributions are identical:

        q_forward(z | x) = q_inverse(x | z) = 1 / n!  for each σ ∈ S_n

    so the entropy contribution to ``log_prob`` cancels and ``log_det = 0`` in
    both directions. The transform is volume-preserving in the discrete sense:
    every output is a permutation of the input.

    Args:
        shape: Event shape of a single input (no leading batch dim — callers
            vmap externally; see the project-wide single-event convention in
            ``CLAUDE.md``).
        axis: Axis along which to permute (default ``0``). Wrapped modulo
            ``len(shape)``.

    Shape:
        - Input ``x``: ``shape``
        - Output ``z``: ``shape`` (a permutation of ``x`` along ``axis``)
        - ``log_det``: scalar (shape ``()``)

    Examples:
        Permute a 1D event:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import StochasticPermutation
        >>>
        >>> surj = StochasticPermutation((4,))
        >>> z, log_det = surj.forward_and_log_det(jnp.arange(4.0), jr.key(0))
        >>> # z is a permutation of [0., 1., 2., 3.]; log_det == 0.

        Permute the columns of a 2D event:

        >>> surj2 = StochasticPermutation((3, 4), axis=1)
        >>> z, _ = surj2.forward_and_log_det(jnp.zeros((3, 4)), jr.key(0))
        >>> # z.shape == (3, 4); same permutation applied to every row.

        Use inside a SurVAEFlow (the container vmaps over leading sample dims):

        >>> from gauss_flows import SurVAEFlow
        >>> from flowjax.distributions import Normal
        >>> base = Normal(jnp.zeros(4))
        >>> flow = SurVAEFlow(base, [StochasticPermutation((4,))])
    """

    shape: tuple[int, ...]
    axis: int
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...], axis: int = 0):
        if not shape:
            raise ValueError("Shape must be non-empty for StochasticPermutation.")
        self.shape = shape
        self.axis = axis % len(shape)

    def _permute(self, key: PRNGKeyArray, x: Array) -> Array:
        # x: shape; perm: (shape[axis],); out: shape
        perm = jr.permutation(key, self.shape[self.axis])
        return jnp.take(x, perm, axis=self.axis)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample z = σ_forward(x) with σ_forward ~ Uniform(S_n)."""
        del cond
        z = self._permute(key, jnp.asarray(x))
        return z, jnp.zeros(())

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample x = σ_inverse(z) with σ_inverse ~ Uniform(S_n)."""
        del cond
        x = self._permute(key, jnp.asarray(z))
        return x, jnp.zeros(())


__all__ = ["StochasticPermutation"]
