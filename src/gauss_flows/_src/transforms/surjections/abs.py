"""Elementwise absolute-value surjection with a random-sign inverse."""

from __future__ import annotations

import math
from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms.base import AbstractSurjection


class SimpleAbsSurjection(AbstractSurjection):
    """Elementwise absolute-value surjection with a random-sign inverse.

    Inference-style surjection (Nielsen et al. 2020 §3.1). The forward map is
    deterministic; the inverse draws a uniform sign per dimension:

        forward:  z = |x|                            (deterministic)
        inverse:  s ~ Bernoulli(½)^D,  x = (2s − 1) · z

    The log-det contribution is the entropy of the uniform sign prior:

        log_det = log p(s) = −D · log 2,    D = prod(shape)

    This is exact, not a bound: every collapsed pair ``{+x, −x}`` has equal
    prior mass, so the inverse uniform sampler matches the true posterior over
    pre-images.

    Args:
        shape: Event shape of a single input (no leading batch — see the
            project-wide single-event convention).

    Shape:
        - Input ``x``:   ``shape``
        - Output ``z``:  ``shape``    (non-negative)
        - ``log_det``:   scalar (shape ``()``)

    Examples:
        Single-event call:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import SimpleAbsSurjection
        >>>
        >>> surj = SimpleAbsSurjection((3,))
        >>> z, log_det = surj.forward_and_log_det(
        ...     jnp.array([1.0, -2.0, 3.0]), jr.key(0)
        ... )
        >>> # z == [1., 2., 3.]; log_det == -3 * log(2)

        Inside a SurVAEFlow (container vmaps over batch):

        >>> from gauss_flows import SurVAEFlow
        >>> from flowjax.distributions import Normal
        >>> flow = SurVAEFlow(Normal(jnp.zeros(3)), [SimpleAbsSurjection((3,))])
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    @property
    def _log_det(self) -> float:
        return -math.prod(self.shape) * math.log(2.0)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Compute ``z = |x|`` and the constant log-det ``-D · log 2``."""
        del key, cond
        # x: shape -> z: shape (non-negative).
        z = jnp.abs(jnp.asarray(x))
        return z, jnp.asarray(self._log_det)

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample ``x = ±z`` element-wise with uniform signs."""
        del cond
        z_arr = jnp.asarray(z)
        # signs: shape, in {-1, +1}; x: shape.
        signs = jr.bernoulli(key, 0.5, shape=z_arr.shape)
        signs = jnp.where(signs, 1.0, -1.0)
        x = signs * z_arr
        return x, jnp.asarray(self._log_det)


__all__ = ["SimpleAbsSurjection"]
