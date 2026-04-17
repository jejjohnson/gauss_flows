"""Planar normalizing flow (Rezende & Mohamed 2015).

No algebraic inverse; use only in the generative / variational-inference
direction via :meth:`flowjax.distributions.Transformed.sample_and_log_prob`.
"""

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array
from jax.nn import softplus
from jaxtyping import ArrayLike, PRNGKeyArray


class PlanarFlow(AbstractBijection):
    """Planar flow bijection (Rezende & Mohamed 2015).

    Applies ``y = x + u_hat * tanh(w . x + b)`` where ``u_hat`` is the
    projection of ``u`` that guarantees ``w . u_hat >= -1`` (the
    invertibility condition from the original paper).

    Args:
        key: JAX random key.
        shape: Shape of the input ``(n_dims,)``.
        scale_init: Std of the Gaussian used to initialise ``u`` and ``w``.

    Note:
        No algebraic inverse; ``inverse_and_log_det`` raises
        :class:`NotImplementedError`. Use only in the generative
        (sample / VI) direction.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    u: Array
    w: Array
    b: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        scale_init: float = 0.01,
    ):
        if len(shape) != 1:
            raise ValueError("PlanarFlow only supports 1D inputs.")
        d = shape[0]
        k_u, k_w = jr.split(key)
        self.shape = shape
        self.u = jr.normal(k_u, (d,)) * scale_init
        self.w = jr.normal(k_w, (d,)) * scale_init
        self.b = jnp.zeros(())

    def _u_hat(self) -> Array:
        wu = jnp.dot(self.w, self.u)
        m = -1.0 + softplus(wu)
        w_norm_sq = jnp.sum(self.w**2) + 1e-12
        return self.u + (m - wu) * self.w / w_norm_sq

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x = jnp.asarray(x)
        u_hat = self._u_hat()
        a = jnp.dot(self.w, x) + self.b
        y = x + u_hat * jnp.tanh(a)
        psi_dot_w = (1.0 - jnp.tanh(a) ** 2) * jnp.dot(u_hat, self.w)
        log_det = jnp.log(jnp.abs(1.0 + psi_dot_w) + 1e-12)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        raise NotImplementedError(
            "PlanarFlow has no algebraic inverse; use sample_and_log_prob in the "
            "generative direction (VI). log_prob of arbitrary points is not supported."
        )


__all__ = ["PlanarFlow"]
