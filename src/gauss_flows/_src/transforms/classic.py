"""Classic low-expressivity normalizing flow bijections.

Both transforms predate coupling and spline flows. They are useful as
teaching baselines and as variational-inference-only flows: they have no
algebraic inverse, so they can be used to sample from ``base -> data`` but
cannot evaluate ``log_prob`` on arbitrary points. Use
:meth:`flowjax.distributions.Transformed.sample_and_log_prob` in the VI
setting.

References:
    Rezende & Mohamed 2015, *Variational Inference with Normalizing Flows*.
    van den Berg et al. 2018, *Sylvester Normalizing Flows for VI*.
"""

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array
from jax.nn import softplus
from jaxtyping import ArrayLike, PRNGKeyArray


_NO_INVERSE_MSG = (
    "{name} has no algebraic inverse; use sample_and_log_prob in the "
    "generative direction (VI). log_prob of arbitrary points is not supported."
)


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
        raise NotImplementedError(_NO_INVERSE_MSG.format(name="PlanarFlow"))


class SylvesterFlow(AbstractBijection):
    """Sylvester normalizing flow (van den Berg et al. 2018).

    Generalises :class:`PlanarFlow` to rank ``M <= D``:

    .. code-block::

        y = x + Q @ R_tilde @ tanh(R @ Q^T @ x + b)

    where ``Q in R^{D x M}`` has orthonormal columns (parameterised via
    ``M`` Householder reflections on ``I_D``), and ``R``, ``R_tilde`` are
    ``M x M`` upper-triangular matrices with diagonal entries constrained
    via ``tanh`` (to keep ``|diag(R) * diag(R_tilde)| < 1``, which is
    sufficient for invertibility).

    Via Sylvester's determinant identity the log-abs-det reduces to a
    sum over the ``M`` diagonal entries.

    Args:
        key: JAX random key.
        shape: Shape of the input ``(n_dims,)``.
        rank: Rank ``M`` of the flow. Defaults to the full input dim.
        scale_init: Std of Gaussian init for the triangular strict upper
            entries.

    Note:
        No algebraic inverse; ``inverse_and_log_det`` raises
        :class:`NotImplementedError`. Use only in the generative direction.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    householder_vectors: Array
    R_raw_diag: Array
    R_upper: Array
    R_tilde_raw_diag: Array
    R_tilde_upper: Array
    b: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        rank: int | None = None,
        scale_init: float = 0.01,
    ):
        if len(shape) != 1:
            raise ValueError("SylvesterFlow only supports 1D inputs.")
        d = shape[0]
        if rank is None:
            rank = d
        if not 1 <= rank <= d:
            raise ValueError(f"rank must be in [1, {d}], got {rank}.")
        m = rank
        self.shape = shape
        n_strict = m * (m - 1) // 2
        k_hh, k_rd, k_ru, k_td, k_tu = jr.split(key, 5)
        # Initialise Householder vectors away from zero to avoid degenerate norms
        self.householder_vectors = (
            jr.normal(k_hh, (m, d)) * scale_init + jnp.eye(m, d) * 1.0
        )
        self.R_raw_diag = jr.normal(k_rd, (m,)) * scale_init
        self.R_upper = jr.normal(k_ru, (n_strict,)) * scale_init
        self.R_tilde_raw_diag = jr.normal(k_td, (m,)) * scale_init
        self.R_tilde_upper = jr.normal(k_tu, (n_strict,)) * scale_init
        self.b = jnp.zeros((m,))

    def _build_Q(self) -> Array:
        """Householder-parametrised matrix with orthonormal columns (D, M)."""
        d = self.shape[0]
        m = self.householder_vectors.shape[0]
        Q = jnp.eye(d)
        for v in self.householder_vectors:
            v = v / (jnp.linalg.norm(v) + 1e-12)
            Q = Q - 2.0 * jnp.outer(v, v @ Q)
        return Q[:, :m]

    def _build_upper_triangular(self, diag: Array, strict_upper: Array) -> Array:
        m = diag.shape[0]
        R = jnp.diag(diag)
        if m > 1:
            upper_idx = jnp.triu_indices(m, k=1)
            R = R.at[upper_idx].set(strict_upper)
        return R

    def _build_RR_tilde(self) -> tuple[Array, Array, Array, Array]:
        # Constrain |diag(R) * diag(R_tilde)| < 1 for invertibility.
        r_diag = jnp.tanh(self.R_raw_diag)
        r_tilde_diag = jnp.tanh(self.R_tilde_raw_diag)
        R = self._build_upper_triangular(r_diag, self.R_upper)
        R_tilde = self._build_upper_triangular(r_tilde_diag, self.R_tilde_upper)
        return R, R_tilde, r_diag, r_tilde_diag

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x = jnp.asarray(x)
        Q = self._build_Q()
        R, R_tilde, r_diag, r_tilde_diag = self._build_RR_tilde()
        z = R @ (Q.T @ x) + self.b
        y = x + Q @ (R_tilde @ jnp.tanh(z))
        # Sylvester's identity: det(I_D + Q R_tilde diag(h') R Q^T)
        # = det(I_M + R_tilde diag(h') R). Both R, R_tilde upper-triangular, so
        # the result is upper-triangular with diagonal
        # 1 + diag(R_tilde) * h'(z) * diag(R).
        h_prime = 1.0 - jnp.tanh(z) ** 2
        diag_term = 1.0 + r_tilde_diag * h_prime * r_diag
        log_det = jnp.sum(jnp.log(jnp.abs(diag_term) + 1e-12))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        raise NotImplementedError(_NO_INVERSE_MSG.format(name="SylvesterFlow"))


__all__ = [
    "PlanarFlow",
    "SylvesterFlow",
]
