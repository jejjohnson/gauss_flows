"""Sylvester normalizing flow (van den Berg et al. 2018).

No algebraic inverse; use only in the generative / variational-inference
direction via :meth:`flowjax.distributions.Transformed.sample_and_log_prob`.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


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
        scale_init: Std of the Gaussian used to initialise the Householder
            vectors (offset from the identity), the strict upper-triangular
            entries of ``R`` and ``R_tilde``, and their raw diagonal
            parameter vectors.

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

    def _q_transpose_apply(self, x: Array) -> Array:
        """Compute ``Q.T @ x`` without materialising Q.

        ``Q = (H_1 H_2 ... H_M) @ E`` where ``E = I_D[:, :M]`` selects the
        first M columns. So ``Q.T @ x = E.T @ (H_M ... H_1) @ x``: apply
        Householders ``H_1, ..., H_M`` in order, then take the leading M
        components.
        """

        def body(carry, vec):
            v_norm = vec / (jnp.linalg.norm(vec) + 1e-12)
            return carry - 2.0 * v_norm * jnp.dot(v_norm, carry), None

        result, _ = jax.lax.scan(body, x, self.householder_vectors)
        m = self.householder_vectors.shape[0]
        return result[:m]

    def _q_apply(self, v: Array) -> Array:
        """Compute ``Q @ v`` without materialising Q.

        Pad ``v`` (shape ``(M,)``) to ``D`` with zeros, then apply the
        Householder reflections in reverse order.
        """
        d = self.shape[0]
        w = jnp.zeros(d).at[: v.shape[0]].set(v)

        def body(carry, vec):
            v_norm = vec / (jnp.linalg.norm(vec) + 1e-12)
            return carry - 2.0 * v_norm * jnp.dot(v_norm, carry), None

        result, _ = jax.lax.scan(body, w, self.householder_vectors[::-1])
        return result

    def _build_Q(self) -> Array:
        """Materialise ``Q`` as a ``(D, M)`` matrix.

        Used by tests to verify orthonormality. The hot path
        (``transform_and_log_det``) uses :meth:`_q_apply` /
        :meth:`_q_transpose_apply` and never builds this matrix.
        """
        m = self.householder_vectors.shape[0]
        return jax.vmap(self._q_apply, in_axes=1, out_axes=1)(jnp.eye(m))

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
        R, R_tilde, r_diag, r_tilde_diag = self._build_RR_tilde()
        z = R @ self._q_transpose_apply(x) + self.b
        y = x + self._q_apply(R_tilde @ jnp.tanh(z))
        # Sylvester's identity: det(I_D + Q R_tilde diag(h') R Q^T)
        # = det(I_M + R_tilde diag(h') R). Both R, R_tilde upper-triangular, so
        # the result is upper-triangular with diagonal
        # 1 + diag(R_tilde) * h'(z) * diag(R).
        h_prime = 1.0 - jnp.tanh(z) ** 2
        diag_term = 1.0 + r_tilde_diag * h_prime * r_diag
        log_det = jnp.sum(jnp.log(jnp.abs(diag_term) + 1e-12))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        raise NotImplementedError(
            "SylvesterFlow has no algebraic inverse; use sample_and_log_prob in "
            "the generative direction (VI). log_prob of arbitrary points is not "
            "supported."
        )


__all__ = ["SylvesterFlow"]
