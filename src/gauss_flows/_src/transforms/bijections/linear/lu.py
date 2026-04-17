"""LU-parameterised invertible linear layer + fixed permutation."""

from typing import ClassVar

import jax
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike


class LULinearPermute(AbstractBijection):
    """Invertible linear layer via LU decomposition composed with a fixed permutation.

    Applies ``y = P @ L @ U @ x``, where ``P`` is a fixed permutation matrix,
    ``L`` is unit lower-triangular, and ``U`` is upper-triangular with a strictly
    positive diagonal (parameterised as ``exp(log_diag)``). The log-absolute
    determinant reduces to ``sum(log_diag)``, avoiding an explicit Jacobian.

    Cheaper than :class:`OrthogonalRotation` in the relevant regime (no Cayley
    solve at forward / inverse time) and widely used as the mixing layer in
    Neural Spline Flows (Durkan et al. 2019).

    Args:
        shape: Shape of the input ``(n_dims,)``.
        permutation: Optional permutation of length ``n_dims``. Defaults to the
            reverse permutation.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    lower_params: Array
    upper_params: Array
    log_diag: Array
    permutation: Array
    inv_permutation: Array

    def __init__(
        self,
        shape: tuple[int, ...],
        permutation: ArrayLike | None = None,
    ):
        if len(shape) != 1:
            raise ValueError("LULinearPermute only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        n_strict = n_dims * (n_dims - 1) // 2
        self.lower_params = jnp.zeros(n_strict)
        self.upper_params = jnp.zeros(n_strict)
        self.log_diag = jnp.zeros(n_dims)
        if permutation is None:
            self.permutation = jnp.arange(n_dims - 1, -1, -1)
        else:
            perm = jnp.asarray(permutation, dtype=jnp.int32)
            if perm.shape != (n_dims,):
                raise ValueError(
                    f"permutation must have shape ({n_dims},), got {perm.shape}."
                )
            if sorted(perm.tolist()) != list(range(n_dims)):
                raise ValueError(
                    f"permutation must be a permutation of 0..{n_dims - 1}, "
                    f"got {perm.tolist()}."
                )
            self.permutation = perm
        # Cache the inverse permutation so inverse_and_log_det avoids an argsort
        # on every call.
        self.inv_permutation = jnp.argsort(self.permutation)

    def _build_lu(self) -> tuple[Array, Array]:
        n = self.shape[0]
        lower_idx = jnp.tril_indices(n, k=-1)
        upper_idx = jnp.triu_indices(n, k=1)
        L = jnp.eye(n).at[lower_idx].set(self.lower_params)
        U = jnp.diag(jnp.exp(self.log_diag)).at[upper_idx].set(self.upper_params)
        return L, U

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        L, U = self._build_lu()
        x = jnp.asarray(x)
        y = L @ (U @ x)
        y = y[self.permutation]
        log_det = jnp.sum(self.log_diag)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        L, U = self._build_lu()
        y = jnp.asarray(y)
        y = y[self.inv_permutation]
        w = jax.scipy.linalg.solve_triangular(L, y, lower=True, unit_diagonal=True)
        x = jax.scipy.linalg.solve_triangular(U, w, lower=False)
        log_det = -jnp.sum(self.log_diag)
        return x, log_det


__all__ = ["LULinearPermute"]
