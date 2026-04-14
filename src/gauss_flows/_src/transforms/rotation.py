"""Rotation transforms for Gaussianization flows.

These bijections implement various rotation/linear transformations used in
RBIG and related methods to decorrelate dimensions between marginal
Gaussianization steps.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike


def _householder(x: Array, v: Array) -> Array:
    """Apply a single Householder reflection: x - 2*(x·v)*v (v is unit vector)."""
    v = v / jnp.linalg.norm(v)
    return x - 2.0 * v * (x @ v)


class HouseholderRotation(AbstractBijection):
    """Rotation via a sequence of Householder reflections.

    Stacks multiple Householder reflections to produce a general orthogonal
    transformation. When n_reflections == dim, this can represent any orthogonal
    matrix.

    Args:
        n_reflections: Number of Householder reflections to compose.
        shape: Shape of the input (n_dims,).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    params: Array

    def __init__(self, n_reflections: int, shape: tuple[int, ...]):
        if len(shape) != 1:
            raise ValueError("HouseholderRotation only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        # Initialize with random-looking but deterministic params
        self.params = jnp.ones((n_reflections, n_dims)) / jnp.sqrt(
            jnp.arange(1, n_dims + 1, dtype=float)
        )

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x = jnp.asarray(x)
        for v in self.params:
            x = _householder(x, v)
        return x, jnp.zeros(())

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        y = jnp.asarray(y)
        # Inverse is applying reflections in reverse order (each H is self-inverse)
        for v in reversed(list(self.params)):
            y = _householder(y, v)
        return y, jnp.zeros(())


class OrthogonalRotation(AbstractBijection):
    """Rotation via a learnable orthogonal matrix using Cayley parameterization.

    Uses the Cayley map to parameterize orthogonal matrices via skew-symmetric
    matrices: Q = (I - A)(I + A)^{-1}, where A is skew-symmetric.

    Args:
        shape: Shape of the input (n_dims,).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    skew_params: Array

    def __init__(self, shape: tuple[int, ...]):
        if len(shape) != 1:
            raise ValueError("OrthogonalRotation only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        n_params = n_dims * (n_dims - 1) // 2
        self.skew_params = jnp.zeros(n_params)

    def _get_rotation_matrix(self) -> Array:
        n_dims = self.shape[0]
        A = jnp.zeros((n_dims, n_dims))
        idx = jnp.tril_indices(n_dims, k=-1)
        A = A.at[idx].set(self.skew_params)
        A = A - A.T  # make skew-symmetric
        eye = jnp.eye(n_dims)
        # Cayley map: Q = (I - A)(I + A)^{-1}
        # Equivalent to: Q = solve(I + A, I - A)^T with A^T = -A gives (I-A)(I+A)^{-1}
        Q = jnp.linalg.solve(eye - A, eye + A).T
        return Q

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        Q = self._get_rotation_matrix()
        y = Q @ jnp.asarray(x)
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        Q = self._get_rotation_matrix()
        x = Q.T @ jnp.asarray(y)
        return x, jnp.zeros(())


class FixedRotation(AbstractBijection):
    """A fixed (non-trainable) rotation matrix.

    Useful when the rotation is pre-computed (e.g. via PCA) and should
    not be updated during training.

    Args:
        matrix: Orthogonal rotation matrix of shape (n_dims, n_dims).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    matrix: Array

    def __init__(self, matrix: ArrayLike):
        matrix = jnp.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square 2D array.")
        self.shape = (matrix.shape[0],)
        self.matrix = matrix

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        y = self.matrix @ jnp.asarray(x)
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        x = self.matrix.T @ jnp.asarray(y)
        return x, jnp.zeros(())


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


__all__ = [
    "FixedRotation",
    "HouseholderRotation",
    "LULinearPermute",
    "OrthogonalRotation",
]
