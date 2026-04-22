"""Learnable and fixed orthogonal rotations for Gaussianization flows."""

from typing import ClassVar

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

    @classmethod
    def from_data(cls, x: ArrayLike) -> "FixedRotation":
        """Build a PCA rotation from the eigenvectors of ``cov(x)``.

        The rotation is the matrix whose rows are the principal axes of
        ``x``, in descending-eigenvalue order, so that ``y = rotation(x)``
        is the decorrelating PCA projection.

        Args:
            x: Training data of shape ``(n, d)``.

        Returns:
            A :class:`FixedRotation` whose matrix is the PCA rotation of ``x``.
        """
        x = jnp.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (n, d); got shape {x.shape}")
        xc = x - jnp.mean(x, axis=0, keepdims=True)
        cov = (xc.T @ xc) / jnp.maximum(x.shape[0] - 1, 1)
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        # eigh returns ascending eigvals; reverse for PCA convention, then
        # transpose so rows are principal axes (y = matrix @ x).
        del eigvals
        matrix = eigvecs[:, ::-1].T
        return cls(matrix)


__all__ = ["FixedRotation", "HouseholderRotation", "OrthogonalRotation"]
