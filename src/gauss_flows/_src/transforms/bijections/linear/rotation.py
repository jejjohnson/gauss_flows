"""Learnable and fixed orthogonal rotations for Gaussianization flows."""

from typing import ClassVar

import jax.numpy as jnp
import paramax
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike


def _householder(x: Array, v: Array) -> Array:
    """Apply a single Householder reflection: x - 2*(x·v)*v (v is unit vector)."""
    v = v / jnp.linalg.norm(v)
    return x - 2.0 * v * (x @ v)


class HouseholderRotation(AbstractBijection):
    """Orthogonal rotation via a sequence of Householder reflections.

    Composes ``n_reflections`` Householder reflections — each of the form
    ``x ↦ x − 2·(x·v̂)·v̂`` with unit vector ``v̂`` — to build a general
    orthogonal map. With ``n_reflections == n_dims`` this can represent any
    orthogonal matrix. Each reflection is its own inverse, so the inverse
    simply applies the reflections in reverse order. Being orthogonal, the map
    is volume-preserving and ``log_det = 0`` exactly.

    Operates on a single ``(n_dims,)`` event; callers vmap over any batch axis.

    Args:
        n_reflections: Number of Householder reflections to compose.
        shape: Event shape ``(n_dims,)``. Only 1-D events are supported.

    Raises:
        ValueError: If ``shape`` is not 1-D.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)``, scalar log_det = 0
        - inverse_and_log_det:   ``(n_dims,)`` → ``(n_dims,)``, scalar log_det = 0

    Examples:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import HouseholderRotation
        >>> t = HouseholderRotation(n_reflections=2, shape=(3,))
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y, log_det = t.transform_and_log_det(x)
        >>> y.shape
        (3,)
        >>> bool(jnp.allclose(log_det, 0.0))
        True
        >>> x_rec, _ = t.inverse_and_log_det(y)
        >>> bool(jnp.allclose(x, x_rec, atol=1e-5))
        True
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
    """Learnable orthogonal rotation via the Cayley map.

    Parameterises an orthogonal matrix ``Q = (I − A)(I + A)⁻¹`` from a
    skew-symmetric ``A`` (built from the ``n_dims·(n_dims−1)/2`` strictly
    lower-triangular free parameters, then antisymmetrised). The Cayley
    transform maps any skew-symmetric ``A`` to a special orthogonal ``Q``,
    so the map ``y = Q·x`` is volume-preserving with ``log_det = 0`` exactly.
    Initialised at ``A = 0`` (``Q = I``), i.e. the identity at step 0.

    Operates on a single ``(n_dims,)`` event; callers vmap over any batch axis.

    Args:
        shape: Event shape ``(n_dims,)``. Only 1-D events are supported.

    Raises:
        ValueError: If ``shape`` is not 1-D.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)``, scalar log_det = 0
        - inverse_and_log_det:   ``(n_dims,)`` → ``(n_dims,)``, scalar log_det = 0

    Examples:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import OrthogonalRotation
        >>> t = OrthogonalRotation(shape=(3,))  # Q = I at init
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y, log_det = t.transform_and_log_det(x)
        >>> bool(jnp.allclose(y, x))  # identity at zero init
        True
        >>> bool(jnp.allclose(log_det, 0.0))
        True
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
    """Fixed (non-trainable) orthogonal rotation matrix.

    Applies a pre-computed orthogonal matrix ``y = matrix·x`` (e.g. a PCA
    rotation from `from_data`) that is held constant during training.
    Orthogonality makes the map volume-preserving, so ``log_det = 0`` exactly
    and the inverse is the transpose.

    The matrix is stored wrapped in `paramax.NonTrainable` so
    ``flowjax.train.fit_to_data`` skips it during gradient descent — without
    the wrapper Adam drifts the rows off the orthogonal manifold and sampling
    silently collapses while ``log_prob`` keeps reporting plausible numbers.
    The public ``matrix`` property transparently returns the unwrapped
    `jax.Array` so downstream code can still do ``rot.matrix @ x``.

    Operates on a single ``(n_dims,)`` event; callers vmap over any batch axis.

    Args:
        matrix: Orthogonal rotation matrix of shape ``(n_dims, n_dims)``.

    Raises:
        ValueError: If ``matrix`` is not a square 2-D array.

    Shape:
        - transform_and_log_det: ``(n_dims,)`` → ``(n_dims,)``, scalar log_det = 0
        - inverse_and_log_det:   ``(n_dims,)`` → ``(n_dims,)``, scalar log_det = 0

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import FixedRotation
        >>> Q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (3, 3)))
        >>> t = FixedRotation(Q)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y, log_det = t.transform_and_log_det(x)
        >>> y.shape
        (3,)
        >>> x_rec, _ = t.inverse_and_log_det(y)
        >>> bool(jnp.allclose(x, x_rec, atol=1e-5))
        True
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    _matrix: paramax.NonTrainable

    def __init__(self, matrix: ArrayLike):
        matrix = jnp.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square 2D array.")
        self.shape = (matrix.shape[0],)
        self._matrix = paramax.non_trainable(matrix)

    @property
    def matrix(self) -> Array:
        """The orthogonal rotation matrix (unwrapped from NonTrainable)."""
        return paramax.unwrap(self._matrix)

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
            A `FixedRotation` whose matrix is the PCA rotation of ``x``.
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
