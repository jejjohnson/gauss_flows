"""Orthogonal 1x1 convolution for Gaussianization flows."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array, lax
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms.bijections.linear.rotation import OrthogonalRotation


class Orthogonal1x1Conv(AbstractBijection):
    """Orthogonal 1x1 convolution (Cayley-parameterised orthogonal mixing).

    For a ``(n_channels,)`` single event, applies an orthogonal matrix Q as
    ``y = Q·x``; the inverse is ``x = Q.T·y``; ``log|det Q| = 0`` (volume-
    preserving). Callers that operate on ``(H, W, C)`` image events should
    vmap externally — this matches the single-event convention shared with
    :class:`Invertible1x1Conv` and the rest of the package (see
    ``CLAUDE.md``).

    **Learnable path** (default): delegates to :class:`OrthogonalRotation`'s
    Cayley parameterisation ``Q = (I − A)(I + A)^{-1}`` with skew-symmetric
    ``A``. Unlike ``OrthogonalRotation``'s zero init (which starts at
    ``Q = I``), this class re-initialises the skew parameters with small
    Gaussian noise so the initial mixing is near-identity but not exactly
    identity — breaking the symmetry that would otherwise pin gradients to
    zero for a Householder-like conditioner.

    **Fixed path** (``fixed_matrix`` supplied): holds Q constant at the
    supplied matrix. Gradients into ``fixed_matrix`` are blocked via
    :func:`jax.lax.stop_gradient`, so Optax steps against a frozen flow
    leave the matrix exactly unchanged — unlike plain :class:`FixedRotation`
    whose ``matrix`` field is a mutable leaf.

    Args:
        key: PRNG key used for learnable Cayley initialisation. Ignored
            when ``fixed_matrix`` is supplied (but still required for API
            symmetry with the other learnable bijections in this
            subpackage).
        n_channels: Number of channels / feature dimensions.
        fixed_matrix: Optional pre-computed ``(n_channels, n_channels)``
            orthogonal matrix (e.g. from PCA or a random SVD rotation).
            When provided, the layer is non-trainable — gradients through
            the stored matrix are blocked. When ``None`` (default), the
            layer is Cayley-parameterised and trainable.

    Shape:
        - Input  ``x``:  ``(n_channels,)``
        - Output ``y``:  ``(n_channels,)``
        - ``log_det``:   scalar (always ``0``)

    Note:
        The public surface is a superset of
        :class:`OrthogonalRotation`: the learnable path produces the same
        family of orthogonal matrices (same ``skew_params`` shape, same
        Cayley map), with a more flow-literature-friendly name and a
        genuinely-fixed alternative branch. Prefer
        :class:`OrthogonalRotation` if you want the exact zero-init
        (``Q = I`` at step 0) used by classic RBIG flows; prefer this
        class when small-random init or a pre-computed fixed rotation is
        wanted.

    Example:
        Learnable (Cayley-parameterised, near-identity init):

        >>> import jax.random as jr
        >>> from gauss_flows import Orthogonal1x1Conv
        >>> conv = Orthogonal1x1Conv(jr.key(0), n_channels=4)
        >>> y, log_det = conv.transform_and_log_det(jr.normal(jr.key(1), (4,)))
        >>> assert y.shape == (4,) and log_det == 0.0

        Fixed (pre-computed orthogonal matrix, non-trainable):

        >>> import jax.numpy as jnp
        >>> Q, _ = jnp.linalg.qr(jr.normal(jr.key(2), (4, 4)))
        >>> conv_fixed = Orthogonal1x1Conv(jr.key(0), n_channels=4, fixed_matrix=Q)
        >>> y_fixed, _ = conv_fixed.transform_and_log_det(jr.normal(jr.key(3), (4,)))
        >>> # Fixed matrix survives gradient steps unchanged (see tests).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    _rotation: OrthogonalRotation | None
    fixed_matrix: Array | None

    def __init__(
        self,
        key: PRNGKeyArray,
        n_channels: int,
        *,
        fixed_matrix: ArrayLike | None = None,
    ):
        self.shape = (n_channels,)
        if fixed_matrix is None:
            # Learnable path — delegate to OrthogonalRotation's Cayley
            # construction, replacing its zero init with small-random skew
            # params so the initial Q is near-identity (breaks symmetry).
            base = OrthogonalRotation(shape=self.shape)
            n_params = n_channels * (n_channels - 1) // 2
            self._rotation = eqx.tree_at(
                lambda m: m.skew_params,
                base,
                jr.normal(key, (n_params,)) * 0.01,
            )
            self.fixed_matrix = None
        else:
            m = jnp.asarray(fixed_matrix, dtype=float)
            if m.shape != (n_channels, n_channels):
                raise ValueError(
                    "fixed_matrix must have shape "
                    f"({n_channels}, {n_channels}); got {m.shape}."
                )
            self._rotation = None
            self.fixed_matrix = m

    def _get_rotation_matrix(self) -> Array:
        """Return the orthogonal matrix Q for the current forward/inverse call.

        Shape: ``(n_channels, n_channels)``. When fixed, gradients through
        Q are blocked via ``lax.stop_gradient`` so the matrix is genuinely
        non-trainable.
        """
        if self.fixed_matrix is not None:
            return lax.stop_gradient(self.fixed_matrix)
        # Learnable path: reuse OrthogonalRotation's Cayley construction.
        return self._rotation._get_rotation_matrix()  # type: ignore[union-attr]

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        Q = self._get_rotation_matrix()
        y = Q @ jnp.asarray(x)
        # Orthogonal => |det Q| = 1 => log|det| = 0 exactly.
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        Q = self._get_rotation_matrix()
        # Orthogonal matrix inverse is its transpose.
        x = Q.T @ jnp.asarray(y)
        return x, jnp.zeros(())


__all__ = ["Orthogonal1x1Conv"]
