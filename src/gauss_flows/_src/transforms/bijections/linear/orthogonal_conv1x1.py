"""Orthogonal 1x1 convolution for Gaussianization flows."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import paramax
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src.transforms.bijections.linear.rotation import OrthogonalRotation


class Orthogonal1x1Conv(AbstractBijection):
    """Orthogonal 1x1 convolution (Cayley-parameterised orthogonal mixing).

    For a ``(n_channels,)`` single event, applies an orthogonal matrix Q as
    ``y = Q·x``; the inverse is ``x = Q.T·y``; ``log|det Q| = 0`` (volume-
    preserving). Callers that operate on ``(H, W, C)`` image events should
    vmap externally — this matches the single-event convention shared with
    `Invertible1x1Conv` and the rest of the package (see
    ``CLAUDE.md``).

    **Learnable path** (default): delegates to `OrthogonalRotation`'s
    Cayley parameterisation ``Q = (I − A)(I + A)^{-1}`` with skew-symmetric
    ``A``. Unlike ``OrthogonalRotation``'s zero init (which starts at
    ``Q = I``), this class re-initialises the skew parameters with small
    Gaussian noise so the initial mixing is near-identity but not exactly
    identity — breaking the symmetry that would otherwise pin gradients to
    zero for a Householder-like conditioner.

    **Fixed path** (``fixed_matrix`` supplied): holds Q constant at the
    supplied matrix. ``fixed_matrix`` is wrapped in
    `paramax.NonTrainable` so the standard flowjax training loops
    (``flowjax.train.fit_to_data`` and friends) and any
    ``eqx.filter(model, eqx.is_inexact_array)``-based optimizer-filter
    combined with ``paramax.unwrap`` treat the matrix as frozen; the
    wrapper also applies ``jax.lax.stop_gradient`` to the inner array
    every time `_get_rotation_matrix` unwraps the module.

    !!! warning
        **Decoupled weight decay bypasses `NonTrainable`.** Optimizers that
        update parameters from the parameter *value* (e.g. ``optax.adamw``
        with a non-zero ``weight_decay``) will still drift ``fixed_matrix``
        away from orthogonality because weight decay does not go through
        the gradient path. To keep ``fixed_matrix`` bit-identical under
        ``adamw``-style training, partition it out of the optimizer state
        before updating:
        ```python
        trainable, frozen = eqx.partition(
            model,
            lambda leaf: eqx.is_inexact_array(leaf)
            and not isinstance(leaf, paramax.NonTrainable),
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        # optimize `trainable`, recombine with `frozen` at step end.
        ```

    Args:
        key: PRNG key used for learnable Cayley initialisation. Ignored
            when ``fixed_matrix`` is supplied (but still required for API
            symmetry with the other learnable bijections in this
            subpackage).
        n_channels: Number of channels / feature dimensions.
        fixed_matrix: Optional pre-computed ``(n_channels, n_channels)``
            orthogonal matrix (e.g. from PCA or a random QR). Validated
            at construction — must satisfy ``Q.T @ Q ≈ I`` within
            ``orthogonality_atol``. When ``None`` (default), the layer is
            Cayley-parameterised and trainable.
        orthogonality_atol: Absolute tolerance for the orthogonality
            check on ``fixed_matrix``. Defaults to ``1e-5`` — loose enough
            for fp32 QR / SVD residuals, tight enough to catch casual
            errors (e.g. passing a non-orthogonal PCA loading matrix).

    Shape:
        - Input  ``x``:  ``(n_channels,)``
        - Output ``y``:  ``(n_channels,)``
        - ``log_det``:   scalar (always ``0``)

    Note:
        The public surface is a superset of
        `OrthogonalRotation`: the learnable path produces the same
        family of orthogonal matrices (same ``skew_params`` shape, same
        Cayley map), with a more flow-literature-friendly name and a
        genuinely-fixed alternative branch. Prefer
        `OrthogonalRotation` if you want the exact zero-init
        (``Q = I`` at step 0) used by classic RBIG flows; prefer this
        class when small-random init or a pre-computed fixed rotation is
        wanted.

    Examples:
        Learnable (Cayley-parameterised, near-identity init):

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import Orthogonal1x1Conv
        >>> conv = Orthogonal1x1Conv(jr.key(0), n_channels=4)
        >>> y, log_det = conv.transform_and_log_det(jr.normal(jr.key(1), (4,)))
        >>> assert y.shape == (4,)
        >>> assert jnp.allclose(log_det, 0.0)

        Fixed (pre-computed orthogonal matrix, non-trainable):

        >>> Q, _ = jnp.linalg.qr(jr.normal(jr.key(2), (4, 4)))
        >>> conv_fixed = Orthogonal1x1Conv(jr.key(0), n_channels=4, fixed_matrix=Q)
        >>> y_fixed, _ = conv_fixed.transform_and_log_det(jr.normal(jr.key(3), (4,)))
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    _rotation: OrthogonalRotation | None
    # ``fixed_matrix`` is stored as ``paramax.NonTrainable[Array]`` at
    # runtime when supplied, None otherwise. Annotated as ``Array | None``
    # to match flowjax/paramax ergonomics (same pragmatic convention used
    # by ``_DeepSigmoidTransformer``'s ``base_scale`` / ``amplitudes``).
    fixed_matrix: Array | None

    def __init__(
        self,
        key: PRNGKeyArray,
        n_channels: int,
        *,
        fixed_matrix: ArrayLike | None = None,
        orthogonality_atol: float = 1e-5,
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
            # Validate Q.T @ Q ≈ I. Evaluated eagerly at construction —
            # outside jit, so the ``.item()`` call is safe.
            eye = jnp.eye(n_channels, dtype=m.dtype)
            residual = jnp.max(jnp.abs(m.T @ m - eye))
            if not bool(jnp.allclose(m.T @ m, eye, atol=orthogonality_atol).item()):
                raise ValueError(
                    "fixed_matrix must be orthogonal (Q.T @ Q ≈ I) within "
                    f"atol={orthogonality_atol}; got max|Q.T @ Q - I| = "
                    f"{float(residual.item()):.3e}."
                )
            self._rotation = None
            # NonTrainable signals to paramax-aware training loops that
            # this leaf should not be updated. Its ``unwrap`` applies
            # ``stop_gradient`` so the gradient path is also blocked.
            self.fixed_matrix = paramax.NonTrainable(m)

    def _get_rotation_matrix(self) -> Array:
        """Return the orthogonal matrix Q for the current forward/inverse call.

        Shape: ``(n_channels, n_channels)``. Unwraps the module so
        ``NonTrainable``-wrapped ``fixed_matrix`` is materialised with
        ``stop_gradient`` applied automatically.
        """
        unwrapped = paramax.unwrap(self)
        if unwrapped.fixed_matrix is not None:
            return unwrapped.fixed_matrix
        # Learnable path: reuse OrthogonalRotation's Cayley construction.
        return unwrapped._rotation._get_rotation_matrix()  # type: ignore[union-attr]

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
