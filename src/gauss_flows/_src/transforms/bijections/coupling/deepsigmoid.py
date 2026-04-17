"""Deep sigmoid coupling layer — single-layer DSF (Huang et al. 2018 NAF §3.2)."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
import paramax
from flowjax.bijections import AbstractBijection, Coupling
from jax import Array
from jax.nn import sigmoid, softplus
from jaxtyping import ArrayLike, PRNGKeyArray
from paramax.utils import inv_softplus


def _softplus_plus_floor(t: Array) -> Array:
    """Positive constraint: softplus(t) + 1e-4. Used by paramax.Parameterize."""
    return softplus(t) + 1e-4


class _DeepSigmoidTransformer(AbstractBijection):
    """Monotone single-layer sigmoidal transformer for a single dimension.

    Implements the per-dim transformer

        y     = base_scale·x + bias + Σᵢ amplitudesᵢ · σ(slopesᵢ·x + shiftsᵢ)
        dy/dx = base_scale + Σᵢ amplitudesᵢ · slopesᵢ · σ(·) · (1 − σ(·))

    with ``base_scale``, ``amplitudes``, ``slopes`` constrained strictly
    positive via :class:`paramax.Parameterize` wrapping ``softplus + 1e-4``
    (mirrors the flowjax convention in ``RQSpline``, ``StudentT``, etc.).
    ``dy/dx > 0`` everywhere → strictly monotone, hence invertible. Matches
    the *single-layer* sigmoid flow of Huang et al. (2018) NAF, §3.2; it
    is **not** the stacked DDSF ("deep dense sigmoid flow") variant from
    the same paper. Single-layer DSF has limited expressiveness for
    multi-modal targets and the loss surface tends to have a strong
    attractor at the affine local minimum (see PR #45).

    The inverse solves ``y = _forward(x)`` via :func:`optimistix.root_find`
    with :class:`optimistix.Bisection` — this gives ``expand_if_necessary``
    bracket growth and proper convergence handling, replacing the previous
    hand-rolled fori_loop bisection.
    """

    n_components: int = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    bias: Array
    # base_scale, amplitudes, slopes are stored as paramax.Parameterize
    # wrappers in __init__ and unwrapped to plain Array at runtime via
    # ``paramax.unwrap(self)``. We annotate them as ``Array`` for the type
    # checker (matching their runtime behaviour in _forward); flowjax uses
    # the same pragmatic convention.
    base_scale: Array
    amplitudes: Array
    slopes: Array
    shifts: Array

    def __init__(self, n_components: int):
        # The init values are just shape/structure declarations: the parent
        # ``Coupling`` rebuilds this transformer with conditioner-supplied
        # leaf values at every forward pass, so these defaults never reach
        # the user.
        self.n_components = n_components
        self.shape = ()
        self.bias = jnp.array(0.0)
        # Raw value r such that softplus(r) + 1e-4 ≈ 1 → init positives to 1.
        raw_for_one = jnp.asarray(inv_softplus(1.0 - 1e-4))
        self.base_scale = paramax.Parameterize(_softplus_plus_floor, raw_for_one)
        self.amplitudes = paramax.Parameterize(
            _softplus_plus_floor, jnp.full((n_components,), raw_for_one)
        )
        self.slopes = paramax.Parameterize(
            _softplus_plus_floor, jnp.full((n_components,), raw_for_one)
        )
        self.shifts = jnp.zeros((n_components,))

    def _forward(self, x: Array):
        """Evaluate y and dy/dx. Assumes ``self`` is already paramax-unwrapped."""
        # x: scalar -> y: scalar, dy/dx: scalar.
        preactivations = self.slopes * x + self.shifts  # (n_components,)
        sig = sigmoid(preactivations)  # (n_components,)
        y = self.base_scale * x + self.bias + jnp.sum(self.amplitudes * sig, axis=-1)
        dy_dx = self.base_scale + jnp.sum(
            self.amplitudes * self.slopes * sig * (1.0 - sig), axis=-1
        )
        return y, dy_dx

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        self = paramax.unwrap(self)
        y, dy_dx = self._forward(jnp.asarray(x))
        log_det = jnp.log(dy_dx)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        # Solve y = self._forward(x) for x via optimistix Bisection. This
        # delegates bracket expansion + convergence checks to the library,
        # replacing the prior hand-rolled fori_loop bisection.
        self = paramax.unwrap(self)
        y = jnp.asarray(y)
        # Affine-approx seed: y ≈ base_scale·x + bias + 0.5·Σ amplitudes.
        approx_center = (
            y - self.bias - 0.5 * jnp.sum(self.amplitudes)
        ) / self.base_scale

        def fn(x, args_y):
            # fn(x) = f(x) - y_target; root where f(x) == y_target.
            return self._forward(x)[0] - args_y

        solver = optx.Bisection(rtol=1e-6, atol=1e-6, expand_if_necessary=True)
        sol = optx.root_find(
            fn,
            solver,
            y0=approx_center,
            args=y,
            options={"lower": approx_center - 1.0, "upper": approx_center + 1.0},
            max_steps=100,
            throw=False,
        )
        x = sol.value
        _, dy_dx = self._forward(x)
        log_det = -jnp.log(dy_dx)
        return x, log_det


class DeepSigmoidCoupling(AbstractBijection):
    """Coupling layer with a monotone single-layer sigmoidal transformer.

    Replaces ``AffineCoupling``'s affine transformer with the
    :class:`_DeepSigmoidTransformer` — a per-dim monotone function
    parametrised by ``n_components`` sigmoid units (single-layer DSF, Huang
    et al. 2018 NAF §3.2). The transformer is *strictly more expressive*
    than affine in the function-class sense: it has nonzero second derivative
    where Affine has zero, and can in principle approximate any smooth
    monotone CDF.

    In practice, single-layer DSF inherits the optimisation difficulties
    Huang et al. flagged: the loss surface around the conditioner's default
    initialisation is dominated by an affine-equivalent basin, and SGD often
    fails to escape it on multi-modal / heavy-tailed targets within a few
    hundred steps. For challenging targets, prefer ``RQSplineCoupling`` (see
    flowjax) or stacked DSF chains.

    Args:
        key: JAX random key for conditioner MLP initialisation.
        shape: Event shape ``(n_dims,)``.
        n_components: Number of sigmoid units in the per-dim transformer.
            More components → more capacity, more parameters per dim.
            Defaults to 8.
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.

    Shape:
        - Input  ``x``:  ``(n_dims,)`` (with ``n_dims`` even-friendly; the
          coupling splits into ``n_dims // 2`` kept + transformed halves).
        - Output ``y``:  ``(n_dims,)``
        - ``log_det``:   scalar (sum over transformed dims of ``log(dy/dx)``)

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import DeepSigmoidCoupling
        >>>
        >>> coupling = DeepSigmoidCoupling(
        ...     key=jr.key(0), shape=(4,), n_components=8, nn_width=32, nn_depth=2,
        ... )
        >>> y, log_det = coupling.transform_and_log_det(jr.normal(jr.key(1), (4,)))
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    _coupling: AbstractBijection

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        n_components: int = 8,
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        if len(shape) != 1:
            raise ValueError("DeepSigmoidCoupling only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape

        transformer = _DeepSigmoidTransformer(n_components=n_components)
        self._coupling = Coupling(
            key=key,
            transformer=transformer,
            untransformed_dim=n_dims // 2,
            dim=n_dims,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        return self._coupling.transform_and_log_det(jnp.asarray(x), condition)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        return self._coupling.inverse_and_log_det(jnp.asarray(y), condition)


__all__ = ["DeepSigmoidCoupling"]
