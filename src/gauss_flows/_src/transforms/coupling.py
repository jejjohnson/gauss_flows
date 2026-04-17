"""Coupling layer transforms for Gaussianization flows.

These bijections implement various coupling-based architectures used in
normalizing flows for density estimation.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
import paramax
from flowjax.bijections import (
    AbstractBijection,
    Coupling,
    RationalQuadraticSpline,
)
from flowjax.bijections.affine import Affine
from jax import Array
from jax.nn import sigmoid, softplus
from jaxtyping import ArrayLike, PRNGKeyArray
from paramax.utils import inv_softplus


class ActNorm1D(AbstractBijection):
    """Activation normalization for 1D inputs.

    Performs per-dimension affine transformation with learnable location and
    log-scale parameters. Similar to batch normalization but with learned
    parameters that are not data-dependent at inference time.

    Args:
        shape: Shape of the input (n_dims,).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    log_scale: Array

    def __init__(self, shape: tuple[int, ...]):
        if len(shape) != 1:
            raise ValueError("ActNorm1D only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape
        self.loc = jnp.zeros(n_dims)
        self.log_scale = jnp.zeros(n_dims)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        scale = softplus(self.log_scale) + 1e-5
        y = (jnp.asarray(x) - self.loc) / scale
        log_det = -jnp.sum(jnp.log(scale))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        scale = softplus(self.log_scale) + 1e-5
        x = jnp.asarray(y) * scale + self.loc
        log_det = jnp.sum(jnp.log(scale))
        return x, log_det


class BatchNorm(AbstractBijection):
    """Invertible batch normalization layer for 1D inputs.

    Like every other bijection in this package, ``transform_and_log_det`` and
    ``inverse_and_log_det`` operate on a **single event** (``x.shape == self.shape``)
    and return a **scalar** ``log_det``. Callers vectorise over a batch with
    ``jax.vmap`` / ``eqx.filter_vmap``. Because a single event has no batch
    statistics of its own, the layer requires an explicit statistics source
    before the forward pass:

    - **Training**: compute batch statistics once outside ``vmap`` via
      ``.with_batch_stats_from_data(batch)``. The returned layer uses those
      stats for every per-event call in that forward pass. Inverse roundtrip
      holds only while the same ``batch_stats`` are in effect.
    - **Evaluation**: warm the running EMA via
      ``.update_running_stats_from_batch(...)`` (outside ``jit``), then flip
      to eval mode with ``.with_running_average(True)``. Forward/inverse are
      exact roundtrips regardless of the current batch.

    Calling either method without having set one of those two sources raises
    ``RuntimeError`` — there is no silent fall-back to computing stats from
    the single-event input.

    **Design note**: running stats and batch stats live as regular (immutable)
    ``Array`` fields updated via ``eqx.tree_at`` rather than ``equinox.nn.State``.
    This keeps the module a plain PyTree that plays nicely with ``Chain``,
    ``flowjax.bijections.Scan``, and the existing flowjax training loop; the
    trade-off is that users must thread updated copies back through their loop
    explicitly instead of mutating in place.

    Args:
        shape: Input shape ``(n_dims,)``.
        momentum: Exponential moving-average factor for running statistics,
            updated as ``running = (1 - momentum) * running + momentum * batch``.
        eps: Numerical stability term added to the variance.
        use_running_average: Whether to use running statistics (evaluation).

    Example:
        Training step pattern (per batch)::

            bn = bn.with_batch_stats_from_data(batch)           # set source
            y, log_det = jax.vmap(bn.transform_and_log_det)(batch)
            # ... loss.backward, optimiser step ...
            bn = bn.update_running_stats_from_batch(batch)      # update EMA

        Evaluation::

            bn_eval = bn.with_running_average(True)
            y, _ = jax.vmap(bn_eval.transform_and_log_det)(batch)
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    momentum: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    # use_running_average / use_batch_stats are Python bools kept in the pytree
    # (no static=True) so eqx.tree_at can target them. Python bools are already
    # static-in-practice for jit purposes since they are not JAX arrays.
    use_running_average: bool
    use_batch_stats: bool
    log_gamma: Array
    beta: Array
    running_mean: Array
    running_var: Array
    batch_stats: tuple[Array, Array]

    def __init__(
        self,
        shape: tuple[int, ...],
        momentum: float = 0.1,
        eps: float = 1e-5,
        *,
        use_running_average: bool = False,
    ):
        if len(shape) != 1:
            raise ValueError("BatchNorm only supports 1D inputs.")
        if not 0.0 < momentum < 1.0:
            raise ValueError(f"momentum must be in (0, 1), got {momentum}.")
        self.shape = shape
        self.momentum = momentum
        self.eps = eps
        self.use_running_average = use_running_average
        self.use_batch_stats = False
        n_dims = shape[0]
        self.log_gamma = jnp.zeros(n_dims)
        self.beta = jnp.zeros(n_dims)
        self.running_mean = jnp.zeros(n_dims)
        self.running_var = jnp.ones(n_dims)
        # Kept as a real tuple (never ``None``) so the pytree structure stays
        # constant across ``.with_batch_stats_from_data`` calls — jit does not
        # retrace and vmap does not see a structure mismatch.
        self.batch_stats = (jnp.zeros(n_dims), jnp.ones(n_dims))

    def _select_stats(self) -> tuple[Array, Array]:
        if self.use_running_average:
            return self.running_mean, self.running_var
        if self.use_batch_stats:
            return self.batch_stats
        raise RuntimeError(
            "BatchNorm needs a statistics source. Call "
            ".with_batch_stats_from_data(batch) before the forward pass for "
            "training, or .with_running_average(True) for evaluation."
        )

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x_arr = jnp.asarray(x)
        mean, var = self._select_stats()
        scale = jnp.exp(self.log_gamma) / jnp.sqrt(var + self.eps)
        y = (x_arr - mean) * scale + self.beta
        log_det = jnp.sum(self.log_gamma - 0.5 * jnp.log(var + self.eps))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        y_arr = jnp.asarray(y)
        mean, var = self._select_stats()
        scale = jnp.exp(-self.log_gamma) * jnp.sqrt(var + self.eps)
        x = (y_arr - self.beta) * scale + mean
        log_det = jnp.sum(-self.log_gamma + 0.5 * jnp.log(var + self.eps))
        return x, log_det

    def update_running_stats(
        self, batch_mean: ArrayLike, batch_var: ArrayLike
    ) -> BatchNorm:
        """Return a copy with running stats updated (call outside ``jit``)."""

        new_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * jnp.asarray(batch_mean)
        new_var = (1 - self.momentum) * self.running_var + self.momentum * jnp.asarray(
            batch_var
        )
        return eqx.tree_at(
            lambda m: (m.running_mean, m.running_var), self, (new_mean, new_var)
        )

    def update_running_stats_from_batch(self, batch: ArrayLike) -> BatchNorm:
        """Compute batch statistics from ``batch`` (shape ``(N, *self.shape)``)
        and update running buffers via the EMA."""

        batch_arr = jnp.asarray(batch)
        batch_mean = jnp.mean(batch_arr, axis=0)
        batch_var = jnp.var(batch_arr, axis=0)
        return self.update_running_stats(batch_mean, batch_var)

    def with_batch_stats_from_data(self, batch: ArrayLike) -> BatchNorm:
        """Return a copy using batch statistics computed from ``batch``
        (shape ``(N, *self.shape)``). Enables the ``use_batch_stats`` code path."""

        batch_arr = jnp.asarray(batch)
        batch_mean = jnp.mean(batch_arr, axis=0)
        batch_var = jnp.var(batch_arr, axis=0)
        return eqx.tree_at(
            lambda m: (m.batch_stats, m.use_batch_stats),
            self,
            ((batch_mean, batch_var), True),
        )

    def with_running_average(self, use_running_average: bool = True) -> BatchNorm:
        """Return a copy toggling the use of running statistics."""

        return eqx.tree_at(lambda m: m.use_running_average, self, use_running_average)


class AffineCoupling(AbstractBijection):
    """Affine coupling layer.

    Splits input into two halves: the first half is unchanged (condition),
    and the second half is transformed by an affine function parameterized
    by the first half.

    Args:
        key: JAX random key.
        shape: Shape of the input (n_dims,).
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    _coupling: AbstractBijection

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        if len(shape) != 1:
            raise ValueError("AffineCoupling only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape

        affine = Affine()
        self._coupling = Coupling(
            key=key,
            transformer=affine,
            untransformed_dim=n_dims // 2,
            dim=n_dims,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        return self._coupling.transform_and_log_det(jnp.asarray(x), condition)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        return self._coupling.inverse_and_log_det(jnp.asarray(y), condition)


class RQSplineCoupling(AbstractBijection):
    """Rational quadratic spline coupling layer.

    Like AffineCoupling, but uses a rational quadratic spline as the transformer.

    Args:
        key: JAX random key.
        shape: Shape of the input (n_dims,).
        n_bins: Number of spline bins. Defaults to 8.
        interval: Spline interval. Defaults to 5.0.
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    _coupling: AbstractBijection

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        n_bins: int = 8,
        interval: float = 5.0,
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        if len(shape) != 1:
            raise ValueError("RQSplineCoupling only supports 1D inputs.")
        n_dims = shape[0]
        self.shape = shape

        spline = RationalQuadraticSpline(knots=n_bins, interval=interval)
        self._coupling = Coupling(
            key=key,
            transformer=spline,
            untransformed_dim=n_dims // 2,
            dim=n_dims,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        return self._coupling.transform_and_log_det(jnp.asarray(x), condition)

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        return self._coupling.inverse_and_log_det(jnp.asarray(y), condition)


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


__all__ = [
    "ActNorm1D",
    "AffineCoupling",
    "BatchNorm",
    "DeepSigmoidCoupling",
    "RQSplineCoupling",
]
