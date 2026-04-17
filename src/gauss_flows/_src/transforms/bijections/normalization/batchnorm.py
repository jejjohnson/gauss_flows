"""Invertible batch normalization for 1D single-event inputs."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike


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

            @eqx.filter_value_and_grad
            def loss_fn(layer, x):
                _, log_dets = jax.vmap(layer.transform_and_log_det)(x)
                return -jnp.mean(log_dets)

            bn = bn.with_batch_stats_from_data(batch)           # set source
            loss, grads = loss_fn(bn, batch)
            bn = eqx.apply_updates(bn, jax.tree.map(lambda g: -1e-3 * g, grads))
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
        (shape ``(N, *self.shape)``). Enables the ``use_batch_stats`` code path
        and atomically clears ``use_running_average`` so the two stats sources
        are mutually exclusive."""

        batch_arr = jnp.asarray(batch)
        batch_mean = jnp.mean(batch_arr, axis=0)
        batch_var = jnp.var(batch_arr, axis=0)
        return eqx.tree_at(
            lambda m: (m.batch_stats, m.use_batch_stats, m.use_running_average),
            self,
            ((batch_mean, batch_var), True, False),
        )

    def with_running_average(self, use_running_average: bool = True) -> BatchNorm:
        """Return a copy toggling the use of running statistics.

        When enabling running-average mode, ``use_batch_stats`` is atomically
        cleared so the two stats sources are mutually exclusive."""

        if use_running_average:
            return eqx.tree_at(
                lambda m: (m.use_running_average, m.use_batch_stats),
                self,
                (True, False),
            )
        return eqx.tree_at(lambda m: m.use_running_average, self, False)


__all__ = ["BatchNorm"]
