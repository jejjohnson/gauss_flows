"""FFJORD continuous normalizing flow bijection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src._divergence import exact_divergence, hutchinson_divergence
from gauss_flows._src.conditioning import time_control_cond_shape
from gauss_flows._src.nn import DiffeqMLP
from gauss_flows._src.transforms.bijections.continuous._ode import (
    solve_augmented_dynamics,
)


_AdjointName = Literal["recursive_checkpoint", "backsolve", "direct"]
_DivergenceMode = Literal["hutchinson", "exact"]
_ProbeDistribution = Literal["rademacher", "normal"]
_SolverName = Literal["dopri5", "tsit5", "heun"]
_VectorField = Callable[[ArrayLike, ArrayLike, ArrayLike | None], Array]


class FFJORD(AbstractBijection):
    """Continuous normalizing flow via a learned ODE.

    FFJORD integrates the augmented dynamics

    ``dx/dt = f(t, x, condition)``
    ``dlog|det J|/dt = tr(∂f/∂x)``

    over a time span ``[t0, t1]`` using ``diffrax``. The trace term is computed
    either exactly with ``jax.jacfwd`` (cost ``O(dim)`` per ODE step — use for
    small ``dim`` or validation) or stochastically with Hutchinson's identity
    via :mod:`matfree` (cost ``O(n_hutchinson_samples)`` per step).

    Hutchinson probes are **fixed per FFJORD instance**: the ``trace_key`` is
    stored once at construction and reused across every call to
    ``transform_and_log_det`` / ``inverse_and_log_det`` *and* across every
    internal ODE step. This is the "fixed noise" FFJORD variant (Grathwohl
    et al. 2019, Appendix C) — it makes ``log_det`` a deterministic function
    of parameters and stabilises gradient optimisation. To draw new probes,
    construct a new :class:`FFJORD` with a freshly split key.

    Args:
        key: PRNG key for the default vector field and Hutchinson probes.
        shape: Event shape ``(dim,)``.
        vector_field: Module with signature
            ``f(ode_time, x, condition) -> dx_dt``. When omitted, a default
            :class:`gauss_flows.DiffeqMLP` is created.
        control_dim: Size of the packed control vector ``c``.
        t_span: Integration interval ``(t0, t1)``.
        solver: ODE solver name.
        adjoint: Diffrax adjoint name.
        rtol: Relative tolerance for adaptive solvers.
        atol: Absolute tolerance for adaptive solvers.
        divergence_mode: ``"hutchinson"`` or ``"exact"``.
        n_hutchinson_samples: Number of Hutchinson probes.
        probe_distribution: Probe distribution for Hutchinson estimation.

    Shape:
        - Input ``x`` / ``y``: ``(dim,)``
        - Condition: ``(1 + control_dim,)`` packed as ``[t_query, c]``
        - Output ``log_det``: scalar ``()``

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import DiffeqMLP, FFJORD, pack_time_control
        >>> vf = DiffeqMLP(jr.key(1), in_dim=2, control_dim=1, hidden=(8, 8))
        >>> bij = FFJORD(jr.key(0), shape=(2,), vector_field=vf, control_dim=1)
        >>> x = jnp.array([0.1, -0.2])
        >>> cond = pack_time_control(0.5, jnp.array([1.0]))
        >>> y, log_det = bij.transform_and_log_det(x, cond)
        >>> x_rec, log_det_inv = bij.inverse_and_log_det(y, cond)
        >>> assert jnp.allclose(x, x_rec, atol=1e-4)
        >>> assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-4)
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    control_dim: int = eqx.field(static=True)
    vector_field: _VectorField
    trace_key: PRNGKeyArray
    t_span: tuple[float, float] = eqx.field(static=True)
    solver: _SolverName = eqx.field(static=True)
    adjoint: _AdjointName = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    divergence_mode: _DivergenceMode = eqx.field(static=True)
    n_hutchinson_samples: int = eqx.field(static=True)
    probe_distribution: _ProbeDistribution = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        *,
        vector_field: _VectorField | None = None,
        control_dim: int = 0,
        t_span: tuple[float, float] = (0.0, 1.0),
        solver: _SolverName = "tsit5",
        adjoint: _AdjointName = "recursive_checkpoint",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        divergence_mode: _DivergenceMode = "hutchinson",
        n_hutchinson_samples: int = 1,
        probe_distribution: _ProbeDistribution = "rademacher",
    ):
        if len(shape) != 1:
            raise ValueError("FFJORD only supports 1D event shapes.")
        if shape[0] < 1:
            raise ValueError(f"shape must have a positive dimension; got {shape}.")
        if control_dim < 0:
            raise ValueError(
                "control_dim must be non-negative "
                f"(got {control_dim}); use 0 for unconditional models."
            )
        if n_hutchinson_samples < 1:
            raise ValueError(
                f"n_hutchinson_samples must be positive; got {n_hutchinson_samples}."
            )
        if t_span[0] == t_span[1]:
            raise ValueError("t_span endpoints must differ.")
        if divergence_mode not in ("hutchinson", "exact"):
            raise ValueError(
                "divergence_mode must be 'hutchinson' or 'exact'; "
                f"got {divergence_mode!r}."
            )
        if probe_distribution not in ("rademacher", "normal"):
            raise ValueError(
                "probe_distribution must be 'rademacher' or 'normal'; "
                f"got {probe_distribution!r}."
            )

        key_vf, key_trace = jr.split(key)
        self.shape = shape
        self.cond_shape = time_control_cond_shape(control_dim)
        self.control_dim = control_dim
        self.vector_field = (
            DiffeqMLP(key_vf, in_dim=shape[0], control_dim=control_dim)
            if vector_field is None
            else vector_field
        )
        self.trace_key = key_trace
        self.t_span = t_span
        self.solver = solver
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol
        self.divergence_mode = divergence_mode
        self.n_hutchinson_samples = n_hutchinson_samples
        self.probe_distribution = probe_distribution

    def transform_and_log_det(
        self,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        return self._solve(
            jnp.asarray(x), condition, t0=self.t_span[0], t1=self.t_span[1]
        )

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        return self._solve(
            jnp.asarray(y), condition, t0=self.t_span[1], t1=self.t_span[0]
        )

    def _solve(
        self,
        x0: Array,
        condition: ArrayLike | None,
        *,
        t0: float,
        t1: float,
    ) -> tuple[Array, Array]:
        if condition is None:
            raise ValueError(
                "FFJORD requires a packed condition. Use pack_time_control(t, control)."
            )

        condition_array = jnp.asarray(condition, dtype=x0.dtype)

        def augmented_vector_field(
            ode_time: Array,
            state: tuple[Array, Array],
            packed_condition: Array | None,
        ) -> tuple[Array, Array]:
            x_state, _ = state
            dx_dt = self.vector_field(ode_time, x_state, packed_condition)
            div = self._divergence(ode_time, x_state, packed_condition)
            return dx_dt, div

        return solve_augmented_dynamics(
            augmented_vector_field,
            x0=x0,
            condition=condition_array,
            t0=t0,
            t1=t1,
            solver=self.solver,
            adjoint=self.adjoint,
            rtol=self.rtol,
            atol=self.atol,
        )

    def _divergence(
        self,
        ode_time: Array,
        x: Array,
        condition: Array | None,
    ) -> Array:
        if self.divergence_mode == "exact":
            return exact_divergence(self.vector_field, ode_time, x, condition)
        return hutchinson_divergence(
            self.vector_field,
            ode_time,
            x,
            condition,
            key=self.trace_key,
            n_samples=self.n_hutchinson_samples,
            distribution=self.probe_distribution,
        )


__all__ = ["FFJORD"]
