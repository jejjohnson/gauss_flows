"""ODE helpers for continuous-time bijections."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import diffrax
import jax
import jax.numpy as jnp
from jax import Array


_AdjointName = Literal["recursive_checkpoint", "backsolve", "direct"]
_SolverName = Literal["dopri5", "tsit5", "heun"]


def solve_augmented_dynamics(
    augmented_vector_field: Callable[
        [Array, tuple[Array, Array], Array | None], tuple[Array, Array]
    ],
    *,
    x0: Array,
    condition: Array | None,
    t0: float,
    t1: float,
    solver: _SolverName,
    adjoint: _AdjointName,
    rtol: float,
    atol: float,
) -> tuple[Array, Array]:
    """Solve the augmented state ``(x, log_det)`` over ``[t0, t1]``."""
    if t0 == t1:
        return x0, jnp.zeros((), dtype=x0.dtype)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(augmented_vector_field),
        solver=_resolve_solver(solver),
        t0=t0,
        t1=t1,
        dt0=_initial_step_size(t0, t1),
        y0=(x0, jnp.zeros((), dtype=x0.dtype)),
        args=condition,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=_resolve_stepsize_controller(solver, rtol=rtol, atol=atol),
        adjoint=_resolve_adjoint(adjoint),
    )
    x_final, log_det_final = jax.tree_util.tree_map(lambda leaf: leaf[-1], solution.ys)
    return x_final, log_det_final


def _initial_step_size(t0: float, t1: float) -> float:
    span = t1 - t0
    return span / 64.0


def _resolve_solver(solver: _SolverName):
    if solver == "tsit5":
        return diffrax.Tsit5()
    if solver == "dopri5":
        return diffrax.Dopri5()
    if solver == "heun":
        return diffrax.Heun()
    raise ValueError("solver must be 'dopri5', 'tsit5', or 'heun'.")


def _resolve_adjoint(adjoint: _AdjointName):
    if adjoint == "recursive_checkpoint":
        return diffrax.RecursiveCheckpointAdjoint()
    if adjoint == "backsolve":
        return diffrax.BacksolveAdjoint()
    if adjoint == "direct":
        return diffrax.DirectAdjoint()
    raise ValueError(
        "adjoint must be 'recursive_checkpoint', 'backsolve', or 'direct'."
    )


def _resolve_stepsize_controller(
    solver: _SolverName,
    *,
    rtol: float,
    atol: float,
):
    if solver == "heun":
        return diffrax.ConstantStepSize()
    return diffrax.PIDController(rtol=rtol, atol=atol)


__all__ = ["solve_augmented_dynamics"]
