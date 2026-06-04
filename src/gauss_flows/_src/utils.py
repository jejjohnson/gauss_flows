"""Utility functions for gauss_flows."""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike


def stable_norm_logpdf(x: ArrayLike) -> Array:
    """Compute log PDF of standard normal in a numerically stable way.

    Args:
        x: Input values.

    Returns:
        Log PDF values.
    """
    x = jnp.asarray(x, dtype=float)
    return -0.5 * (jnp.log(2 * jnp.pi) + x**2)


def bisection_inverse(
    fn,
    y: ArrayLike,
    lower: float = -100.0,
    upper: float = 100.0,
    n_iter: int = 50,
) -> Array:
    """Invert a monotone function, differentiable via the implicit function theorem.

    Brackets the root with an expanding bisection (cheap, derivative-free, robust)
    and then appends a single Newton step from the *detached* root so reverse-mode
    AD is correct. A bare unrolled bisection returns the right value but a **zero
    gradient**: its output depends on the inputs only through piecewise-constant
    ``where(fn(mid) < y, ...)`` comparisons, so ``jax.grad`` through it is
    identically ``0`` w.r.t. both ``y`` and any parameters ``fn`` closes over.

    The Newton step ``x* - (fn(x*) - y) / fn'(x*)`` with ``x*`` detached leaves the
    value unchanged to bracket precision (``fn(x*) ≈ y``) while making the output
    depend smoothly on the inputs, recovering the exact implicit-function gradients

        dx/dy = 1 / fn'(x*)      and      dx/dθ = -∂_θ fn(x*) / fn'(x*).

    Args:
        fn: Monotone (increasing) function to invert, applied element-wise. May
            close over differentiable parameters.
        y: Target value(s).
        lower: Initial lower bracket; expanded automatically if it does not
            bracket the root. Defaults to -100.0.
        upper: Initial upper bracket; likewise expanded. Defaults to 100.0.
        n_iter: Number of bisection iterations. Defaults to 50.

    Returns:
        Approximate inverse value(s) ``x`` such that ``fn(x) ≈ y``.
    """
    y = jnp.asarray(y, dtype=float)

    lo = jnp.full_like(y, lower)
    hi = jnp.full_like(y, upper)

    def _expand(_, state):
        lo, hi = state
        f_lo = fn(lo)
        f_hi = fn(hi)
        span = hi - lo
        lo = jnp.where(f_lo > y, lo - span, lo)
        hi = jnp.where(f_hi < y, hi + span, hi)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, 16, _expand, (lo, hi))

    def _step(_, state):
        lo, hi = state
        mid = (lo + hi) / 2.0
        f_mid = fn(mid)
        lo = jnp.where(f_mid < y, mid, lo)
        hi = jnp.where(f_mid < y, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, n_iter, _step, (lo, hi))

    # One Newton step from the detached root attaches the exact implicit-function
    # gradient to an otherwise piecewise-constant (zero-gradient) bisection. The
    # value is unchanged to bracket precision: fn(x*) ≈ y so the correction is
    # ~the bracket width. df underflows to 0 only deep in a flat tail; guard it so
    # the value falls back to x* there instead of dividing by zero.
    x_star = jax.lax.stop_gradient((lo + hi) / 2.0)
    f_star, df_star = jax.jvp(fn, (x_star,), (jnp.ones_like(x_star),))
    safe = df_star != 0.0
    correction = jnp.where(safe, (f_star - y) / jnp.where(safe, df_star, 1.0), 0.0)
    return x_star - correction
