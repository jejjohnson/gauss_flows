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
    """Compute the inverse of a monotone function using bisection.

    Args:
        fn: Monotone increasing function to invert.
        y: Target values.
        lower: Lower bound for bisection. Defaults to -100.0.
        upper: Upper bound for bisection. Defaults to 100.0.
        n_iter: Number of bisection iterations. Defaults to 50.

    Returns:
        Approximate inverse values x such that fn(x) ≈ y.
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
    return (lo + hi) / 2.0
