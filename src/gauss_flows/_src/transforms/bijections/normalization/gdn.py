"""Generalized Divisive Normalization (GDN).

GDN (Ballé, Laparra & Simoncelli 2016) is an exact-likelihood, dimension-
preserving normalization with a *coupled* (full) Jacobian — the joint counterpart
of the package's marginal Gaussianization. Each output divides by an energy
pooled across coordinates:

    y_i = x_i / sqrt( beta_i + sum_j gamma_ij * x_j^2 )

``GeneralizedDivisiveNormalization`` couples across the trailing (channel) axis of
image-shaped events ``(..., C)`` (a 1x1 GDN, so the dense determinant is O(C^3)),
and ``GeneralizedDivisiveNormalization1D`` couples the full ``(D,)`` vector
(O(D^3)) for use inside ``gaussianization_flow`` / ``iterative_rbig`` stacks.

The forward log-det is exact (``-sum_i log denom_i + slogdet(I - M)``). The inverse
has no closed form: it runs a damped fixed-point iteration with a ``custom_vjp``
that backpropagates through the implicit function theorem with a single linear
solve against the same ``(I - M)`` matrix built for the log-det.
"""

from __future__ import annotations

from functools import partial
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array, lax
from jax.nn import softplus
from jaxtyping import ArrayLike


def _inv_softplus(x: float) -> float:
    """Inverse of ``softplus``: raw value whose softplus is ``x`` (``x > 0``)."""
    return float(jnp.log(jnp.expm1(jnp.asarray(x, dtype=float))))


def _denom(x: Array, beta: Array, gamma: Array) -> Array:
    """Divisive denominator ``sqrt(beta_i + sum_j gamma_ij x_j^2)``.

    Broadcasts over any leading axes of ``x`` (shape ``(..., C)``); the coupling
    contracts only the trailing channel axis.
    """
    energy = jnp.einsum("ij,...j->...i", gamma, x * x)
    return jnp.sqrt(beta + energy)


def _logdet_vec(x_vec: Array, d_vec: Array, gamma: Array) -> tuple[Array, Array]:
    """Per-channel-vector log|det J| and its slogdet sign.

    ``J = diag(1/d) (I - M)`` with ``M_ik = gamma_ik x_i x_k / d_i^2``, so
    ``log|det J| = -sum_i log d_i + slogdet(I - M)``.
    """
    c = x_vec.shape[-1]
    m = gamma * jnp.outer(x_vec, x_vec) / (d_vec[:, None] ** 2)
    sign, logabsdet = jnp.linalg.slogdet(jnp.eye(c, dtype=x_vec.dtype) - m)
    return -jnp.sum(jnp.log(d_vec)) + logabsdet, sign


def _logdet_sum(x: Array, beta: Array, gamma: Array) -> tuple[Array, Array]:
    """Total log|det J| over every channel vector in ``x``; min slogdet sign.

    The 1x1 GDN is block-diagonal across the leading (spatial) positions, so the
    event log-det is the sum of the per-position channel log-dets.
    """
    c = x.shape[-1]
    d = _denom(x, beta, gamma)
    rows = jax.vmap(partial(_logdet_vec, gamma=gamma))(
        jnp.reshape(x, (-1, c)), jnp.reshape(d, (-1, c))
    )
    logdet, sign = rows
    return jnp.sum(logdet), jnp.min(sign)


def _forward(x: Array, beta: Array, gamma: Array) -> tuple[Array, Array, Array]:
    """Forward GDN: ``y = x / denom(x)`` with exact log-det and slogdet sign."""
    y = x / _denom(x, beta, gamma)
    logdet, sign = _logdet_sum(x, beta, gamma)
    return y, logdet, sign


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _invert_vec(
    y_vec: Array,
    beta: Array,
    gamma: Array,
    max_iters: int,
    tol: float,
    damping: float,
) -> Array:
    """Invert one channel vector by a damped fixed point ``x = y * denom(x)``.

    The undamped fixed point is Ballé's reference IGDN scheme; the damping factor
    ``lambda`` (``x <- (1-lambda) x + lambda * y * denom(x)``) widens the
    contraction regime under stronger coupling. The fixed point itself is
    independent of ``lambda`` — only the convergence rate changes — so the
    implicit-function backward pass is unaffected.
    """

    def cond(state):
        _, i, err = state
        return (i < max_iters) & (err > tol)

    def body(state):
        x, i, _ = state
        x_target = y_vec * _denom(x, beta, gamma)
        x_new = (1.0 - damping) * x + damping * x_target
        return x_new, i + 1, jnp.max(jnp.abs(x_new - x))

    x0 = y_vec * jnp.sqrt(beta)
    init = (x0, jnp.array(0), jnp.asarray(jnp.inf, dtype=x0.dtype))
    x_star, _, _ = lax.while_loop(cond, body, init)
    return x_star


def _invert_vec_fwd(y_vec, beta, gamma, max_iters, tol, damping):
    x_star = _invert_vec(y_vec, beta, gamma, max_iters, tol, damping)
    return x_star, (x_star, y_vec, beta, gamma)


def _invert_vec_bwd(max_iters, tol, damping, res, g):
    # Implicit function theorem at the fixed point
    # r(x*, y, theta) = x* - y * denom(x*) = 0. d_x r = I - M (the same matrix as
    # the forward log-det), so solve once and push the cotangent through r's
    # dependence on (y, beta, gamma) by autodiff.
    x_star, y_vec, beta, gamma = res
    c = x_star.shape[-1]
    d = _denom(x_star, beta, gamma)
    m = gamma * jnp.outer(x_star, x_star) / (d[:, None] ** 2)
    a = jnp.eye(c, dtype=x_star.dtype) - m
    s = jnp.linalg.solve(a.T, g)

    def residual(y_in, beta_in, gamma_in):
        return x_star - y_in * _denom(x_star, beta_in, gamma_in)

    _, vjp_fn = jax.vjp(residual, y_vec, beta, gamma)
    return vjp_fn(-s)


_invert_vec.defvjp(_invert_vec_fwd, _invert_vec_bwd)


def _inverse(
    y: Array, beta: Array, gamma: Array, max_iters: int, tol: float, damping: float
) -> tuple[Array, Array, Array]:
    """Inverse GDN over every channel vector in ``y``; log-det of the inverse."""
    c = y.shape[-1]
    x_flat = jax.vmap(
        partial(
            _invert_vec,
            beta=beta,
            gamma=gamma,
            max_iters=max_iters,
            tol=tol,
            damping=damping,
        )
    )(jnp.reshape(y, (-1, c)))
    x = jnp.reshape(x_flat, jnp.shape(y))
    logdet, sign = _logdet_sum(x, beta, gamma)
    return x, -logdet, sign


def _init_raw(c: int, beta_floor: float) -> tuple[Array, Array]:
    """Near-identity init: ``beta ~ 1`` and ``gamma ~ 0`` (small coupling).

    Zeros init (as in ActNorm) would set every ``gamma_ij = softplus(0) ~ 0.69``,
    i.e. strong all-to-all coupling that is generally non-invertible — the inverse
    fixed point would not contract. Starting near the identity keeps the layer
    invertible from step 0 while leaving every parameter trainable.
    """
    raw_beta = jnp.full((c,), _inv_softplus(max(1.0 - beta_floor, beta_floor)))
    raw_gamma = jnp.full((c, c), -5.0)  # softplus(-5) ~ 0.0067
    return raw_beta, raw_gamma


class GeneralizedDivisiveNormalization(AbstractBijection):
    """Generalized divisive normalization for image-shaped events ``(..., C)``.

    Couples across the trailing channel axis (a 1x1 GDN) and broadcasts over any
    leading spatial axes, so the dense log-det is O(C^3) per spatial position
    rather than O((H*W*C)^3). Parameters use the project's positive
    reparameterization: ``beta = softplus(raw_beta) + beta_floor`` (``> 0``) and
    ``gamma = softplus(raw_gamma)`` (``>= 0``, off-diagonal only — the diagonal is
    held at zero so each coordinate is not self-bounded), optionally symmetrized.
    The inverse runs a damped fixed point with an implicit-function-theorem
    ``custom_vjp``, so backprop through ``inverse`` / ``sample`` is exact.

    Like any divisive-normalization flow, GDN is invertible within a contraction
    regime (modest coupling); very strong ``gamma`` can make the fixed point
    diverge, so keep the coupling moderate (or regularize its spectral norm).

    Args:
        shape: Event shape ``(..., C)``; the last axis is the channel axis.
        symmetric_gamma: If True, symmetrize ``gamma`` (Ballé's convention).
        beta_floor: Floor added after ``softplus`` on ``beta``. Defaults to 1e-5
            (matches `ActNorm`).
        inverse_max_iters: Fixed-point iteration cap. Defaults to 100.
        inverse_tol: Fixed-point convergence tolerance. Defaults to 1e-6.
        inverse_damping: Damping ``lambda`` in ``(0, 1]`` for the inverse
            fixed point; smaller is more robust but slower. Defaults to 0.5.

    Shape:
        - transform_and_log_det: ``(..., C)`` -> ``(..., C)``, scalar log_det
        - inverse_and_log_det:   ``(..., C)`` -> ``(..., C)``, scalar log_det

    Examples:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import GeneralizedDivisiveNormalization
        >>> layer = GeneralizedDivisiveNormalization(shape=(4, 4, 3))
        >>> y, log_det = layer.transform_and_log_det(jnp.ones((4, 4, 3)))
        >>> y.shape, log_det.shape
        ((4, 4, 3), ())
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    raw_beta: Array
    raw_gamma: Array
    symmetric_gamma: bool = eqx.field(static=True)
    beta_floor: float = eqx.field(static=True)
    inverse_max_iters: int = eqx.field(static=True)
    inverse_tol: float = eqx.field(static=True)
    inverse_damping: float = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        symmetric_gamma: bool = False,
        beta_floor: float = 1e-5,
        inverse_max_iters: int = 100,
        inverse_tol: float = 1e-6,
        inverse_damping: float = 0.5,
    ):
        self.shape = shape
        self.symmetric_gamma = symmetric_gamma
        self.beta_floor = beta_floor
        self.inverse_max_iters = inverse_max_iters
        self.inverse_tol = inverse_tol
        self.inverse_damping = inverse_damping
        self.raw_beta, self.raw_gamma = _init_raw(shape[-1], beta_floor)

    def _params(self) -> tuple[Array, Array]:
        beta = softplus(self.raw_beta) + self.beta_floor
        gamma = softplus(self.raw_gamma)
        if self.symmetric_gamma:
            gamma = 0.5 * (gamma + gamma.T)
        # Hold the diagonal at zero: a positive gamma_ii would let a coordinate
        # divide by its own energy, capping |y_i| < 1 / sqrt(gamma_ii). The
        # forward map would then land in a bounded box rather than all of R^D,
        # so it would not be a bijection and out-of-box targets would have no
        # preimage for the inverse fixed point. Cross-coordinate (off-diagonal)
        # gain control is what GDN is for; the self-term is dropped.
        gamma = gamma * (1.0 - jnp.eye(gamma.shape[-1], dtype=gamma.dtype))
        return beta, gamma

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        beta, gamma = self._params()
        y, log_det, _ = _forward(jnp.asarray(x), beta, gamma)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        beta, gamma = self._params()
        x, log_det, _ = _inverse(
            jnp.asarray(y),
            beta,
            gamma,
            self.inverse_max_iters,
            self.inverse_tol,
            self.inverse_damping,
        )
        return x, log_det


class GeneralizedDivisiveNormalization1D(AbstractBijection):
    """1-D Generalized divisive normalization with full ``D x D`` coupling.

    The 1-D specialization of `GeneralizedDivisiveNormalization` for use inside
    ``gaussianization_flow`` / ``iterative_rbig`` stacks, where there are no
    spatial axes. Couples the whole ``(D,)`` event, so the exact log-det is
    O(D^3) — practical for moderate ``D``. Raises on non-1-D ``shape`` (matches
    the `ActNorm1D` convention).

    Args:
        shape: Event shape ``(D,)``. Must be 1-D.
        symmetric_gamma: If True, symmetrize ``gamma`` (Ballé's convention).
        beta_floor: Floor added after ``softplus`` on ``beta``. Defaults to 1e-5.
        inverse_max_iters: Fixed-point iteration cap. Defaults to 100.
        inverse_tol: Fixed-point convergence tolerance. Defaults to 1e-6.
        inverse_damping: Damping ``lambda`` in ``(0, 1]`` for the inverse
            fixed point. Defaults to 0.5.

    Raises:
        ValueError: If ``shape`` is not 1-D.

    Shape:
        - transform_and_log_det: ``(D,)`` -> ``(D,)``, scalar log_det
        - inverse_and_log_det:   ``(D,)`` -> ``(D,)``, scalar log_det

    Examples:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import GeneralizedDivisiveNormalization1D
        >>> layer = GeneralizedDivisiveNormalization1D(shape=(3,))
        >>> y, log_det = layer.transform_and_log_det(jnp.ones((3,)))
        >>> y.shape, log_det.shape
        ((3,), ())
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    raw_beta: Array
    raw_gamma: Array
    symmetric_gamma: bool = eqx.field(static=True)
    beta_floor: float = eqx.field(static=True)
    inverse_max_iters: int = eqx.field(static=True)
    inverse_tol: float = eqx.field(static=True)
    inverse_damping: float = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        symmetric_gamma: bool = False,
        beta_floor: float = 1e-5,
        inverse_max_iters: int = 100,
        inverse_tol: float = 1e-6,
        inverse_damping: float = 0.5,
    ):
        if len(shape) != 1:
            raise ValueError(
                "GeneralizedDivisiveNormalization1D only supports 1D inputs."
            )
        self.shape = shape
        self.symmetric_gamma = symmetric_gamma
        self.beta_floor = beta_floor
        self.inverse_max_iters = inverse_max_iters
        self.inverse_tol = inverse_tol
        self.inverse_damping = inverse_damping
        self.raw_beta, self.raw_gamma = _init_raw(shape[0], beta_floor)

    def _params(self) -> tuple[Array, Array]:
        beta = softplus(self.raw_beta) + self.beta_floor
        gamma = softplus(self.raw_gamma)
        if self.symmetric_gamma:
            gamma = 0.5 * (gamma + gamma.T)
        # Hold the diagonal at zero: a positive gamma_ii would let a coordinate
        # divide by its own energy, capping |y_i| < 1 / sqrt(gamma_ii). The
        # forward map would then land in a bounded box rather than all of R^D,
        # so it would not be a bijection and out-of-box targets would have no
        # preimage for the inverse fixed point. Cross-coordinate (off-diagonal)
        # gain control is what GDN is for; the self-term is dropped.
        gamma = gamma * (1.0 - jnp.eye(gamma.shape[-1], dtype=gamma.dtype))
        return beta, gamma

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        beta, gamma = self._params()
        y, log_det, _ = _forward(jnp.asarray(x), beta, gamma)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        beta, gamma = self._params()
        x, log_det, _ = _inverse(
            jnp.asarray(y),
            beta,
            gamma,
            self.inverse_max_iters,
            self.inverse_tol,
            self.inverse_damping,
        )
        return x, log_det


__all__ = [
    "GeneralizedDivisiveNormalization",
    "GeneralizedDivisiveNormalization1D",
]
