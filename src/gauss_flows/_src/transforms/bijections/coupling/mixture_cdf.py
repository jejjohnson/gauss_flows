"""Coupling layer with a mixture-of-Gaussians CDF transformer.

Given a mask that splits the input into an untransformed half ``x_a`` and a
transformed half ``x_b``, a conditioner MLP maps ``x_a`` to per-dim mixture
parameters ``(logits, means, log_scales)`` and the coupling applies the
mixture-CDF Gaussianization

    z_b = Φ⁻¹(F(x_b; π, μ, σ))
    log |det J| = Σᵢ [log f_i(x_{b,i}) − log φ(z_{b,i})]

The inverse is a per-dim monotone bisection on ``F_i(x_{b,i}) = Φ(z_{b,i})``.
We solve it with :func:`optimistix.root_find` backed by
:class:`optimistix.Bisection` with ``expand_if_necessary=True``, matching the
convention already established for :class:`DeepSigmoidCoupling`.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import jax.scipy.stats as jstats
import optimistix as optx
from flowjax.bijections import AbstractBijection, Coupling
from jax import Array
from jax.nn import log_softmax, softmax
from jaxtyping import ArrayLike, PRNGKeyArray


def _clamp_log_scale(raw: Array, bound: float) -> Array:
    """Tanh-bounded log-scale: ``bound · tanh(raw)`` → σ ∈ [exp(-bound), exp(bound)]."""
    return bound * jnp.tanh(raw)


def _invert_clamp_log_scale(clamped: ArrayLike, bound: float) -> Array:
    """Invert :func:`_clamp_log_scale`. Used by IG warm-start init."""
    y = jnp.clip(jnp.asarray(clamped) / bound, -1.0 + 1e-6, 1.0 - 1e-6)
    return jnp.arctanh(y)


class _MixtureGaussianCDFTransformer(AbstractBijection):
    """Scalar mixture-CDF Gaussianization transformer (shape ``()``).

    A single-dim bijection ``z = Φ⁻¹(F(x; π, μ, σ))`` with trainable
    ``(logits, means, log_scales)``. Vectorised across dims by wrapping it
    in :class:`flowjax.bijections.Coupling`.
    """

    n_components: int = eqx.field(static=True)
    log_scale_bound: float = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    logits: Array
    means: Array
    log_scales: Array

    def __init__(self, n_components: int, log_scale_bound: float = 5.0):
        self.n_components = int(n_components)
        self.log_scale_bound = float(log_scale_bound)
        self.shape = ()
        self.logits = jnp.zeros((n_components,))
        self.means = jnp.zeros((n_components,))
        self.log_scales = jnp.zeros((n_components,))

    def _params(self) -> tuple[Array, Array, Array]:
        weights = softmax(self.logits, axis=-1)
        means = self.means
        scales = jnp.exp(_clamp_log_scale(self.log_scales, self.log_scale_bound))
        return weights, means, scales

    def _mixture_cdf(self, x: Array) -> Array:
        weights, means, scales = self._params()
        return jnp.sum(weights * jstats.norm.cdf(x, loc=means, scale=scales), axis=-1)

    def _mixture_log_pdf(self, x: Array) -> Array:
        log_weights = log_softmax(self.logits, axis=-1)
        means = self.means
        scales = jnp.exp(_clamp_log_scale(self.log_scales, self.log_scale_bound))
        log_comps = jstats.norm.logpdf(x, loc=means, scale=scales)
        return jsp_special.logsumexp(log_weights + log_comps, axis=-1)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x = jnp.asarray(x)
        u = jnp.clip(self._mixture_cdf(x), 1e-6, 1.0 - 1e-6)
        z = jsp_special.ndtri(u)
        log_det = self._mixture_log_pdf(x) - jstats.norm.logpdf(z)
        return z, log_det

    def inverse_and_log_det(self, z: ArrayLike, condition=None):
        z = jnp.asarray(z)
        weights, means, _ = self._params()

        # Affine seed: E[x] under the mixture. optx.Bisection with
        # expand_if_necessary=True widens the bracket until it spans the root.
        approx_center = jnp.sum(weights * means, axis=-1)
        u_target = jsp_special.ndtr(z)

        def residual(x, args):
            _z_target = args
            del _z_target  # fn returns cdf(x) - u; same root as ndtri(cdf) = z.
            return self._mixture_cdf(x) - u_target

        solver = optx.Bisection(rtol=1e-6, atol=1e-6, expand_if_necessary=True)
        sol = optx.root_find(
            residual,
            solver,
            y0=approx_center,
            args=z,
            options={"lower": approx_center - 1.0, "upper": approx_center + 1.0},
            max_steps=100,
            throw=False,
        )
        x = sol.value
        log_det = jstats.norm.logpdf(z) - self._mixture_log_pdf(x)
        return x, log_det


class MixtureGaussianCDFCoupling(AbstractBijection):
    """Coupling layer whose transformed half goes through a mixture-CDF.

    Splits input into two halves along the leading (1D) axis. The first
    ``n_dims // 2`` dims are passed through and feed a conditioner MLP that
    emits per-dim mixture parameters for the remaining dims, which are
    then Gaussianized via ``z = Φ⁻¹(F(x; π, μ, σ))``.

    The per-dim log-scales are passed through a tanh clamp
    ``bound · tanh(raw)`` for training stability and for RBIG warm-start
    inversion — see :func:`gauss_flows._src.init.rbig.fit_rbig_coupling`.

    Args:
        key: JAX random key for the conditioner MLP.
        shape: Event shape ``(n_dims,)``.
        n_components: Mixture components ``K`` per transformed dim.
            Defaults to 8.
        nn_width: Hidden layer width of the conditioner MLP. Defaults to 64.
        nn_depth: Depth of the conditioner MLP. Defaults to 2.
        log_scale_bound: Tanh bound applied to per-dim log-scales. Defaults
            to 5.0 (σ ∈ [exp(-5), exp(5)] ≈ [6.7e-3, 148]).

    Shape:
        - Input ``x``: ``(n_dims,)``
        - Output ``y``: ``(n_dims,)``
        - ``log_det``: scalar ``()``
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    n_components: int = eqx.field(static=True)
    log_scale_bound: float = eqx.field(static=True)
    _coupling: AbstractBijection

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        n_components: int = 8,
        nn_width: int = 64,
        nn_depth: int = 2,
        log_scale_bound: float = 5.0,
    ):
        if len(shape) != 1:
            raise ValueError("MixtureGaussianCDFCoupling only supports 1D inputs.")
        n_dims = shape[0]
        if n_dims < 2:
            raise ValueError(
                "MixtureGaussianCDFCoupling needs n_dims >= 2 for a non-trivial split."
            )
        self.shape = shape
        self.n_components = int(n_components)
        self.log_scale_bound = float(log_scale_bound)

        transformer = _MixtureGaussianCDFTransformer(
            n_components=n_components, log_scale_bound=log_scale_bound
        )
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


__all__ = ["MixtureGaussianCDFCoupling"]
