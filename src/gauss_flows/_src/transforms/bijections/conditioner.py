"""Context-amortised affine modulation around any bijection (FiLM-style)."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


class Conditioner(AbstractBijection):
    """Wrap an unconditional bijection with a context-amortised affine.

    Computes

    $$
    y = \\exp(\\log\\sigma(c)) \\cdot f(x) + \\mu(c)
    $$

    where ``f`` is the inner bijection and
    ``(\\mu, \\log\\sigma) = \\text{split}(g(c))`` are produced by a single
    user-supplied (or default `equinox.nn.MLP`) network with
    ``2 * D`` outputs — same coupling-net pattern used by
    `AffineCoupling` and `ConditionalDiagGaussian`.

    Use this for transforms that genuinely cannot accept a condition
    (rotations, normalisations, fixed orthogonal layers, …). For coupling
    layers and other transforms that natively take a condition, pass
    ``cond_dim`` directly rather than wrapping in `Conditioner`.

    Args:
        key: PRNG key used to initialise the default `equinox.nn.MLP`.
            Ignored when a pre-built ``context_net`` is supplied.
        inner: Any bijection whose ``cond_shape`` is ``None``. Strict — wrap
            only genuinely unconditional inner transforms; raise otherwise.
        cond_shape: Context shape ``(cond_dim,)``. 1-D only.
        context_net: Optional pre-built callable
            ``(cond_dim,) -> (2 * D,)``. Default builds an
            `equinox.nn.MLP`.
        nn_width: Width of the default MLP.
        nn_depth: Depth of the default MLP.
        activation: Activation between hidden layers of the default MLP.

    Shape:
        - Input ``x`` / output ``y``: ``inner.shape`` (1-D)
        - Condition: ``(cond_dim,)``
        - ``log_det``: scalar ``()``

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import Conditioner, OrthogonalRotation
        >>> rot = OrthogonalRotation(shape=(4,))
        >>> bij = Conditioner(jr.key(1), inner=rot, cond_shape=(2,))
        >>> x = jr.normal(jr.key(2), (4,))
        >>> y, log_det = bij.transform_and_log_det(x, jnp.array([0.5, -0.5]))
        >>> x_rec, log_det_inv = bij.inverse_and_log_det(y, jnp.array([0.5, -0.5]))
        >>> assert jnp.allclose(x, x_rec, atol=1e-5)
        >>> assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    inner: AbstractBijection
    context_net: Callable[[Array], Array]

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        inner: AbstractBijection,
        cond_shape: tuple[int, ...],
        context_net: Callable[[Array], Array] | None = None,
        nn_width: int = 64,
        nn_depth: int = 2,
        activation: Callable[[Array], Array] = jnn.tanh,
    ):
        if inner.cond_shape is not None:
            raise ValueError(
                "Conditioner wraps strictly unconditional bijections "
                f"(inner.cond_shape must be None); got "
                f"inner.cond_shape={inner.cond_shape}. For transforms that "
                "natively accept a condition, pass cond_dim directly rather "
                "than wrapping in Conditioner."
            )
        if len(inner.shape) != 1:
            raise ValueError(
                "Conditioner only supports 1D inner event shapes; got "
                f"inner.shape={inner.shape}."
            )
        if len(cond_shape) != 1:
            raise ValueError(
                f"Conditioner only supports 1D cond_shape; got {cond_shape}."
            )
        if cond_shape[0] < 1:
            raise ValueError(
                f"cond_shape must have a positive dimension; got {cond_shape}."
            )

        self.shape = inner.shape
        self.cond_shape = cond_shape
        self.inner = inner

        d = inner.shape[0]
        cond_dim = cond_shape[0]
        if context_net is None:
            context_net = eqx.nn.MLP(
                in_size=cond_dim,
                out_size=2 * d,
                width_size=nn_width,
                depth=nn_depth,
                activation=activation,
                key=key,
            )
        else:
            test_out = jnp.asarray(context_net(jnp.zeros(cond_shape)))
            if test_out.shape != (2 * d,):
                raise ValueError(
                    "context_net output must have shape "
                    f"(2 * inner.shape[0],) = ({2 * d},); got {test_out.shape}."
                )

        self.context_net = context_net

    def _resolve_film(self, condition: Array) -> tuple[Array, Array]:
        out = self.context_net(condition)
        shift, log_scale = jnp.split(out, 2, axis=-1)
        return shift, log_scale

    def transform_and_log_det(
        self,
        x: ArrayLike,
        condition: ArrayLike,
    ) -> tuple[Array, Array]:
        x = jnp.asarray(x)
        y_inner, log_det_inner = self.inner.transform_and_log_det(x)
        shift, log_scale = self._resolve_film(jnp.asarray(condition, dtype=x.dtype))
        y = jnp.exp(log_scale) * y_inner + shift
        return y, log_det_inner + jnp.sum(log_scale)

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        condition: ArrayLike,
    ) -> tuple[Array, Array]:
        y = jnp.asarray(y)
        shift, log_scale = self._resolve_film(jnp.asarray(condition, dtype=y.dtype))
        y_inner = (y - shift) * jnp.exp(-log_scale)
        x, log_det_inner_inv = self.inner.inverse_and_log_det(y_inner)
        return x, log_det_inner_inv - jnp.sum(log_scale)


__all__ = ["Conditioner"]
