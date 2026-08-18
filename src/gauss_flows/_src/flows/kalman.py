r"""Normalizing Kalman Filter: a state-space base under a per-timestep warp.

The model of de Bézenac et al. (2020). A linear-Gaussian state-space model
supplies the base density over a latent series ``z_{1:T}``, and an invertible
warp ``G`` maps each timestep into observation space:

$$
\log p(y_{1:T}) = \log p_{\mathrm{LGSSM}}\!\left(G^{-1}(y_{1:T})\right)
  + \sum_t \log\left|\det \frac{\partial G^{-1}}{\partial y_t}\right| .
$$

Because ``G`` acts on one timestep at a time, the Markov structure of the base
is untouched — the Kalman recursion still applies in latent space.

Building this correctly takes **two nested** `flowjax.bijections.Vmap` calls,
and getting them wrong fails in two different ways.

**The trap that bites first: the channel axis.** Scalar bijections such as
`flowjax.bijections.RationalQuadraticSpline` have event shape ``()``, so they
cannot be chained with an ``M``-vector `flowjax.bijections.Affine`. Do it
anyway and flowjax raises

```text
ValueError: Expected shapes to match, but index 0 had shape (),
            and index 1 had shape (4,)
```

which names neither ``Vmap`` nor the axis at fault, and reads like a bug in
`flowjax.bijections.Chain`. Lift the scalar bijection over the channel axis
*before* chaining:

```python
warp = Chain([
    Vmap(RationalQuadraticSpline(knots=6, interval=4), in_axes=None, axis_size=M),
    Affine(loc=jnp.zeros(M), scale=jnp.ones(M)),
])
```

**The second axis: time.** `normalizing_kalman_filter` supplies this one — it
lifts the ``(M,)``-shaped warp across the ``T`` timesteps for you. That is the
whole reason this constructor exists rather than a line in the docs.

## Masking and the commutation requirement

With partially-observed series the base marginalises the unobserved channels,
and marginalisation only commutes with the warp when the warp's Jacobian is
**diagonal** in the channel axis. Writing ``P_m`` for the projection onto the
observed channels, the base's exact marginal likelihood is the right density
only if

```text
P_m ∘ G⁻¹  =  G⁻¹_m ∘ P_m       for some well-defined G⁻¹_m
```

A coupling bijection has a *triangular*, not diagonal, Jacobian: the transform
of channel ``j`` reads the untransformed conditioning half, so a missing entry
there corrupts the transform of channels that **were** observed. Measured on a
2-layer coupling flow with ``M = 6`` and ~40% missing, the error on the
observed channels alone had median ``0.49``, p90 ``1.45``, max ``4.35`` over
4000 random ``(y, mask)`` draws — against ``E|z| = 0.43``, i.e. the corruption
is the size of the signal. The same measurement on an elementwise warp gives
exactly ``0.0``.

Three combinations are coherent, and `normalizing_kalman_filter` treats them
differently:

1. **Elementwise warp + masked base** — masking stays exact. Cross-channel
   structure lives in the state-space model's ``H`` and ``R``, where the Kalman
   recursion handles it exactly. **This is the recommendation**, and it is a
   deliberate deviation from the source paper, which uses a coupling flow
   specifically to model cross-series dependence.
2. **Mask-conditioned warp + masked base** — permitted. The transform sees the
   mask through ``condition``, so it is at least a well-defined function of what
   was observed. No exactness guarantee, and the warp must learn up to ``2^M``
   masking patterns.
3. **Unconditional channel-mixing warp + masked base** — **rejected at
   construction.** This is the case measured above. The failure is silent,
   data-dependent, and looks like underfitting rather than a bug.

An unmasked base places no restriction on the warp; case 3 is refused only when
the base consumes a condition.

Note on direction: flowjax's ``transform`` maps base → data and ``inverse``
maps data → base, so the likelihood above uses ``inverse``.

References:
    de Bézenac, E., et al. (2020). Normalizing Kalman Filters for Multivariate
    Time Series Analysis. NeurIPS 2020.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection, Chain, Invert, Vmap
from flowjax.distributions import AbstractDistribution, Transformed


# Bijections whose Jacobian is diagonal in the event axis by construction.
# Anything absent from this list is decided by the numerical probe rather than
# assumed safe, so a new or third-party bijection is measured, not trusted.
_ELEMENTWISE_NAMES = frozenset(
    {
        "Affine",
        "Exp",
        "Identity",
        "LeakyTanh",
        "Loc",
        "Power",
        "RationalQuadraticSpline",
        "Scale",
        "Sigmoid",
        "SoftPlus",
        "Tanh",
    }
)

# Number of random points at which the numerical probe evaluates the Jacobian.
# A triangular Jacobian can look diagonal at an unlucky single point.
_N_PROBE_POINTS = 4
_PROBE_SEED = 0
_PROBE_RTOL = 1e-6


def _structurally_diagonal(bijection: AbstractBijection) -> bool | None:
    """Whether ``bijection`` is elementwise, if that is decidable structurally.

    Returns ``True`` / ``False`` when the answer follows from the bijection's
    type, and ``None`` when it does not — in which case the caller falls back
    to `_probe_mixes_channels`.

    Args:
        bijection: The bijection to classify.

    Returns:
        ``True`` if provably diagonal, ``False`` if provably not, ``None`` if
        undecidable from structure alone.
    """
    if isinstance(bijection, Chain):
        members = [_structurally_diagonal(b) for b in bijection.bijections]
        if any(m is False for m in members):
            return False  # one mixing member is enough
        return None if any(m is None for m in members) else True
    if isinstance(bijection, Invert):
        # Inverting transposes the Jacobian's sparsity pattern; diagonal stays
        # diagonal, triangular stays triangular.
        return _structurally_diagonal(bijection.bijection)
    if isinstance(bijection, Vmap):
        # Vmap gives a block-diagonal Jacobian whose blocks are the inner
        # bijection's — diagonal overall exactly when the block is.
        return _structurally_diagonal(bijection.bijection)
    if type(bijection).__name__ in _ELEMENTWISE_NAMES:
        return True
    return None


def _probe_mixes_channels(bijection: AbstractBijection) -> bool:
    """Test numerically whether the Jacobian has off-diagonal mass.

    Evaluates ``jax.jacfwd`` of ``bijection.transform`` at several random
    points and compares the largest off-diagonal entry to the largest diagonal
    one. Run once at construction, never in a training loop.

    Args:
        bijection: An unconditional bijection with event shape ``(M,)``.

    Returns:
        ``True`` if any probe point shows off-diagonal mass above the
        tolerance, or if the Jacobian could not be evaluated at all — an
        unverifiable warp is treated as mixing rather than assumed safe.
    """
    keys = jr.split(jr.key(_PROBE_SEED), _N_PROBE_POINTS)
    try:
        for key in keys:
            point = jr.normal(key, bijection.shape)  # (M,)
            jacobian = jax.jacfwd(bijection.transform)(point)  # (M, M)
            diagonal = jnp.diagonal(jacobian)
            off_diagonal = jacobian - jnp.diag(diagonal)
            scale = jnp.maximum(jnp.max(jnp.abs(diagonal)), 1.0)
            if bool(jnp.max(jnp.abs(off_diagonal)) > _PROBE_RTOL * scale):
                return True
    except Exception:  # any failure to evaluate means "unverified"
        return True
    return False


def _mixes_channels(bijection: AbstractBijection) -> bool:
    """Whether the bijection's Jacobian is not diagonal in the event axis.

    Args:
        bijection: An unconditional bijection with event shape ``(M,)``.

    Returns:
        ``True`` if the bijection mixes channels.
    """
    structural = _structurally_diagonal(bijection)
    if structural is not None:
        return not structural
    return _probe_mixes_channels(bijection)


_MASKED_COUPLING_MESSAGE = """\
A channel-mixing warp cannot be combined with a base that consumes an \
observation mask: the two do not commute, so the observed channels are \
silently corrupted by entries that were never measured (measured median error \
0.49 against a signal scale of 0.43, on a 2-layer coupling flow with ~40% \
missing). The failure looks like underfitting, not like a bug, which is why it \
is refused here.

Three combinations are coherent — pick one:

  1. Use an elementwise warp, e.g.
     ``Vmap(RationalQuadraticSpline(...), in_axes=None, axis_size=M)``, and put
     the cross-channel structure in the state-space model's H and R, where the
     Kalman recursion handles it exactly. Masking then stays exact.
  2. Condition the warp on the mask, so the transform is at least a
     well-defined function of what was observed. The warp must then learn up
     to 2**M masking patterns, and exactness is not recovered.
  3. Use an unmasked base, which places no restriction on the warp.\
"""


def normalizing_kalman_filter(
    base: AbstractDistribution,
    warp: AbstractBijection,
    *,
    n_steps: int | None = None,
) -> Transformed:
    r"""Compose a state-space base with a per-timestep observation warp.

    Builds the Normalizing Kalman Filter of de Bézenac et al. (2020) by lifting
    ``warp`` over the time axis, so the base's Markov structure is preserved and
    the Kalman recursion still applies in latent space. See the module docstring
    for the two-nested-`flowjax.bijections.Vmap` construction and the reason a
    masked base restricts which warps are allowed.

    Args:
        base: Any distribution with event shape ``(T, M)`` — typically
            ``NumpyroBase(dist=gaussx.LGSSM(...))``, or the ``dist_factory``
            form wrapping ``gaussx.LGSSMFactory`` for the masked case. A
            non-``None`` ``cond_shape`` is taken to mean the base consumes an
            observation mask.
        warp: Bijection with event shape ``(M,)``, mapping base values to
            observations. It is applied independently at each of the ``T``
            timesteps.
        n_steps: Time-axis size ``T``. Defaults to ``base.shape[0]``.

    Returns:
        A `flowjax.distributions.Transformed` over ``(T, M)`` observations.

    Raises:
        ValueError: If ``base`` is not ``(T, M)``-shaped, if ``warp`` is not
            ``(M,)``-shaped or does not match the base's channel count, if
            ``n_steps`` disagrees with ``base.shape[0]``, or if ``base``
            consumes a condition while ``warp`` mixes channels without
            consuming that condition itself. The last case silently corrupts
            the observed channels; the message names the three coherent
            alternatives.

    Shape:
        - Base event: ``(T, M)``
        - Warp event: ``(M,)`` (single timestep, single event)
        - Returned distribution event: ``(T, M)``
        - ``log_prob`` of a ``(T, M)`` observation: scalar ``()``
        - ``sample(key, (S,))``: ``(S, T, M)``

    Examples:
        >>> import gaussx
        >>> import jax.numpy as jnp
        >>> from flowjax.bijections import Affine, Chain, RationalQuadraticSpline, Vmap
        >>> from gauss_flows import NumpyroBase, normalizing_kalman_filter
        >>> M, T = 4, 8
        >>> # 1. Channel axis: lift the scalar spline to an M-vector, THEN chain.
        >>> warp = Chain([
        ...     Vmap(RationalQuadraticSpline(knots=6, interval=4),
        ...          in_axes=None, axis_size=M),
        ...     Affine(loc=jnp.zeros(M), scale=jnp.ones(M)),
        ... ])
        >>> base = NumpyroBase(dist=gaussx.LGSSM(
        ...     0.9 * jnp.eye(M), jnp.eye(M), 0.1 * jnp.eye(M), 0.1 * jnp.eye(M),
        ...     jnp.zeros(M), jnp.eye(M), n_steps=T,
        ... ))
        >>> nkf = normalizing_kalman_filter(base, warp)   # 2. Time axis, for you.
        >>> nkf.shape
        (8, 4)
    """
    if len(base.shape) != 2:
        raise ValueError(
            "base must have event shape (T, M) — a time axis and a channel "
            f"axis; got {base.shape}."
        )
    if len(warp.shape) != 1:
        raise ValueError(
            "warp must have event shape (M,), i.e. a single timestep. Scalar "
            "bijections need lifting over the channel axis first, e.g. "
            "Vmap(RationalQuadraticSpline(...), in_axes=None, axis_size=M); "
            f"got warp.shape={warp.shape}."
        )

    n_base_steps, n_channels = base.shape
    if warp.shape[0] != n_channels:
        raise ValueError(
            f"warp event shape {warp.shape} does not match the base's channel "
            f"count M={n_channels} (base event shape {base.shape})."
        )
    if n_steps is not None and n_steps != n_base_steps:
        raise ValueError(
            f"n_steps={n_steps} disagrees with the base's time axis T={n_base_steps}."
        )

    # Case 3: a masked base with an unconditional channel-mixing warp. The warp
    # is unconditional exactly when cond_shape is None; a conditioned warp is
    # case 2 and permitted, so it is never probed.
    if (
        base.cond_shape is not None
        and warp.cond_shape is None
        and _mixes_channels(warp)
    ):
        raise ValueError(_MASKED_COUPLING_MESSAGE)

    if warp.cond_shape is not None:
        # The warp's condition is one timestep's worth; the base's is the whole
        # series. Check that here, where the time axis can be named, rather
        # than letting flowjax report a bare cond_shape mismatch.
        if base.cond_shape is None:
            raise ValueError(
                f"warp consumes a condition of shape {warp.cond_shape} but base "
                "does not consume one at all. A conditional warp needs a base "
                "that takes the same per-timestep condition (the observation "
                "mask), e.g. NumpyroBase(dist_factory=gaussx.LGSSMFactory(...))."
            )
        if tuple(base.cond_shape) != (n_base_steps, *warp.cond_shape):
            raise ValueError(
                f"base.cond_shape={base.cond_shape} does not match the warp's "
                f"per-timestep condition {warp.cond_shape} lifted over the time "
                f"axis, which would be {(n_base_steps, *warp.cond_shape)}."
            )

    # Time axis: apply the same (M,)-shaped warp at each of the T steps. A
    # conditional warp (case 2) additionally needs its condition mapped over
    # that axis, so the lifted warp's cond_shape becomes (T, M) and matches the
    # per-timestep mask the base consumes. Without in_axes_condition the lifted
    # warp keeps cond_shape (M,) and `Transformed` rejects the pair.
    lifted = Vmap(
        warp,
        in_axes=None,
        axis_size=n_steps or n_base_steps,
        in_axes_condition=None if warp.cond_shape is None else 0,
    )
    return Transformed(base, lifted)


__all__ = ["normalizing_kalman_filter"]
