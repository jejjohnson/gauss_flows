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

   Mind the direction, and mind where each bijection lands. ``warp`` maps the
   Gaussian base to observations, whereas a Gaussianiser like
   `gauss_flows.MixtureGaussianCDF` maps data to Gaussian — its *inverse* is
   the warp, so wrap it: ``Invert(MixtureGaussianCDF(...))``. Same for
   `MixtureLogisticCDF`. This is the convention `gauss_flows.fit_rbig` already
   follows when it returns ``Transformed(base, Invert(Chain(...)))``.

   `HistogramCDF` needs one more link. It maps data to a **uniform** variable,
   not to a Gaussian one, so ``Invert(HistogramCDF(...))`` expects inputs in
   ``[0, 1]`` while the base hands it values across all of ℝ — everything
   outside gets clamped to the fitted bin edges, giving boundary atoms and an
   unnormalised density. Compose it with the probit step that
   `MixtureGaussianCDF` already has built in::

       Invert(Chain([HistogramCDF(...), InverseGaussCDF(shape=(M,))]))

   `RQSplineMarginal` and a lifted
   `flowjax.bijections.RationalQuadraticSpline` carry no Gaussianising
   direction and can be used either way round. The change of variables is
   restricted to the observed channels automatically — see `_MaskedLogDet`,
   without which the log-det would pick up terms for entries the base
   marginalised away and the density would stop being a marginal likelihood.
2. **Mask-conditioned warp + masked base** — permitted. The transform sees the
   mask through ``condition``, so it is at least a well-defined function of what
   was observed. No exactness guarantee, and the warp must learn up to ``2^M``
   masking patterns. The log-det is **not** restricted to the observed channels
   here: a channel-mixing Jacobian does not decompose per channel, so there is
   no exact marginal to compute. ``log_prob`` is a training objective in this
   case, not a marginal likelihood.
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
from flowjax.bijections import (
    AbstractBijection,
    Affine,
    Chain,
    Exp,
    Identity,
    Invert,
    LeakyTanh,
    Loc,
    Power,
    RationalQuadraticSpline,
    Scale,
    Sigmoid,
    SoftPlus,
    Stack,
    Tanh,
    Vmap,
)
from flowjax.distributions import AbstractDistribution, Transformed


# Bijections whose Jacobian is diagonal in the event axis by construction, for
# EVERY parameter value — not merely at initialisation. That distinction is the
# point: a numerical probe describes one point in parameter space, and a warp
# that trains can leave it (see `_mixes_channels`).
# Matched by class identity, never by ``__name__``: a third-party bijection
# called "Affine" that implements a dense map would otherwise be waved through
# the mixing guard and into `_MaskedLogDet`, whose ``J @ 1 == diag(J)`` identity
# holds only for a genuinely diagonal Jacobian. Exact identity rather than
# ``isinstance`` for the same reason — a subclass is free to override
# ``transform`` into something dense.
_ELEMENTWISE_TYPES = frozenset(
    {
        Affine,
        Exp,
        Identity,
        LeakyTanh,
        Loc,
        Power,
        RationalQuadraticSpline,
        Scale,
        Sigmoid,
        SoftPlus,
        Tanh,
    }
)

# Every bijection in this subpackage "acts independently on every dimension of
# its input" (its own module docstring), which is exactly the property needed
# here — and it is a contract of the subpackage rather than of one parameter
# value, so it holds under training. Classifying by module keeps
# `MixtureGaussianCDF`, `MixtureLogisticCDF`, `RQSplineMarginal`,
# `HistogramCDF` and friends usable as warps, which a name allowlist would
# silently exclude. Coupling variants live under `.coupling` and are unaffected.
_ELEMENTWISE_MODULE = "gauss_flows._src.transforms.bijections.elementwise"

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
    # Exact type, not isinstance, for the containers too: a Chain subclass is
    # free to override transform/inverse with a dense map, and inspecting only
    # its stored children would then wave that override straight past the
    # guard. Same reasoning as the leaf allowlist below.
    kind = type(bijection)
    if kind is Chain:
        members = [_structurally_diagonal(b) for b in bijection.bijections]
        if any(m is False for m in members):
            return False  # one mixing member is enough
        return None if any(m is None for m in members) else True
    if kind is Invert:
        # Inverting transposes the Jacobian's sparsity pattern; diagonal stays
        # diagonal, triangular stays triangular.
        return _structurally_diagonal(bijection.bijection)
    if kind is Vmap:
        # Vmap gives a block-diagonal Jacobian whose blocks are the inner
        # bijection's — diagonal overall exactly when the block is. A scalar
        # inner makes those blocks 1x1, which is diagonal no matter what the
        # inner does, so lifting an unrecognised scalar bijection over the
        # channel axis is provably elementwise: no allowlist entry needed, and
        # no assume_elementwise_warp.
        if bijection.bijection.shape == ():
            return True
        return _structurally_diagonal(bijection.bijection)
    if kind is Stack:
        # Stack puts each block on its own slice of the stacked axis. Scalar
        # blocks therefore occupy one channel each and the Jacobian is
        # strictly diagonal, whatever the blocks are — which is what makes
        # heterogeneous marginals like Stack([Exp(), Sigmoid()]) safe over a
        # masked base. Non-scalar blocks give a 2-D event, outside what this
        # module's (M,) warps cover, so leave those undecided.
        blocks = bijection.bijections
        if all(b.shape == () for b in blocks):
            return True
        return None
    if kind.__module__.startswith(_ELEMENTWISE_MODULE):
        return True
    if kind in _ELEMENTWISE_TYPES:
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

    Classification is **structural**, never by measurement. Sampling the
    Jacobian cannot establish diagonality, in two independent ways:

    - *across parameters* — `gauss_flows.OrthogonalRotation` initialises its
      Cayley parameters to zero, so it literally is the identity when built and
      probes as diagonal; any nonzero learned value makes it a dense rotation;
    - *across inputs* — a parameter-free bijection need not have a constant
      Jacobian either. A fixed triangular shear that only engages past a
      threshold looks diagonal at every standard-normal probe point and mixes
      channels in the tail, where no finite sample of probes will find it.

    So an unrecognised bijection is reported as mixing regardless of what a
    probe would say. `normalizing_kalman_filter` offers
    ``assume_elementwise_warp`` for callers who know better than the type
    system does.

    Args:
        bijection: An unconditional bijection with event shape ``(M,)``.

    Returns:
        ``True`` if the bijection mixes channels, or if being elementwise
        cannot be established from its structure.
    """
    structural = _structurally_diagonal(bijection)
    return True if structural is None else not structural


class _MaskedLogDet(AbstractBijection):
    r"""Wrap an elementwise warp so its log-det only counts observed channels.

    Change of variables applies to the coordinates actually being modelled. A
    masked base marginalises the unobserved channels, so the density of the
    observed ones is

    $$
    \log p(y_{\mathrm{obs}}) = \log p_{\mathrm{base}}(z_{\mathrm{obs}})
      + \sum_{i \in \mathrm{obs}} \log\left|\frac{\partial z_i}{\partial y_i}\right| ,
    $$

    with the sum over the **observed** channels only. A plain
    `flowjax.distributions.Transformed` sums it over all ``M`` of them, adding
    Jacobian terms for entries that were never measured — and evaluating them
    at whatever placeholder the caller left in those slots. The result is not a
    marginal likelihood, and the error is a data-dependent offset that no test
    of the unmasked path can catch.

    This wrapper is applied automatically by `normalizing_kalman_filter` when
    the base consumes a mask and the warp is elementwise. It relies on that
    elementwise property: for a diagonal Jacobian $J$, a single
    forward-mode pass gives $J \mathbf{1} = \mathrm{diag}(J)$, so the
    per-channel log-dets come out in one JVP rather than ``M`` of them.

    The wrapped bijection's own log-determinant is preserved — only the
    unobserved channels' contributions are subtracted from it — so a warp that
    reports a log-det differing from the autodiff derivative of its map keeps
    its declared value, and an all-observed mask reproduces the ordinary
    `flowjax.distributions.Transformed` density exactly. See
    `_masked_log_det`.

    Attributes:
        bijection: The elementwise warp, event shape ``(M,)``.
        shape: ``(M,)``, taken from ``bijection``.
        cond_shape: ``(M,)`` — the per-timestep observation mask.
    """

    bijection: AbstractBijection
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]

    def __init__(self, bijection: AbstractBijection):
        if bijection.cond_shape is not None:
            raise ValueError(
                "_MaskedLogDet wraps an unconditional elementwise warp; got "
                f"cond_shape={bijection.cond_shape}."
            )
        self.bijection = bijection
        self.shape = bijection.shape
        self.cond_shape = bijection.shape

    def _safe(self, x, mask, reference):
        """Replace unobserved entries with a value inside the warp's support.

        Unobserved slots hold whatever the caller parked there, and the two
        commonest choices — ``NaN`` and an out-of-range sentinel like ``-1`` or
        ``-999`` — are exactly the ones that break a bounded warp. The
        bijection reduces its per-channel log-dets to a scalar internally, so
        one bad entry turns the whole timestep's log-determinant into ``NaN``:
        measured on ``Vmap(Exp())`` and ``Vmap(Sigmoid())``, whose inverses
        take a logarithm of the placeholder. No later masking recovers the
        observed channels from that — the information is gone before the mask
        is applied.

        Substituting first is what makes the masked density well defined at
        all. The reference is derived from the warp itself, so it is in-support
        per channel by construction, and it depends on the mask rather than on
        the discarded values, so the result stays independent of whatever the
        placeholder happened to be.
        """
        return jnp.where(mask, x, reference)  # (M,)

    def _masked_log_det(self, fn, declared, x, mask):
        """Drop the unobserved channels from the wrapped log-determinant.

        Subtracts the missing channels' contributions from ``declared`` rather
        than rebuilding the total from the per-channel derivatives. The two
        differ whenever a bijection reports a log-det that is not exactly the
        autodiff derivative of the map it applies — `HistogramCDF` with
        ``method="monotonic"`` does this deliberately, its inverse reporting
        ``-log(fwd'(x_rec))`` while differentiating ``inverse`` goes through a
        separately fitted interpolator. Rebuilding would then silently change
        the model even with nothing missing, and the discrepancy (~4e-3 per
        step, measured) would accumulate over the time axis.

        Starting from ``declared`` makes the all-observed case exact by
        construction, and leaves the ordinary case — where the declared total
        already equals the sum of per-channel derivatives — untouched. Only the
        *dropped* terms come from the JVP, so a bijection with a bespoke
        log-det keeps its own convention for everything it still counts.
        """
        # J is diagonal, so J @ 1 == diag(J): one forward-mode pass, not M.
        _, diagonal = jax.jvp(fn, (x,), (jnp.ones_like(x),))  # (M,)
        per_channel = jnp.log(jnp.abs(diagonal))  # (M,)
        return declared - jnp.sum(jnp.where(mask, 0.0, per_channel))

    def transform_and_log_det(self, x, condition=None):
        # Base space: the state-space base is Gaussian, so zero is in support.
        safe = self._safe(x, condition, jnp.zeros(self.shape))  # (M,)
        y, declared = self.bijection.transform_and_log_det(safe)  # (M,), ()
        return y, self._masked_log_det(
            self.bijection.transform, declared, safe, condition
        )

    def inverse_and_log_det(self, y, condition=None):
        # Data space: the warp's image is exactly the support of the
        # observations, so pushing a base-space zero forward lands inside it,
        # per channel, whatever each channel's range happens to be.
        reference = self.bijection.transform(jnp.zeros(self.shape))  # (M,)
        safe = self._safe(y, condition, reference)  # (M,)
        x, declared = self.bijection.inverse_and_log_det(safe)  # (M,), ()
        return x, self._masked_log_det(
            self.bijection.inverse, declared, safe, condition
        )


_MASKED_COUPLING_MESSAGE = """\
A channel-mixing warp cannot be combined with a base that consumes an \
observation mask: the two do not commute, so the observed channels are \
silently corrupted by entries that were never measured (measured median error \
0.49 against a signal scale of 0.43, on a 2-layer coupling flow with ~40% \
missing). The failure looks like underfitting, not like a bug, which is why it \
is refused here.

A warp counts as elementwise only when that follows from its type, because
measuring the Jacobian cannot establish it — not across parameter values
(OrthogonalRotation starts at exactly the identity and trains into a dense
rotation) and not across inputs (a fixed shear can engage only in the tail,
where no finite set of probe points will look). Everything in flowjax's
elementwise family and in gauss_flows' own
`transforms.bijections.elementwise` subpackage is recognised, which covers
MixtureGaussianCDF, MixtureLogisticCDF, RQSplineMarginal and HistogramCDF. For
anything else, pass assume_elementwise_warp=True to assert the property
yourself.

Three combinations are coherent — pick one:

  1. Use an elementwise warp and put the cross-channel structure in the
     state-space model's H and R, where the Kalman recursion handles it
     exactly. Masking then stays exact. Mind the direction — ``warp`` maps the
     Gaussian base to observations, so a Gaussianising CDF, which maps data to
     Gaussian, has to be inverted:

         Invert(MixtureGaussianCDF(n_components=8, shape=(M,)))
         RQSplineMarginal(n_bins=8, shape=(M,))       # direction-neutral
         Vmap(RationalQuadraticSpline(...), in_axes=None, axis_size=M)
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
    assume_elementwise_warp: bool = False,
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
        assume_elementwise_warp: Assert that ``warp`` is elementwise when its
            type is not recognised as such. Only consulted for a masked base.
            gauss_flows cannot verify the property — see `_mixes_channels` for
            why sampling the Jacobian cannot establish it — so this is a claim
            you are making, and getting it wrong silently corrupts the observed
            channels and invalidates the marginal likelihood. A cheap numerical
            probe still runs and refuses an assertion it can immediately
            disprove, but passing it is not evidence of correctness.

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
    # case 2 and permitted, so it is never classified.
    masked = base.cond_shape is not None
    elementwise = warp.cond_shape is None and not _mixes_channels(warp)
    if masked and warp.cond_shape is None and not elementwise:
        if not assume_elementwise_warp:
            raise ValueError(_MASKED_COUPLING_MESSAGE)
        # Trust the caller, but reject an assertion the probe already refutes.
        if _probe_mixes_channels(warp):
            raise ValueError(
                "assume_elementwise_warp=True, but a numerical probe found "
                "off-diagonal Jacobian mass in this warp, so the assertion is "
                "false as written. (The probe is only a sanity check — it can "
                "miss channel mixing that appears at other inputs or other "
                "parameter values, so passing it would not have proved the "
                "warp elementwise.)"
            )
        elementwise = True

    if warp.cond_shape is not None:
        # Case 2 only. The warp's condition is one timestep's worth; the base's
        # is the whole series. Check that here, where the time axis can be
        # named, rather than letting flowjax report a bare cond_shape mismatch.
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

    # Case 1 with a masked base: the change of variables must count only the
    # observed channels, or `Transformed` adds Jacobian terms for entries the
    # base marginalised away and the result stops being a marginal likelihood.
    # The wrapper consumes the same per-timestep mask the base does.
    per_step = _MaskedLogDet(warp) if masked and elementwise else warp

    # Time axis: apply the same (M,)-shaped warp at each of the T steps. A
    # conditional warp additionally needs its condition mapped over that axis,
    # so the lifted warp's cond_shape becomes (T, M) and matches the
    # per-timestep mask the base consumes. Without in_axes_condition the lifted
    # warp keeps cond_shape (M,) and `Transformed` rejects the pair.
    lifted = Vmap(
        per_step,
        in_axes=None,
        axis_size=n_steps or n_base_steps,
        in_axes_condition=None if per_step.cond_shape is None else 0,
    )
    return Transformed(base, lifted)


__all__ = ["normalizing_kalman_filter"]
