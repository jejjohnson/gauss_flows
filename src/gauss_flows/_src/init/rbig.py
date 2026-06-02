"""Iterative-Gaussianization (RBIG) warm-start builders.

These functions return a fully-configured flow by greedily fitting each
block to the data (non-gradient), then wrapping the stack into a
:class:`flowjax.distributions.Transformed`. The result is already a
reasonable density estimator before any gradient training, and can be
refined further by the usual NLL minimisation.

Two variants:

- :func:`fit_rbig` — the canonical diagonal flow: alternating PCA rotation
  and per-dim mixture-of-Gaussians CDF Gaussianization. Mirrors the
  Laparra & Malo (2011) RBIG algorithm.
- :func:`fit_rbig_coupling` — warm-start for a coupling-based flow:
  each block is rotation + :class:`MixtureGaussianCDFCoupling` where the
  conditioner is initialised so the layer acts like a diagonal mixture
  Gaussianization on the transformed half. Training then lets the
  conditioner learn to modulate on the untransformed half.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from flowjax.bijections import Chain, Flip, Invert
from flowjax.distributions import Normal, Transformed
from jaxtyping import ArrayLike, PRNGKeyArray
from paramax.utils import inv_softplus
from sklearn.mixture import GaussianMixture

from gauss_flows._src.transforms.bijections.coupling.mixture_cdf import (
    MixtureGaussianCDFCoupling,
    _invert_clamp_log_scale,
)
from gauss_flows._src.transforms.bijections.elementwise.mixture_cdf import (
    MixtureGaussianCDF,
)
from gauss_flows._src.transforms.bijections.linear.rotation import FixedRotation


def _dim_seed(block_idx: int, dim_idx: int, random_state: int) -> int:
    """Per-(block, dim) EM seed — matches the Keras IG init convention."""
    return int(random_state) + int(block_idx) * 1000 + int(dim_idx)


def _fit_gmm_per_dim(
    col: np.ndarray, n_components: int, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 1-D diagonal GMM. Returns ``(weights, means, scales)``."""
    col = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=int(n_components),
        covariance_type="diag",
        random_state=int(random_state),
        reg_covar=1e-6,
    ).fit(col)
    weights = gmm.weights_.astype(np.float64)
    means = gmm.means_.reshape(-1).astype(np.float64)
    variances = gmm.covariances_.reshape(-1).astype(np.float64)
    scales = np.sqrt(np.maximum(variances, 1e-12))
    return weights, means, scales


def _fit_marginal(
    y: np.ndarray,
    n_components: int,
    block_idx: int,
    random_state: int,
) -> MixtureGaussianCDF:
    """Build a :class:`MixtureGaussianCDF` with per-dim GMM-fitted params.

    Translates sklearn's ``(weights, means, scales)`` into the
    :class:`MixtureGaussianCDF` convention, where the stored
    ``log_weights``, ``means``, ``log_scales`` are used as
    ``softmax(log_weights)`` and ``softplus(log_scales) + 5e-3`` at
    forward time.
    """
    n, d = y.shape
    del n
    log_weights = np.zeros((d, n_components), dtype=np.float32)
    means = np.zeros((d, n_components), dtype=np.float32)
    raw_log_scales = np.zeros((d, n_components), dtype=np.float32)
    for i in range(d):
        w, mu, sigma = _fit_gmm_per_dim(
            y[:, i], n_components, _dim_seed(block_idx, i, random_state)
        )
        log_weights[i] = np.log(np.maximum(w, 1e-20))
        means[i] = mu
        # softplus(raw) + 5e-3 ≈ sigma → raw = inv_softplus(sigma - 5e-3).
        # Floor at a tiny positive epsilon (not 5e-3) so components with
        # fitted sigma in (5e-3, 1e-2) don't get silently widened to
        # ≥1e-2 — we want sigma_target → 0⁺ to produce the architectural
        # minimum scale of 5e-3, not a sigma_target of 5e-3 (which gives
        # an actual scale of ~1e-2).
        sigma_target = np.maximum(sigma - 5e-3, 1e-6)
        raw_log_scales[i] = np.asarray(inv_softplus(sigma_target))

    marg = MixtureGaussianCDF(n_components=n_components, shape=(d,))
    marg = eqx.tree_at(lambda m: m.log_weights, marg, jnp.asarray(log_weights))
    marg = eqx.tree_at(lambda m: m.means, marg, jnp.asarray(means))
    marg = eqx.tree_at(lambda m: m.log_scales, marg, jnp.asarray(raw_log_scales))
    return marg


def _push_forward(bij, y: np.ndarray) -> np.ndarray:
    """Apply ``bij.transform_and_log_det`` to each row of ``y`` via vmap."""
    y_j = jnp.asarray(y)
    transformed = jax.vmap(lambda yi: bij.transform_and_log_det(yi)[0])(y_j)
    return np.asarray(transformed)


def fit_rbig(
    x: ArrayLike,
    *,
    n_layers: int = 10,
    n_components: int = 8,
    random_state: int = 0,
) -> Transformed:
    """Fit an iterative-Gaussianization (RBIG) flow to data in closed form.

    Greedily walks ``n_layers`` blocks of ``[FixedRotation (PCA),
    MixtureGaussianCDF (GMM per-dim)]``, fitting each block directly from
    the current data state — a per-dimension sklearn ``GaussianMixture``
    EM fit, no gradient training — and propagating the data forward
    before the next block. The result is a warm start: a
    :class:`flowjax.distributions.Transformed` whose ``log_prob`` is
    already a reasonable density estimator and can be refined further by
    the usual NLL minimisation (see
    :func:`~gauss_flows.fit_gaussianization_flow`).

    Args:
        x: Training data of shape ``(n, d)``.
        n_layers: Number of ``(rotation, marginal)`` blocks. Defaults to 10.
        n_components: Mixture components ``K`` per marginal layer. Defaults
            to 8.
        random_state: Base seed for per-dim GMM EM fits. Each fit uses
            ``random_state + block_idx * 1000 + dim_idx``. Defaults to 0.

    Returns:
        A flowjax ``Transformed`` distribution over ℝ^d whose bijection
        Gaussianises ``x``, with ``log_prob`` and ``sample``.

    Raises:
        ValueError: If ``x`` is not 2-D.

    Example:
        >>> import jax.random as jr
        >>> from gauss_flows import fit_rbig
        >>> x = jr.normal(jr.key(0), (500, 2))
        >>> flow = fit_rbig(x, n_layers=4, n_components=4)
        >>> flow.sample(jr.key(1), (8,)).shape
        (8, 2)
    """
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"x must be 2-D (n, d); got shape {x_np.shape}")
    n_dims = x_np.shape[-1]

    bijections = []
    y = x_np
    for block_idx in range(int(n_layers)):
        rot = FixedRotation.from_data(y)
        bijections.append(rot)
        y = _push_forward(rot, y)

        marg = _fit_marginal(y, n_components, block_idx, random_state)
        bijections.append(marg)
        y = _push_forward(marg, y)

    base_dist = Normal(jnp.zeros(n_dims))
    bijection = Invert(Chain(bijections))
    return Transformed(base_dist, bijection)


def _pack_coupling_bias(
    log_weights: np.ndarray,
    means: np.ndarray,
    raw_log_scales: np.ndarray,
) -> np.ndarray:
    """Flatten per-b-dim mixture params into the conditioner's bias vector.

    :class:`flowjax.bijections.Coupling` reshapes the conditioner output
    as ``(d_b, params_per_dim)`` and feeds each row through
    ``transformer_constructor``. The ravelled transformer has three
    inexact-array fields in declaration order: ``logits``, ``means``,
    ``log_scales``. We stack in that order.

    Args:
        log_weights: ``(d_b, K)`` — written straight into the transformer's
            ``logits`` field (flowjax does not apply a softmax pre-clamp,
            the transformer does that internally).
        means: ``(d_b, K)``.
        raw_log_scales: ``(d_b, K)`` — pre-clamp log-scales that, when run
            through ``bound * tanh(·)``, give the fitted log-scales.

    Returns:
        A flat ``float32`` array of shape ``(d_b * 3 * K,)``.
    """
    stacked = np.concatenate(
        [
            log_weights.astype(np.float32),
            means.astype(np.float32),
            raw_log_scales.astype(np.float32),
        ],
        axis=-1,
    )  # (d_b, 3 * K)
    return stacked.reshape(-1)


def _push_forward_b_half_diag(
    y: np.ndarray,
    b_idx: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Advance state via the ``MixtureGaussianCDF`` forward on ``y[:, b_idx]``.

    Used by :func:`fit_rbig_coupling` to propagate state between blocks
    along the *same* numerical path as :func:`fit_rbig`, so the two warm
    starts see identical ``y`` states at every block and their GMM fits
    agree exactly. Without this, each coupling's slightly different
    numerical forward (different log-pdf formula, different scale
    parameterisation) produces accumulated drift in ``y`` that breaks
    the zero-kernel equivalence at higher block counts.

    Args:
        y: Current data state ``(n, d)``.
        b_idx: Indices of dims the coupling transforms (``d_b`` of them).
        weights: Mixture weights ``(d_b, K)`` (already softmaxed).
        means: Mixture means ``(d_b, K)``.
        sigmas: Mixture scales ``(d_b, K)`` (σ, not log σ).

    Returns:
        ``y_new`` with the ``b_idx`` columns replaced by the Gaussianized
        values and the rest unchanged.
    """
    import jax.scipy.special as jsp_special
    import jax.scipy.stats as jstats

    y_b = jnp.asarray(y[:, b_idx])  # (n, d_b)
    weights_j = jnp.asarray(weights)  # (d_b, K)
    means_j = jnp.asarray(means)
    sigmas_j = jnp.asarray(sigmas)
    cdfs = jstats.norm.cdf(y_b[:, :, None], loc=means_j, scale=sigmas_j)
    u = jnp.sum(weights_j * cdfs, axis=-1)
    u = jnp.clip(u, 1e-6, 1 - 1e-6)
    z_b = jsp_special.ndtri(u)
    y_new = np.asarray(y).copy()
    y_new[:, b_idx] = np.asarray(z_b)
    return y_new


def _init_coupling_from_fits(
    bij: MixtureGaussianCDFCoupling,
    y: np.ndarray,
    block_idx: int,
    random_state: int,
    seed_dim_idx: np.ndarray | None = None,
) -> tuple[MixtureGaussianCDFCoupling, np.ndarray]:
    """Overwrite the coupling's conditioner so forward ≈ per-dim mixture fit.

    Args:
        bij: Coupling bijection to re-init.
        y: Current data state ``(n, d)``. GMM is fit on ``y[:, b_idx]``.
        block_idx: Block index used to seed GMM EM.
        random_state: Base seed.
        seed_dim_idx: Original dim indices (in the pre-any-flip frame) that
            the current ``b`` positions correspond to. Length must equal
            ``d_b``. Used only to seed the GMM EM — matches the convention
            in the research-notebook mask-based coupling, so that zero-kernel
            couplings paired via flips produce the *same* fits as the
            diagonal flow on all dims. Defaults to ``arange(d_a, d)`` (the
            identity mapping when no flips have happened yet).
    """
    d = bij.shape[0]
    d_a = d // 2
    k = bij.n_components
    b_idx = np.arange(d_a, d)
    d_b = b_idx.size

    if seed_dim_idx is None:
        seed_dim_idx = b_idx
    seed_dim_idx = np.asarray(seed_dim_idx, dtype=np.int64)
    if seed_dim_idx.size != d_b:
        raise ValueError(f"seed_dim_idx length {seed_dim_idx.size} != d_b {d_b}")

    log_weights = np.zeros((d_b, k), dtype=np.float32)
    means = np.zeros((d_b, k), dtype=np.float32)
    clamped_log_scales = np.zeros((d_b, k), dtype=np.float32)
    weights = np.zeros((d_b, k), dtype=np.float32)
    sigmas = np.zeros((d_b, k), dtype=np.float32)
    for j, dim_idx in enumerate(b_idx):
        seed_i = int(seed_dim_idx[j])
        w, mu, sigma = _fit_gmm_per_dim(
            y[:, dim_idx], k, _dim_seed(block_idx, seed_i, random_state)
        )
        log_weights[j] = np.log(np.maximum(w, 1e-20))
        means[j] = mu
        clamped_log_scales[j] = np.log(np.maximum(sigma, 1e-12))
        weights[j] = w
        sigmas[j] = sigma

    # The coupling applies `bound * tanh(raw_log_scales)` to the raw
    # values emitted by the conditioner. Invert that so the clamped
    # output equals the GMM-fit log-scales.
    raw_log_scales = np.asarray(
        _invert_clamp_log_scale(jnp.asarray(clamped_log_scales), bij.log_scale_bound)
    )

    # Sigmas that the coupling's forward will actually use, after the
    # tanh bound is applied. Identical to `sigmas` when |log sigma| <
    # bound, otherwise clipped to exp(+/-bound). State propagation below
    # must use these (not raw `sigmas`) so we match the flow's real
    # trajectory when any fitted component falls outside exp(+/-bound).
    effective_log_scales = bij.log_scale_bound * np.tanh(raw_log_scales)
    effective_sigmas = np.exp(effective_log_scales).astype(np.float32)

    bias = _pack_coupling_bias(log_weights, means, raw_log_scales)

    # Zero the final Dense kernel and set its bias to the packed params.
    # flowjax.Coupling stores the MLP as self._coupling.conditioner.
    mlp = bij._coupling.conditioner  # type: ignore[attr-defined]
    final = mlp.layers[-1]

    def _zero_kernel(f):
        return jnp.zeros_like(f.weight)

    new_final = eqx.tree_at(
        lambda f: (f.weight, f.bias), final, (_zero_kernel(final), jnp.asarray(bias))
    )
    new_mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, new_final)
    new_coupling = eqx.tree_at(lambda c: c.conditioner, bij._coupling, new_mlp)
    new_bij = eqx.tree_at(lambda b: b._coupling, bij, new_coupling)

    # Advance y through a MixtureGaussianCDF-style forward on the b-dims
    # using the *effective* sigmas (bound-clamped), matching what the
    # initialized coupling's forward pass will actually compute.
    y_new = _push_forward_b_half_diag(y, b_idx, weights, means, effective_sigmas)
    return new_bij, y_new


def fit_rbig_coupling(
    x: ArrayLike,
    key: PRNGKeyArray,
    *,
    n_layers: int = 6,
    n_components: int = 8,
    nn_width: int = 64,
    nn_depth: int = 2,
    log_scale_bound: float = 5.0,
    random_state: int = 0,
) -> Transformed:
    """Warm-start an RBIG-style coupling flow in closed form.

    Each block is ``[FixedRotation (PCA), MixtureGaussianCDFCoupling]``.
    The coupling's conditioner is initialised in closed form — no gradient
    training — with a zero kernel and a bias set from per-b-dim sklearn
    ``GaussianMixture`` fits (via :func:`_init_coupling_from_fits`), so
    the layer starts as a constant-in-``x_a`` mixture-CDF transform on the
    ``x_b`` half — numerically equivalent to a diagonal marginal fit on
    the ``x_b`` dims. Gradient training (see
    :func:`~gauss_flows.fit_gaussianization_flow`) can then break the
    constancy and let the conditioner modulate on ``x_a``.

    Args:
        x: Training data of shape ``(n, d)`` with even ``d ≥ 2``.
        key: JAX random key for the conditioner MLPs' hidden-layer init.
        n_layers: Number of ``(rotation, coupling)`` blocks. Defaults to 6.
        n_components: Mixture components per transformed dim. Defaults to 8.
        nn_width: Hidden layer width of each conditioner MLP. Defaults to 64.
        nn_depth: Depth of each conditioner MLP. Defaults to 2.
        log_scale_bound: Tanh bound on per-dim log-scales. Must match the
            value used at gradient-training time. Defaults to 5.0.
        random_state: Base seed for per-dim GMM EM fits. Defaults to 0.

    Returns:
        A flowjax ``Transformed`` distribution with ``log_prob`` and ``sample``.

    Raises:
        ValueError: If ``x`` is not 2-D, or ``d`` is less than 2 or odd
            (the half-swap coupling pair requires an even number of dims).

    Example:
        >>> import jax.random as jr
        >>> from gauss_flows import fit_rbig_coupling
        >>> x = jr.normal(jr.key(0), (500, 2))
        >>> flow = fit_rbig_coupling(x, jr.key(1), n_layers=2, n_components=4)
        >>> flow.sample(jr.key(2), (8,)).shape
        (8, 2)
    """
    import jax.random as jr

    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"x must be 2-D (n, d); got shape {x_np.shape}")
    n_dims = x_np.shape[-1]
    if n_dims < 2:
        raise ValueError("fit_rbig_coupling needs n_dims >= 2 for a non-trivial split.")
    if n_dims % 2 != 0:
        # With odd n_dims the two coupling passes do not cover complementary
        # halves: `MixtureGaussianCDFCoupling` always uses
        # `untransformed_dim = n_dims // 2 = floor(n_dims/2)`, so each pass
        # transforms `ceil(n_dims/2)` coordinates. After the intervening Flip,
        # one dim ends up in *both* b-halves and gets transformed twice per
        # block, breaking the zero-kernel ≡ diagonal-RBIG equivalence.
        raise ValueError(
            f"fit_rbig_coupling requires even n_dims for the half-swap pair to "
            f"cover each dim exactly once; got n_dims={n_dims}. Pad the data "
            f"(e.g. via Augment) or use fit_rbig for odd dimensions."
        )

    keys = jr.split(key, int(n_layers))
    bijections = []
    y = x_np
    d_a = n_dims // 2
    for block_idx in range(int(n_layers)):
        rot = FixedRotation.from_data(y)
        bijections.append(rot)
        y = _push_forward(rot, y)

        # Two couplings with opposing halves (the flowjax equivalent of
        # the Keras `[coupling(mask=m), coupling(mask=~m)]` pair). Between
        # them a `Flip` swaps the halves so the second coupling sees the
        # original untransformed half as its b-half. A trailing `Flip`
        # restores dim order so the next block's PCA fit sees the
        # expected orientation.
        #
        # `current_perm[i]` = the original-dim (pre-any-flip) index that
        # currently sits at position `i`. At block start this is
        # `[0, 1, ..., d-1]`; each Flip reverses it. We use it to seed
        # each coupling's per-b-dim GMM fit by its *original* dim index,
        # so the paired couplings reproduce the same per-dim seeding a
        # diagonal-marginal block would use (mirroring the research-
        # notebook mask-based code, where `_dim_seed` takes the original
        # dim index).
        current_perm = np.arange(n_dims)
        coup_keys = jr.split(keys[block_idx], 2)
        for side_idx in range(2):
            seed_dims = current_perm[d_a:]
            bij = MixtureGaussianCDFCoupling(
                coup_keys[side_idx],
                shape=(n_dims,),
                n_components=n_components,
                nn_width=nn_width,
                nn_depth=nn_depth,
                log_scale_bound=log_scale_bound,
            )
            bij, y = _init_coupling_from_fits(
                bij, y, block_idx, random_state, seed_dim_idx=seed_dims
            )
            bijections.append(bij)

            flip = Flip(shape=(n_dims,))
            bijections.append(flip)
            y = _push_forward(flip, y)
            current_perm = current_perm[::-1].copy()

    base_dist = Normal(jnp.zeros(n_dims))
    bijection = Invert(Chain(bijections))
    return Transformed(base_dist, bijection)


__all__ = ["fit_rbig", "fit_rbig_coupling"]
