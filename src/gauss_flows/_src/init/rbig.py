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
from flowjax.bijections import Chain, Invert
from flowjax.distributions import Normal, Transformed
from jaxtyping import ArrayLike
from paramax.utils import inv_softplus
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from gauss_flows._src.transforms.bijections.coupling.mixture_cdf import (
    MixtureGaussianCDFCoupling,
    _invert_clamp_log_scale,
)
from gauss_flows._src.transforms.bijections.elementwise.mixture_cdf import (
    MixtureGaussianCDF,
)
from gauss_flows._src.transforms.bijections.linear.rotation import FixedRotation


_SCALE_FLOOR = 1e-5


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
    ``softmax(log_weights)`` and ``softplus(log_scales) + 1e-5`` at
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
        # softplus(raw) + 1e-5 ≈ sigma → raw = inv_softplus(sigma - 1e-5).
        sigma_target = np.maximum(sigma - _SCALE_FLOOR, _SCALE_FLOOR)
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
    """Fit an iterative-Gaussianization (RBIG) flow to data.

    Greedily walks ``n_layers`` blocks of ``[FixedRotation (PCA),
    MixtureGaussianCDF (GMM per-dim)]``, fitting each from the current
    state and propagating the data forward before the next block. The
    result is a :class:`flowjax.distributions.Transformed` whose
    ``log_prob`` is already a reasonable density estimator before any
    gradient training.

    Args:
        x: Training data of shape ``(n, d)``.
        n_layers: Number of ``(rotation, marginal)`` blocks. Defaults to 10.
        n_components: Mixture components ``K`` per marginal layer. Defaults
            to 8.
        random_state: Base seed for per-dim GMM EM fits. Each fit uses
            ``random_state + block_idx * 1000 + dim_idx``. Defaults to 0.

    Returns:
        A :class:`flowjax.distributions.Transformed` distribution over
        ``R^d`` whose bijection Gaussianises ``x``.

    Example:
        >>> import jax.random as jr
        >>> from gauss_flows import fit_rbig
        >>> x = jr.normal(jr.key(0), (500, 2))
        >>> flow = fit_rbig(x, n_layers=4, n_components=4)
        >>> float(flow.log_prob(x).mean())  # doctest: +SKIP
        -2.8...
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


def _coupling_params_shape(bij: MixtureGaussianCDFCoupling) -> tuple[int, int]:
    """Return ``(d_b, params_per_dim)`` for the coupling's conditioner."""
    d = bij.shape[0]
    d_a = d // 2
    d_b = d - d_a
    params_per_dim = 3 * bij.n_components
    return d_b, params_per_dim


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


def _init_coupling_from_fits(
    bij: MixtureGaussianCDFCoupling,
    y: np.ndarray,
    block_idx: int,
    random_state: int,
) -> MixtureGaussianCDFCoupling:
    """Overwrite the coupling's conditioner so forward ≈ per-dim mixture fit."""
    d = bij.shape[0]
    d_a = d // 2
    k = bij.n_components
    b_idx = np.arange(d_a, d)
    d_b = b_idx.size

    log_weights = np.zeros((d_b, k), dtype=np.float32)
    means = np.zeros((d_b, k), dtype=np.float32)
    clamped_log_scales = np.zeros((d_b, k), dtype=np.float32)
    for j, dim_idx in enumerate(b_idx):
        w, mu, sigma = _fit_gmm_per_dim(
            y[:, dim_idx], k, _dim_seed(block_idx, int(dim_idx), random_state)
        )
        log_weights[j] = np.log(np.maximum(w, 1e-20))
        means[j] = mu
        clamped_log_scales[j] = np.log(np.maximum(sigma, 1e-12))

    # The coupling applies `bound * tanh(raw_log_scales)` to the raw
    # values emitted by the conditioner. Invert that so the clamped
    # output equals the GMM-fit log-scales.
    raw_log_scales = np.asarray(
        _invert_clamp_log_scale(
            jnp.asarray(clamped_log_scales), bij.log_scale_bound
        )
    )

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
    new_coupling = eqx.tree_at(
        lambda c: c.conditioner, bij._coupling, new_mlp
    )
    return eqx.tree_at(lambda b: b._coupling, bij, new_coupling)


def fit_rbig_coupling(
    x: ArrayLike,
    key,
    *,
    n_layers: int = 6,
    n_components: int = 8,
    nn_width: int = 64,
    nn_depth: int = 2,
    log_scale_bound: float = 5.0,
    random_state: int = 0,
) -> Transformed:
    """Warm-start an RBIG-style coupling flow.

    Each block is ``[FixedRotation (PCA), MixtureGaussianCDFCoupling]``.
    The coupling's conditioner is initialised with a zero kernel and a
    bias set from per-b-dim GMM fits (via :func:`_init_coupling_from_fits`),
    so the layer starts as a constant-in-``x_a`` mixture-CDF transform on
    the ``x_b`` half — numerically equivalent to a diagonal marginal fit
    on the ``x_b`` dims. Gradient training can then break the constancy
    and let the conditioner modulate on ``x_a``.

    Args:
        x: Training data of shape ``(n, d)``.
        key: JAX random key for the conditioner MLPs' hidden-layer init.
        n_layers: Number of ``(rotation, coupling)`` blocks. Defaults to 6.
        n_components: Mixture components per transformed dim. Defaults to 8.
        nn_width: Hidden layer width of each conditioner MLP. Defaults to 64.
        nn_depth: Depth of each conditioner MLP. Defaults to 2.
        log_scale_bound: Tanh bound on per-dim log-scales. Must match the
            value used at gradient-training time. Defaults to 5.0.
        random_state: Base seed for per-dim GMM EM fits. Defaults to 0.

    Returns:
        A :class:`flowjax.distributions.Transformed` distribution.
    """
    import jax.random as jr

    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"x must be 2-D (n, d); got shape {x_np.shape}")
    n_dims = x_np.shape[-1]
    if n_dims < 2:
        raise ValueError(
            "fit_rbig_coupling needs n_dims >= 2 for a non-trivial split."
        )

    keys = jr.split(key, int(n_layers))
    bijections = []
    y = x_np
    for block_idx in range(int(n_layers)):
        rot = FixedRotation.from_data(y)
        bijections.append(rot)
        y = _push_forward(rot, y)

        bij = MixtureGaussianCDFCoupling(
            keys[block_idx],
            shape=(n_dims,),
            n_components=n_components,
            nn_width=nn_width,
            nn_depth=nn_depth,
            log_scale_bound=log_scale_bound,
        )
        bij = _init_coupling_from_fits(bij, y, block_idx, random_state)
        bijections.append(bij)
        y = _push_forward(bij, y)

    base_dist = Normal(jnp.zeros(n_dims))
    bijection = Invert(Chain(bijections))
    return Transformed(base_dist, bijection)


__all__ = ["fit_rbig", "fit_rbig_coupling"]
