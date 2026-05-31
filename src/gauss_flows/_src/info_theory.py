"""Information-theoretic measures using normalizing flows.

These functions compute various IT measures in three complementary ways:

* **Analytical** closed forms for multivariate Gaussians
  (:func:`gaussian_entropy`, :func:`gaussian_total_correlation`,
  :func:`gaussian_mutual_information`, :func:`gaussian_kl_divergence`).
* **Change-of-variables** Monte-Carlo estimators that exploit the exact
  log-density of a fitted flow (:func:`entropy`, :func:`total_correlation`,
  :func:`mutual_information`, :func:`kl_divergence`, :func:`negentropy`).
* **RBIG-way** estimators that accumulate the total-correlation removed by a
  Gaussianization flow (:func:`information_reduction`,
  :func:`total_correlation_reduction`, :func:`entropy_reduction`).
"""

import jax
import jax.numpy as jnp
from flowjax.distributions import AbstractDistribution
from jax import Array
from jaxtyping import ArrayLike


def entropy(dist: AbstractDistribution, n_samples: int = 10000, *, key) -> Array:
    """Estimate the differential entropy of a distribution.

    Uses Monte Carlo estimation: H(X) = -E[log p(X)] where samples are drawn
    from the distribution.

    Args:
        dist: A flowjax distribution.
        n_samples: Number of Monte Carlo samples. Defaults to 10000.
        key: JAX random key.

    Returns:
        Scalar estimate of the differential entropy.
    """
    samples = dist.sample(key, (n_samples,))
    log_probs = dist.log_prob(samples)
    return -jnp.mean(log_probs)


def total_correlation(
    dist: AbstractDistribution, n_samples: int = 10000, *, key
) -> Array:
    """Estimate the total correlation (multi-information) of a distribution.

    Total correlation is defined as:
        TC(X) = sum_i H(X_i) - H(X)
    where H(X_i) is the marginal entropy of each dimension.

    Args:
        dist: A flowjax distribution with 1D shape.
        n_samples: Number of Monte Carlo samples. Defaults to 10000.
        key: JAX random key.

    Returns:
        Scalar estimate of total correlation (in nats).
    """
    import jax.random as jr

    key1, _key2 = jr.split(key)
    samples = dist.sample(key1, (n_samples,))
    log_probs = dist.log_prob(samples)
    joint_entropy = -jnp.mean(log_probs)

    from flowjax.distributions import Normal

    n_dims = samples.shape[-1]
    marginal_entropies = jnp.zeros(())
    for i in range(n_dims):
        xi = samples[:, i]
        # Fit a simple Gaussian to the marginal
        mean_i = jnp.mean(xi)
        std_i = jnp.std(xi)
        marginal_dist = Normal(mean_i, std_i)
        log_p_marginal = marginal_dist.log_prob(xi[:, None])
        marginal_entropies = marginal_entropies + (-jnp.mean(log_p_marginal))

    return marginal_entropies - joint_entropy


def mutual_information(
    dist_xy: AbstractDistribution,
    dist_x: AbstractDistribution,
    dist_y: AbstractDistribution,
    n_samples: int = 10000,
    *,
    key,
) -> Array:
    """Estimate the mutual information I(X; Y).

    I(X; Y) = H(X) + H(Y) - H(X, Y)

    Args:
        dist_xy: Joint distribution of (X, Y).
        dist_x: Marginal distribution of X.
        dist_y: Marginal distribution of Y.
        n_samples: Number of Monte Carlo samples. Defaults to 10000.
        key: JAX random key.

    Returns:
        Scalar estimate of mutual information.
    """
    import jax.random as jr

    key1, key2, key3 = jr.split(key, 3)
    h_xy = entropy(dist_xy, n_samples=n_samples, key=key1)
    h_x = entropy(dist_x, n_samples=n_samples, key=key2)
    h_y = entropy(dist_y, n_samples=n_samples, key=key3)
    return h_x + h_y - h_xy


def kl_divergence(
    dist_p: AbstractDistribution,
    dist_q: AbstractDistribution,
    n_samples: int = 10000,
    *,
    key,
) -> Array:
    """Estimate the KL divergence KL(P || Q) using Monte Carlo.

    KL(P || Q) = E_P[log p(X) - log q(X)]

    Args:
        dist_p: Distribution P (samples are drawn from this).
        dist_q: Distribution Q (log-probs are evaluated under this).
        n_samples: Number of Monte Carlo samples. Defaults to 10000.
        key: JAX random key.

    Returns:
        Scalar estimate of the KL divergence.
    """
    samples = dist_p.sample(key, (n_samples,))
    log_p = dist_p.log_prob(samples)
    log_q = dist_q.log_prob(samples)
    return jnp.mean(log_p - log_q)


def negentropy(dist: AbstractDistribution, n_samples: int = 10000, *, key) -> Array:
    """Estimate the negentropy of a distribution.

    Negentropy is defined as:
        J(X) = H(X_gauss) - H(X)
    where X_gauss is a Gaussian with the same mean and variance as X.

    Args:
        dist: A flowjax distribution.
        n_samples: Number of Monte Carlo samples. Defaults to 10000.
        key: JAX random key.

    Returns:
        Scalar estimate of negentropy.
    """
    import jax.random as jr
    from flowjax.distributions import Normal

    key1, key2 = jr.split(key)
    samples = dist.sample(key1, (n_samples,))

    # Estimate entropy of the distribution
    h = entropy(dist, n_samples=n_samples, key=key2)

    # Entropy of a Gaussian with the same mean and variance
    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0)
    gauss = Normal(mean, std)
    h_gauss = entropy(gauss, n_samples=n_samples, key=key2)

    return h_gauss - h


# ---------------------------------------------------------------------------
# Analytical (closed-form) Gaussian measures
# ---------------------------------------------------------------------------


def gaussian_entropy(covariance: ArrayLike) -> Array:
    """Differential entropy of a multivariate Gaussian (analytical).

    H(X) = 1/2 log |2 pi e Sigma|

    Args:
        covariance: Covariance matrix Sigma of shape ``(d, d)``.

    Returns:
        Scalar differential entropy in nats.
    """
    cov = jnp.asarray(covariance)
    d = cov.shape[-1]
    _sign, logdet = jnp.linalg.slogdet(cov)
    return 0.5 * (d * jnp.log(2.0 * jnp.pi * jnp.e) + logdet)


def gaussian_total_correlation(covariance: ArrayLike) -> Array:
    """Total correlation of a multivariate Gaussian (analytical).

    TC(X) = 1/2 [ sum_i log Sigma_ii - log |Sigma| ]

    Args:
        covariance: Covariance matrix Sigma of shape ``(d, d)``.

    Returns:
        Scalar total correlation in nats.
    """
    cov = jnp.asarray(covariance)
    variances = jnp.diagonal(cov)
    _sign, logdet = jnp.linalg.slogdet(cov)
    return 0.5 * (jnp.sum(jnp.log(variances)) - logdet)


def gaussian_mutual_information(covariance: ArrayLike, dim_x: int) -> Array:
    """Mutual information between two Gaussian blocks X and Y (analytical).

    I(X; Y) = H(X) + H(Y) - H(X, Y), where the joint covariance is partitioned
    into the leading ``dim_x`` dimensions (X) and the rest (Y).

    Args:
        covariance: Joint covariance matrix of ``[X, Y]`` of shape ``(d, d)``.
        dim_x: Number of leading dimensions belonging to X.

    Returns:
        Scalar mutual information in nats.
    """
    cov = jnp.asarray(covariance)
    cov_xx = cov[:dim_x, :dim_x]
    cov_yy = cov[dim_x:, dim_x:]
    return gaussian_entropy(cov_xx) + gaussian_entropy(cov_yy) - gaussian_entropy(cov)


def gaussian_kl_divergence(
    mean_p: ArrayLike,
    cov_p: ArrayLike,
    mean_q: ArrayLike,
    cov_q: ArrayLike,
) -> Array:
    """KL divergence between two multivariate Gaussians (analytical).

    KL(N_p || N_q) = 1/2 [ tr(Sigma_q^-1 Sigma_p)
        + (mu_q - mu_p)^T Sigma_q^-1 (mu_q - mu_p) - d
        + log(|Sigma_q| / |Sigma_p|) ]

    Args:
        mean_p: Mean of P, shape ``(d,)``.
        cov_p: Covariance of P, shape ``(d, d)``.
        mean_q: Mean of Q, shape ``(d,)``.
        cov_q: Covariance of Q, shape ``(d, d)``.

    Returns:
        Scalar KL divergence in nats.
    """
    mu_p = jnp.asarray(mean_p)
    mu_q = jnp.asarray(mean_q)
    sigma_p = jnp.asarray(cov_p)
    sigma_q = jnp.asarray(cov_q)
    d = sigma_p.shape[-1]
    sigma_q_inv = jnp.linalg.inv(sigma_q)
    _sp, logdet_p = jnp.linalg.slogdet(sigma_p)
    _sq, logdet_q = jnp.linalg.slogdet(sigma_q)
    diff = mu_q - mu_p
    trace_term = jnp.trace(sigma_q_inv @ sigma_p)
    maha_term = diff @ sigma_q_inv @ diff
    return 0.5 * (trace_term + maha_term - d + logdet_q - logdet_p)


# ---------------------------------------------------------------------------
# RBIG-way measures (accumulated total-correlation reduction)
# ---------------------------------------------------------------------------


def _n_bins(n_samples: int, n_bins: int | None) -> int:
    """Histogram bin count (square-root rule when ``n_bins`` is ``None``)."""
    if n_bins is not None:
        return int(n_bins)
    return max(2, round(n_samples**0.5))


def _marginal_entropy_hist(x: Array, n_bins: int) -> Array:
    """1-D differential entropy of ``x`` via a histogram plug-in estimator.

    Args:
        x: Samples of shape ``(n,)``.
        n_bins: Number of histogram bins.

    Returns:
        Scalar entropy estimate in nats.
    """
    counts, edges = jnp.histogram(x, bins=n_bins)
    total = jnp.sum(counts)
    probs = counts / total
    bin_width = edges[1] - edges[0]
    nonzero = probs > 0
    safe_probs = jnp.where(nonzero, probs, 1.0)
    # H = -sum p log p + log(bin_width)
    return -jnp.sum(jnp.where(nonzero, probs * jnp.log(safe_probs), 0.0)) + jnp.log(
        bin_width
    )


def _marginal_entropy_sum(x: Array, n_bins: int) -> Array:
    """Sum of per-dimension histogram marginal entropies.

    Args:
        x: Data of shape ``(n, d)``.
        n_bins: Number of histogram bins per dimension.

    Returns:
        Scalar sum of marginal entropies in nats.
    """
    return jnp.sum(
        jnp.stack([_marginal_entropy_hist(x[:, i], n_bins) for i in range(x.shape[-1])])
    )


def _joint_entropy_gaussian(x: Array) -> Array:
    """Gaussian (maximum-entropy) approximation of the joint entropy of ``x``.

    Args:
        x: Data of shape ``(n, d)``.

    Returns:
        Scalar joint-entropy estimate in nats.
    """
    cov = jnp.cov(x, rowvar=False)
    cov = jnp.atleast_2d(cov)
    return gaussian_entropy(cov)


def _total_correlation_estimate(x: Array, n_bins: int) -> Array:
    """Total-correlation estimate ``sum_i H(X_i) - H(X)`` for raw samples."""
    return _marginal_entropy_sum(x, n_bins) - _joint_entropy_gaussian(x)


def information_reduction(
    x_before: ArrayLike, x_after: ArrayLike, *, n_bins: int | None = None
) -> Array:
    """Total-correlation removed between two representations of the data.

    This is the building block of the "RBIG-way" estimators. The reduction is
    the drop in total correlation when moving from ``x_before`` to ``x_after``::

        Delta_TC = TC(x_before) - TC(x_after)

    with ``TC(X) = sum_i H(X_i) - H(X)``, marginal entropies estimated from
    histograms and the joint entropy from the Gaussian (maximum-entropy)
    approximation.

    Args:
        x_before: Data before the transformation, shape ``(n, d)``.
        x_after: Data after the transformation, shape ``(n, d)``.
        n_bins: Histogram bins per marginal. Defaults to ``round(sqrt(n))``.

    Returns:
        Scalar total-correlation reduction in nats.
    """
    xb = jnp.asarray(x_before)
    xa = jnp.asarray(x_after)
    bins = _n_bins(xb.shape[0], n_bins)
    return _total_correlation_estimate(xb, bins) - _total_correlation_estimate(xa, bins)


def total_correlation_reduction(
    flow: AbstractDistribution, x: ArrayLike, *, n_bins: int | None = None
) -> Array:
    """Total correlation of ``x`` via a Gaussianization flow (RBIG-way).

    The flow Gaussianizes the data, ``z = f(x)``, via ``flow.bijection.inverse``.
    The total correlation removed is the drop from the data to the
    (approximately independent) latent representation::

        TC(X) ~= TC(x) - TC(z)

    which mirrors RBIG's ``total_correlation_reduction``. No Jacobian is needed.

    Args:
        flow: A fitted flowjax ``Transformed`` distribution whose bijection
            maps the latent base to the data space.
        x: Data of shape ``(n, d)``.
        n_bins: Histogram bins per marginal. Defaults to ``round(sqrt(n))``.

    Returns:
        Scalar total-correlation estimate in nats.
    """
    xx = jnp.asarray(x)
    z = jax.vmap(flow.bijection.inverse)(xx)
    return information_reduction(xx, z, n_bins=n_bins)


def entropy_reduction(
    flow: AbstractDistribution, x: ArrayLike, *, n_bins: int | None = None
) -> Array:
    """Differential entropy of ``x`` via a Gaussianization flow (RBIG-way).

    Uses the decomposition ``H(X) = sum_i H(X_i) - TC(X)`` with the marginal
    entropies estimated from histograms and ``TC(X)`` obtained from
    :func:`total_correlation_reduction`.

    Args:
        flow: A fitted flowjax ``Transformed`` distribution.
        x: Data of shape ``(n, d)``.
        n_bins: Histogram bins per marginal. Defaults to ``round(sqrt(n))``.

    Returns:
        Scalar differential-entropy estimate in nats.
    """
    xx = jnp.asarray(x)
    bins = _n_bins(xx.shape[0], n_bins)
    marginal = _marginal_entropy_sum(xx, bins)
    tc = total_correlation_reduction(flow, xx, n_bins=bins)
    return marginal - tc


def _kl_hist_to_normal(v: Array, n_bins: int) -> Array:
    """Discrete KL between the histogram of ``v`` and a standard normal."""
    counts, edges = jnp.histogram(v, bins=n_bins)
    total = jnp.sum(counts)
    p = counts / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    # Probability mass of N(0, 1) within each bin (mid-point rule).
    q = jnp.exp(-0.5 * centers**2) / jnp.sqrt(2.0 * jnp.pi) * bin_width
    nonzero = p > 0
    safe_p = jnp.where(nonzero, p, 1.0)
    safe_q = jnp.where(q > 0, q, 1.0)
    return jnp.sum(jnp.where(nonzero, p * jnp.log(safe_p / safe_q), 0.0))


def kl_divergence_reduction(
    flow_ref: AbstractDistribution, x_query: ArrayLike, *, n_bins: int | None = None
) -> Array:
    """KL divergence ``KL(P || Q)`` via a reference Gaussianization (RBIG-way).

    The reference flow ``flow_ref`` Gaussianizes ``Q`` (the distribution it was
    fitted to). Applying that same map to samples ``x_query`` drawn from ``P``
    gives ``z = f_Q(x)``; if ``P == Q`` then ``z`` is standard normal. The
    divergence is recovered as the per-marginal KL to a standard normal plus the
    residual total correlation::

        KL(P || Q) ~= sum_d KL(z_d || N(0, 1)) + TC(z)

    This mirrors RBIG's ``estimate_kld`` and requires no Jacobian.

    Args:
        flow_ref: A flow fitted to the reference distribution ``Q``.
        x_query: Samples from ``P`` of shape ``(n, d)``.
        n_bins: Histogram bins per marginal. Defaults to ``round(sqrt(n))``.

    Returns:
        Scalar KL-divergence estimate in nats.
    """
    xq = jnp.asarray(x_query)
    z = jax.vmap(flow_ref.bijection.inverse)(xq)
    bins = _n_bins(z.shape[0], n_bins)
    marginal_kl = jnp.sum(
        jnp.stack([_kl_hist_to_normal(z[:, i], bins) for i in range(z.shape[-1])])
    )
    residual_tc = _total_correlation_estimate(z, bins)
    return marginal_kl + residual_tc


__all__ = [
    "entropy",
    "entropy_reduction",
    "gaussian_entropy",
    "gaussian_kl_divergence",
    "gaussian_mutual_information",
    "gaussian_total_correlation",
    "information_reduction",
    "kl_divergence",
    "kl_divergence_reduction",
    "mutual_information",
    "negentropy",
    "total_correlation",
    "total_correlation_reduction",
]
