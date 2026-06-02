"""Information-theoretic measures for normalizing flows.

Three complementary families of estimators are provided, each measuring the
same quantities (entropy, total correlation, mutual information, KL divergence,
negentropy) but under different assumptions. All entropies and divergences are
returned in **nats** (natural logarithm).

* **Flow-based Monte-Carlo** estimators draw ``n_samples`` from a *trained*
  flow and average its exact change-of-variables log-density. They require a
  ``key`` and converge as O(1/√N) (:func:`entropy`, :func:`total_correlation`,
  :func:`mutual_information`, :func:`kl_divergence`, :func:`negentropy`).
* **Analytical Gaussian** closed forms evaluate the exact expression for a
  multivariate Gaussian from its covariance (and mean). They are deterministic
  and take no samples (:func:`gaussian_entropy`,
  :func:`gaussian_total_correlation`, :func:`gaussian_mutual_information`,
  :func:`gaussian_kl_divergence`).
* **RBIG-way reductions** accumulate the total correlation removed by a
  Gaussianization flow, pairing histogram marginal entropies with a Gaussian
  joint entropy (:func:`information_reduction`,
  :func:`total_correlation_reduction`, :func:`entropy_reduction`,
  :func:`kl_divergence_reduction`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flowjax.distributions import AbstractDistribution
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray


def entropy(
    dist: AbstractDistribution, n_samples: int = 10000, *, key: PRNGKeyArray
) -> Array:
    """Differential entropy of a flow (flow-based Monte-Carlo estimator).

    Draws samples from the trained ``dist`` and averages its exact log-density::

        H(X) = −E[log p(X)] ≈ −(1/N) Σᵢ log p(xᵢ),   xᵢ ∼ dist.

    The estimate converges as O(1/√N) in ``n_samples``.

    Args:
        dist: A trained flowjax distribution exposing ``sample`` and
            ``log_prob``.
        n_samples: Number of Monte-Carlo samples. Defaults to 10000.
        key: JAX PRNG key used to draw the samples.

    Returns:
        Scalar differential-entropy estimate (in nats).

    Example:
        Requires a fitted flow, so the call is slow and non-deterministic::

            import jax.random as jr
            from gauss_flows import entropy

            h = entropy(trained_flow, n_samples=10000, key=jr.key(0))
    """
    samples = dist.sample(key, (n_samples,))
    log_probs = dist.log_prob(samples)
    return -jnp.mean(log_probs)


def total_correlation(
    dist: AbstractDistribution, n_samples: int = 10000, *, key: PRNGKeyArray
) -> Array:
    """Total correlation of a flow (flow-based Monte-Carlo estimator).

    Total correlation (multi-information) is the gap between the sum of marginal
    entropies and the joint entropy::

        TC(X) = Σᵢ H(Xᵢ) − H(X).

    Samples are drawn from the trained ``dist``; the joint entropy is the
    Monte-Carlo average −(1/N) Σ log p(xᵢ), while each marginal entropy uses a
    Gaussian fitted to the corresponding sample column. Converges as O(1/√N).

    Args:
        dist: A trained flowjax distribution with a 1-D event shape.
        n_samples: Number of Monte-Carlo samples. Defaults to 10000.
        key: JAX PRNG key used to draw the samples.

    Returns:
        Scalar total-correlation estimate (in nats).

    Example:
        Requires a fitted flow, so the call is slow and non-deterministic::

            import jax.random as jr
            from gauss_flows import total_correlation

            tc = total_correlation(trained_flow, n_samples=10000, key=jr.key(0))
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
    key: PRNGKeyArray,
) -> Array:
    """Mutual information I(X; Y) of flows (flow-based Monte-Carlo estimator).

    Combines three Monte-Carlo entropy estimates of the supplied trained
    flows::

        I(X; Y) = H(X) + H(Y) − H(X, Y).

    Each entropy is estimated via :func:`entropy` from independent sample
    batches, so the result converges as O(1/√N).

    Args:
        dist_xy: Trained joint distribution of (X, Y).
        dist_x: Trained marginal distribution of X.
        dist_y: Trained marginal distribution of Y.
        n_samples: Number of Monte-Carlo samples per entropy term.
            Defaults to 10000.
        key: JAX PRNG key; split three ways for the three entropy estimates.

    Returns:
        Scalar mutual-information estimate (in nats).

    Example:
        Requires fitted flows, so the call is slow and non-deterministic::

            import jax.random as jr
            from gauss_flows import mutual_information

            mi = mutual_information(flow_xy, flow_x, flow_y, key=jr.key(0))
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
    key: PRNGKeyArray,
) -> Array:
    """KL divergence KL(P ‖ Q) of flows (flow-based Monte-Carlo estimator).

    Samples are drawn from ``dist_p`` and both flows' exact log-densities are
    averaged::

        KL(P ‖ Q) = E_P[log p(X) − log q(X)]
                  ≈ (1/N) Σᵢ [log p(xᵢ) − log q(xᵢ)],   xᵢ ∼ dist_p.

    Converges as O(1/√N) in ``n_samples``.

    Args:
        dist_p: Trained distribution P; samples are drawn from this.
        dist_q: Trained distribution Q; log-probs are evaluated under this.
        n_samples: Number of Monte-Carlo samples. Defaults to 10000.
        key: JAX PRNG key used to draw the samples from P.

    Returns:
        Scalar KL-divergence estimate (in nats).

    Example:
        Requires fitted flows, so the call is slow and non-deterministic::

            import jax.random as jr
            from gauss_flows import kl_divergence

            kl = kl_divergence(flow_p, flow_q, n_samples=10000, key=jr.key(0))
    """
    samples = dist_p.sample(key, (n_samples,))
    log_p = dist_p.log_prob(samples)
    log_q = dist_q.log_prob(samples)
    return jnp.mean(log_p - log_q)


def negentropy(
    dist: AbstractDistribution, n_samples: int = 10000, *, key: PRNGKeyArray
) -> Array:
    """Negentropy of a flow (flow-based Monte-Carlo estimator).

    Negentropy is the entropy gap to the Gaussian of matched moments — a
    non-negative measure of non-Gaussianity::

        J(X) = H(X_gauss) − H(X),

    where ``X_gauss`` is the Normal sharing the per-dimension mean and standard
    deviation of the samples drawn from ``dist``. Both entropies are estimated
    via :func:`entropy`, so the result converges as O(1/√N).

    Args:
        dist: A trained flowjax distribution.
        n_samples: Number of Monte-Carlo samples. Defaults to 10000.
        key: JAX PRNG key used to draw the samples and estimate both entropies.

    Returns:
        Scalar negentropy estimate (in nats).

    Example:
        Requires a fitted flow, so the call is slow and non-deterministic::

            import jax.random as jr
            from gauss_flows import negentropy

            j = negentropy(trained_flow, n_samples=10000, key=jr.key(0))
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
    """Differential entropy of a Gaussian (analytical Gaussian closed form).

    Deterministic, sample-free closed form in the covariance::

        H(X) = ½ log |2π e Σ| = ½ (d log(2π e) + log|Σ|).

    Args:
        covariance: Covariance matrix Σ of shape ``(d, d)``.

    Returns:
        Scalar differential entropy (in nats).

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import gaussian_entropy
        >>> round(float(gaussian_entropy(jnp.eye(3))), 4)
        4.2568
    """
    cov = jnp.asarray(covariance)
    d = cov.shape[-1]
    _sign, logdet = jnp.linalg.slogdet(cov)
    return 0.5 * (d * jnp.log(2.0 * jnp.pi * jnp.e) + logdet)


def gaussian_total_correlation(covariance: ArrayLike) -> Array:
    """Total correlation of a Gaussian (analytical Gaussian closed form).

    Deterministic, sample-free closed form comparing the diagonal variances to
    the full covariance::

        TC(X) = ½ [ Σᵢ log Σᵢᵢ − log|Σ| ].

    It is zero exactly when Σ is diagonal (independent components).

    Args:
        covariance: Covariance matrix Σ of shape ``(d, d)``.

    Returns:
        Scalar total correlation (in nats).

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import gaussian_total_correlation
        >>> cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        >>> round(float(gaussian_total_correlation(cov)), 4)
        0.1438
        >>> float(gaussian_total_correlation(jnp.eye(2)))  # diagonal → 0
        0.0
    """
    cov = jnp.asarray(covariance)
    variances = jnp.diagonal(cov)
    _sign, logdet = jnp.linalg.slogdet(cov)
    return 0.5 * (jnp.sum(jnp.log(variances)) - logdet)


def gaussian_mutual_information(covariance: ArrayLike, dim_x: int) -> Array:
    """Mutual information of Gaussian blocks (analytical Gaussian closed form).

    Deterministic, sample-free closed form. The joint covariance is partitioned
    into the leading ``dim_x`` dimensions (X) and the remainder (Y), then::

        I(X; Y) = H(X) + H(Y) − H(X, Y),

    each entropy evaluated via :func:`gaussian_entropy`.

    Args:
        covariance: Joint covariance matrix of ``[X, Y]`` of shape ``(d, d)``.
        dim_x: Number of leading dimensions belonging to X.

    Returns:
        Scalar mutual information (in nats).

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import gaussian_mutual_information
        >>> cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        >>> round(float(gaussian_mutual_information(cov, dim_x=1)), 4)
        0.1438
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
    """KL divergence of two Gaussians (analytical Gaussian closed form).

    Deterministic, sample-free closed form::

        KL(N_p ‖ N_q) = ½ [ tr(Σ_q⁻¹ Σ_p)
                            + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) − d
                            + log(|Σ_q| / |Σ_p|) ].

    It is zero exactly when the two Gaussians coincide.

    Args:
        mean_p: Mean of P, shape ``(d,)``.
        cov_p: Covariance of P, shape ``(d, d)``.
        mean_q: Mean of Q, shape ``(d,)``.
        cov_q: Covariance of Q, shape ``(d, d)``.

    Returns:
        Scalar KL divergence (in nats).

    Example:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import gaussian_kl_divergence
        >>> mu, eye = jnp.zeros(2), jnp.eye(2)
        >>> float(gaussian_kl_divergence(mu, eye, mu, eye))  # identical → 0
        0.0
    """
    mu_p = jnp.asarray(mean_p)
    mu_q = jnp.asarray(mean_q)
    sigma_p = jnp.asarray(cov_p)
    sigma_q = jnp.asarray(cov_q)
    d = sigma_p.shape[-1]
    _sp, logdet_p = jnp.linalg.slogdet(sigma_p)
    _sq, logdet_q = jnp.linalg.slogdet(sigma_q)
    diff = mu_q - mu_p
    # Solve instead of forming Sigma_q^-1 (faster, more numerically stable).
    trace_term = jnp.trace(jnp.linalg.solve(sigma_q, sigma_p))
    maha_term = diff @ jnp.linalg.solve(sigma_q, diff)
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

    Note:
        Pairing histogram marginal entropies with a Gaussian joint entropy is
        only exact when ``x`` is Gaussian (the regime targeted by the
        Gaussianization flows here, where the latent ``z`` is standard normal).
        For strongly non-Gaussian marginals the two estimators are inconsistent
        and the resulting total correlation may be slightly negative even for
        independent data. This matches RBIG's reference behaviour; pass a fitted
        Gaussianization flow (so the joint is evaluated on the near-Gaussian
        latent) for reliable estimates.

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
    """Total correlation removed between two representations (RBIG-way reduction).

    The building block of the RBIG-way estimators: the drop in total
    correlation when moving from ``x_before`` to ``x_after``::

        Δ_TC = TC(x_before) − TC(x_after),

    with ``TC(X) = Σᵢ H(Xᵢ) − H(X)``, marginal entropies estimated from
    histograms and the joint entropy from the Gaussian (maximum-entropy)
    approximation.

    Args:
        x_before: Data before the transformation, shape ``(n, d)``.
        x_after: Data after the transformation, shape ``(n, d)``.
        n_bins: Histogram bins per marginal. Defaults to ``round(√n)``.

    Returns:
        Scalar total-correlation reduction (in nats).

    Example:
        Plain sampled data, no flow required::

            from gauss_flows import information_reduction

            delta_tc = information_reduction(x_before, x_after)
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
    # Reference mass per bin from exact N(0, 1) CDF differences, with the two
    # outer bins absorbing the tail mass below/above the finite support so that
    # q integrates to 1 over (-inf, inf) and the result is a proper KL.
    cdf = jax.scipy.stats.norm.cdf(edges)
    q = cdf[1:] - cdf[:-1]
    q = q.at[0].add(cdf[0])
    q = q.at[-1].add(1.0 - cdf[-1])
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
