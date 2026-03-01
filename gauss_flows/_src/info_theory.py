"""Information-theoretic measures using normalizing flows.

These functions compute various IT measures by leveraging the change-of-variables
formula provided by normalizing flows.
"""

import jax.numpy as jnp
from flowjax.distributions import AbstractDistribution
from jax import Array


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


__all__ = [
    "entropy",
    "kl_divergence",
    "mutual_information",
    "negentropy",
    "total_correlation",
]
