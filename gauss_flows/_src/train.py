"""Training utilities for Gaussianization flows."""

from flowjax.distributions import AbstractDistribution
from flowjax.train.loops import fit_to_data
from jaxtyping import ArrayLike, PRNGKeyArray


def fit_gaussianization_flow(
    key: PRNGKeyArray,
    dist: AbstractDistribution,
    data: ArrayLike,
    *,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 256,
    val_prop: float = 0.1,
    show_progress: bool = True,
):
    """Fit a Gaussianization flow to data using maximum likelihood.

    Wraps ``flowjax.train.fit_to_data`` with sensible defaults for
    Gaussianization flows.

    Args:
        key: JAX random key.
        dist: A flowjax distribution (e.g. from ``gaussianization_flow``).
        data: Training data array of shape (n_samples, n_dims).
        learning_rate: Adam optimizer learning rate. Defaults to 5e-4.
        max_epochs: Maximum training epochs. Defaults to 100.
        max_patience: Early stopping patience in epochs. Defaults to 5.
        batch_size: Batch size for training. Defaults to 256.
        val_prop: Proportion of data to use for validation. Defaults to 0.1.
        show_progress: Whether to show a progress bar. Defaults to True.

    Returns:
        Tuple of (trained_dist, losses) where losses is a dict with "train" and
        "val" lists of per-epoch losses.
    """
    return fit_to_data(
        key,
        dist,
        data,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
        val_prop=val_prop,
        show_progress=show_progress,
    )


__all__ = ["fit_gaussianization_flow"]
