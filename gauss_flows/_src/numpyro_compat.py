"""NumPyro compatibility for Gaussianization flows.

Provides a wrapper that makes a flowjax Transformed distribution compatible
with the NumPyro distribution interface for use in probabilistic programs.
"""

from typing import ClassVar

import numpyro.distributions as dist_lib
from flowjax.distributions import Transformed
from jax import Array
from jaxtyping import PRNGKeyArray


class FlowDist(dist_lib.Distribution):
    """Wraps a flowjax Transformed distribution as a NumPyro distribution.

    This allows Gaussianization flows (and other flowjax distributions) to be
    used directly in NumPyro probabilistic programs with standard MCMC
    or variational inference algorithms.

    Args:
        flow: A flowjax Transformed distribution.

    Example:
        >>> import jax.random as jr
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from gauss_flows import gaussianization_flow, FlowDist
        >>> key = jr.key(0)
        >>> flow = gaussianization_flow(key, n_dims=2)
        >>> flow_dist = FlowDist(flow)
        >>> # Use in a NumPyro model:
        >>> # numpyro.sample("x", flow_dist)
    """

    arg_constraints: ClassVar[dict] = {}

    def __init__(self, flow: Transformed):
        self.flow = flow
        batch_shape = ()
        event_shape = flow.shape
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()) -> Array:
        """Sample from the flow distribution.

        Args:
            key: JAX random key.
            sample_shape: Shape of samples to draw. Defaults to ().

        Returns:
            Samples of shape sample_shape + event_shape.
        """
        return self.flow.sample(key, sample_shape)

    def log_prob(self, value: Array) -> Array:
        """Compute log probability under the flow.

        Args:
            value: Points at which to evaluate the log probability.

        Returns:
            Log probability values.
        """
        return self.flow.log_prob(value)


__all__ = ["FlowDist"]
