"""NumPyro compatibility for Gaussianization flows.

Provides a wrapper that makes a flowjax Transformed distribution compatible
with the NumPyro distribution interface for use in probabilistic programs.
"""

from typing import ClassVar

import numpyro.distributions as dist_lib
import numpyro.distributions.constraints as constraints
from flowjax.distributions import Transformed
from jax import Array
from jaxtyping import PRNGKeyArray


class FlowDist(dist_lib.Distribution):
    """Wrap a flowjax ``Transformed`` distribution as a NumPyro distribution.

    Subclasses :class:`numpyro.distributions.Distribution`, delegating
    ``sample`` and ``log_prob`` to the wrapped flow. This lets
    Gaussianization flows (and other flowjax distributions) be used
    directly in NumPyro probabilistic programs with standard MCMC or
    variational inference algorithms. The batch shape is empty and the
    event shape is taken from ``flow.shape``; the support is a real
    vector.

    Args:
        flow: A flowjax ``Transformed`` distribution.

    Example:
        >>> import jax.random as jr
        >>> from gauss_flows import gaussianization_flow, FlowDist
        >>> flow = gaussianization_flow(jr.key(0), n_dims=2)
        >>> flow_dist = FlowDist(flow)
        >>> flow_dist.event_shape
        (2,)
        >>> x = flow_dist.sample(jr.key(1), (5,))
        >>> x.shape
        (5, 2)
        >>> flow_dist.log_prob(x).shape
        (5,)

        Use inside a NumPyro model with ``numpyro.sample("x", flow_dist)``.
    """

    arg_constraints: ClassVar[dict] = {}
    support = constraints.real_vector

    def __init__(self, flow: Transformed):
        self.flow = flow
        batch_shape = ()
        event_shape = flow.shape
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()) -> Array:
        """Sample from the flow distribution.

        Args:
            key: JAX random key.
            sample_shape: Shape of samples to draw. Defaults to ``()``.

        Returns:
            Samples of shape ``sample_shape + event_shape``.
        """
        return self.flow.sample(key, sample_shape)

    def log_prob(self, value: Array) -> Array:
        """Compute log probability under the flow.

        Args:
            value: Points of shape ``sample_shape + event_shape`` at which
                to evaluate the log probability.

        Returns:
            Log-probability values of shape ``sample_shape``.
        """
        return self.flow.log_prob(value)


__all__ = ["FlowDist"]
