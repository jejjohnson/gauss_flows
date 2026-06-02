"""NumPyro guide using Gaussianization flows for SVI.

Provides :class:`FlowGuide`, a variational guide that uses a normalizing flow
as the approximate posterior for Stochastic Variational Inference (SVI).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.random as jr
from flowjax.distributions import Transformed
from flowjax.experimental.numpyro import register_params
from jaxtyping import PRNGKeyArray
from numpyro.infer.autoguide import AutoContinuous
from numpyro.infer.initialization import init_to_uniform

from gauss_flows._src.flows.gaussianization import gaussianization_flow
from gauss_flows._src.inference.numpyro_compat import FlowDist


class FlowGuide(AutoContinuous):
    """NumPyro variational guide using a Gaussianization flow.

    Subclasses :class:`numpyro.infer.autoguide.AutoContinuous`, using a
    normalizing flow as the variational distribution for SVI. The flow
    models a joint distribution over the concatenated unconstrained latent
    space of the model, so it captures cross-variable correlations because
    all latent sites are handled jointly. The flow is built lazily once
    ``latent_dim`` is known (at prototype setup), and its parameters are
    registered with NumPyro so the SVI optimiser can update them.

    Args:
        model: A NumPyro model.
        flow_factory: Callable that constructs a flowjax ``Transformed``
            distribution. Must accept ``key`` and ``n_dims`` positional /
            keyword arguments, plus any ``flow_kwargs``. Defaults to
            :func:`~gauss_flows.gaussianization_flow`.
        flow_kwargs: Extra keyword arguments forwarded to ``flow_factory``
            (e.g. ``n_layers``, ``n_components``). Defaults to ``{}``.
        init_key: JAX random key used to initialise the flow parameters.
            Defaults to ``jr.PRNGKey(0)`` when not provided.  The SVI
            optimiser will update these parameters, so the exact choice of
            seed only affects the starting point of optimisation.
        prefix: Prefix used for numpyro parameter names. Defaults to
            ``"auto"``.
        init_loc_fn: Per-site initialization strategy. Defaults to
            :func:`~numpyro.infer.initialization.init_to_uniform`.

    Example::

        import jax.numpy as jnp
        import jax.random as jr
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.optim import Adam
        from gauss_flows import FlowGuide

        def model(obs=None):
            mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
            numpyro.sample("obs", dist.Normal(mu, 1.0), obs=obs)

        guide = FlowGuide(model)
        svi = SVI(model, guide, Adam(1e-3), loss=Trace_ELBO())
        result = svi.run(jr.key(0), num_steps=100, obs=jnp.array([1.0]))
    """

    def __init__(
        self,
        model: Callable[..., Any],
        *,
        flow_factory: Callable[..., Transformed] = gaussianization_flow,
        flow_kwargs: dict[str, Any] | None = None,
        init_key: PRNGKeyArray | None = None,
        prefix: str = "auto",
        init_loc_fn: Callable[..., Any] = init_to_uniform,
    ):
        self._flow_factory = flow_factory
        self._flow_kwargs = flow_kwargs if flow_kwargs is not None else {}
        self._init_key = jr.PRNGKey(0) if init_key is None else init_key
        self._flow_init = None
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Build the initial flow now that latent_dim is known.
        self._flow_init = self._flow_factory(
            self._init_key, n_dims=self.latent_dim, **self._flow_kwargs
        )

    def _get_posterior(self):
        flow = register_params(f"{self.prefix}_flow", self._flow_init)
        return FlowDist(flow)

    # sample_posterior is inherited from AutoContinuous and works correctly:
    # it substitutes the numpyro param store with the trained params and
    # then calls _unpack_and_constrain to map latent samples back to the
    # constrained per-site values.


__all__ = ["FlowGuide"]
