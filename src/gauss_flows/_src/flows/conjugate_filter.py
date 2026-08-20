r"""Ensemble Kalman analysis performed in a Gaussianised latent space.

The Ensemble Conjugate Transform Filter of Chipilski (2025). The ensemble
Kalman filter's Gaussian update is not wrong in itself — it is applied in the
wrong **coordinates**. If the prior is the pushforward of a Gaussian under a
bijection $\Gamma$, and the observation model is affine with additive Gaussian
noise in that same latent space, then the Kalman update performed in latent
space and mapped back is *exactly Bayes* in the population limit, not an
approximation:

$$
X^a = \Gamma\!\left(\mathrm{EnKF}\!\left(\Gamma^{-1}(X^f),\;
      \Gamma^{-1}(Y^f),\; \Gamma^{-1}(y)\right)\right).
$$

What that buys, on the reference paper's lognormal / logit-normal example: the
physical-space update plateaus several percent away from the exact posterior
mean and stays there as the ensemble grows, because the error is one of
coordinates rather than of sampling. Conjugating the update with $\Gamma$
removes it, and the remaining error does shrink with ensemble size.

`rbig_conjugate_filter` pairs this with RBIG, which is unusually well suited to
supplying $\Gamma$: it produces a closed-form Gaussianising map from the prior
ensemble with no gradient training, which is exactly what the method needs and
exactly what the reference implementation hand-specifies per experiment
(a hardcoded ``exp`` / ``logistic`` pair).

Note on direction: flowjax's ``transform`` maps base → data, so ``warp.transform``
is latent → physical and ``warp.inverse`` is physical → latent.

References:
    Chipilski, H. G. (2025). Exact Nonlinear State Estimation.
    *Journal of the Atmospheric Sciences* 82(4), 809–827.
    doi:10.1175/JAS-D-24-0171.1
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import PRNGKeyArray

# The lazy gaussx import and the warp / EnKF / unwarp plumbing are shared
# with `TransformFilter` -- this class is its observation-warp special case.
# `_require_gaussx` is re-exported here because this module's tests exercise
# the import-failure path through it.
from gauss_flows._src.flows.transform_filter import (
    _require_gaussx as _require_gaussx,
    _warped_enkf_analysis,
)
from gauss_flows._src.init.rbig import fit_rbig


class ConjugateTransformFilter(eqx.Module):
    r"""Ensemble Kalman analysis performed in a Gaussianised latent space.

    Implements the Ensemble Conjugate Transform Filter of Chipilski (2025).
    Given a bijection $\Gamma$ that maps a Gaussian latent space onto the
    physical state space, `analysis` warps the ensemble into latent
    coordinates, applies `gaussx.enkf_analysis` there, and warps the result
    back. See the module docstring for the accuracy argument.

    **Exactness caveat.** The update is exact Bayes only in the population
    limit, and only when the prior really is the pushforward of a Gaussian under
    ``warp`` *and* the observation model is affine with additive Gaussian noise
    in those same latent coordinates. With a finite ensemble the gain is
    empirical and the perturbations are Monte Carlo, so even then the result is
    an estimate. "Gaussian likelihood" is not sufficient on its own: a model
    like $y = \zeta^2 + \varepsilon$ has Gaussian noise but a non-Gaussian
    posterior this update does not reproduce.

    When the assumption does not hold, this is an approximation with **no
    guaranteed ordering** against the physical-space update. A poorly matched
    warp can make the latent joint *less* Gaussian and give a worse answer than
    doing nothing — "Gaussianise first" is not monotone. It typically helps, and
    helps a great deal when the prior is genuinely a warped Gaussian (see the
    module docstring), but a degraded result is an expected possibility rather
    than a sign of a bug in the implementation or of too few members.
    gauss_flows cannot check the condition for you — nothing in the ensemble
    reveals it — so treat exactness as conditional on a modelling assumption you
    have made, not as a guarantee the library enforces.

    A practical note on ``warp``: `analysis` calls ``warp.inverse`` under
    `jax.vmap` once per ensemble member, so a warp whose inverse is itself
    iterative (`flowjax.bijections.NumericalInverse`, root-finding splines)
    makes the analysis step expensive in proportion to the ensemble size.
    Closed-form warps — RBIG's, or an explicit ``exp`` / ``logistic`` pair —
    are what this method is designed around.

    Attributes:
        warp: Bijection mapping latent (Gaussian) values to physical state
            values, with event shape ``(N,)``.
        obs_warp: Optional separate bijection for observation space, event
            shape ``(M,)``. Defaults to ``None``, meaning ``warp`` is reused —
            correct when state and observations live in the same space.

    Shape:
        - ``analysis`` particles: ``(J, N)`` in, ``(J, N)`` out
        - ``analysis`` obs_particles: ``(J, M)``
        - ``analysis`` observation: ``(M,)``

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> from flowjax.bijections import Exp
        >>> from gauss_flows import ConjugateTransformFilter
        >>> # A strictly-positive state: the prior is lognormal in physical
        >>> # space, Gaussian under log.
        >>> filt = ConjugateTransformFilter(warp=Exp(shape=(2,)))
        >>> prior = jnp.exp(jr.normal(jr.key(0), (256, 2)))       # (J, N)
        >>> noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> posterior = filt.analysis(
        ...     prior, prior, jnp.array([1.0, 2.0]), noise, key=jr.key(1)
        ... )
        >>> posterior.shape
        (256, 2)
    """

    warp: AbstractBijection
    obs_warp: AbstractBijection | None = None

    def analysis(
        self,
        particles: Array,
        obs_particles: Array,
        observation: Array,
        obs_noise: Any,
        *,
        key: PRNGKeyArray | None = None,
        perturbed_obs: Array | None = None,
        localization: Array | None = None,
    ) -> Array:
        r"""Analysis step. Returns the posterior ensemble in physical space.

        Everything is supplied in **physical** coordinates and returned in
        physical coordinates; the latent detour is internal. ``obs_noise`` is
        the observation error covariance **in latent observation coordinates**,
        because that is where the Gaussian likelihood is assumed to live.

        Args:
            particles: Prior ensemble in physical state space, shape ``(J, N)``.
            obs_particles: Prior ensemble in physical observation space, i.e.
                the image of ``particles`` under the observation operator,
                shape ``(J, M)``.
            observation: The observation in physical space, shape ``(M,)``.
            obs_noise: Observation error covariance operator in *latent*
                observation coordinates, a ``lineax`` operator of shape
                ``(M, M)``.
            key: PRNG key, forwarded to `gaussx.enkf_analysis` to draw the
                observation perturbations in latent space. Mutually exclusive
                with ``perturbed_obs``.
            perturbed_obs: Pre-built perturbed observation ensemble in
                **physical** space, shape ``(J, M)``. Warped into latent
                coordinates before use. Pass this to compare against a
                physical-space filter on one shared noise realisation.
                Mutually exclusive with ``key``.
            localization: Optional state-observation taper, shape ``(N, M)``,
                forwarded to `gaussx.enkf_analysis`. It applies in latent
                coordinates, where the analysis happens.

        Returns:
            Posterior ensemble in physical state space, shape ``(J, N)``.

        Raises:
            ImportError: If ``gaussx`` is not installed.
            ValueError: If the observation-space warp's event shape does not
                match ``obs_particles``, or if neither or both of ``key`` /
                ``perturbed_obs`` are given (the latter raised by
                `gaussx.enkf_analysis`).
        """
        obs_warp = self.warp if self.obs_warp is None else self.obs_warp

        if obs_warp.shape != obs_particles.shape[1:]:
            detail = (
                "obs_warp"
                if self.obs_warp is not None
                else "warp, which is reused for the observation space because "
                "obs_warp is None"
            )
            raise ValueError(
                f"{detail} has event shape {obs_warp.shape}, but obs_particles "
                f"has M={obs_particles.shape[1]} channels. When the observation "
                "space differs from the state space, pass a matching obs_warp "
                "— the Gaussianising map for the observed quantity, not for "
                "the whole state."
            )

        return _warped_enkf_analysis(
            self.warp,
            obs_warp,
            particles,
            obs_particles,
            observation,
            obs_noise,
            key=key,
            perturbed_obs=perturbed_obs,
            localization=localization,
        )


def rbig_conjugate_filter(
    particles: Array,
    *,
    n_layers: int = 10,
    n_components: int = 8,
    random_state: int = 0,
) -> ConjugateTransformFilter:
    r"""Fit a Gaussianising warp to a prior ensemble with RBIG.

    Convenience constructor. RBIG gives a closed-form $\Gamma$ from the
    ensemble with no gradient-based training, which is what the conjugate
    transform needs and what the reference implementation specifies by hand
    for each experiment. The fitted warp is used for both the state and the
    observation space.

    Unlike a hand-specified warp, an RBIG warp is *fitted*, so the resulting
    filter is an approximation even when the prior really is a pushforward of a
    Gaussian — expect a bias reduction rather than the exact update.

    Args:
        particles: Prior ensemble in physical state space, shape ``(J, N)``.
            Used only to fit the warp.
        n_layers: Number of RBIG ``(rotation, marginal)`` blocks. Defaults
            to 10.
        n_components: Mixture components per marginal layer. Defaults to 8.
        random_state: Base seed for the per-dimension GMM fits. RBIG's fit is
            deterministic given this seed, so no PRNG key is taken.

    Returns:
        A `ConjugateTransformFilter` whose ``warp`` Gaussianises ``particles``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import rbig_conjugate_filter
        >>> prior = jnp.exp(jr.normal(jr.key(0), (512, 2)))       # lognormal
        >>> filt = rbig_conjugate_filter(prior, n_layers=4, n_components=4)
        >>> filt.warp.shape
        (2,)
    """
    flow = fit_rbig(
        particles,
        n_layers=n_layers,
        n_components=n_components,
        random_state=random_state,
    )
    return ConjugateTransformFilter(warp=flow.bijection)


__all__ = ["ConjugateTransformFilter", "rbig_conjugate_filter"]
