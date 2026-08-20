r"""Run any Gaussian filter in warped coordinates.

Filtering a strictly positive, bounded, or otherwise non-Gaussian state in
its native coordinates puts posterior mass outside the support: intervals
cross zero, ensemble members go negative, and the Gaussian likelihood is
evaluated where the state cannot be. `TransformFilter` fixes the
*coordinates* instead of the algorithm — it composes bijections into the
dynamics and observation operators and delegates the actual filtering to
[`gaussx`](https://github.com/jejjohnson/gaussx):

$$
\tilde f = \Gamma_x^{-1} \circ f \circ \Gamma_x, \qquad
\tilde h = \Gamma_y^{-1} \circ h \circ \Gamma_x .
$$

The method is function composition, not a new algorithm, so every inner
filter works unchanged. Warping only the observation preserves conjugacy and
is already covered by `gauss_flows.normalizing_kalman_filter` and
`gauss_flows.ConjugateTransformFilter`; warping the **state** breaks
conjugacy and makes the predict step non-Gaussian, which is why this wrapper
delegates to `gaussx.nonlinear_kalman_filter` — the choice of integrator
there is the choice of filter (EKF, UKF, CKF, GHKF, Monte Carlo).

**What this buys, and what it does not.** On a strictly positive state with
multiplicative noise, filtering in latent coordinates uniformly improves
predictive density and support — credible intervals stop crossing zero and
ensemble members stop going negative — but point-estimate RMSE can get
*worse*, because the latent filter answers a different (better-posed)
question than the physical one. This feature buys calibration and support,
not point accuracy. A user who adopts it expecting RMSE wins will be
disappointed, and will be right to be.

Prior art: Gaussian anamorphosis (Bertino et al., 2003; Simon & Bertino,
2009), the normal-score EnKF (Zhou et al., 2011), and — for the conjugate
observation-only special case — Chipilski (2025), *Exact Nonlinear State
Estimation*, J. Atmos. Sci. 82(4), 809-827.
"""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jax import Array
from jaxtyping import PRNGKeyArray

from gauss_flows._src.flows.kalman import _mixes_channels


def _require_gaussx():
    """Import gaussx, with an actionable error if absent or broken."""
    try:
        # gaussx is deliberately not a declared dependency (see the
        # ImportError below), so it is absent from the typecheck environment.
        import gaussx  # ty: ignore[unresolved-import]
    # Catching Exception, not ImportError: gaussx can be *present* yet fail
    # partway through its own import when its pins are unmet (seen in
    # practice as an AttributeError from a matfree version mismatch).
    # Narrowing to ImportError would let that surface as an opaque traceback
    # from a package the caller never imported.
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "The state-space transform filters need the 'gaussx' package for "
            "their filtering and ensemble analysis steps. gaussx is NOT a "
            "declared dependency of gauss_flows -- it is not published to "
            "PyPI, and the optional `interp` extra's interpax caps lineax at "
            "<=0.1.0 while gaussx needs >=0.1.1. It co-installs cleanly with "
            "core gauss_flows: "
            "`pip install git+https://github.com/jejjohnson/gaussx.git`. "
            f"The underlying failure was {type(exc).__name__}: {exc}"
        ) from exc
    return gaussx


def _identity(value: Array) -> Array:
    return value


def _warped_enkf_analysis(
    warp_state: AbstractBijection | None,
    warp_obs: AbstractBijection | None,
    particles: Array,
    obs_particles: Array,
    observation: Array,
    obs_noise: Any,
    *,
    key: PRNGKeyArray | None = None,
    perturbed_obs: Array | None = None,
    localization: Array | None = None,
) -> Array:
    """Shared warp / EnKF / unwarp plumbing for the ensemble analysis step.

    Used by both `TransformFilter.analysis` and
    `gauss_flows.ConjugateTransformFilter.analysis`; the two classes differ
    only in how they resolve their warps (``None`` means identity here).
    ``obs_noise`` is taken to live in latent observation coordinates.
    """
    gaussx = _require_gaussx()
    inv_state = _identity if warp_state is None else warp_state.inverse
    fwd_state = _identity if warp_state is None else warp_state.transform
    inv_obs = _identity if warp_obs is None else warp_obs.inverse

    latent = jax.vmap(inv_state)(particles)  # (J, N) latent state
    latent_obs = jax.vmap(inv_obs)(obs_particles)  # (J, M)
    latent_observation = inv_obs(observation)  # (M,)
    latent_perturbed = (
        None if perturbed_obs is None else jax.vmap(inv_obs)(perturbed_obs)  # (J, M)
    )

    latent_analysis = gaussx.enkf_analysis(
        latent,
        latent_obs,
        latent_observation,
        obs_noise,
        key=key,
        perturbed_obs=latent_perturbed,
        localization=localization,
    )  # (J, N)
    return jax.vmap(fwd_state)(latent_analysis)  # back to physical


def _broadcast_noise(noise: Any, n_steps: int, dim: int, name: str) -> Array:
    """Materialise a noise covariance and broadcast it along time.

    The trailing shape is checked against ``dim`` rather than only the rank:
    a ``(1, 1)`` covariance would otherwise pass and then silently broadcast
    inside ``cov + noise``.
    """
    dense = noise.as_matrix() if hasattr(noise, "as_matrix") else jnp.asarray(noise)
    if dense.ndim == 2 and dense.shape == (dim, dim):
        return jnp.broadcast_to(dense, (n_steps, dim, dim))
    if dense.ndim == 3 and dense.shape == (n_steps, dim, dim):
        return dense
    raise ValueError(
        f"{name} must have shape ({dim}, {dim}) or ({n_steps}, {dim}, {dim}); "
        f"got {dense.shape}."
    )


def _symmetrize(matrix: Array) -> Array:
    # Bit-exact identity on an already-symmetric input (x + x and 0.5 * 2x
    # are exact in floating point), so the identity-warp reduction stays
    # bit-for-bit.
    return 0.5 * (matrix + matrix.T)


class TransformFilter(eqx.Module):
    r"""Run a Gaussian filter in warped coordinates.

    Composes ``warp_state`` and ``warp_obs`` into the dynamics and
    observation operators and delegates to gaussx:

    $$
    \tilde f = \Gamma_x^{-1} \circ f \circ \Gamma_x, \qquad
    \tilde h = \Gamma_y^{-1} \circ h \circ \Gamma_x .
    $$

    `filter` runs `gaussx.nonlinear_kalman_filter` on the composed maps (the
    integrator passed through selects EKF / UKF / CKF / GHKF / Monte Carlo);
    `analysis` runs `gaussx.enkf_analysis` for the ensemble case. Latent
    moments are pushed back to physical space with `predictive` (quadrature
    mean and covariance) and `predictive_interval` (transformed quantiles).

    **What this buys, measured — and what it does not.** Filtering a
    positive state in log coordinates removes impossible posterior mass
    (zero credible intervals crossing zero, zero negative ensemble members)
    and improves predictive density, but point RMSE can *worsen* relative to
    the physical-space filter. Calibration and support, not point accuracy.

    A practical note on warps: every filter step calls the warps' forward
    and inverse maps, so a ``warp_state`` whose inverse is itself iterative
    (`flowjax.bijections.NumericalInverse`, root-finding splines) puts a
    root-find inside the innermost filter loop — a performance cliff.
    Closed-form warps (``Exp``, ``Affine`` chains, RBIG's) are what this
    wrapper is designed around.

    Attributes:
        warp_state: Bijection, latent → physical state, event shape ``(N,)``.
            ``None`` for an observation-only warp, which preserves conjugacy
            and reduces to the conjugate case.
        warp_obs: Bijection, latent → physical observation, event shape
            ``(M,)``. ``None`` for a state-only warp.
        noise_frame: Where the noise covariances are Gaussian — ``"latent"``
            (default) or ``"physical"``. With ``"latent"`` the noise is
            Gaussian in warped coordinates, so physical noise is
            multiplicative or otherwise support-respecting for free; this is
            the recommended default and costs nothing. ``"physical"``
            re-expresses Gaussian physical noise in latent coordinates with
            one extra linearisation per step, and is approximate even for
            linear dynamics.

    Shape:
        - ``filter`` observations: ``(T, M)`` physical; init moments
          ``(N,)`` / ``(N, N)`` **latent**; returned moments latent.
        - ``analysis`` particles: ``(J, N)`` in, ``(J, N)`` out, physical.
        - ``predictive`` / ``predictive_interval``: ``(N,)`` / ``(N, N)``
          latent in, physical out; a leading time axis is vmapped.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from flowjax.bijections import Exp
        >>> from gauss_flows import TransformFilter
        >>> # A strictly positive 2-D state filtered in log coordinates.
        >>> filt = TransformFilter(warp_state=Exp(shape=(2,)))
        >>> growth = lambda x: x * jnp.exp(0.3 * (1.0 - x / 3.0))
        >>> observations = jnp.exp(jr.normal(jr.key(0), (40, 2)))  # (T, M)
        >>> state = filt.filter(
        ...     growth,                     # physical dynamics, (N,) -> (N,)
        ...     lambda x: x,                # physical observation operator
        ...     0.05 * jnp.eye(2),          # Q, latent frame
        ...     0.10 * jnp.eye(2),          # R, latent frame
        ...     observations,
        ...     jnp.zeros(2),               # latent init mean
        ...     jnp.eye(2),                 # latent init cov
        ... )
        >>> mean, cov = filt.predictive(
        ...     state.filtered_means, state.filtered_covs
        ... )  # physical-space moments, (T, 2) / (T, 2, 2)
        >>> lo, hi = filt.predictive_interval(
        ...     state.filtered_means, state.filtered_covs
        ... )  # strictly positive by construction
    """

    warp_state: AbstractBijection | None = None
    warp_obs: AbstractBijection | None = None
    noise_frame: Literal["latent", "physical"] = "latent"

    def __check_init__(self):
        if self.noise_frame not in ("latent", "physical"):
            raise ValueError(
                f"noise_frame must be 'latent' or 'physical'; got {self.noise_frame!r}."
            )

    def _check_channel_mask(self, mask: Array | None, n_obs: int) -> None:
        """Refuse a per-channel mask when the observation warp mixes channels.

        ``warp_obs.inverse`` runs on the raw observation vector *before* the
        filter's mask machinery sees it, so with a channel-mixing warp the
        placeholder garbage in masked channels contaminates the observed
        ones. An elementwise warp keeps each channel's garbage in its own
        channel, where the mask then removes it.
        """
        if mask is None or mask.ndim != 2:
            return
        if self.warp_obs is not None and _mixes_channels(self.warp_obs):
            raise ValueError(
                "A per-channel observation mask cannot be combined with a "
                "warp_obs that mixes channels: the warp runs on the raw "
                "observation vector before masking, so placeholder values in "
                "masked channels would contaminate the observed ones. Use an "
                "elementwise warp_obs, or a per-step (T,) mask."
            )
        del n_obs

    def filter(
        self,
        dynamics: Any,
        obs_fn: Any,
        process_noise: Any,
        obs_noise: Any,
        observations: Array,
        init_mean: Array,
        init_cov: Array,
        *,
        integrator: Any | None = None,
        mask: Array | None = None,
        joseph: bool = True,
        solver: Any | None = None,
    ) -> Any:
        r"""Filter in latent coordinates. Moments returned are latent.

        ``dynamics``, ``obs_fn``, and ``observations`` are all supplied in
        **physical** coordinates; the composition and the warping of the
        data are internal. The initial moments are **latent** — they
        describe a Gaussian, and latent space is where the Gaussian lives.
        Push the returned moments to physical space with `predictive` /
        `predictive_interval`.

        Args:
            dynamics: Physical state transition ``(N,) -> (N,)``.
            obs_fn: Physical observation operator ``(N,) -> (M,)``.
            process_noise: Process covariance, ``(N, N)`` or ``(T, N, N)``
                or a lineax operator — Gaussian in the frame selected by
                ``noise_frame``.
            obs_noise: Observation covariance, ``(M, M)`` or ``(T, M, M)``
                or a lineax operator, in the ``noise_frame`` frame.
            observations: Physical observations, shape ``(T, M)``.
            init_mean: Latent initial mean, shape ``(N,)``.
            init_cov: Latent initial covariance, shape ``(N, N)``.
            integrator: gaussx integrator; the choice of integrator is the
                choice of filter. Defaults to gaussx's default (unscented).
            mask: Optional ``(T,)`` step mask or ``(T, M)`` channel mask,
                forwarded to the inner filter. A ``(T, M)`` mask requires an
                elementwise ``warp_obs`` (see Raises).
            joseph: Use the Joseph-form covariance update. Default ``True``.
            solver: gaussx solver strategy for the innovation solve.

        Returns:
            ``gaussx.FilterState`` with **latent** filtered and predicted
            moments and the latent-space log-likelihood.

        Raises:
            ImportError: If gaussx is not installed.
            ValueError: If a ``(T, M)`` channel mask is combined with a
                channel-mixing ``warp_obs``, or a noise shape is wrong.
        """
        gaussx = _require_gaussx()
        observations = jnp.asarray(observations)
        self._check_channel_mask(mask, observations.shape[-1])

        fwd_state = _identity if self.warp_state is None else self.warp_state.transform
        inv_state = _identity if self.warp_state is None else self.warp_state.inverse
        inv_obs = _identity if self.warp_obs is None else self.warp_obs.inverse

        def latent_dynamics(z: Array) -> Array:
            # z: (N,) latent -> physical -> stepped physical -> latent
            return inv_state(dynamics(fwd_state(z)))

        def latent_obs_fn(z: Array) -> Array:
            # z: (N,) latent state -> (M,) latent observation
            return inv_obs(obs_fn(fwd_state(z)))

        latent_observations = jax.vmap(inv_obs)(observations)  # (T, M)

        if self.noise_frame == "latent":
            return gaussx.nonlinear_kalman_filter(
                latent_dynamics,
                latent_obs_fn,
                process_noise,
                obs_noise,
                latent_observations,
                init_mean,
                init_cov,
                integrator=integrator,
                mask=mask,
                joseph=joseph,
                solver=solver,
            )

        return self._filter_physical_noise(
            gaussx,
            latent_dynamics,
            latent_obs_fn,
            dynamics,
            obs_fn,
            process_noise,
            obs_noise,
            latent_observations,
            init_mean,
            init_cov,
            integrator=integrator,
            mask=mask,
            joseph=joseph,
            solver=solver,
        )

    def _filter_physical_noise(
        self,
        gaussx: Any,
        latent_dynamics: Any,
        latent_obs_fn: Any,
        dynamics: Any,
        obs_fn: Any,
        process_noise: Any,
        obs_noise: Any,
        latent_observations: Array,
        init_mean: Array,
        init_cov: Array,
        *,
        integrator: Any | None,
        mask: Array | None,
        joseph: bool,
        solver: Any | None,
    ) -> Any:
        r"""The ``noise_frame="physical"`` loop: one linearisation per step.

        The model is $x' = f(x) + w$, $y = h(x) + v$ with $w, v$ Gaussian in
        **physical** coordinates. In latent coordinates the noise enters
        through $\Gamma^{-1}$, so it is re-expressed by linearising at the
        predicted physical mean:

        $$
        \tilde Q_t = J_x Q_t J_x^\top, \quad
        J_x = \partial \Gamma_x^{-1}(\hat x_t), \qquad
        \tilde R_t = J_y R_t J_y^\top, \quad
        J_y = \partial \Gamma_y^{-1}(\hat y_t),
        $$

        with $\hat x_t = \Gamma_x(m^-_t)$ and $\hat y_t = h(\hat x_t)$. This
        is a first-order approximation even when the dynamics are linear.
        For an identity warp both Jacobians are exactly the identity, so the
        loop reduces bit-for-bit to the ``"latent"`` path.
        """
        n_steps, n_obs = latent_observations.shape
        n_state = init_mean.shape[0]
        q_seq = _broadcast_noise(process_noise, n_steps, n_state, "process_noise")
        r_seq = _broadcast_noise(obs_noise, n_steps, n_obs, "obs_noise")
        channel_mask = mask is not None and mask.ndim == 2
        if mask is None:
            mask_seq = jnp.ones((n_steps, 1), dtype=bool)
        else:
            mask_seq = jnp.asarray(mask)
            if mask_seq.ndim == 1:
                # (T,) step gate -> (T, 1) so the scan slice stays indexable.
                mask_seq = mask_seq[:, None]

        fwd_state = _identity if self.warp_state is None else self.warp_state.transform
        inv_state = _identity if self.warp_state is None else self.warp_state.inverse
        inv_obs = _identity if self.warp_obs is None else self.warp_obs.inverse
        no_noise = jnp.zeros((n_state, n_state), dtype=init_cov.dtype)

        def step(carry, inputs):
            mean, cov, ll = carry
            q_t, r_t, y_t, mask_t = inputs

            mean_pred, cov_dyn = gaussx.nonlinear_kalman_predict(
                latent_dynamics, mean, cov, no_noise, integrator=integrator
            )
            # Linearise the warps at the predicted physical mean.
            x_hat = fwd_state(mean_pred)  # (N,) physical predicted state
            jac_state = jax.jacobian(inv_state)(x_hat)  # (N, N)
            cov_pred = cov_dyn + _symmetrize(jac_state @ q_t @ jac_state.T)

            y_hat = obs_fn(x_hat)  # (M,) physical predicted observation
            jac_obs = jax.jacobian(inv_obs)(y_hat)  # (M, M)
            r_lat = _symmetrize(jac_obs @ r_t @ jac_obs.T)

            def _update(_):
                return gaussx.nonlinear_kalman_update(
                    latent_obs_fn,
                    mean_pred,
                    cov_pred,
                    y_t,
                    r_lat,
                    integrator=integrator,
                    mask=mask_t if channel_mask else None,
                    joseph=joseph,
                    solver=solver,
                )

            def _skip(_):
                # Gated-off step: keep the prediction, contribute no
                # likelihood — same convention as the inner filter.
                return mean_pred, cov_pred, jnp.zeros((), dtype=cov_pred.dtype)

            if channel_mask:
                mean_new, cov_new, ll_inc = _update(None)
            else:
                mean_new, cov_new, ll_inc = jax.lax.cond(
                    mask_t[0], _update, _skip, operand=None
                )

            carry_new = (mean_new, cov_new, ll + ll_inc)
            return carry_new, (mean_new, cov_new, mean_pred, cov_pred)

        init_carry = (init_mean, init_cov, jnp.zeros((), dtype=init_cov.dtype))
        final_carry, (f_means, f_covs, p_means, p_covs) = jax.lax.scan(
            step, init_carry, (q_seq, r_seq, latent_observations, mask_seq)
        )
        return gaussx.FilterState(
            filtered_means=f_means,
            filtered_covs=f_covs,
            predicted_means=p_means,
            predicted_covs=p_covs,
            log_likelihood=final_carry[2],
        )

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
        r"""Ensemble Kalman analysis in latent coordinates.

        The ensemble leg: warp the prior ensemble into latent coordinates,
        run `gaussx.enkf_analysis` there, warp the posterior ensemble back.
        Everything is supplied and returned in **physical** coordinates.

        ``obs_noise`` follows ``noise_frame``: with ``"latent"`` (default)
        it is the observation covariance in latent observation coordinates;
        with ``"physical"`` it is Gaussian in physical coordinates and is
        re-expressed by linearising $\Gamma_y^{-1}$ at the observation.

        Args:
            particles: Prior ensemble in physical state space, ``(J, N)``.
            obs_particles: Image of the ensemble under the observation
                operator, physical, ``(J, M)``.
            observation: The observation, physical, ``(M,)``.
            obs_noise: Observation error covariance, ``(M, M)`` array or
                lineax operator, in the ``noise_frame`` frame.
            key: PRNG key for the observation perturbations, drawn in latent
                space. Mutually exclusive with ``perturbed_obs``.
            perturbed_obs: Pre-built perturbed observations in **physical**
                space, ``(J, M)``; warped into latent coordinates before
                use. Mutually exclusive with ``key``.
            localization: Optional state-observation taper, ``(N, M)``,
                applied in latent coordinates.

        Returns:
            Posterior ensemble in physical state space, ``(J, N)``.

        Raises:
            ImportError: If gaussx is not installed.
            ValueError: If a warp's event shape does not match its ensemble.
        """
        particles = jnp.asarray(particles)
        obs_particles = jnp.asarray(obs_particles)
        if self.warp_state is not None and self.warp_state.shape != particles.shape[1:]:
            raise ValueError(
                f"warp_state has event shape {self.warp_state.shape}, but "
                f"particles has N={particles.shape[1]} channels."
            )
        if self.warp_obs is not None and self.warp_obs.shape != obs_particles.shape[1:]:
            raise ValueError(
                f"warp_obs has event shape {self.warp_obs.shape}, but "
                f"obs_particles has M={obs_particles.shape[1]} channels."
            )

        if self.noise_frame == "physical":
            # lineax is not a declared gauss_flows dependency, but it is a
            # gaussx one, and this path already requires gaussx.
            _require_gaussx()
            import lineax as lx

            inv_obs = _identity if self.warp_obs is None else self.warp_obs.inverse
            jac_obs = jax.jacobian(inv_obs)(jnp.asarray(observation))  # (M, M)
            dense = (
                obs_noise.as_matrix()
                if hasattr(obs_noise, "as_matrix")
                else jnp.asarray(obs_noise)
            )
            obs_noise = lx.MatrixLinearOperator(
                _symmetrize(jac_obs @ dense @ jac_obs.T),
                lx.positive_semidefinite_tag,
            )

        return _warped_enkf_analysis(
            self.warp_state,
            self.warp_obs,
            particles,
            obs_particles,
            observation,
            obs_noise,
            key=key,
            perturbed_obs=perturbed_obs,
            localization=localization,
        )

    def predictive(
        self, mean: Array, cov: Array, *, order: int = 32
    ) -> tuple[Array, Array]:
        r"""Push latent moments to physical space by Gauss-Hermite quadrature.

        Returns $(\mathbb{E}[x], \mathrm{Cov}[x])$ for
        $x = \Gamma_x(z)$, $z \sim \mathcal{N}(m, P)$ — **not**
        $\Gamma_x(m)$, which is the pushforward *median*: for a non-affine
        warp the two differ, and comparing $\Gamma_x(m)$ against a
        physical-space mean silently mixes estimands.

        The quadrature grid is a tensor product, ``order ** N`` points, so
        this is practical for small state dimension (``N <= ~5``). Lower
        ``order`` before reaching for it in higher dimension.

        Args:
            mean: Latent mean, shape ``(N,)`` or ``(T, N)``.
            cov: Latent covariance, shape ``(N, N)`` or ``(T, N, N)``.
            order: Gauss-Hermite points per dimension. Default ``32``.

        Returns:
            Tuple ``(physical_mean, physical_cov)`` with the input's leading
            shape. With ``warp_state=None`` the inputs are returned as-is.
        """
        if self.warp_state is None:
            return mean, cov
        gaussx = _require_gaussx()
        integrator = gaussx.GaussHermiteIntegrator(order=order)
        transform = self.warp_state.transform

        def push(m: Array, p: Array) -> tuple[Array, Array]:
            # m: (N,), p: (N, N) latent -> physical mean / cov
            phys_mean, phys_cov, _ = gaussx.moment_transform(
                transform, m, p, integrator=integrator
            )
            return phys_mean, phys_cov

        if jnp.asarray(mean).ndim == 2:
            return jax.vmap(push)(mean, cov)
        return push(mean, cov)

    def predictive_interval(
        self, mean: Array, cov: Array, *, level: float = 0.95
    ) -> tuple[Array, Array]:
        r"""Per-dimension credible interval via transformed quantiles.

        Computes $\Gamma_x(m \pm z_\alpha s)$ with $s$ the latent marginal
        standard deviations — the latent Gaussian quantiles pushed through
        the warp. Monotone maps preserve quantiles, so the result is an
        exact equal-tailed interval in physical space and it respects the
        support by construction. Deliberately offers **no** symmetric
        moment-based option: $\hat\mu \pm z_\alpha \hat\sigma$ in physical
        space is what reintroduces impossible values.

        Requires an elementwise ``warp_state`` (quantiles do not commute
        with channel mixing). A decreasing warp flips the endpoints; they
        are returned sorted.

        Args:
            mean: Latent mean, shape ``(N,)`` or ``(T, N)``.
            cov: Latent covariance, shape ``(N, N)`` or ``(T, N, N)``.
            level: Two-sided coverage level in ``(0, 1)``. Default ``0.95``.

        Returns:
            Tuple ``(lower, upper)``, each with ``mean``'s shape, physical.

        Raises:
            ValueError: If ``warp_state`` mixes channels, or ``level`` is
                outside ``(0, 1)``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1); got {level}.")
        if self.warp_state is not None and _mixes_channels(self.warp_state):
            raise ValueError(
                "predictive_interval needs an elementwise warp_state: "
                "per-dimension quantiles do not commute with a warp that "
                "mixes channels. Use predictive() for moments, or sample."
            )
        mean = jnp.asarray(mean)
        cov = jnp.asarray(cov)
        z_alpha = jnp.sqrt(2.0) * jax.scipy.special.erfinv(level)
        # scale: marginal std devs, matching mean's shape
        scale = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))
        lo_latent = mean - z_alpha * scale
        hi_latent = mean + z_alpha * scale

        transform = _identity if self.warp_state is None else self.warp_state.transform
        if mean.ndim == 2:
            transform = jax.vmap(transform)
        endpoint_a = transform(lo_latent)
        endpoint_b = transform(hi_latent)
        return jnp.minimum(endpoint_a, endpoint_b), jnp.maximum(endpoint_a, endpoint_b)


__all__ = ["TransformFilter"]
