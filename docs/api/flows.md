# Flows & RBIG

These are the **Layer 1** assemblies — containers that stack transforms into a full flow, and
one-call factories that build a ready-to-train (or already-fitted) Gaussianization flow. A
Gaussianization flow alternates a marginal CDF step $G$ with a rotation $R$,

$$
z \;=\; (R_L \circ G_L)\circ\cdots\circ(R_1 \circ G_1)\,(x),
$$

driving the data toward a standard normal one block at a time. After $L$ blocks the flow is a
density estimator (`log_prob`), a sampler (`sample`), and — because the latent is Gaussian — a
plug-in estimator for the information-theoretic measures in
[Information Theory](info_theory.md).

## Containers

`SurVAEFlow` is the general container: it threads a `key` through a chain that may mix
bijections, surjections, and stochastic transforms, and vmaps `log_prob` / `sample` over any
leading `sample_shape`. For a bijection-only chain it matches `flowjax.Transformed` exactly.

::: gauss_flows.SurVAEFlow

::: gauss_flows.ClassCondFlow

## Flow factories

`gaussianization_flow` builds the classic marginal-mixture + rotation stack;
`coupling_gaussianization_flow` swaps the marginal step for spline coupling layers;
`iterative_rbig` constructs the trainable iterative-RBIG flow. Each returns a FlowJax
`Transformed` distribution.

::: gauss_flows.gaussianization_flow

::: gauss_flows.coupling_gaussianization_flow

::: gauss_flows.iterative_rbig

## RBIG warm-start

Rotation-Based Iterative Gaussianization fits each block in closed form from data — per-dimension
Gaussian-mixture CDFs followed by a random/PCA rotation — giving a strong **warm start** with no
gradient training. The result is an ordinary flow that can be used as-is or fine-tuned with
[`fit_gaussianization_flow`](inference.md#gauss_flows.fit_gaussianization_flow).

::: gauss_flows.fit_rbig

::: gauss_flows.fit_rbig_coupling

## State-space flows

Two constructions that put a normalizing flow and a state-space model together.
Both take their state-space pieces from [`gaussx`](https://github.com/jejjohnson/gaussx),
which is **not a declared dependency** of gauss_flows: gaussx is not published to
PyPI, and its pins currently conflict with this package's (interpax caps `lineax`
at `<=0.1.0` while gaussx needs `>=0.1.1`; gaussx needs `matfree>=0.6` while
`_src/_divergence.py` still calls the pre-0.6 `sampler_rademacher`). Install it
deliberately, into an environment where those are resolved:

```bash
pip install git+https://github.com/jejjohnson/gaussx.git
```

`normalizing_kalman_filter` itself never imports gaussx — it duck-types on the
base distribution's `cond_shape` — so it works with any `(T, M)`-shaped base.
`ConjugateTransformFilter` imports `gaussx.enkf_analysis` lazily and raises an
actionable `ImportError` when it is missing.

### Normalizing Kalman Filter

`normalizing_kalman_filter` composes a linear-Gaussian state-space base with a
per-timestep observation warp (de Bézenac et al., 2020). The base supplies the
density over the latent series and the warp maps each timestep into observation
space, so the Markov structure survives and the Kalman recursion still applies in
latent space.

Two things it exists to get right. First, the construction needs **two nested**
`Vmap`s — one over channels, one over time — and skipping the channel one raises a
shape error that mentions neither `Vmap` nor the axis at fault. The constructor
supplies the time axis and validates the channel axis with an error that names the
fix.

Second, and more consequential: **the default recommendation deviates from the
source paper.** The paper uses a coupling flow specifically to model cross-series
dependence. With a *masked* base — partially-observed series — that is unsound.
Marginalising the unobserved channels commutes with the warp only when the warp's
Jacobian is diagonal; a coupling bijection's is triangular, so a missing entry in
the conditioning half corrupts the transform of channels that **were** observed.
Measured on a 2-layer coupling flow with `M = 6` and ~40% missing, the error on the
observed channels alone had median `0.49` against a signal scale of `0.43` — the
corruption is the size of the signal, and it looks like underfitting rather than a
bug. An elementwise warp gives exactly `0.0`.

So the recommendation is an **elementwise warp with the cross-channel structure in
the state-space model's `H` and `R`**, where the Kalman recursion handles it
exactly. A mask-conditioned warp is permitted; an unconditional channel-mixing warp
over a masked base is refused at construction. An unmasked base places no
restriction on the warp.

Two details that follow from this and are easy to get wrong:

- **The log-det is restricted to the observed channels.** Change of variables
  applies to the coordinates being modelled, so with a masked base the Jacobian
  term is summed under the mask. A plain `Transformed` sums it over all `T × M`
  entries, adding terms for values that were never measured — the density then
  stops being a marginal likelihood, off by a data-dependent offset. The
  constructor wraps elementwise warps to handle this automatically. In the
  mask-conditioned case (2) it cannot: a channel-mixing Jacobian does not
  decompose per channel, so there `log_prob` is a training objective, not a
  marginal likelihood.
- **A warp is classified structurally, never by measurement.** Sampling the
  Jacobian cannot establish diagonality, in two independent ways.
  `OrthogonalRotation` starts with zero Cayley parameters, so it *is* the identity
  when you build it and would pass any probe — then becomes a dense rotation the
  moment an optimiser touches it. And a *parameter-free* bijection need not have a
  constant Jacobian either: a fixed shear that engages only past a threshold looks
  diagonal at every probe point drawn from the bulk and mixes channels in the tail.
  So classification comes from the type. Recognised are flowjax's elementwise
  family and everything in gauss_flows' own `transforms.bijections.elementwise`
  subpackage — `MixtureGaussianCDF`, `MixtureLogisticCDF`, `RQSplineMarginal`,
  `HistogramCDF`, `InverseGaussCDF`. A scalar bijection lifted with
  `Vmap(..., axis_size=M)` is admitted whatever it contains, since 1x1 Jacobian
  blocks are diagonal by construction. For a warp outside all of that, pass
  `assume_elementwise_warp=True` to assert the property yourself; a probe still
  refuses assertions it can immediately disprove, but passing it proves nothing.
- **Mind the warp's direction.** `warp` maps the Gaussian base **to**
  observations. The Gaussianising CDFs go the other way — `MixtureGaussianCDF`
  maps data to Gaussian — so they belong inside an `Invert`:
  `Invert(MixtureGaussianCDF(n_components=8, shape=(M,)))`. Passing one directly
  builds a model of the pushforward through the Gaussianiser rather than of the
  observations. `fit_rbig` already follows this convention, returning
  `Transformed(base, Invert(Chain(...)))`. `RQSplineMarginal` and a lifted
  `RationalQuadraticSpline` are direction-neutral.

::: gauss_flows.normalizing_kalman_filter

### Ensemble Conjugate Transform Filter

`ConjugateTransformFilter` performs the ensemble Kalman analysis in a Gaussianised
latent space (Chipilski, 2025): warp the ensemble, update, warp back. The framing
worth keeping is that **the EnKF's Gaussian assumption is a statement about
coordinates, not about the algorithm**. Applied to a non-Gaussian prior the
physical-space update is biased, and the bias does not shrink with ensemble size —
no number of extra members will fix it. Conjugating the update with a bijection
removes it.

Worth stating the exactness conditions precisely, because they are easy to
over-claim. The conjugated update is exact Bayes only **in the population limit**,
and only when the observation model is **affine with additive Gaussian noise** in
the same latent coordinates that Gaussianise the prior. A merely "Gaussian
likelihood" is not enough — `y = ζ² + ε` has Gaussian noise and a non-Gaussian
posterior no Kalman update reproduces. With a finite ensemble the gain is empirical
and the perturbations are Monte Carlo, so the result is an estimate regardless.
Outside those conditions there is **no guaranteed ordering** against the
physical-space update: it usually helps, and helps a lot when the prior really is a
warped Gaussian, but a badly matched warp can make the latent joint less Gaussian
and do worse. gauss_flows cannot check the condition for you.

On the reference paper's lognormal / logit-normal problem, whose exact posterior
mean is `[0.548062, 0.353937]` (error in the posterior mean, averaged over 4 draws):

| ensemble size `J` | conjugated update | physical-space update |
| --- | --- | --- |
| 2,000 | 0.0030 | 0.0239 |
| 20,000 | 0.0011 | 0.0257 |
| 200,000 | 0.0004 | 0.0260 |

One column converges; the other plateaus.

`rbig_conjugate_filter` fits the warp with RBIG, which suits the method unusually
well: a closed-form Gaussianising map straight from the prior ensemble, no training
loop — exactly the `Γ` the method needs, and exactly what the reference
implementation hand-specifies per experiment.

::: gauss_flows.ConjugateTransformFilter

::: gauss_flows.rbig_conjugate_filter
