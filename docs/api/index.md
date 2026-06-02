# API Reference

`gauss_flows` is a [JAX](https://docs.jax.dev/) / [FlowJax](https://github.com/danielward27/flowjax)
toolkit for **Gaussianization flows** — normalizing flows that iteratively map data to a
standard normal by alternating marginal CDF transforms with learned rotations (the RBIG
pattern), together with the [SurVAE](https://arxiv.org/abs/2007.02731) hierarchy of
bijections, surjections, and stochastic transforms, and the information-theoretic measures
those flows unlock. The reference is organised by theme rather than dumped as one flat page:

| Section | What's inside |
|---------|---------------|
| [Bijections](bijections.md) | Exactly-invertible transforms — coupling, marginal CDFs, linear/rotation, normalisation, periodic, classic residual, and continuous (FFJORD) layers |
| [Surjections & Stochastic](surjections.md) | Deterministic surjections (abs, sort, max-pool, augment/slice) and stochastic transforms (VAE, permutation) with lower-bound likelihoods |
| [Base Distributions](distributions.md) | Latent distributions for the flow base — Gaussian mixtures, low-rank PCA Gaussian, conditional/class-conditional diagonals, and manifold (sphere) distributions |
| [Flows & RBIG](flows.md) | Flow containers (`SurVAEFlow`) and one-call factories — `gaussianization_flow`, `coupling_gaussianization_flow`, `iterative_rbig`, plus the `fit_rbig` warm-start |
| [NumPyro Inference](inference.md) | `FlowDist` distribution wrapper, `FlowGuide` variational guide, and the `fit_gaussianization_flow` training loop |
| [Information Theory](info_theory.md) | Entropy, total correlation, mutual information, KL, negentropy — Monte-Carlo (flow-based), analytical Gaussian closed forms, and RBIG-way reductions |

## Conventions

A few patterns hold across the whole package:

- **Built on FlowJax.** Every bijection subclasses
  [`flowjax.bijections.AbstractBijection`](https://danielward27.github.io/flowjax/);
  base distributions subclass `flowjax`/`equinox` modules. Compose transforms with
  FlowJax [`Chain`](https://danielward27.github.io/flowjax/) / `Scan` and wrap them with
  [`Transformed`](https://danielward27.github.io/flowjax/) — or use the package's
  `SurVAEFlow` container when a chain mixes surjections and stochastic steps.

- **Single-event, not batched.** Transform methods
  (`transform_and_log_det` / `inverse_and_log_det` / `forward_and_log_det`) assume
  `x.shape == self.shape` and return a **scalar** `log_det`. Vectorise over a batch with
  [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) /
  `eqx.filter_vmap`; the `SurVAEFlow` container already vmaps `log_prob` / `sample` over
  any leading `sample_shape`. This matches the FlowJax convention, which runtime-enforces
  `x.shape == bijection.shape`.

    ```python
    import jax, jax.numpy as jnp, jax.random as jr
    from gauss_flows import MixtureGaussianCDF

    t = MixtureGaussianCDF(n_components=8, shape=(4,))
    x = jr.normal(jr.key(0), (4,))          # one event, shape == t.shape
    y, log_det = t.transform_and_log_det(x)  # y: (4,), log_det: scalar

    xs = jr.normal(jr.key(1), (32, 4))       # a batch
    ys, log_dets = jax.vmap(t.transform_and_log_det)(xs)  # (32, 4), (32,)
    ```

- **Immutable Equinox modules.** Transforms and distributions are
  [`equinox.Module`](https://docs.kidger.site/equinox/) pytrees — immutable, JIT/grad/vmap
  friendly. Trainable parameters are array leaves; filter them with
  `eqx.filter(model, eqx.is_inexact_array)` when building an optimiser. Constructors that
  draw random weights take a `key` (`jax.random.PRNGKey`/`jax.random.key`).

- **Unicode math in docstrings.** Docstrings use mathematical Unicode freely
  (`−`, `×`, `σ`, `ℝ`, `∑`) — `RUF002` is ignored project-wide for this reason. Public
  transforms and distributions document a **Shape:** section (per-method input/output
  shapes, single-event convention) and a runnable **Example:** block.

The public `gauss_flows.*` surface re-exports every leaf — import from `gauss_flows`
directly rather than the internal `_src/` paths.
