# gauss_flows

[![Tests](https://github.com/jejjohnson/gauss_flows/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/gauss_flows/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/gauss_flows/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/gauss_flows/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/gauss_flows/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/gauss_flows/actions/workflows/typecheck.yml)
[![codecov](https://codecov.io/gh/jejjohnson/gauss_flows/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/gauss_flows)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Gaussianization flows in JAX** — density estimation, normalizing flows, and
information-theoretic measures, built on [FlowJax](https://github.com/danielward27/flowjax)
and compatible with [NumPyro](https://num.pyro.ai/). `gauss_flows` implements
Rotation-Based Iterative Gaussianization (RBIG) and related flows that map data to a standard
normal by alternating marginal CDF transforms with learned rotations, plus the
[SurVAE](https://arxiv.org/abs/2007.02731) hierarchy of bijections, surjections, and
stochastic transforms.

## Install

Not yet on PyPI — install from source:

```bash
pip install git+https://github.com/jejjohnson/gauss_flows
```

For development, see [Contributing](#contributing).

## What's inside

| Module | Contents |
|--------|----------|
| Bijections — coupling | `AffineCoupling`, `GINCoupling`, `RQSplineCoupling`, `CircularRQSplineCoupling`, `DeepSigmoidCoupling`, `MixtureGaussianCDFCoupling`, `ContinuousAffineCoupling` |
| Bijections — marginal | `MixtureGaussianCDF`, `MixtureLogisticCDF`, `RQSplineMarginal`, `CircularRationalQuadraticSpline`, `HistogramCDF`, `InverseGaussCDF` |
| Bijections — linear | `HouseholderRotation`, `OrthogonalRotation`, `FixedRotation`, `LULinearPermute`, `Invertible1x1Conv`, `OrthogonalConvExponential`, `MatrixExponential`, `HaarWavelet` |
| Bijections — other | `ActNorm`/`BatchNorm`, `PeriodicShift`/`PeriodicWrap`, `Squeeze`, `PlanarFlow`/`SylvesterFlow`, `FFJORD` (continuous) |
| Surjections & stochastic | `SimpleAbsSurjection`, `SimpleSortSurjection`, `SimpleMaxPoolSurjection2d`, `Augment`/`Slice`, `VAE`, `StochasticPermutation` |
| Base distributions | `GaussianPCA`, `GaussianMixture`, `ConditionalDiagGaussian`, `ClassCondDiagGaussian`, `VonMisesFisher`, `UniformOnSphere` |
| Flows & RBIG | `SurVAEFlow`, `gaussianization_flow`, `coupling_gaussianization_flow`, `iterative_rbig`, `fit_rbig`, `fit_rbig_coupling` |
| NumPyro inference | `FlowDist`, `FlowGuide`, `fit_gaussianization_flow` |
| Information theory | `entropy`, `total_correlation`, `mutual_information`, `kl_divergence`, `negentropy` (+ analytical & RBIG-way variants) |

The public `gauss_flows.*` surface re-exports every leaf class and function, so the internal
`_src/` layout is just organisation — import from `gauss_flows` directly.

## Quick start

Build a Gaussianization flow, fit it to data, then read off a density and an entropy estimate:

```python
import jax.random as jr
from gauss_flows import gaussianization_flow, fit_gaussianization_flow, entropy

key = jr.key(0)

# Build an 8-block Gaussianization flow over 10-D data.
flow = gaussianization_flow(key, n_dims=10, n_layers=8, n_components=8)

# Fit to samples by maximum likelihood.
data = jr.normal(key, (1000, 10))
trained_flow, losses = fit_gaussianization_flow(key, flow, data)

log_probs = trained_flow.log_prob(data)          # (1000,)
h = entropy(trained_flow, n_samples=10_000, key=jr.key(1))   # scalar, in nats
print(f"Entropy: {h:.3f} nats")
```

Transforms act on a **single event** (`x.shape == transform.shape`) and return a scalar
`log_det`; batch with `jax.vmap`:

```python
import jax, jax.numpy as jnp, jax.random as jr
from gauss_flows import MixtureGaussianCDF, HouseholderRotation, RQSplineCoupling

marginal = MixtureGaussianCDF(n_components=8, shape=(10,))   # per-dim Gaussianization
rotation = HouseholderRotation(n_reflections=10, shape=(10,))  # decorrelating rotation
spline = RQSplineCoupling(jr.key(0), shape=(10,), n_bins=8)  # expressive coupling

x = jr.normal(jr.key(1), (10,))             # one event
y, log_det = marginal.transform_and_log_det(x)        # y: (10,), log_det: scalar
ys, log_dets = jax.vmap(marginal.transform_and_log_det)(jnp.ones((32, 10)))  # batch
```

For a closed-form **warm start** with no gradient training, fit RBIG directly from data:

```python
from gauss_flows import fit_rbig

flow = fit_rbig(data, n_layers=8, n_components=8)   # per-block GMM CDFs + rotations
log_probs = flow.log_prob(data)
```

`gauss_flows` flows also drop into NumPyro models as a distribution (`FlowDist`) or a
variational guide (`FlowGuide`) — see the [docs](https://jejjohnson.github.io/gauss_flows/).

## Contributing

Built with `uv`, `ruff`, `ty`, `pytest`, and MkDocs.

```bash
make install      # install all dependency groups + pre-commit hooks
make test         # run the test suite (use `make test-fast` to skip slow + integration)
make format       # auto-format and fix lint
make lint         # ruff check
make typecheck    # ty check
make docs-serve   # preview docs locally
```

Before committing, all four gates must pass: tests, `ruff check .`,
`ruff format --check .`, and `ty check src/gauss_flows`. See
[`AGENTS.md`](AGENTS.md) for the full workflow and coding conventions.

## References

- Meng, C., Song, Y., Song, J., & Ermon, S. (2020). *Gaussianization Flows*. arXiv:2003.01941
- Laparra, V., Camps-Valls, G., & Malo, J. (2011). *Iterative Gaussianization: From ICA to Random Rotations*. IEEE TNNLS.
- Nielsen, D., Jaini, P., Hoogeboom, E., Winther, O., & Welling, M. (2020). *SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows*. NeurIPS. arXiv:2007.02731
- Ward, D. (2023). *FlowJAX: Distributions and Normalizing Flows in JAX*.

## License

MIT — see [LICENSE](LICENSE).

Author: [J. Emmanuel Johnson](https://jejjohnson.netlify.com) ·
Repo: <https://github.com/jejjohnson/gauss_flows>
