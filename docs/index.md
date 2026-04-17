# gauss_flows

> JAX/FlowJax-based Gaussianization flows, SurVAE transforms, and information-theoretic utilities.

`gauss_flows` is a specialised extension of [FlowJax](https://danielward27.github.io/flowjax/) providing:

- **Gaussianization transforms** — alternating marginal CDF transforms and learned rotations (the RBIG pattern)
- **SurVAE hierarchy** — bijections, surjections, and stochastic transforms with exact or lower-bound likelihoods
- **Information theory** — first-class `entropy`, `mutual_information`, `total_correlation`, `kl_divergence`, `negentropy`
- **NumPyro integration** — `FlowDist` (Distribution wrapper) and `FlowGuide` (`AutoContinuous` variational guide)

## Package layout

The implementation follows the SurVAE Flows taxonomy (Nielsen et al. 2020).
Every transform is classified as a bijection, surjection, or stochastic
transform; each category is further split by technique:

```text
_src/
├── transforms/
│   ├── base.py                        # AbstractSurjection / AbstractStochastic / _IdentitySurjection
│   ├── bijections/
│   │   ├── coupling/                  # Affine / Spline / CircularSpline / DeepSigmoid
│   │   ├── elementwise/               # Histogram / InverseGauss / MixtureCDFs / (Circular)Spline
│   │   ├── linear/                    # Rotation / LU / Conv1x1 / ConvExp / Haar
│   │   ├── normalization/             # ActNorm(2D+1D) / BatchNorm
│   │   ├── reshape/                   # Squeeze
│   │   ├── periodic/                  # PeriodicShift / PeriodicWrap (+ shared _utils)
│   │   └── classic/                   # PlanarFlow / SylvesterFlow
│   ├── surjections/
│   │   ├── dimension/                 # Augment / Slice
│   │   ├── abs.py / sort.py / pool.py
│   └── stochastic/
│       └── vae.py / permutation.py
├── flows/
│   ├── survae.py                      # SurVAEFlow container
│   ├── gaussianization.py             # gaussianization_flow, coupling_gaussianization_flow
│   └── rbig.py                        # iterative_rbig
├── inference/
│   ├── numpyro_compat.py              # FlowDist
│   ├── numpyro_guide.py               # FlowGuide
│   └── train.py                       # fit_gaussianization_flow
└── nn/                                # reserved placeholder for future NN building blocks
```

The public `gauss_flows.*` surface re-exports every leaf class; import
from `gauss_flows` directly rather than the `_src/` paths.

## Installation

```bash
uv add gauss_flows
```

Or with pip:

```bash
pip install gauss_flows
```

## Quickstart

```python
import jax
from gauss_flows import gaussianization_flow, fit_gaussianization_flow, entropy

key = jax.random.key(0)
flow = gaussianization_flow(key, n_dims=4, n_layers=8, n_components=8)

# Sample / score
samples = flow.sample(key, (100,))
log_probs = flow.log_prob(samples)

# Information theory
H = entropy(flow, n_samples=10_000, key=key)
```

## Links

- [API Reference](api/reference.md)
- [GitHub](https://github.com/jejjohnson/gauss_flows)
- [Changelog](https://github.com/jejjohnson/gauss_flows/blob/main/CHANGELOG.md)
