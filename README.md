# gauss_flows

**JAX/FlowJax-based RBIG — density estimation and IT measures with normalizing flows**

[![CI](https://github.com/jejjohnson/gauss_flows/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/gauss_flows/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jejjohnson/gauss_flows/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/gauss_flows)

`gauss_flows` implements Rotation-Based Iterative Gaussianization (RBIG) and related
Gaussianization flows in JAX, built on top of the
[FlowJax](https://github.com/danielward27/flowjax) library.

## Features

- **Gaussianization Flows**: Iterative RBIG and coupling-based Gaussianization flows
- **Marginal Transforms**: Mixture Gaussian/Logistic CDFs, rational quadratic splines, histogram CDF
- **Rotation Transforms**: Householder, orthogonal, and fixed rotations
- **Coupling Transforms**: Affine coupling, RQ spline coupling, deep sigmoid coupling
- **Conv Transforms**: Invertible 1x1 convolution, ActNorm, Haar wavelet, Squeeze
- **Information Theory**: Entropy, total correlation, mutual information, KL divergence, negentropy
- **NumPyro Integration**: `FlowDist` wrapper for use in NumPyro probabilistic programs

## Installation

```bash
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Quick Start

```python
import jax.random as jr
from gauss_flows import gaussianization_flow, fit_gaussianization_flow, entropy

key = jr.key(0)

# Build a Gaussianization flow
flow = gaussianization_flow(key, n_dims=10, n_layers=8, n_components=8)

# Fit to data
data = jr.normal(key, (1000, 10))
trained_flow, losses = fit_gaussianization_flow(key, flow, data)

# Compute log-probability
log_probs = trained_flow.log_prob(data)

# Estimate entropy (in nats)
h = entropy(trained_flow, n_samples=10000, key=jr.key(1))
print(f"Entropy: {h:.3f} nats")
```

### Information Theory

```python
from gauss_flows import entropy, total_correlation, kl_divergence, mutual_information

# Entropy
h = entropy(flow, n_samples=10000, key=key)

# Total correlation (multi-information)
tc = total_correlation(flow, n_samples=10000, key=key)

# KL divergence between two distributions
kl = kl_divergence(flow1, flow2, n_samples=10000, key=key)
```

### NumPyro Integration

```python
import numpyro
from gauss_flows import gaussianization_flow, FlowDist

flow = gaussianization_flow(key, n_dims=5)
flow_dist = FlowDist(flow)

# Use in a NumPyro model:
def model(obs=None):
    x = numpyro.sample("x", flow_dist, obs=obs)
```

### Transforms

```python
from gauss_flows import (
    MixtureGaussianCDF,
    HouseholderRotation,
    AffineCoupling,
    CircularRQSplineCoupling,
    PeriodicShift,
    PeriodicWrap,
    RQSplineCoupling,
)

# Marginal Gaussianization
marginal = MixtureGaussianCDF(n_components=8, shape=(10,))

# Rotation
rotation = HouseholderRotation(n_reflections=10, shape=(10,))

# Coupling layers
affine = AffineCoupling(key, shape=(10,))
spline = RQSplineCoupling(key, shape=(10,), n_bins=8)
circular = CircularRQSplineCoupling(key, shape=(10,))  # all dims periodic

# Periodic utilities
wrap = PeriodicWrap(ind=(0,), shape=(10,))
shift = PeriodicShift(ind=(0,), shape=(10,))
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
make test

# Run linter
make lint

# Run all checks
make check
```

## References

- Meng, C., Song, Y., Song, J., & Ermon, S. (2020). *Gaussianization Flows*. arXiv:2003.01941
- Laparra, V., Camps-Valls, G., & Malo, J. (2011). *Iterative Gaussianization: From ICA to Random Rotations*. IEEE TNNLS.
- Ward, D. (2023). *FlowJAX: Distributions and Normalizing Flows in JAX*.
