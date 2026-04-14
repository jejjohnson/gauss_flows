# gauss_flows

> JAX/FlowJax-based Gaussianization flows, SurVAE transforms, and information-theoretic utilities.

`gauss_flows` is a specialised extension of [FlowJax](https://danielward27.github.io/flowjax/) providing:

- **Gaussianization transforms** — alternating marginal CDF transforms and learned rotations (the RBIG pattern)
- **SurVAE hierarchy** — bijections, surjections, and stochastic transforms with exact or lower-bound likelihoods
- **Information theory** — first-class `entropy`, `mutual_information`, `total_correlation`, `kl_divergence`, `negentropy`
- **NumPyro integration** — `FlowDist` (Distribution wrapper) and `FlowGuide` (`AutoContinuous` variational guide)

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
