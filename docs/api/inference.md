# NumPyro Inference

`gauss_flows` flows interoperate with [NumPyro](https://num.pyro.ai/) so a learned flow can
serve as either a **prior/likelihood** in a probabilistic model or as a **variational guide**
for an arbitrary model.

- `FlowDist` wraps any FlowJax flow as a `numpyro.distributions.Distribution`, exposing
  `sample` and `log_prob` so it can be used directly at a `numpyro.sample` site.
- `FlowGuide` is an `AutoContinuous` variational guide whose posterior is a normalizing flow —
  a flexible, full-rank alternative to a mean-field `AutoNormal` guide for SVI.
- `fit_gaussianization_flow` is the maximum-likelihood training loop (a thin wrapper over
  `flowjax.train.fit_to_data`) for fitting a flow to samples.

## Distribution wrapper

::: gauss_flows.FlowDist

## Variational guide

::: gauss_flows.FlowGuide

## Training

::: gauss_flows.fit_gaussianization_flow
