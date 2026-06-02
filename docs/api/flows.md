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
