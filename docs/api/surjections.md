# Surjections & Stochastic transforms

The [SurVAE](https://arxiv.org/abs/2007.02731) framework generalises a normalizing flow
beyond exact bijections. A **surjection** is deterministic in one direction and
stochastic (or many-to-one) in the other; a **stochastic** transform is random in both. Both
relax the change-of-variables equality into a bound, contributing a *likelihood
contribution* term in place of an exact `log_det`:

$$
\log p_X(x) \;\geq\; \mathbb{E}_{q(z\mid x)}\!\left[\log p_Z(z) + \mathcal{L}(x, z)\right],
$$

which is exact when the inverse direction is deterministic. This lets a flow change
dimensionality (augment/slice, max-pool), encode invariances (abs, sort), or insert
VAE-style stochastic layers — while still composing inside a `SurVAEFlow`.

Surjection and stochastic methods take an explicit `key` (the direction may sample) and follow
the **single-event** convention; the `SurVAEFlow` container vmaps `log_prob` / `sample` over
any leading `sample_shape`.

## Base classes

::: gauss_flows.AbstractSurjection

::: gauss_flows.AbstractStochastic

## Surjections

::: gauss_flows.SimpleAbsSurjection

::: gauss_flows.SimpleSortSurjection

::: gauss_flows.SimpleMaxPoolSurjection2d

::: gauss_flows.Augment

::: gauss_flows.Slice

## Stochastic transforms

::: gauss_flows.VAE

::: gauss_flows.StochasticPermutation
