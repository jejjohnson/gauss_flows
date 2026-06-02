# Information Theory

A Gaussianization flow is also an **information-theoretic estimator**. Because the flow gives a
tractable density, differential entropy follows from the Monte-Carlo identity

$$
H(X) \;=\; -\,\mathbb{E}_{x\sim p_X}\big[\log p_X(x)\big]
\;\approx\; -\frac{1}{N}\sum_{i=1}^{N}\log p_X(x_i),\qquad x_i \sim p_X,
$$

and total correlation, mutual information, KL, and negentropy reduce to combinations of
entropies. This module offers three families that share the measures but differ in how the
density is obtained.

## Flow-based (Monte-Carlo) estimators

Estimate a measure from a trained flow by sampling and scoring. Each takes the flow, a sample
count `n_samples`, and a `key`; the estimate's variance shrinks as $O(1/\sqrt{N})$.

::: gauss_flows.entropy

::: gauss_flows.total_correlation

::: gauss_flows.mutual_information

::: gauss_flows.kl_divergence

::: gauss_flows.negentropy

## Analytical Gaussian closed forms

Exact values for (multivariate) Gaussians — useful as ground truth in tests and as a baseline.
For a covariance $\Sigma$ the differential entropy is
$H = \tfrac12\log\!\big((2\pi e)^D\,|\Sigma|\big)$, and total correlation / mutual information
follow from the log-determinant gap between the joint and its marginals.

::: gauss_flows.gaussian_entropy

::: gauss_flows.gaussian_total_correlation

::: gauss_flows.gaussian_mutual_information

::: gauss_flows.gaussian_kl_divergence

## RBIG-way reductions

The original RBIG estimators read a measure off the **total-correlation reduction** achieved by
each Gaussianization layer, accumulated across the stack rather than scored from the final
density. `information_reduction` is the per-pair building block; the others sum it along a
fitted flow.

::: gauss_flows.information_reduction

::: gauss_flows.total_correlation_reduction

::: gauss_flows.entropy_reduction

::: gauss_flows.kl_divergence_reduction
