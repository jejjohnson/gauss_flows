# Base Distributions

A normalizing flow needs a tractable **base distribution** $p_Z$ to push forward through its
bijections. The default is a standard normal (`flowjax.distributions.Normal`), but a richer
or structured base can shorten the flow it has to learn. These distributions expose
`log_prob` and `sample` and plug into any FlowJax
[`Transformed`](https://danielward27.github.io/flowjax/) or the package's `SurVAEFlow`
container as the `base_dist` argument.

## Euclidean bases

`GaussianPCA` is a low-rank-plus-diagonal Gaussian, $\Sigma = W W^\top + \sigma^2 I$, whose
`log_prob` uses the Woodbury identity to avoid forming the full $D\times D$ covariance.
`GaussianMixture` is a 1-D mixture used as a marginal target. The conditional variants emit
mean/scale from a context vector for amortised inference and class-conditional flows.

::: gauss_flows.GaussianPCA

::: gauss_flows.GaussianMixture

::: gauss_flows.ConditionalDiagGaussian

::: gauss_flows.ClassCondDiagGaussian

::: gauss_flows.NumpyroBase

## Manifold (sphere) distributions

Distributions on the unit sphere $\mathbb{S}^{d}\subset\mathbb{R}^{d+1}$ for directional data
and global flows. The von Mises–Fisher density $p(x)\propto\exp(\kappa\,\mu^\top x)$
concentrates around a mean direction $\mu$ with concentration $\kappa$; `UniformOnSphere` is
its $\kappa\to 0$ limit. The exp/log maps and tangent basis below move between the manifold
and its tangent plane for building flows in local coordinates.

::: gauss_flows.VonMisesFisher

::: gauss_flows.UniformOnSphere

::: gauss_flows.expmap_sphere

::: gauss_flows.logmap_sphere

::: gauss_flows.tangent_basis
