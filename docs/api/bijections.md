# Bijections

A bijection is an exactly invertible map $f:\mathbb{R}^D\!\to\mathbb{R}^D$ paired with the
log-absolute-determinant of its Jacobian. Under the change-of-variables formula a base
density $p_Z$ pushes forward to

$$
\log p_X(x) \;=\; \log p_Z\!\big(f(x)\big) \;+\; \log\left|\det \frac{\partial f}{\partial x}\right|,
$$

so each layer exposes its map and the log-determinant of its Jacobian in **both** directions —
`transform_and_log_det` and `inverse_and_log_det`, each returning `(y, log_det)`. Per the
[FlowJax](https://danielward27.github.io/flowjax/) convention a
[`Transformed`](https://danielward27.github.io/flowjax/) distribution applies `transform` when
**sampling** (base → data) and `inverse` when **evaluating density** (data → base — the $f$ in
the formula above), so a layer's data/latent orientation is set by how it is composed, not by
the method name. Generative-only layers such as `PlanarFlow` / `SylvesterFlow` implement only
`transform_and_log_det`. All layers below subclass
[`flowjax.bijections.AbstractBijection`](https://danielward27.github.io/flowjax/) and follow
the **single-event** convention (`x.shape == self.shape`, scalar `log_det`); batch with
`jax.vmap`.

## Coupling layers

Coupling layers split the input into two halves, leave one half untouched, and transform the
other half with an elementwise bijection whose parameters are predicted by a neural network
from the untouched half. The Jacobian is triangular, so `log_det` is a cheap sum and the
inverse is closed-form — the workhorse of expressive flows.

::: gauss_flows.AffineCoupling

::: gauss_flows.GINCoupling

::: gauss_flows.RQSplineCoupling

::: gauss_flows.CircularRQSplineCoupling

::: gauss_flows.DeepSigmoidCoupling

::: gauss_flows.MixtureGaussianCDFCoupling

::: gauss_flows.ContinuousAffineCoupling

## Marginal (elementwise) transforms

Marginal transforms act independently on each coordinate — the Gaussianization "G" step that
reshapes per-dimension marginals toward a target (usually a standard normal or a uniform).
Because they are diagonal, the Jacobian determinant is the product of per-coordinate
derivatives.

::: gauss_flows.MixtureGaussianCDF

::: gauss_flows.MixtureLogisticCDF

::: gauss_flows.RQSplineMarginal

::: gauss_flows.CircularRationalQuadraticSpline

::: gauss_flows.HistogramCDF

::: gauss_flows.InverseGaussCDF

## Linear & rotation layers

Linear layers mix information across coordinates — the rotation "R" step of RBIG. A rotation
$W$ contributes $\log|\det W|$ (zero for an orthogonal $W$) and, by decorrelating dimensions,
lets the next marginal step make progress.

::: gauss_flows.HouseholderRotation

::: gauss_flows.OrthogonalRotation

::: gauss_flows.FixedRotation

::: gauss_flows.LULinearPermute

::: gauss_flows.Invertible1x1Conv

::: gauss_flows.Orthogonal1x1Conv

::: gauss_flows.OrthogonalConvExponential

::: gauss_flows.MatrixExponential

::: gauss_flows.HaarWavelet

## Normalisation layers

Normalisation layers rescale and recentre activations between flow blocks, stabilising both
training and the conditioning of downstream rotations.

::: gauss_flows.ActNorm

::: gauss_flows.ActNorm1D

::: gauss_flows.BatchNorm

::: gauss_flows.GeneralizedDivisiveNormalization

::: gauss_flows.GeneralizedDivisiveNormalization1D

## Periodic layers

Periodic layers operate on angular (circular) coordinates living on $[-\pi, \pi)$, for flows
on the torus or sphere.

::: gauss_flows.PeriodicShift

::: gauss_flows.PeriodicWrap

## Reshape layers

::: gauss_flows.Squeeze

## Classic residual flows

Residual flows apply a low-rank perturbation $x + u\,h(w^\top x + b)$. The determinant follows
from the matrix-determinant lemma, so `log_det` stays cheap despite the full-rank coupling
across coordinates.

::: gauss_flows.PlanarFlow

::: gauss_flows.SylvesterFlow

## Continuous (FFJORD) flows

A continuous normalizing flow replaces the discrete layer stack with an ODE
$\dot{x} = v_\theta(t, x)$ integrated over $t\in[0,1]$. The log-density change accumulates the
instantaneous trace of the Jacobian,

$$
\log p_X(x(0)) \;=\; \log p_Z(x(1)) + \int_0^1 \nabla\!\cdot\, v_\theta\big(t, x(t)\big)\,\mathrm{d}t,
$$

estimated exactly (small $D$) or with the Hutchinson trace estimator. The `Diffeq*` networks
parameterise $v_\theta$, the `Time*` nets embed $t$, and `pack_time_control` bundles the time
and optional control signal into the `condition` FlowJax threads through the solver.

::: gauss_flows.FFJORD

::: gauss_flows.DiffeqMLP

::: gauss_flows.DiffeqConcat

::: gauss_flows.TimeIdentity

::: gauss_flows.TimeTanh

::: gauss_flows.TimeFourier

::: gauss_flows.pack_time_control

::: gauss_flows.unpack_time_control

::: gauss_flows.time_control_cond_shape

## Conditional wrappers

::: gauss_flows.Conditioner
