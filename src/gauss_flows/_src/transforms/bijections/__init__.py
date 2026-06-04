"""Bijection transforms: invertible layers with exact log-determinant.

Subpackages group bijections by technique so the hierarchy mirrors the
SurVAE taxonomy:

- `.coupling` — masked-conditioner couplings.
- `.continuous` — ODE-defined continuous normalizing flows.
- `.elementwise` — pointwise CDF / spline layers that act on each dim.
- `.linear` — matrix-shaped layers: rotations, LU, 1×1 conv, conv-exp.
- `.normalization` — ActNorm + invertible BatchNorm.
- `.reshape` — volume-preserving shape rewriters (Squeeze).
- `.periodic` — wrap / shift primitives on the torus.
- `.classic` — low-expressivity VI-only flows (Planar, Sylvester).
"""

from __future__ import annotations

from gauss_flows._src.transforms.bijections.classic import PlanarFlow, SylvesterFlow
from gauss_flows._src.transforms.bijections.conditioner import Conditioner
from gauss_flows._src.transforms.bijections.continuous import FFJORD
from gauss_flows._src.transforms.bijections.coupling import (
    AffineCoupling,
    CircularRQSplineCoupling,
    ContinuousAffineCoupling,
    DeepSigmoidCoupling,
    GINCoupling,
    MixtureGaussianCDFCoupling,
    RQSplineCoupling,
)
from gauss_flows._src.transforms.bijections.elementwise import (
    CircularRationalQuadraticSpline,
    HistogramCDF,
    InverseGaussCDF,
    MixtureGaussianCDF,
    MixtureLogisticCDF,
    RQSplineMarginal,
)
from gauss_flows._src.transforms.bijections.linear import (
    FixedRotation,
    HaarWavelet,
    HouseholderRotation,
    Invertible1x1Conv,
    LULinearPermute,
    MatrixExponential,
    Orthogonal1x1Conv,
    OrthogonalConvExponential,
    OrthogonalRotation,
)
from gauss_flows._src.transforms.bijections.normalization import (
    ActNorm,
    ActNorm1D,
    BatchNorm,
    GeneralizedDivisiveNormalization,
    GeneralizedDivisiveNormalization1D,
)
from gauss_flows._src.transforms.bijections.periodic import PeriodicShift, PeriodicWrap
from gauss_flows._src.transforms.bijections.reshape import Squeeze


__all__ = [
    "FFJORD",
    "ActNorm",
    "ActNorm1D",
    "AffineCoupling",
    "BatchNorm",
    "CircularRQSplineCoupling",
    "CircularRationalQuadraticSpline",
    "Conditioner",
    "ContinuousAffineCoupling",
    "DeepSigmoidCoupling",
    "FixedRotation",
    "GINCoupling",
    "GeneralizedDivisiveNormalization",
    "GeneralizedDivisiveNormalization1D",
    "HaarWavelet",
    "HistogramCDF",
    "HouseholderRotation",
    "InverseGaussCDF",
    "Invertible1x1Conv",
    "LULinearPermute",
    "MatrixExponential",
    "MixtureGaussianCDF",
    "MixtureGaussianCDFCoupling",
    "MixtureLogisticCDF",
    "Orthogonal1x1Conv",
    "OrthogonalConvExponential",
    "OrthogonalRotation",
    "PeriodicShift",
    "PeriodicWrap",
    "PlanarFlow",
    "RQSplineCoupling",
    "RQSplineMarginal",
    "Squeeze",
    "SylvesterFlow",
]
