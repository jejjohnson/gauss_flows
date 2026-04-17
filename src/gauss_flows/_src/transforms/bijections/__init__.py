"""Bijection transforms: invertible layers with exact log-determinant.

Subpackages group bijections by technique so the hierarchy mirrors the
SurVAE taxonomy:

- :mod:`.coupling` — masked-conditioner couplings.
- :mod:`.elementwise` — pointwise CDF / spline layers that act on each dim.
- :mod:`.linear` — matrix-shaped layers: rotations, LU, 1×1 conv, conv-exp.
- :mod:`.normalization` — ActNorm + invertible BatchNorm.
- :mod:`.reshape` — volume-preserving shape rewriters (Squeeze).
- :mod:`.periodic` — wrap / shift primitives on the torus.
- :mod:`.classic` — low-expressivity VI-only flows (Planar, Sylvester).
"""

from gauss_flows._src.transforms.bijections.classic import PlanarFlow, SylvesterFlow
from gauss_flows._src.transforms.bijections.coupling import (
    AffineCoupling,
    CircularRQSplineCoupling,
    DeepSigmoidCoupling,
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
    Orthogonal1x1Conv,
    OrthogonalConvExponential,
    OrthogonalRotation,
)
from gauss_flows._src.transforms.bijections.normalization import (
    ActNorm,
    ActNorm1D,
    BatchNorm,
)
from gauss_flows._src.transforms.bijections.periodic import PeriodicShift, PeriodicWrap
from gauss_flows._src.transforms.bijections.reshape import Squeeze


__all__ = [
    "ActNorm",
    "ActNorm1D",
    "AffineCoupling",
    "BatchNorm",
    "CircularRQSplineCoupling",
    "CircularRationalQuadraticSpline",
    "DeepSigmoidCoupling",
    "FixedRotation",
    "HaarWavelet",
    "HistogramCDF",
    "HouseholderRotation",
    "InverseGaussCDF",
    "Invertible1x1Conv",
    "LULinearPermute",
    "MixtureGaussianCDF",
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
