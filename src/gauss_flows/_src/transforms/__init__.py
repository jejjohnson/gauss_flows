"""Transforms sub-package for gauss_flows.

Re-exports all transform classes from the sub-modules.
"""

from gauss_flows._src.transforms.classic import PlanarFlow, SylvesterFlow
from gauss_flows._src.transforms.conv import (
    ActNorm,
    HaarWavelet,
    Invertible1x1Conv,
    OrthogonalConvExponential,
    Squeeze,
)
from gauss_flows._src.transforms.coupling import (
    ActNorm1D,
    AffineCoupling,
    BatchNorm,
    DeepSigmoidCoupling,
    RQSplineCoupling,
)
from gauss_flows._src.transforms.marginal import (
    HistogramCDF,
    InverseGaussCDF,
    MixtureGaussianCDF,
    MixtureLogisticCDF,
    RQSplineMarginal,
)
from gauss_flows._src.transforms.periodic import (
    CircularRationalQuadraticSpline,
    CircularRQSplineCoupling,
    PeriodicShift,
    PeriodicWrap,
)
from gauss_flows._src.transforms.rotation import (
    FixedRotation,
    HouseholderRotation,
    LULinearPermute,
    OrthogonalRotation,
)
from gauss_flows._src.transforms.stochastic import StochasticPermutation
from gauss_flows._src.transforms.surjections import (
    Augment,
    SimpleAbsSurjection,
    SimpleMaxPoolSurjection2d,
    SimpleSortSurjection,
    Slice,
)


__all__ = [
    "ActNorm",
    "ActNorm1D",
    "AffineCoupling",
    "Augment",
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
    "OrthogonalConvExponential",
    "OrthogonalRotation",
    "PeriodicShift",
    "PeriodicWrap",
    "PlanarFlow",
    "RQSplineCoupling",
    "RQSplineMarginal",
    "SimpleAbsSurjection",
    "SimpleMaxPoolSurjection2d",
    "SimpleSortSurjection",
    "Slice",
    "Squeeze",
    "StochasticPermutation",
    "SylvesterFlow",
]
