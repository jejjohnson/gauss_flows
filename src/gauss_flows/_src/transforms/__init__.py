"""Transforms sub-package for gauss_flows.

Re-exports every transform class from the nested SurVAE-taxonomy
subpackages (:mod:`.bijections`, :mod:`.surjections`, :mod:`.stochastic`)
so downstream code can keep importing from ``gauss_flows._src.transforms``
without caring which leaf module a class lives in.
"""

from __future__ import annotations

from gauss_flows._src.transforms._sphere_utils import (
    expmap_sphere,
    logmap_sphere,
    tangent_basis,
)
from gauss_flows._src.transforms.bijections import (
    FFJORD,
    ActNorm,
    ActNorm1D,
    AffineCoupling,
    BatchNorm,
    CircularRationalQuadraticSpline,
    CircularRQSplineCoupling,
    ContinuousAffineCoupling,
    DeepSigmoidCoupling,
    FixedRotation,
    HaarWavelet,
    HistogramCDF,
    HouseholderRotation,
    InverseGaussCDF,
    Invertible1x1Conv,
    LULinearPermute,
    MixtureGaussianCDF,
    MixtureGaussianCDFCoupling,
    MixtureLogisticCDF,
    Orthogonal1x1Conv,
    OrthogonalConvExponential,
    OrthogonalRotation,
    PeriodicShift,
    PeriodicWrap,
    PlanarFlow,
    RQSplineCoupling,
    RQSplineMarginal,
    Squeeze,
    SylvesterFlow,
)
from gauss_flows._src.transforms.stochastic import VAE, StochasticPermutation
from gauss_flows._src.transforms.surjections import (
    Augment,
    SimpleAbsSurjection,
    SimpleMaxPoolSurjection2d,
    SimpleSortSurjection,
    Slice,
)


__all__ = [
    "FFJORD",
    "VAE",
    "ActNorm",
    "ActNorm1D",
    "AffineCoupling",
    "Augment",
    "BatchNorm",
    "CircularRQSplineCoupling",
    "CircularRationalQuadraticSpline",
    "ContinuousAffineCoupling",
    "DeepSigmoidCoupling",
    "FixedRotation",
    "HaarWavelet",
    "HistogramCDF",
    "HouseholderRotation",
    "InverseGaussCDF",
    "Invertible1x1Conv",
    "LULinearPermute",
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
    "SimpleAbsSurjection",
    "SimpleMaxPoolSurjection2d",
    "SimpleSortSurjection",
    "Slice",
    "Squeeze",
    "StochasticPermutation",
    "SylvesterFlow",
    "expmap_sphere",
    "logmap_sphere",
    "tangent_basis",
]
