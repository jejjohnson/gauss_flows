"""gauss_flows — JAX/FlowJax-based RBIG and Gaussianization flows.

Public API re-exported from the private _src/ implementation directory.
"""

from gauss_flows._src.distributions import (
    GaussianMixture,
    GaussianPCA,
)
from gauss_flows._src.flows import (
    SurVAEFlow,
    coupling_gaussianization_flow,
    gaussianization_flow,
    iterative_rbig,
)
from gauss_flows._src.info_theory import (
    entropy,
    kl_divergence,
    mutual_information,
    negentropy,
    total_correlation,
)
from gauss_flows._src.numpyro_compat import FlowDist
from gauss_flows._src.numpyro_guide import FlowGuide
from gauss_flows._src.train import fit_gaussianization_flow
from gauss_flows._src.transforms import (
    ActNorm,
    ActNorm1D,
    AffineCoupling,
    CircularRationalQuadraticSpline,
    CircularRQSplineCoupling,
    DeepSigmoidCoupling,
    FixedRotation,
    HaarWavelet,
    HouseholderRotation,
    InverseGaussCDF,
    Invertible1x1Conv,
    LULinearPermute,
    MixtureGaussianCDF,
    MixtureLogisticCDF,
    OrthogonalRotation,
    PeriodicShift,
    PeriodicWrap,
    PlanarFlow,
    RQSplineCoupling,
    RQSplineMarginal,
    Squeeze,
    SylvesterFlow,
)
from gauss_flows._src.transforms.base import (
    AbstractStochastic,
    AbstractSurjection,
)


__all__ = [
    "AbstractStochastic",
    "AbstractSurjection",
    "ActNorm",
    "ActNorm1D",
    "AffineCoupling",
    "CircularRQSplineCoupling",
    "CircularRationalQuadraticSpline",
    "DeepSigmoidCoupling",
    "FixedRotation",
    "FlowDist",
    "FlowGuide",
    "GaussianMixture",
    "GaussianPCA",
    "HaarWavelet",
    "HouseholderRotation",
    "InverseGaussCDF",
    "Invertible1x1Conv",
    "LULinearPermute",
    "MixtureGaussianCDF",
    "MixtureLogisticCDF",
    "OrthogonalRotation",
    "PeriodicShift",
    "PeriodicWrap",
    "PlanarFlow",
    "RQSplineCoupling",
    "RQSplineMarginal",
    "Squeeze",
    "SurVAEFlow",
    "SylvesterFlow",
    "coupling_gaussianization_flow",
    "entropy",
    "fit_gaussianization_flow",
    "gaussianization_flow",
    "iterative_rbig",
    "kl_divergence",
    "mutual_information",
    "negentropy",
    "total_correlation",
]
