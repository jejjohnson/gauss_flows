"""gauss_flows — JAX/FlowJax-based RBIG and Gaussianization flows.

Public API re-exported from the private _src/ implementation directory.
"""

from gauss_flows._src.flows import (
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
    PlanarFlow,
    RQSplineCoupling,
    RQSplineMarginal,
    Squeeze,
    SylvesterFlow,
)


__all__ = [
    "ActNorm",
    "ActNorm1D",
    "AffineCoupling",
    "DeepSigmoidCoupling",
    "FixedRotation",
    "FlowDist",
    "FlowGuide",
    "HaarWavelet",
    "HouseholderRotation",
    "InverseGaussCDF",
    "Invertible1x1Conv",
    "LULinearPermute",
    "MixtureGaussianCDF",
    "MixtureLogisticCDF",
    "OrthogonalRotation",
    "PlanarFlow",
    "RQSplineCoupling",
    "RQSplineMarginal",
    "Squeeze",
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
