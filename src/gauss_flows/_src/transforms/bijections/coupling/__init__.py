"""Coupling bijections.

All coupling layers in this subpackage split the input along its last
(leading) dim into an untransformed "condition" half and a "transformed"
half whose elementwise update is parameterised by an MLP that sees the
condition half.
"""

from __future__ import annotations

from gauss_flows._src.transforms.bijections.coupling.affine import AffineCoupling
from gauss_flows._src.transforms.bijections.coupling.circular_spline import (
    CircularRQSplineCoupling,
)
from gauss_flows._src.transforms.bijections.coupling.continuous_affine_coupling import (
    ContinuousAffineCoupling,
)
from gauss_flows._src.transforms.bijections.coupling.deepsigmoid import (
    DeepSigmoidCoupling,
)
from gauss_flows._src.transforms.bijections.coupling.gin_coupling import GINCoupling
from gauss_flows._src.transforms.bijections.coupling.mixture_cdf import (
    MixtureGaussianCDFCoupling,
)
from gauss_flows._src.transforms.bijections.coupling.spline import RQSplineCoupling


__all__ = [
    "AffineCoupling",
    "CircularRQSplineCoupling",
    "ContinuousAffineCoupling",
    "DeepSigmoidCoupling",
    "GINCoupling",
    "MixtureGaussianCDFCoupling",
    "RQSplineCoupling",
]
