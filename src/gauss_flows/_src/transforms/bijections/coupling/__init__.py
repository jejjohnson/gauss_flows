"""Coupling bijections.

All coupling layers in this subpackage split the input along its last
(leading) dim into an untransformed "condition" half and a "transformed"
half whose elementwise update is parameterised by an MLP that sees the
condition half.
"""

from gauss_flows._src.transforms.bijections.coupling.affine import AffineCoupling
from gauss_flows._src.transforms.bijections.coupling.circular_spline import (
    CircularRQSplineCoupling,
)
from gauss_flows._src.transforms.bijections.coupling.deepsigmoid import (
    DeepSigmoidCoupling,
)
from gauss_flows._src.transforms.bijections.coupling.spline import RQSplineCoupling


__all__ = [
    "AffineCoupling",
    "CircularRQSplineCoupling",
    "DeepSigmoidCoupling",
    "RQSplineCoupling",
]
