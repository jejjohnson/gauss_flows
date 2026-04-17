"""Linear / convolutional bijections.

Every transform here is ultimately a multiplication by a matrix (or the
matrix exponential of one): rotations, LU-parameterised linear layers,
1x1 convs, full-kernel orthogonal convs, and the Haar wavelet operator.
"""

from gauss_flows._src.transforms.bijections.linear.conv1x1 import Invertible1x1Conv
from gauss_flows._src.transforms.bijections.linear.conv_exp import (
    OrthogonalConvExponential,
)
from gauss_flows._src.transforms.bijections.linear.haar import HaarWavelet
from gauss_flows._src.transforms.bijections.linear.lu import LULinearPermute
from gauss_flows._src.transforms.bijections.linear.orthogonal_conv1x1 import (
    Orthogonal1x1Conv,
)
from gauss_flows._src.transforms.bijections.linear.rotation import (
    FixedRotation,
    HouseholderRotation,
    OrthogonalRotation,
)


__all__ = [
    "FixedRotation",
    "HaarWavelet",
    "HouseholderRotation",
    "Invertible1x1Conv",
    "LULinearPermute",
    "Orthogonal1x1Conv",
    "OrthogonalConvExponential",
    "OrthogonalRotation",
]
