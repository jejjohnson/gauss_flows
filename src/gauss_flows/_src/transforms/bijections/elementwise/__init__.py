"""Elementwise (per-dim) bijections.

Each transform in this subpackage acts independently on every dimension of
its input — typically as a learned or fitted 1D CDF / inverse CDF. Used as
the "marginal Gaussianization" half of RBIG-style flows and as drop-in
pointwise layers inside a `SurVAEFlow` chain.
"""

from gauss_flows._src.transforms.bijections.elementwise.circular_spline import (
    CircularRationalQuadraticSpline,
)
from gauss_flows._src.transforms.bijections.elementwise.histogram import HistogramCDF
from gauss_flows._src.transforms.bijections.elementwise.inverse_gauss import (
    InverseGaussCDF,
)
from gauss_flows._src.transforms.bijections.elementwise.mixture_cdf import (
    MixtureGaussianCDF,
    MixtureLogisticCDF,
)
from gauss_flows._src.transforms.bijections.elementwise.spline import RQSplineMarginal


__all__ = [
    "CircularRationalQuadraticSpline",
    "HistogramCDF",
    "InverseGaussCDF",
    "MixtureGaussianCDF",
    "MixtureLogisticCDF",
    "RQSplineMarginal",
]
