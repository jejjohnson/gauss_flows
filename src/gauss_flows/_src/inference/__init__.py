"""Inference utilities: NumPyro interop + training loops.

- :mod:`.numpyro_compat` adapts a flowjax ``Transformed`` as a NumPyro distribution.
- :mod:`.numpyro_guide` plugs that adapter into an ``AutoContinuous`` SVI guide.
- :mod:`.train` wraps ``flowjax.train.fit_to_data`` with Gaussianization defaults.
"""

from gauss_flows._src.inference.numpyro_compat import FlowDist
from gauss_flows._src.inference.numpyro_guide import FlowGuide
from gauss_flows._src.inference.train import fit_gaussianization_flow


__all__ = ["FlowDist", "FlowGuide", "fit_gaussianization_flow"]
