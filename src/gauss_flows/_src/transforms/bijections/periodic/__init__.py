"""Periodic / circular bijections suited to angular data on the torus.

The circular-spline classes (scalar transformer + coupling layer) live in
the ``elementwise/`` and ``coupling/`` subpackages respectively; this
subpackage hosts only the wrap / shift primitives that are specific to the
periodic setting.
"""

from gauss_flows._src.transforms.bijections.periodic.shift import PeriodicShift
from gauss_flows._src.transforms.bijections.periodic.wrap import PeriodicWrap


__all__ = ["PeriodicShift", "PeriodicWrap"]
