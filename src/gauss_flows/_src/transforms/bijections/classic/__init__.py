"""Classic low-expressivity normalizing flow bijections.

Both transforms predate coupling and spline flows. They are useful as
teaching baselines and as variational-inference-only flows: they have no
algebraic inverse, so they can be used to sample from ``base -> data`` but
cannot evaluate ``log_prob`` on arbitrary points. Use
:meth:`flowjax.distributions.Transformed.sample_and_log_prob` in the VI
setting.

References:
    Rezende & Mohamed 2015, *Variational Inference with Normalizing Flows*.
    van den Berg et al. 2018, *Sylvester Normalizing Flows for VI*.
"""

from gauss_flows._src.transforms.bijections.classic.planar import PlanarFlow
from gauss_flows._src.transforms.bijections.classic.sylvester import SylvesterFlow


__all__ = ["PlanarFlow", "SylvesterFlow"]
