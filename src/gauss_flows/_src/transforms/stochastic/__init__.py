"""Stochastic transforms — both directions sample from a non-degenerate kernel.

A "stochastic transform" in the SurVAE hierarchy is one where ``forward``
and ``inverse`` are both random — neither direction is a deterministic
function of its input. The two canonical examples ship here: :class:`VAE`
(encoder / decoder pair as a single chain link) and
:class:`StochasticPermutation` (uniform draw from the symmetric group).
"""

from gauss_flows._src.transforms.stochastic.permutation import StochasticPermutation
from gauss_flows._src.transforms.stochastic.vae import VAE


__all__ = ["VAE", "StochasticPermutation"]
