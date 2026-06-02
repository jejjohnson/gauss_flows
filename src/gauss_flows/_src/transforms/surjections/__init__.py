"""SurVAE surjections — many-to-one and one-to-many transforms.

A "surjection" in the SurVAE hierarchy (Nielsen et al. 2020) is a
many-to-one map: deterministic in one direction, stochastic in the other.
This subpackage ships two families:

**Simple surjections** — pure shape math, no learned encoder/decoder beyond
an optional fixed conditional distribution: `SimpleAbsSurjection`,
`SimpleSortSurjection`, `SimpleMaxPoolSurjection2d`. All
three are *inference* surjections (deterministic forward, stochastic
inverse).

**Encoder/decoder surjections** — change event dimensionality, take a
conditional distribution as a constructor argument: `Slice` (drops
dims under a decoder), `Augment` (adds dims under an encoder).

References:
    Nielsen et al. (2020), *SurVAE Flows*, NeurIPS.
"""

from gauss_flows._src.transforms.surjections.abs import SimpleAbsSurjection
from gauss_flows._src.transforms.surjections.dimension import Augment, Slice
from gauss_flows._src.transforms.surjections.pool import SimpleMaxPoolSurjection2d
from gauss_flows._src.transforms.surjections.sort import SimpleSortSurjection


__all__ = [
    "Augment",
    "SimpleAbsSurjection",
    "SimpleMaxPoolSurjection2d",
    "SimpleSortSurjection",
    "Slice",
]
