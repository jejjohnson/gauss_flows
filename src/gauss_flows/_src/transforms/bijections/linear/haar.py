"""Haar wavelet transform bijection (1D, last-axis)."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jaxtyping import ArrayLike


class HaarWavelet(AbstractBijection):
    """Haar wavelet transform bijection.

    Implements the 1D Haar wavelet transform as a bijection. This is used
    in multi-scale flow architectures to factorize spatial information.

    Args:
        shape: Shape of the input. The last dimension must be divisible by 2.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...]):
        if shape[-1] % 2 != 0:
            raise ValueError("Last dimension must be divisible by 2 for HaarWavelet.")
        self.shape = shape

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        x = jnp.asarray(x)
        # Haar wavelet: split into even/odd, compute averages and differences
        even = x[..., ::2]
        odd = x[..., 1::2]
        avg = (even + odd) / 2.0
        diff = (even - odd) / 2.0
        y = jnp.concatenate([avg, diff], axis=-1)
        # Log det: both avg and diff scale by 1/2, giving -n*log(2) total.
        log_det = -x.shape[-1] * jnp.log(2.0)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        y = jnp.asarray(y)
        n = y.shape[-1] // 2
        avg = y[..., :n]
        diff = y[..., n:]
        even = avg + diff
        odd = avg - diff
        # Interleave even and odd
        x = jnp.stack([even, odd], axis=-1).reshape((*y.shape[:-1], y.shape[-1]))
        log_det = x.shape[-1] * jnp.log(2.0)
        return x, log_det


__all__ = ["HaarWavelet"]
