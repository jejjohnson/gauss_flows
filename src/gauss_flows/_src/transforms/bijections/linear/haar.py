"""Haar wavelet transform bijection (1D, last-axis)."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jaxtyping import ArrayLike


class HaarWavelet(AbstractBijection):
    """Haar wavelet transform bijection (1-D, last axis).

    Applies the single-level 1-D Haar wavelet transform along the last axis:
    splits each consecutive even/odd pair into its average and half-difference
    ``avg = (even + odd)/2``, ``diff = (even − odd)/2``, then concatenates all
    averages followed by all differences. This factorises an event into a
    coarse (low-pass) and a detail (high-pass) half, as used in multi-scale
    flow architectures. Both halves scale by ``1/2``, so the forward
    log-determinant is the constant ``−n·log 2`` where ``n`` is the size of
    the last axis.

    Operates on a single event whose last axis is divisible by 2; the
    transform acts independently along any leading axes.

    Args:
        shape: Event shape. The last dimension must be divisible by 2.

    Raises:
        ValueError: If the last dimension of ``shape`` is not divisible by 2.

    Shape:
        - transform_and_log_det: ``(…, n)`` → ``(…, n)``, scalar log_det = −n·log 2
        - inverse_and_log_det:   ``(…, n)`` → ``(…, n)``, scalar log_det = +n·log 2

    Examples:
        >>> import jax.numpy as jnp
        >>> from gauss_flows import HaarWavelet
        >>> t = HaarWavelet(shape=(4,))
        >>> x = jnp.array([1.0, 3.0, 5.0, 9.0])
        >>> y, log_det = t.transform_and_log_det(x)
        >>> y  # [avg0, avg1, diff0, diff1]
        Array([ 2.,  7., -1., -2.], dtype=float32)
        >>> x_rec, _ = t.inverse_and_log_det(y)
        >>> bool(jnp.allclose(x, x_rec))
        True
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
