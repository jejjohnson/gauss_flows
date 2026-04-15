"""SurVAE surjections — many-to-one and one-to-many transforms.

A "surjection" in the SurVAE hierarchy (Nielsen et al. 2020) is a
many-to-one map: deterministic in one direction, stochastic in the other.
This module ships two families:

**Simple surjections** — pure shape math, no learned encoder/decoder beyond
an optional fixed conditional distribution:

    +-------------------------------+----------+----------+--------------------+
    | Class                         | forward  | inverse  | log_det per event  |
    +===============================+==========+==========+====================+
    | :class:`SimpleAbsSurjection`  | det.     | random   | -D · log 2         |
    | :class:`SimpleSortSurjection` | det.     | random   | -n_sl · log(D!)    |
    | :class:`SimpleMaxPoolSurjec…` | det.     | random   | log q(res) − N·log4|
    +-------------------------------+----------+----------+--------------------+

All three are *inference* surjections (deterministic forward, stochastic
inverse): the forward direction is the one used by ``log_prob``, and the
inverse only needs to sample plausible pre-images from a uniform prior over
the collapsed equivalence class. The log-det contribution is the entropy of
that uniform prior — exact, not a bound.

**Encoder/decoder surjections** — change event dimensionality, take a
conditional distribution as a constructor argument:

    +---------------------+----------+----------+--------------------------+
    | Class               | forward  | inverse  | log_det per event        |
    +=====================+==========+==========+==========================+
    | :class:`Slice`      | det.     | random   | log q(dropped | kept)    |
    | :class:`Augment`    | random   | det.     | -log q(z_aug | x)        |
    +---------------------+----------+----------+--------------------------+

``Slice`` is an inference surjection (data → latent): it drops trailing dims
and scores them under a decoder. ``Augment`` is a generative surjection
(latent → data direction is the deterministic one): it samples extra
dimensions from an encoder, contributing a lower-bound term to ``log_prob``.

References:
    Nielsen et al. (2020), *SurVAE Flows*, NeurIPS.
"""

from __future__ import annotations

import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from gauss_flows._src._protocols import ConditionalDistribution
from gauss_flows._src.transforms.base import AbstractSurjection


class SimpleAbsSurjection(AbstractSurjection):
    """Elementwise absolute-value surjection with a random-sign inverse.

    Inference-style surjection (Nielsen et al. 2020 §3.1). The forward map is
    deterministic; the inverse draws a uniform sign per dimension:

        forward:  z = |x|                            (deterministic)
        inverse:  s ~ Bernoulli(½)^D,  x = (2s − 1) · z

    The log-det contribution is the entropy of the uniform sign prior:

        log_det = log p(s) = −D · log 2,    D = prod(shape)

    This is exact, not a bound: every collapsed pair ``{+x, −x}`` has equal
    prior mass, so the inverse uniform sampler matches the true posterior over
    pre-images.

    Args:
        shape: Event shape of a single input (no leading batch — see the
            project-wide single-event convention).

    Shape:
        - Input ``x``:   ``shape``
        - Output ``z``:  ``shape``    (non-negative)
        - ``log_det``:   scalar (shape ``()``)

    Example:
        Single-event call:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import SimpleAbsSurjection
        >>>
        >>> surj = SimpleAbsSurjection((3,))
        >>> z, log_det = surj.forward_and_log_det(
        ...     jnp.array([1.0, -2.0, 3.0]), jr.key(0)
        ... )
        >>> # z == [1., 2., 3.]; log_det == -3 * log(2)

        Inside a SurVAEFlow (container vmaps over batch):

        >>> from gauss_flows import SurVAEFlow
        >>> from flowjax.distributions import Normal
        >>> flow = SurVAEFlow(Normal(jnp.zeros(3)), [SimpleAbsSurjection((3,))])
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    @property
    def _log_det(self) -> float:
        return -math.prod(self.shape) * math.log(2.0)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Compute ``z = |x|`` and the constant log-det ``-D · log 2``."""
        del key, cond
        # x: shape -> z: shape (non-negative).
        z = jnp.abs(jnp.asarray(x))
        return z, jnp.asarray(self._log_det)

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample ``x = ±z`` element-wise with uniform signs."""
        del cond
        z_arr = jnp.asarray(z)
        # signs: shape, in {-1, +1}; x: shape.
        signs = jr.bernoulli(key, 0.5, shape=z_arr.shape)
        signs = jnp.where(signs, 1.0, -1.0)
        x = signs * z_arr
        return x, jnp.asarray(self._log_det)


class SimpleSortSurjection(AbstractSurjection):
    """Sort along one axis; invert via independent per-slice permutations.

    Inference-style surjection. The forward map sorts each 1D slice along
    ``axis`` independently — ``jnp.sort(x, axis=axis)`` is per-slice — so the
    inverse draws an independent uniform permutation σ_i ∈ S_{shape[axis]} for
    each of the ``n_slices = prod(shape) // shape[axis]`` slices and applies it
    along ``axis``:

        forward:  z = sort(x, axis)                  (deterministic, per-slice)
        inverse:  σ_i ~ Uniform(S_n)  for i ∈ {0, …, n_slices − 1}
                  x_i = z_i ∘ σ_i

    The log-det contribution is the joint entropy of the n_slices independent
    uniform permutation priors:

        log_det = −n_slices · log(shape[axis]!)

    Args:
        shape: Event shape of a single input (no leading batch).
        axis: Axis to sort along (default ``0``). Wrapped modulo ``len(shape)``.

    Shape:
        - Input ``x``:   ``shape``
        - Output ``z``:  ``shape``    (sorted along ``axis``)
        - ``log_det``:   scalar (shape ``()``)

    Example:
        1D event:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import SimpleSortSurjection
        >>>
        >>> surj = SimpleSortSurjection((4,))
        >>> z, _ = surj.forward_and_log_det(
        ...     jnp.array([3.0, 1.0, 4.0, 1.5]), jr.key(0)
        ... )
        >>> # z == [1., 1.5, 3., 4.]; log_det == -log(4!)

        2D event sorted along axis=0 (each column is its own slice):

        >>> surj2 = SimpleSortSurjection((4, 3), axis=0)
        >>> # n_slices = 3 (the W axis); log_det == -3 * log(4!)
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    shape: tuple[int, ...]
    axis: int
    cond_shape: ClassVar[None] = None
    _pack_pattern: str
    _unpack_pattern: str

    def __init__(self, shape: tuple[int, ...], axis: int = 0):
        if not shape:
            raise ValueError("Shape must be non-empty for SimpleSortSurjection.")
        self.shape = shape
        self.axis = axis % len(shape)
        # Build einops patterns that move ``axis`` to the front and collapse the
        # remaining dims into a single ``n_slices`` axis. Stored at init time
        # since the shape is static. Example: shape=(4, 3), axis=0 →
        # pack = "d0 d1 -> d0 (d1)"; unpack = "d0 (d1) -> d0 d1".
        names = [f"d{i}" for i in range(len(shape))]
        target = names[self.axis]
        others = [n for i, n in enumerate(names) if i != self.axis]
        all_dims = " ".join(names)
        if others:
            other_dims = " ".join(others)
            self._pack_pattern = f"{all_dims} -> {target} ({other_dims})"
            self._unpack_pattern = f"{target} ({other_dims}) -> {all_dims}"
        else:
            self._pack_pattern = f"{all_dims} -> {target}"
            self._unpack_pattern = f"{target} -> {all_dims}"

    @property
    def _axis_size(self) -> int:
        return self.shape[self.axis]

    @property
    def _n_slices(self) -> int:
        return math.prod(self.shape) // self._axis_size

    @property
    def _log_det(self) -> float:
        return -self._n_slices * math.lgamma(self._axis_size + 1)

    def _unpack_sizes(self) -> dict[str, int]:
        return {f"d{i}": s for i, s in enumerate(self.shape) if i != self.axis}

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sort each slice along ``axis``; return ``(z, -n_slices · log(D!))``."""
        del key, cond
        # x: shape -> z: shape (sorted along ``axis``).
        z = jnp.sort(jnp.asarray(x), axis=self.axis)
        return z, jnp.asarray(self._log_det)

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Apply an independent uniform permutation to each slice along ``axis``."""
        del cond
        z_arr = jnp.asarray(z)
        if len(self.shape) == 1:
            # 1D fast path — single slice, single permutation.
            perm = jr.permutation(key, self._axis_size)
            x = z_arr[perm]
        else:
            # z_arr: shape -> z_flat: (axis_size, n_slices).
            z_flat = rearrange(z_arr, self._pack_pattern)
            keys = jr.split(key, self._n_slices)

            def _permute_slice(k: PRNGKeyArray, col: Array) -> Array:
                # col: (axis_size,); out: (axis_size,)
                perm = jr.permutation(k, self._axis_size)
                return col[perm]

            # vmap over n_slices: x_flat has same shape as z_flat.
            x_flat = jax.vmap(_permute_slice, in_axes=(0, 1), out_axes=1)(keys, z_flat)
            # x_flat: (axis_size, n_slices) -> x: shape.
            x = rearrange(x_flat, self._unpack_pattern, **self._unpack_sizes())
        return x, jnp.asarray(self._log_det)


class SimpleMaxPoolSurjection2d(AbstractSurjection):
    """2D max-pooling surjection with a learned residual decoder.

    Inference-style surjection on a single image event ``x ∈ ℝ^{H × W × C}``.
    Each pool window of shape ``(pool_size, pool_size)`` collapses to its max,
    and the ``pool_area − 1`` non-max values are scored by a conditional
    decoder. With ``pool_area = pool_size²`` and
    ``N = (H/pool_size) · (W/pool_size) · C`` pooled positions:

        forward:  z[h, w, c]  = max(x[h·ps:(h+1)·ps, w·ps:(w+1)·ps, c])
                  residuals   = z − {non-max values per pool}    (≥ 0)
                  log_det     = log q(residuals | z) − N · log(pool_area)

        inverse:  k       ~ Uniform({0..pool_area−1})^N          (argmax slot)
                  resid   ~ q(· | z)                              (decoder)
                  x       = unpool(z, resid, k)

    The ``−N · log(pool_area)`` term is the entropy of the uniform prior on the
    argmax location ``k``; it is marginalised out of the log-det. The decoder
    contribution is exact under the residual parameterisation.

    Args:
        shape: Image event shape ``(H, W, C)``. Both spatial dims must be
            divisible by ``pool_size``.
        decoder: Object satisfying :class:`ConditionalDistribution` —
            ``sample(key, *, condition=z)`` must produce an array of shape
            ``(H/ps, W/ps, C, pool_area − 1)`` and ``log_prob(value, *,
            condition=z)`` must return a scalar.
        pool_size: Spatial pool size (default ``2`` → 2×2 windows).

    Shape:
        - Input  ``x``:  ``(H, W, C)``
        - Output ``z``:  ``(H/pool_size, W/pool_size, C)``
        - ``log_det``:   scalar (shape ``()``)

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gauss_flows import SimpleMaxPoolSurjection2d
        >>>
        >>> # ``my_decoder`` must implement the ConditionalDistribution
        >>> # protocol — see _src/_protocols.py.
        >>> surj = SimpleMaxPoolSurjection2d((8, 8, 3), my_decoder, pool_size=2)
        >>> # z.shape == (4, 4, 3); residuals.shape == (4, 4, 3, 3) (pool_area-1).
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    shape: tuple[int, ...]
    decoder: ConditionalDistribution
    pool_size: int
    cond_shape: ClassVar[None] = None

    def __init__(
        self,
        shape: tuple[int, ...],
        decoder: ConditionalDistribution,
        pool_size: int = 2,
    ):
        if pool_size <= 0:
            raise ValueError("pool_size must be positive.")
        if len(shape) != 3:
            raise ValueError(
                "SimpleMaxPoolSurjection2d requires a (H, W, C) event shape."
            )
        h, w, _ = shape
        if h % pool_size != 0 or w % pool_size != 0:
            raise ValueError("Spatial dimensions must be divisible by pool_size.")
        self.shape = shape
        self.decoder = decoder
        self.pool_size = pool_size

    @property
    def _pool_area(self) -> int:
        return self.pool_size * self.pool_size

    @property
    def _num_positions(self) -> int:
        h, w, c = self.shape
        return (h // self.pool_size) * (w // self.pool_size) * c

    @property
    def _ldj_k(self) -> float:
        return -math.log(self._pool_area) * self._num_positions

    def _squeeze(self, x: Array) -> Array:
        # x: (H, W, C) -> (H/ps, W/ps, C, pool_area)
        return rearrange(
            x,
            "(h ph) (w pw) c -> h w c (ph pw)",
            ph=self.pool_size,
            pw=self.pool_size,
        )

    def _unsqueeze(self, xs: Array) -> Array:
        # xs: (H/ps, W/ps, C, pool_area) -> (H, W, C)
        return rearrange(
            xs,
            "h w c (ph pw) -> (h ph) (w pw) c",
            ph=self.pool_size,
            pw=self.pool_size,
        )

    def _deconstruct_x(self, x: Array) -> tuple[Array, Array, Array]:
        """x → (z, x_residuals, k); shapes documented inline."""
        # xs: (H/ps, W/ps, C, pool_area)
        xs = self._squeeze(x)
        # z, k: (H/ps, W/ps, C); k is the argmax slot in [0, pool_area).
        z = xs.max(axis=-1)
        k = jnp.argmax(xs, axis=-1)
        # Static gather of the (pool_area - 1) non-argmax positions per pool —
        # boolean masking on a runtime mask isn't traceable under jit/vmap
        # (NonConcreteBooleanIndexError). gather_idx[..., i] = i if i < k else
        # i + 1 skips the argmax slot in O(1).
        idx = jnp.arange(self._pool_area - 1)
        gather_idx = jnp.where(idx < k[..., None], idx, idx + 1)
        # residuals: (H/ps, W/ps, C, pool_area - 1) — non-max values
        residuals = jnp.take_along_axis(xs, gather_idx, axis=-1)
        # x_residuals: same shape, encoded as (max - value) ≥ 0.
        x_residuals = z[..., None] - residuals
        return z, x_residuals, k

    def _construct_x(self, z: Array, x_residuals: Array, k: Array) -> Array:
        """(z, x_residuals, k) → x; the inverse of _deconstruct_x."""
        # ``k`` is marginalised out of the log-det via the ``-N·log(pool_area)``
        # entropy term; here it only decides which slot the pooled max ``z``
        # gets slotted into when we reassemble the pool-sized window.
        # others: (H/ps, W/ps, C, pool_area - 1) — the actual non-max values.
        others = z[..., None] - x_residuals
        # Flatten spatial+channel dims into a single batch for vmap.
        flat_z = rearrange(z, "h w c -> (h w c)")
        flat_k = rearrange(k, "h w c -> (h w c)")
        flat_others = rearrange(others, "h w c p -> (h w c) p")

        def _assemble(z_val: Array, k_val: Array, others_row: Array) -> Array:
            # Place ``others_row`` (length pool_area - 1) into a window of
            # length pool_area, with ``z_val`` slotted at position k_val.
            # combined: (pool_area,) where the last entry is z_val.
            combined = jnp.concatenate([others_row, jnp.array([z_val])])
            idx = jnp.arange(self._pool_area)
            shift = (idx > k_val).astype(idx.dtype)
            gather_idx = idx - shift
            gather_idx = jnp.where(idx == k_val, self._pool_area - 1, gather_idx)
            return combined[gather_idx]

        # xs_flat: ((H/ps)·(W/ps)·C, pool_area)
        xs_flat = jax.vmap(_assemble)(flat_z, flat_k, flat_others)
        # xs: (H/ps, W/ps, C, pool_area)
        xs = rearrange(
            xs_flat,
            "(h w c) p -> h w c p",
            h=z.shape[0],
            w=z.shape[1],
            c=z.shape[2],
        )
        # back to (H, W, C)
        return self._unsqueeze(xs)

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Pool x to z; return ``(z, log q(residuals|z) - N·log(pool_area))``."""
        del key, cond
        z, x_residuals, _k = self._deconstruct_x(jnp.asarray(x))
        log_det = self.decoder.log_prob(x_residuals, condition=z) + self._ldj_k
        return z, log_det

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample residuals + uniform argmax slot; reconstruct x."""
        del cond
        z_arr = jnp.asarray(z)
        key_k, key_res = jr.split(key)
        # k: (H/ps, W/ps, C) uniform over {0, …, pool_area - 1}.
        k = jr.randint(key_k, z_arr.shape, 0, self._pool_area)
        # x_residuals: (H/ps, W/ps, C, pool_area - 1) from the decoder.
        x_residuals = self.decoder.sample(key_res, condition=z_arr)
        x = self._construct_x(z_arr, x_residuals, k)
        log_det = self.decoder.log_prob(x_residuals, condition=z_arr) + self._ldj_k
        return x, log_det


class Slice(AbstractSurjection):
    """Inference surjection that drops the trailing dimensions of a 1D event.

    Forward keeps the first ``keep_dims`` entries of ``x`` and scores the dropped
    tail under ``decoder`` conditioned on the kept prefix:

        forward:  z = x[:keep_dims]                  (deterministic)
                  log_det = log q(x[keep_dims:] | z)

        inverse:  dropped ~ q(· | z)                 (decoder sample)
                  x = concat([z, dropped])

    The forward log-det contribution is exact when the decoder ``q`` matches the
    true conditional density of the dropped dims given the kept dims; otherwise
    it is a (typically loose) lower bound on ``log p(x)``.

    Args:
        keep_dims: Number of leading entries to keep. Must be positive.
        decoder: Object satisfying :class:`ConditionalDistribution` —
            ``sample(key, *, condition=z)`` must produce arrays of shape
            ``(D − keep_dims,)`` for the dropped tail and
            ``log_prob(value, *, condition=z)`` must return a scalar.

    Shape:
        - Input  ``x``:  ``(D,)`` (with ``D > keep_dims``)
        - Output ``z``:  ``(keep_dims,)``
        - ``log_det``:   scalar (shape ``()``)

    Note:
        Unlike most transforms, ``Slice`` does **not** expose a ``shape``
        attribute. Its forward-input dim ``D`` is determined by whatever the
        upstream transform produces — it is a chain-context property, not a
        constructor parameter, so there is no fixed ``shape`` to declare. The
        full data shape lives on :class:`SurVAEFlow.data_shape` instead.

    Example:
        Score the dropped tail under a unit Gaussian decoder:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from flowjax.distributions import Normal
        >>> from gauss_flows import Slice
        >>>
        >>> dec = Normal(jnp.zeros(2))   # decoder for the dropped 2 dims
        >>> surj = Slice(keep_dims=3, decoder=dec)
        >>> x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> z, log_det = surj.forward_and_log_det(x, jr.key(0))
        >>> # z == [1., 2., 3.]; log_det == sum of Normal(0,1).log_prob([4., 5.])

        Inside a SurVAEFlow that maps (4,) data → (2,) latent — note the
        explicit ``data_shape`` since the chain changes dimensionality:

        >>> from gauss_flows import SurVAEFlow
        >>> base = Normal(jnp.zeros(2))
        >>> flow = SurVAEFlow(
        ...     base,
        ...     [Slice(keep_dims=2, decoder=Normal(jnp.zeros(2)))],
        ...     data_shape=(4,),
        ... )
    """

    stochastic_forward: ClassVar[bool] = False
    stochastic_inverse: ClassVar[bool] = True
    lower_bound: ClassVar[bool] = False
    keep_dims: int
    decoder: ConditionalDistribution

    def __init__(self, keep_dims: int, decoder: ConditionalDistribution):
        if keep_dims <= 0:
            raise ValueError("keep_dims must be positive.")
        self.keep_dims = keep_dims
        self.decoder = decoder

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Drop trailing dims; return ``(kept, log q(dropped | kept))``."""
        del key, cond
        x_arr = jnp.asarray(x)
        # x: (D,) -> kept: (keep_dims,), dropped: (D - keep_dims,)
        kept = x_arr[: self.keep_dims]
        dropped = x_arr[self.keep_dims :]
        log_det = self.decoder.log_prob(dropped, condition=kept)
        return kept, log_det

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample the missing tail from the decoder; concat back.

        Returns ``log q(dropped_sampled | z)`` to mirror
        ``forward_and_log_det`` and match :class:`SimpleMaxPoolSurjection2d`'s
        inverse convention. ``SurVAEFlow.log_prob`` doesn't consume this value
        (it only uses ``forward_and_log_det``); the consistent return shape is
        for callers using the inverse standalone (e.g. ELBO computations).
        """
        del cond
        z_arr = jnp.asarray(z)
        # dropped: (D - keep_dims,) sampled from decoder; x: (D,)
        dropped = self.decoder.sample(key, condition=z_arr)
        x = jnp.concatenate([z_arr, dropped])
        log_det = self.decoder.log_prob(dropped, condition=z_arr)
        return x, log_det


class Augment(AbstractSurjection):
    """Generative surjection that appends conditionally sampled dimensions.

    Forward draws ``augment_size`` extra dims from ``encoder`` conditioned on
    the input and concatenates them; inverse drops the augmented tail:

        forward:  z_aug ~ q(· | x)                   (encoder sample)
                  z       = concat([x, z_aug])
                  log_det = − log q(z_aug | x)

        inverse:  x = z[:x_size]                     (deterministic drop)

    Because the forward direction is stochastic, the contribution to
    ``log_prob`` is an ELBO over the encoder's samples, not an exact density.
    Use Augment to give a flow extra latent dimensions to model multi-modal
    structure that would not fit in the data dim alone.

    Args:
        encoder: Object satisfying :class:`ConditionalDistribution` —
            ``sample(key, *, condition=x)`` must produce arrays of shape
            ``(augment_size,)`` and ``log_prob(value, *, condition=x)`` must
            return a scalar.
        x_size: Size of the data input event. Determines ``self.shape``.
        augment_size: Number of latent dims to append.

    Attributes:
        shape: ``(x_size,)`` — the forward-input event shape, matching the
            project-wide ``x.shape == self.shape`` convention. ``forward``
            runtime-validates this.

    Shape:
        - Input  ``x``:  ``(x_size,)``
        - Output ``z``:  ``(x_size + augment_size,)``
        - ``log_det``:   scalar (shape ``()``)

    Example:
        Augment a 2-D data event with a 2-D learned latent:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from flowjax.distributions import Normal
        >>> from gauss_flows import Augment
        >>>
        >>> enc = Normal(jnp.zeros(2))
        >>> surj = Augment(encoder=enc, x_size=2, augment_size=2)
        >>> x = jnp.array([0.5, -0.3])
        >>> z, log_det = surj.forward_and_log_det(x, jr.key(0))
        >>> # z.shape == (4,); log_det == -Normal(0,1).log_prob(z[2:]).sum()

        Inside a SurVAEFlow that maps (2,) data → (4,) latent — supply the
        explicit ``data_shape`` since the chain changes dimensionality:

        >>> from gauss_flows import SurVAEFlow
        >>> base = Normal(jnp.zeros(4))
        >>> flow = SurVAEFlow(
        ...     base,
        ...     [Augment(encoder=Normal(jnp.zeros(2)), x_size=2, augment_size=2)],
        ...     data_shape=(2,),
        ... )
    """

    stochastic_forward: ClassVar[bool] = True
    stochastic_inverse: ClassVar[bool] = False
    lower_bound: ClassVar[bool] = True
    shape: tuple[int, ...]
    encoder: ConditionalDistribution
    x_size: int
    augment_size: int

    def __init__(
        self,
        encoder: ConditionalDistribution,
        x_size: int,
        augment_size: int,
    ):
        if x_size <= 0:
            raise ValueError("x_size must be positive.")
        if augment_size <= 0:
            raise ValueError("augment_size must be positive.")
        self.shape = (x_size,)
        self.encoder = encoder
        self.x_size = x_size
        self.augment_size = augment_size

    def forward_and_log_det(
        self,
        x: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample ``z_aug`` from encoder; ``z = concat([x, z_aug])``."""
        del cond
        x_arr = jnp.asarray(x)
        if x_arr.shape != self.shape:
            raise ValueError(
                f"Augment.forward expected x.shape == {self.shape}; got {x_arr.shape}."
            )
        # z_aug: (augment_size,); z: (x_size + augment_size,)
        z_aug = self.encoder.sample(key, condition=x_arr)
        log_det = -self.encoder.log_prob(z_aug, condition=x_arr)
        z = jnp.concatenate([x_arr, z_aug])
        return z, log_det

    def inverse_and_log_det(
        self,
        z: ArrayLike,
        key: PRNGKeyArray,
        cond: Array | None = None,
    ) -> tuple[Array, Array]:
        """Drop the augmented tail; deterministic, log_det = 0."""
        del key, cond
        z_arr = jnp.asarray(z)
        # x: (x_size,)
        x = z_arr[: self.x_size]
        return x, jnp.zeros(())


__all__ = [
    "Augment",
    "SimpleAbsSurjection",
    "SimpleMaxPoolSurjection2d",
    "SimpleSortSurjection",
    "Slice",
]
