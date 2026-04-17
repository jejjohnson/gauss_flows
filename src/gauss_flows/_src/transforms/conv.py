"""Convolutional transforms for Gaussianization flows.

These bijections implement invertible convolution operations used in
image-based normalizing flows (e.g., Glow-style architectures).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array, lax
from jaxtyping import ArrayLike, PRNGKeyArray


class Invertible1x1Conv(AbstractBijection):
    """Invertible 1x1 convolution as used in Glow (https://arxiv.org/abs/1807.03039).

    Parameterizes an invertible linear mixing across channels/features using
    an LU decomposition to ensure invertibility and efficient log-det computation.
    The weight matrix is implicitly W = L @ U where L is lower triangular with
    unit diagonal and U is upper triangular with positive diagonal (stored as
    log_diag_u). Log-det is computed cheaply as sum(log_diag_u).

    Args:
        key: JAX random key.
        n_channels: Number of channels/features.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    lower_off: Array  # lower-triangular off-diagonal entries of L
    upper_off: Array  # upper-triangular off-diagonal entries of U
    log_diag_u: Array  # log of diagonal of U (ensures non-zero det)

    def __init__(self, key: PRNGKeyArray, n_channels: int):
        self.shape = (n_channels,)
        n = n_channels
        # Initialize near identity: small random off-diagonals, unit diagonal
        k1, k2 = jr.split(key)
        n_off = n * (n - 1) // 2
        self.lower_off = jr.normal(k1, (n_off,)) * 0.01
        self.upper_off = jr.normal(k2, (n_off,)) * 0.01
        self.log_diag_u = jnp.zeros(n)

    def _get_weight(self) -> Array:
        n = self.shape[0]
        idx_lower = jnp.tril_indices(n, k=-1)
        idx_upper = jnp.triu_indices(n, k=1)
        L = jnp.eye(n).at[idx_lower].set(self.lower_off)
        # Build U: zero off-diagonal, then set upper triangle and diagonal explicitly
        U = jnp.zeros((n, n)).at[idx_upper].set(self.upper_off)
        U = U.at[jnp.diag_indices(n)].set(jnp.exp(self.log_diag_u))
        return L @ U

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        W = self._get_weight()
        y = W @ jnp.asarray(x)
        log_det = jnp.sum(self.log_diag_u)
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        W = self._get_weight()
        x = jnp.linalg.solve(W, jnp.asarray(y))
        log_det = -jnp.sum(self.log_diag_u)
        return x, log_det


class ActNorm(AbstractBijection):
    """Activation normalization (ActNorm).

    Performs per-channel normalization with learnable location and scale.
    For images, operates on the channel dimension.

    Args:
        shape: Shape of the input.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    log_scale: Array

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        n_dims = shape[-1]
        self.loc = jnp.zeros(n_dims)
        self.log_scale = jnp.zeros(n_dims)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        from jax.nn import softplus

        scale = softplus(self.log_scale) + 1e-5
        y = (jnp.asarray(x) - self.loc) / scale
        # Multiply by number of non-channel positions (spatial dims) so that
        # the log-det accounts for each broadcasted application of the scale.
        n_spatial = (
            int(jnp.prod(jnp.array(self.shape[:-1]))) if len(self.shape) > 1 else 1
        )
        log_det = -n_spatial * jnp.sum(jnp.log(scale))
        return y, log_det

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        from jax.nn import softplus

        scale = softplus(self.log_scale) + 1e-5
        x = jnp.asarray(y) * scale + self.loc
        n_spatial = (
            int(jnp.prod(jnp.array(self.shape[:-1]))) if len(self.shape) > 1 else 1
        )
        log_det = n_spatial * jnp.sum(jnp.log(scale))
        return x, log_det


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


class Squeeze(AbstractBijection):
    """Squeeze operation for image-like inputs.

    Reshapes spatial dimensions into channels, halving the spatial resolution
    and quadrupling the channels. Commonly used in multi-scale flow architectures.

    Args:
        shape: Shape of the input (H, W, C) or (N, C).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    out_shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        if len(shape) == 1:
            # 1D case: split into two halves → (n//2, 2)
            n = shape[0]
            if n % 2 != 0:
                raise ValueError("1D squeeze requires even dimension size.")
            self.out_shape = (n // 2, 2)
        elif len(shape) == 2:
            h, c = shape
            if h % 2 != 0:
                raise ValueError("Height must be divisible by 2 for Squeeze.")
            self.out_shape = (h // 2, c * 2)
        else:
            self.out_shape = shape

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        # Volume-preserving: log det = 0
        return jnp.asarray(x).reshape(self.out_shape), jnp.zeros(())

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        return jnp.asarray(y).reshape(self.shape), jnp.zeros(())


class OrthogonalConvExponential(AbstractBijection):
    r"""Orthogonal convolution via the exponential of a skew-symmetric operator.

    Motivation
    ----------
    Glow-style image flows need an invertible linear mixer across pixels *and*
    channels whose log-determinant is cheap to evaluate. A convolution with a
    spatially-extended kernel has the desired receptive field, but is not
    invertible in general. The convolution exponential (Hoogeboom et al., 2020)
    lifts a possibly non-invertible conv operator :math:`M` to

    .. math::

        \exp(M) \;=\; \sum_{k=0}^{\infty} \frac{M^k}{k!},

    which is *always* invertible (its inverse is :math:`\exp(-M)`). Taking
    :math:`M` skew-symmetric (:math:`M^\top = -M`) further makes :math:`\exp(M)`
    an **orthogonal** matrix, so the bijection is norm-preserving and has
    Jacobian determinant exactly :math:`+1`.

    Construction
    ------------
    **1. Skew-symmetric conv operator.** For a :math:`(k_h, k_w, C, C)` kernel
    :math:`K` indexed relative to its center, the induced circular-convolution
    Toeplitz operator :math:`M_K` acting on flattened images has the transpose

    .. math::

        (M_K^\top)_{p,q} \;=\; K[k_h{-}1{-}i,\,k_w{-}1{-}j,\,b,\,a]\quad
            \text{when}\quad
        (M_K)_{p,q} \;=\; K[i,\,j,\,a,\,b].

    Skew-symmetry :math:`M_K = -M_K^\top` therefore holds iff

    .. math::

        K[i, j, a, b] \;=\; -\,K[k_h{-}1{-}i,\,k_w{-}1{-}j,\,b,\,a].

    We enforce this exactly by setting :math:`K = W - \widetilde{W}` where
    :math:`\widetilde{W}[i,j,a,b] = W[k_h{-}1{-}i,\,k_w{-}1{-}j,\,b,\,a]`
    (flip spatial axes + swap input/output channels). See :meth:`_skew`.

    **2. Truncated Taylor series.** The forward map applies :math:`\exp(M_K)`
    via the :math:`N`-term truncation

    .. math::

        p_N(M_K)\,x
            \;=\; \sum_{k=0}^{N} \frac{M_K^k\, x}{k!}
            \;=\; \bigl(I + M_K + \tfrac{1}{2!}M_K^2 + \dots\bigr)\,x,

    which we compute with a :func:`jax.lax.scan` using a separate
    Taylor-term accumulator :math:`t_k`: starting from
    :math:`t_0 = r_0 = x`, we update

    .. math::

        t_{k+1} \;=\; \frac{1}{k+1}\, M_K\, t_k, \qquad
        r_{k+1} \;=\; r_k + t_{k+1},

    so that :math:`r_N = p_N(M_K)\,x` (see
    :meth:`_convolution_exponential`). The truncation error is

    .. math::

        \lVert p_N(M) - \exp(M) \rVert
            \;=\; \mathcal{O}\!\bigl(\|M\|^{N+1}/(N{+}1)!\bigr),

    so with ``n_terms = 8`` and ``||M|| <= 1`` the residual is below
    :math:`1/9! \approx 2.8\times 10^{-6}`.

    **3. Spectral normalization.** To keep :math:`\|M_K\| \leq 1`, we estimate
    the operator norm via image-space power iteration and divide by
    :math:`\max(\sigma, 1)`. Gradients are blocked through the
    :math:`\sigma`-estimate (:func:`jax.lax.stop_gradient`), so training does
    *not* differentiate through the power iteration itself — only through the
    explicit ``kernel / sigma`` rescale. See :meth:`_spectral_normalize`.

    **4. Log-determinant.** Because :math:`M_K` is skew-symmetric its trace
    vanishes, and

    .. math::

        \log\!\bigl|\det \exp(M_K)\bigr|
            \;=\; \operatorname{tr}(M_K) \;=\; 0

    exactly. We return ``jnp.zeros(())`` — the truncated operator's log-det
    differs from this by :math:`\mathcal{O}(\|M\|^{N+1}/(N{+}1)!)`, the same
    negligible quantity as the round-trip residual.

    **5. Inverse.** Since :math:`\exp(M)^{-1} = \exp(-M)` for any :math:`M`,
    the inverse applies the same truncated series with negated kernel.
    Composing :math:`p_N(-M_K)\circ p_N(M_K)` leaves an
    :math:`\mathcal{O}(\|M\|^{N+1}/(N{+}1)!)` residual, acceptable for any
    reasonable :math:`N`.

    Shapes
    ------
    ::

        x, y                     : (H, W, C)
        weight / skew kernel K   : (k_h, k_w, C, C) in flowjax/JAX HWIO
        log_det                  : ()                 (scalar, always 0)

    Args:
        key: JAX random key for kernel initialization.
        shape: Image shape ``(H, W, C)`` — single event (no batch axis).
        kernel_size: Spatial kernel size (must be a positive **odd** integer
            so the kernel has a unique center pixel).
        n_terms: Number of Taylor-series terms ``N`` (``>= 1``). Truncation
            error scales as :math:`\mathcal{O}(1 / (N{+}1)!)`.
        n_power_iterations: Power-iteration steps for the spectral-norm
            estimate (``>= 1``). Two iterations is usually enough because
            :math:`\sigma` only needs an upper bound, not high accuracy.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> key = jr.key(0)
        >>> transform = OrthogonalConvExponential(key, shape=(8, 8, 3))
        >>> x = jr.normal(key, (8, 8, 3))
        >>> y, log_det = transform.transform_and_log_det(x)
        >>> x_rec, log_det_inv = transform.inverse_and_log_det(y)
        >>> assert jnp.allclose(x, x_rec, atol=1e-4)
        >>> assert jnp.isclose(log_det, 0.0)

    References:
        Hoogeboom, E., van den Berg, R., & Welling, M.
        *The Convolution Exponential and Generalized Sylvester Flows*,
        NeurIPS 2020. https://arxiv.org/abs/2006.01910
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    weight: Array  # (k_h, k_w, C, C) free parameter; skew kernel derived per-call
    spectral_seed: Array  # (H, W, C) persistent unit-norm seed for power iteration
    n_terms: int
    n_power_iterations: int

    def __init__(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, int, int],
        *,
        kernel_size: int = 3,
        n_terms: int = 8,
        n_power_iterations: int = 2,
    ):
        if len(shape) != 3:
            raise ValueError(
                f"OrthogonalConvExponential expects shape (H, W, C); got {shape!r}."
            )
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer; got {kernel_size}."
            )
        if n_terms < 1:
            raise ValueError(f"n_terms must be >= 1; got {n_terms}.")
        if n_power_iterations < 1:
            raise ValueError(
                f"n_power_iterations must be >= 1; got {n_power_iterations}."
            )

        self.shape = tuple(shape)
        c = int(shape[-1])
        self.n_terms = int(n_terms)
        self.n_power_iterations = int(n_power_iterations)
        # Split the user's key: weight init and the persistent spectral seed
        # must be independent so a pathologically-aligned weight/seed pair is
        # measure-zero rather than correlated by construction.
        w_key, s_key = jr.split(key)
        # Small weight init: the skew kernel K = W - flip(W)^T doubles scale,
        # and spectral norm caps ||K|| <= 1, so near-identity behavior at
        # step 0. weight: (k_h, k_w, C, C).
        self.weight = jr.normal(w_key, (kernel_size, kernel_size, c, c)) * 0.05
        # Persistent random (H, W, C) seed for power iteration. A random
        # vector has non-zero projection on every fixed singular direction
        # with probability 1; a deterministic seed (constant, sin(arange),
        # ...) can collide with ker(M_K) for specific kernel configurations
        # (e.g. shape=(1,1,3) + kernel_size=1 yields a 3x3 skew matrix
        # whose 1-D nullspace could align with a fixed pattern). Grads are
        # blocked in _spectral_normalize so this field stays constant.
        raw_seed = jr.normal(s_key, self.shape)
        self.spectral_seed = raw_seed / (jnp.linalg.norm(raw_seed) + 1e-8)

    @staticmethod
    def _skew(weight: Array) -> Array:
        r"""Project a free weight tensor to a skew-symmetric conv kernel.

        Given free parameter ``W`` of shape ``(k_h, k_w, C, C)``, returns

        .. math::

            K \;=\; W \,-\, \widetilde{W},\qquad
            \widetilde{W}[i, j, a, b] \;=\; W[k_h{-}1{-}i,\,k_w{-}1{-}j,\,b,\,a].

        Under circular padding, the induced Toeplitz operator ``M_K`` satisfies
        :math:`M_K = -M_K^\top`, so :math:`\exp(M_K)` is orthogonal.

        Shape:
            ``weight``: ``(k_h, k_w, C, C)``
            returns:    ``(k_h, k_w, C, C)`` (same shape, skew-symmetric)
        """
        # flip spatial dims -> reverse (i, j) indices relative to center;
        # swapaxes(2, 3) -> swap input/output channel axes (the "transpose" ^T).
        flipped = jnp.swapaxes(jnp.flip(weight, axis=(0, 1)), 2, 3)
        return weight - flipped  # (k_h, k_w, C, C)

    def _circular_conv(self, x: Array, kernel: Array) -> Array:
        """Single-event circular 2D convolution.

        Pads ``x`` with wrap-mode boundary so that VALID-padded convolution
        produces the circular (periodic) result, then delegates to
        :func:`jax.lax.conv_general_dilated` with NHWC/HWIO layout.

        Shape:
            ``x``      : ``(H, W, C_in)``
            ``kernel`` : ``(k_h, k_w, C_in, C_out)`` (HWIO)
            returns    : ``(H, W, C_out)``
        """
        k_h, k_w = kernel.shape[0], kernel.shape[1]
        pad_h, pad_w = k_h // 2, k_w // 2
        # x:        (H, W, C)          -> x_padded: (H + 2*pad_h, W + 2*pad_w, C)
        x_padded = jnp.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="wrap")
        # x_padded[None]: (1, Hp, Wp, C) -> y: (1, H, W, C_out) under VALID padding.
        y = lax.conv_general_dilated(
            x_padded[None, ...],
            kernel,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        return y[0]  # (H, W, C_out)

    def _circular_conv_transpose(self, x: Array, kernel: Array) -> Array:
        r"""Mathematical transpose (adjoint) of circular conv.

        For a circular conv with kernel :math:`K` the operator adjoint is the
        circular conv with the *index-flipped, channel-transposed* kernel
        :math:`K^\top[i, j, o, c] = K[k_h{-}1{-}i, k_w{-}1{-}j, c, o]`.

        Shape:
            ``x``      : ``(H, W, C_out)``
            ``kernel`` : ``(k_h, k_w, C_in, C_out)``
            returns    : ``(H, W, C_in)``
        """
        # Flip spatial -> reverse (i, j); swapaxes(2, 3) -> swap I<->O.
        kernel_t = jnp.swapaxes(jnp.flip(kernel, axis=(0, 1)), 2, 3)
        return self._circular_conv(x, kernel_t)

    def _spectral_normalize(self, kernel: Array) -> Array:
        r"""Clip kernel spectral norm to at most 1 via power iteration.

        Estimates :math:`\sigma \approx \|M_K\|_{op}` using the standard SVD
        power iteration in image space: alternately apply :math:`M_K` and
        :math:`M_K^\top`, renormalizing after each step. The estimate is
        wrapped in :func:`jax.lax.stop_gradient` so backprop does *not*
        differentiate through the iteration itself — gradients flow only
        through the explicit ``kernel / sigma`` rescale.

        After :math:`n` iterations, :math:`\sigma` converges to
        :math:`\sigma_1 = \|M_K\|` with ratio :math:`(\sigma_2/\sigma_1)^n`.
        Two iterations suffice because we only need an *upper bound* (we
        take :math:`\max(\sigma, 1)`), not high accuracy.

        **Seed choice.** The seed must have non-zero projection on the
        leading singular vector of :math:`M_K`. Any *deterministic* seed
        (constant, :math:`\sin(\text{arange})`, ...) can collide with
        :math:`\ker(M_K)` for specific kernel configurations — e.g. a
        constant vector is in :math:`\ker(M_K)` for skew-symmetric kernels
        with :math:`C = 1`, and any fixed pattern in :math:`\mathbb{R}^3`
        can land in the 1-D nullspace of a ``shape=(1,1,3)`` +
        ``kernel_size=1`` skew kernel. We therefore use a **persistent
        random** seed stored at construction time
        (:attr:`spectral_seed`). A random Gaussian has non-zero projection
        on every fixed direction with probability 1, so alignment with
        :math:`\ker(M_K)` over the kernel trajectory during training is
        measure-zero.

        Shape:
            ``kernel`` : ``(k_h, k_w, C, C)``
            ``u``, ``v`` (power-iteration vectors): ``(H, W, C)``
            returns    : ``(k_h, k_w, C, C)``  (rescaled, grad-flowing)
        """
        # Persistent random seed: measure-zero probability of alignment
        # with ker(M_K). stop_gradient blocks gradient flow into the seed
        # field (it stays constant across training steps).
        u0 = lax.stop_gradient(self.spectral_seed)

        # Block gradients through the power iteration: we only want grads
        # through the *rescale* `kernel / sigma`, not through the estimate.
        kernel_sg = lax.stop_gradient(kernel)

        def body(_, u_vec):
            # v = M^T u, normalized  -> singular direction on input side.
            v_vec = self._circular_conv_transpose(u_vec, kernel_sg)  # (H, W, C)
            v_vec = v_vec / (jnp.linalg.norm(v_vec) + 1e-8)
            # u = M v, normalized    -> singular direction on output side.
            u_new = self._circular_conv(v_vec, kernel_sg)  # (H, W, C)
            return u_new / (jnp.linalg.norm(u_new) + 1e-8)

        # After n_power_iterations steps, u ~ leading left-singular vector.
        u = lax.fori_loop(0, self.n_power_iterations, body, u0)  # (H, W, C)
        # Final pairing to estimate sigma = ||M v|| where v = M^T u normalized.
        v = self._circular_conv_transpose(u, kernel_sg)  # (H, W, C)
        v = v / (jnp.linalg.norm(v) + 1e-8)
        sigma = jnp.linalg.norm(self._circular_conv(v, kernel_sg))  # scalar

        # max(sigma, 1): only rescale if the norm *exceeds* 1; otherwise
        # leave the kernel alone so near-identity init isn't perturbed.
        scale = jnp.maximum(lax.stop_gradient(sigma), 1.0)
        return kernel / (scale + 1e-8)  # (k_h, k_w, C, C)

    def _convolution_exponential(self, x: Array, kernel: Array) -> Array:
        r"""Apply the truncated matrix exponential :math:`p_N(M_K)\,x`.

        Computes the partial Taylor sum

        .. math::

            r_N \;=\; \sum_{k=0}^{N} \frac{M_K^k\, x}{k!},

        via the recurrence :math:`t_0 = x`, :math:`t_{k+1} = \tfrac{1}{k+1} M_K\, t_k`,
        :math:`r_{k+1} = r_k + t_{k+1}`. Each iteration costs one circular conv
        (``O(H W k_h k_w C^2)``); total cost scales linearly in ``n_terms``.

        Shape:
            ``x``, ``kernel``, returns as in :meth:`_circular_conv`.
        """

        def body(carry, i):
            prod, acc = carry  # both (H, W, C)
            # t_{k+1} = (1/(k+1)) * M_K * t_k
            prod = self._circular_conv(prod, kernel) / (i + 1)
            return (prod, acc + prod), None

        # carry0 = (t_0, r_0) = (x, x)  — the k=0 term is I*x = x.
        (_, result), _ = lax.scan(body, (x, x), jnp.arange(self.n_terms))
        return result  # (H, W, C)

    def transform_and_log_det(self, x: ArrayLike, condition=None):
        """Forward map :math:`y = \\exp(M_K)\\,x` and scalar log-det (=0).

        Shape:
            ``x``    : ``(H, W, C)``
            returns  : ``((H, W, C), ())``
        """
        x_arr = jnp.asarray(x)  # (H, W, C)
        # 1) W -> K (skew)  2) K -> K / max(sigma, 1)  3) apply p_N(M_K)
        kernel = self._spectral_normalize(self._skew(self.weight))
        y = self._convolution_exponential(x_arr, kernel)  # (H, W, C)
        # Skew-symmetric M  =>  tr(M) = 0  =>  log|det exp(M)| = 0 exactly.
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y: ArrayLike, condition=None):
        """Inverse map :math:`x = \\exp(-M_K)\\,y` and scalar log-det (=0).

        Uses the identity :math:`\\exp(M)^{-1} = \\exp(-M)` — truncated at
        the same ``n_terms`` as forward, so the round-trip residual is
        ``O(||M||^{n_terms+1} / (n_terms+1)!)``.

        Shape:
            ``y``    : ``(H, W, C)``
            returns  : ``((H, W, C), ())``
        """
        y_arr = jnp.asarray(y)  # (H, W, C)
        kernel = self._spectral_normalize(self._skew(self.weight))
        # exp(-M_K) via the same truncation with negated kernel.
        x = self._convolution_exponential(y_arr, -kernel)  # (H, W, C)
        return x, jnp.zeros(())


__all__ = [
    "ActNorm",
    "HaarWavelet",
    "Invertible1x1Conv",
    "OrthogonalConvExponential",
    "Squeeze",
]
