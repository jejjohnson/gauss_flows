"""Orthogonal convolution via matrix exponential of a skew-symmetric operator."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import AbstractBijection
from jax import Array, lax
from jaxtyping import ArrayLike, PRNGKeyArray


class OrthogonalConvExponential(AbstractBijection):
    r"""Orthogonal convolution via the exponential of a skew-symmetric operator.

    **Motivation**
    Glow-style image flows need an invertible linear mixer across pixels *and*
    channels whose log-determinant is cheap to evaluate. A convolution with a
    spatially-extended kernel has the desired receptive field, but is not
    invertible in general. The convolution exponential (Hoogeboom et al., 2020)
    lifts a possibly non-invertible conv operator $M$ to

    $$
    \exp(M) \;=\; \sum_{k=0}^{\infty} \frac{M^k}{k!},
    $$

    which is *always* invertible (its inverse is $\exp(-M)$). Taking
    $M$ skew-symmetric ($M^\top = -M$) further makes $\exp(M)$
    an **orthogonal** matrix, so the bijection is norm-preserving and has
    Jacobian determinant exactly $+1$.

    **Construction**
    **1. Skew-symmetric conv operator.** For a $(k_h, k_w, C, C)$ kernel
    $K$ indexed relative to its center, the induced circular-convolution
    Toeplitz operator $M_K$ acting on flattened images has the transpose

    $$
    (M_K^\top)_{p,q} \;=\; K[k_h{-}1{-}i,\,k_w{-}1{-}j,\,b,\,a]\quad
        \text{when}\quad
    (M_K)_{p,q} \;=\; K[i,\,j,\,a,\,b].
    $$

    Skew-symmetry $M_K = -M_K^\top$ therefore holds iff

    $$
    K[i, j, a, b] \;=\; -\,K[k_h{-}1{-}i,\,k_w{-}1{-}j,\,b,\,a].
    $$

    We enforce this exactly by setting $K = W - \widetilde{W}$ where
    $\widetilde{W}[i,j,a,b] = W[k_h{-}1{-}i,\,k_w{-}1{-}j,\,b,\,a]$
    (flip spatial axes + swap input/output channels). See `_skew`.

    **2. Truncated Taylor series.** The forward map applies $\exp(M_K)$
    via the $N$-term truncation

    $$
    p_N(M_K)\,x
        \;=\; \sum_{k=0}^{N} \frac{M_K^k\, x}{k!}
        \;=\; \bigl(I + M_K + \tfrac{1}{2!}M_K^2 + \dots\bigr)\,x,
    $$

    which we compute with a `jax.lax.scan` using a separate
    Taylor-term accumulator $t_k$: starting from
    $t_0 = r_0 = x$, we update

    $$
    t_{k+1} \;=\; \frac{1}{k+1}\, M_K\, t_k, \qquad
    r_{k+1} \;=\; r_k + t_{k+1},
    $$

    so that $r_N = p_N(M_K)\,x$ (see
    `_convolution_exponential`). The truncation error is

    $$
    \lVert p_N(M) - \exp(M) \rVert
        \;=\; \mathcal{O}\!\bigl(\|M\|^{N+1}/(N{+}1)!\bigr),
    $$

    so with ``n_terms = 8`` and ``||M|| <= 1`` the residual is below
    $1/9! \approx 2.8\times 10^{-6}$.

    **3. Spectral normalization.** To keep $\|M_K\| \leq 1$, we estimate
    the operator norm via image-space power iteration and divide by
    $\max(\sigma, 1)$. Gradients are blocked through the
    $\sigma$-estimate (`jax.lax.stop_gradient`), so training does
    *not* differentiate through the power iteration itself — only through the
    explicit ``kernel / sigma`` rescale. See `_spectral_normalize`.

    **4. Log-determinant.** Because $M_K$ is skew-symmetric its trace
    vanishes, and

    $$
    \log\!\bigl|\det \exp(M_K)\bigr|
        \;=\; \operatorname{tr}(M_K) \;=\; 0
    $$

    exactly. We return ``jnp.zeros(())`` — the truncated operator's log-det
    differs from this by $\mathcal{O}(\|M\|^{N+1}/(N{+}1)!)$, the same
    negligible quantity as the round-trip residual.

    **5. Inverse.** Since $\exp(M)^{-1} = \exp(-M)$ for any $M$,
    the inverse applies the same truncated series with negated kernel.
    Composing $p_N(-M_K)\circ p_N(M_K)$ leaves an
    $\mathcal{O}(\|M\|^{N+1}/(N{+}1)!)$ residual, acceptable for any
    reasonable $N$.

    **Shapes**
    ```python
    x, y                     : (H, W, C)
    weight / skew kernel K   : (k_h, k_w, C, C) in flowjax/JAX HWIO
    log_det                  : ()                 (scalar, always 0)
    ```

    Args:
        key: JAX random key for kernel initialization.
        shape: Image shape ``(H, W, C)`` — single event (no batch axis).
        kernel_size: Spatial kernel size (must be a positive **odd** integer
            so the kernel has a unique center pixel).
        n_terms: Number of Taylor-series terms ``N`` (``>= 1``). Truncation
            error scales as $\mathcal{O}(1 / (N{+}1)!)$.
        n_power_iterations: Power-iteration steps for the spectral-norm
            estimate (``>= 1``). Two iterations is usually enough because
            $\sigma$ only needs an upper bound, not high accuracy.

    Examples:
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

        $$
        K \;=\; W \,-\, \widetilde{W},\qquad
        \widetilde{W}[i, j, a, b] \;=\; W[k_h{-}1{-}i,\,k_w{-}1{-}j,\,b,\,a].
        $$

        Under circular padding, the induced Toeplitz operator ``M_K`` satisfies
        $M_K = -M_K^\top$, so $\exp(M_K)$ is orthogonal.

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
        `jax.lax.conv_general_dilated` with NHWC/HWIO layout.

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

        For a circular conv with kernel $K$ the operator adjoint is the
        circular conv with the *index-flipped, channel-transposed* kernel
        $K^\top[i, j, o, c] = K[k_h{-}1{-}i, k_w{-}1{-}j, c, o]$.

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

        Estimates $\sigma \approx \|M_K\|_{op}$ using the standard SVD
        power iteration in image space: alternately apply $M_K$ and
        $M_K^\top$, renormalizing after each step. The estimate is
        wrapped in `jax.lax.stop_gradient` so backprop does *not*
        differentiate through the iteration itself — gradients flow only
        through the explicit ``kernel / sigma`` rescale.

        After $n$ iterations, $\sigma$ converges to
        $\sigma_1 = \|M_K\|$ with ratio $(\sigma_2/\sigma_1)^n$.
        Two iterations suffice because we only need an *upper bound* (we
        take $\max(\sigma, 1)$), not high accuracy.

        **Seed choice.** The seed must have non-zero projection on the
        leading singular vector of $M_K$. Any *deterministic* seed
        (constant, $\sin(\text{arange})$, ...) can collide with
        $\ker(M_K)$ for specific kernel configurations — e.g. a
        constant vector is in $\ker(M_K)$ for skew-symmetric kernels
        with $C = 1$, and any fixed pattern in $\mathbb{R}^3$
        can land in the 1-D nullspace of a ``shape=(1,1,3)`` +
        ``kernel_size=1`` skew kernel. We therefore use a **persistent
        random** seed stored at construction time
        (`spectral_seed`). A random Gaussian has non-zero projection
        on every fixed direction with probability 1, so alignment with
        $\ker(M_K)$ over the kernel trajectory during training is
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
        r"""Apply the truncated matrix exponential $p_N(M_K)\,x$.

        Computes the partial Taylor sum

        $$
        r_N \;=\; \sum_{k=0}^{N} \frac{M_K^k\, x}{k!},
        $$

        via the recurrence $t_0 = x$, $t_{k+1} = \tfrac{1}{k+1} M_K\, t_k$,
        $r_{k+1} = r_k + t_{k+1}$. Each iteration costs one circular conv
        (``O(H W k_h k_w C^2)``); total cost scales linearly in ``n_terms``.

        Shape:
            ``x``, ``kernel``, returns as in `_circular_conv`.
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
        """Forward map $y = \\exp(M_K)\\,x$ and scalar log-det (=0).

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
        """Inverse map $x = \\exp(-M_K)\\,y$ and scalar log-det (=0).

        Uses the identity $\\exp(M)^{-1} = \\exp(-M)$ — truncated at
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


__all__ = ["OrthogonalConvExponential"]
