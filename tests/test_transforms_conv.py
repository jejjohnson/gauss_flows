"""Tests for conv transforms."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax

from gauss_flows import (
    ActNorm,
    HaarWavelet,
    Invertible1x1Conv,
    Orthogonal1x1Conv,
    OrthogonalRotation,
    Squeeze,
)


def test_act_norm_forward_inverse(key):
    shape = (4,)
    act_norm = ActNorm(shape)
    x = jr.normal(key, shape)
    y, log_det = act_norm.transform_and_log_det(x)
    x_rec, log_det_inv = act_norm.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_act_norm_shape(key):
    act_norm = ActNorm((8,))
    x = jr.normal(key, (8,))
    y, log_det = act_norm.transform_and_log_det(x)
    assert y.shape == (8,)
    assert log_det.shape == ()


def test_invertible1x1conv_forward_inverse(key):
    n_channels = 4
    conv = Invertible1x1Conv(key, n_channels)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    x_rec, log_det_inv = conv.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_invertible1x1conv_shape(key):
    n_channels = 4
    conv = Invertible1x1Conv(key, n_channels)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    assert y.shape == (n_channels,)
    assert log_det.shape == ()


def test_haar_wavelet_forward_inverse(key):
    shape = (8,)
    haar = HaarWavelet(shape)
    x = jr.normal(key, shape)
    y, _log_det = haar.transform_and_log_det(x)
    x_rec, _log_det_inv = haar.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)


def test_squeeze_forward_inverse(key):
    shape = (4,)
    squeeze = Squeeze(shape)
    x = jr.normal(key, shape)
    # Call the underlying (unwrapped) methods to bypass flowjax's shape-check,
    # since Squeeze changes the array shape (volume-preserving reshape).
    y, log_det = Squeeze.transform_and_log_det.__wrapped__(squeeze, x)
    assert y.shape == (2, 2)  # 1D squeeze: (n,) → (n//2, 2)
    x_rec, log_det_inv = Squeeze.inverse_and_log_det.__wrapped__(squeeze, y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)
    assert log_det.shape == ()


def test_orthogonal1x1conv_learnable_forward_inverse(key):
    """Learnable path roundtrips exactly."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    x_rec, log_det_inv = conv.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_orthogonal1x1conv_learnable_shape(key):
    """Learnable path preserves shape."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    assert y.shape == (n_channels,)
    assert log_det.shape == ()


def test_orthogonal1x1conv_learnable_volume_preserving(key):
    """Learnable path reports log_det == 0 in both directions."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels)
    x = jr.normal(key, (n_channels,))
    _, log_det_f = conv.transform_and_log_det(x)
    _, log_det_i = conv.inverse_and_log_det(x)
    assert jnp.allclose(log_det_f, 0.0, atol=1e-5)
    assert jnp.allclose(log_det_i, 0.0, atol=1e-5)


def test_orthogonal1x1conv_fixed_forward_inverse(key):
    """Fixed path roundtrips exactly when given a valid orthogonal matrix."""
    n_channels = 4
    Q, _ = jnp.linalg.qr(jr.normal(jr.fold_in(key, 0), (n_channels, n_channels)))
    conv = Orthogonal1x1Conv(key, n_channels, fixed_matrix=Q)
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    x_rec, log_det_inv = conv.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_orthogonal1x1conv_fixed_identity_is_noop(key):
    """Supplying the identity matrix makes the layer a pass-through."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels, fixed_matrix=jnp.eye(n_channels))
    x = jr.normal(key, (n_channels,))
    y, log_det = conv.transform_and_log_det(x)
    assert jnp.allclose(x, y, atol=1e-6)
    assert jnp.allclose(log_det, 0.0, atol=1e-6)


def test_orthogonal1x1conv_fixed_matrix_shape_validation(key):
    """Wrong-shape fixed_matrix raises ValueError at construction."""
    import pytest

    with pytest.raises(ValueError, match=r"fixed_matrix must have shape"):
        Orthogonal1x1Conv(key, n_channels=4, fixed_matrix=jnp.eye(3))


def test_orthogonal1x1conv_fixed_matrix_orthogonality_validation(key):
    """Non-orthogonal fixed_matrix raises ValueError at construction."""
    import pytest

    # Diagonal matrix with non-unit entries — shape OK, orthogonality fails.
    non_orth = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match=r"fixed_matrix must be orthogonal"):
        Orthogonal1x1Conv(key, n_channels=4, fixed_matrix=non_orth)

    # A generic non-orthogonal matrix (random Gaussian, not QR'd).
    random_nonorth = jr.normal(jr.fold_in(key, 11), (4, 4))
    with pytest.raises(ValueError, match=r"fixed_matrix must be orthogonal"):
        Orthogonal1x1Conv(key, n_channels=4, fixed_matrix=random_nonorth)


def test_orthogonal1x1conv_fixed_matrix_accepts_custom_tolerance(key):
    """Loosened orthogonality_atol allows near-orthogonal inputs."""
    # Start from a QR rotation and perturb slightly to break tight tolerance.
    Q, _ = jnp.linalg.qr(jr.normal(jr.fold_in(key, 12), (4, 4)))
    Q_noisy = Q + 1e-3 * jr.normal(jr.fold_in(key, 13), Q.shape)
    # Default atol (1e-5) rejects; loose atol (1e-2) accepts.
    import pytest

    with pytest.raises(ValueError, match=r"fixed_matrix must be orthogonal"):
        Orthogonal1x1Conv(key, n_channels=4, fixed_matrix=Q_noisy)
    # No error — loose tolerance accepts the perturbed matrix (but the user
    # accepts the slight non-orthogonality of the inverse as a consequence).
    Orthogonal1x1Conv(key, n_channels=4, fixed_matrix=Q_noisy, orthogonality_atol=1e-2)


def test_orthogonal1x1conv_fixed_stays_fixed_under_sgd_step(key):
    """fixed_matrix survives a plain-SGD Optax step unchanged.

    Load-bearing regression test for the ``NonTrainable`` wrapping +
    ``paramax.unwrap``-based ``stop_gradient``. If that chain breaks,
    this test catches it.

    NB: This does **not** test ``optax.adamw``-style decoupled weight
    decay, which bypasses the gradient path and would drift
    ``fixed_matrix`` regardless. The docstring documents that users
    applying weight decay must ``eqx.partition`` out the fixed matrix
    before updating.
    """
    import paramax

    n_channels = 4
    Q0, _ = jnp.linalg.qr(jr.normal(jr.fold_in(key, 7), (n_channels, n_channels)))
    conv = Orthogonal1x1Conv(key, n_channels, fixed_matrix=Q0)
    x = jr.normal(jr.fold_in(key, 8), (n_channels,))

    def loss(m):
        y, _ = m.transform_and_log_det(x)
        return jnp.sum(y**2)

    grads = eqx.filter_grad(loss)(conv)
    optim = optax.sgd(1e-1)
    opt_state = optim.init(eqx.filter(conv, eqx.is_inexact_array))
    updates, _ = optim.update(grads, opt_state)
    updated = eqx.apply_updates(conv, updates)

    # After update, the (still-wrapped) fixed_matrix must unwrap to the
    # original Q0. Compare the materialised Q via _get_rotation_matrix
    # which goes through paramax.unwrap on both sides.
    before_Q = paramax.unwrap(conv).fixed_matrix
    after_Q = paramax.unwrap(updated).fixed_matrix
    assert jnp.allclose(after_Q, Q0, atol=0.0)
    assert jnp.allclose(before_Q, after_Q, atol=0.0)
    assert jnp.allclose(
        updated._get_rotation_matrix(), conv._get_rotation_matrix(), atol=0.0
    )


def test_orthogonal1x1conv_learnable_moves_under_gradient_step(key):
    """Learnable path's skew_params DO move under a gradient step.

    Use a symmetry-breaking target rather than ``||y||²`` — the latter is
    constant w.r.t. ``skew_params`` because orthogonal Q preserves norm, so
    the gradient vanishes and nothing would move.
    """
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels)
    x = jr.normal(jr.fold_in(key, 1), (n_channels,))
    target = jnp.arange(n_channels, dtype=float)

    def loss(m):
        y, _ = m.transform_and_log_det(x)
        return jnp.sum((y - target) ** 2)

    grads = eqx.filter_grad(loss)(conv)
    optim = optax.sgd(1e-1)
    opt_state = optim.init(eqx.filter(conv, eqx.is_inexact_array))
    updates, _ = optim.update(grads, opt_state)
    updated = eqx.apply_updates(conv, updates)
    assert not jnp.allclose(updated._rotation.skew_params, conv._rotation.skew_params)


def test_orthogonal1x1conv_orthogonality(key):
    """Weight matrix is orthogonal (Q @ Q.T ≈ I) in both paths."""
    n_channels = 4
    conv_learn = Orthogonal1x1Conv(key, n_channels)
    Q_learn = conv_learn._get_rotation_matrix()
    eye = jnp.eye(n_channels)
    assert jnp.allclose(Q_learn @ Q_learn.T, eye, atol=1e-5)
    assert jnp.allclose(Q_learn.T @ Q_learn, eye, atol=1e-5)

    Q, _ = jnp.linalg.qr(jr.normal(jr.fold_in(key, 2), (n_channels, n_channels)))
    conv_fixed = Orthogonal1x1Conv(key, n_channels, fixed_matrix=Q)
    Q_fixed = conv_fixed._get_rotation_matrix()
    assert jnp.allclose(Q_fixed @ Q_fixed.T, eye, atol=1e-5)


def test_orthogonal1x1conv_preserves_norm(key):
    """Orthogonal transformation preserves vector norm."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels)
    x = jr.normal(key, (n_channels,))
    y, _ = conv.transform_and_log_det(x)
    assert jnp.allclose(jnp.linalg.norm(x), jnp.linalg.norm(y), atol=1e-5)


def test_orthogonal1x1conv_jit(key):
    """Both modes survive eqx.filter_jit (guards against PyTree-structure bugs)."""
    n_channels = 4
    x = jr.normal(key, (n_channels,))

    @eqx.filter_jit
    def forward(m, x):
        return m.transform_and_log_det(x)

    conv_learn = Orthogonal1x1Conv(key, n_channels)
    y1, ld1 = forward(conv_learn, x)
    assert y1.shape == (n_channels,)
    assert jnp.allclose(ld1, 0.0, atol=1e-5)

    Q, _ = jnp.linalg.qr(jr.normal(jr.fold_in(key, 3), (n_channels, n_channels)))
    conv_fixed = Orthogonal1x1Conv(key, n_channels, fixed_matrix=Q)
    y2, ld2 = forward(conv_fixed, x)
    assert y2.shape == (n_channels,)
    assert jnp.allclose(ld2, 0.0, atol=1e-5)


def test_orthogonal1x1conv_matches_orthogonal_rotation_with_shared_params(key):
    """Learnable Orthogonal1x1Conv and OrthogonalRotation compute the same Q
    when they share ``skew_params`` — documents that the two classes are the
    same family, just with different init conventions."""
    n_channels = 4
    conv = Orthogonal1x1Conv(key, n_channels)
    # Copy skew_params across so both classes see identical inputs.
    rot = OrthogonalRotation(shape=(n_channels,))
    rot = eqx.tree_at(lambda m: m.skew_params, rot, conv._rotation.skew_params)
    assert jnp.allclose(
        conv._get_rotation_matrix(),
        rot._get_rotation_matrix(),
        atol=1e-6,
    )
