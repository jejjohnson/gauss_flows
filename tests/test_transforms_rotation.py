"""Tests for rotation transforms."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from gauss_flows import (
    FixedRotation,
    HouseholderRotation,
    LULinearPermute,
    OrthogonalRotation,
    gaussianization_flow,
)


def test_householder_rotation_forward_inverse(key):
    shape = (4,)
    rotation = HouseholderRotation(n_reflections=4, shape=shape)
    x = jr.normal(key, shape)
    y, log_det_f = rotation.transform_and_log_det(x)
    x_rec, log_det_i = rotation.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det_f, 0.0, atol=1e-5)
    assert jnp.allclose(log_det_i, 0.0, atol=1e-5)


def test_householder_rotation_shape(key):
    shape = (4,)
    rotation = HouseholderRotation(n_reflections=4, shape=shape)
    x = jr.normal(key, shape)
    y, log_det = rotation.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_orthogonal_rotation_forward_inverse(key):
    shape = (4,)
    rotation = OrthogonalRotation(shape=shape)
    x = jr.normal(key, shape)
    y, log_det_f = rotation.transform_and_log_det(x)
    x_rec, _log_det_i = rotation.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.allclose(log_det_f, 0.0, atol=1e-5)


def test_orthogonal_rotation_volume_preserving(key):
    shape = (4,)
    rotation = OrthogonalRotation(shape=shape)
    x = jr.normal(key, shape)
    _, log_det = rotation.transform_and_log_det(x)
    assert jnp.allclose(log_det, 0.0, atol=1e-4)


def test_fixed_rotation_forward_inverse(key):
    shape = (4,)
    # Use a random orthogonal matrix
    W = jr.normal(key, (4, 4))
    Q, _ = jnp.linalg.qr(W)
    rotation = FixedRotation(Q)
    x = jr.normal(key, shape)
    y, log_det_f = rotation.transform_and_log_det(x)
    x_rec, log_det_i = rotation.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-5)


def test_fixed_rotation_orthogonal_matrix_preserves_volume(key):
    W = jr.normal(key, (4, 4))
    Q, _ = jnp.linalg.qr(W)
    rotation = FixedRotation(Q)
    x = jr.normal(key, (4,))
    _, log_det = rotation.transform_and_log_det(x)
    assert jnp.allclose(log_det, 0.0, atol=1e-4)


def test_fixed_rotation_is_non_trainable(key):
    """``FixedRotation.matrix`` must sit in the non-trainable partition, else
    ``fit_to_data`` will drift the matrix off the orthogonal manifold and
    ``flow.sample`` will silently produce collapsed output (while log_prob
    still reports plausible numbers — see `fit_rbig_coupling` regression).
    """
    import paramax

    W = jr.normal(key, (4, 4))
    Q, _ = jnp.linalg.qr(W)
    rotation = FixedRotation(Q)

    params, _static = eqx.partition(
        rotation,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    trainable_leaves = [
        leaf
        for leaf in __import__("jax").tree_util.tree_leaves(params)
        if eqx.is_inexact_array(leaf)
    ]
    assert trainable_leaves == [], (
        "FixedRotation has trainable inexact-array leaves; "
        "matrix must be paramax.non_trainable-wrapped."
    )


def _perturb_lu(rotation: LULinearPermute, key) -> LULinearPermute:
    """Helper to give the LU bijection non-trivial parameters for testing."""
    k1, k2, k3 = jr.split(key, 3)
    lower = jr.normal(k1, rotation.lower_params.shape) * 0.3
    upper = jr.normal(k2, rotation.upper_params.shape) * 0.3
    log_diag = jr.normal(k3, rotation.log_diag.shape) * 0.1

    def where(r):
        return (r.lower_params, r.upper_params, r.log_diag)

    return eqx.tree_at(where, rotation, (lower, upper, log_diag))


def test_lu_linear_permute_forward_inverse(key):
    shape = (4,)
    rotation = _perturb_lu(LULinearPermute(shape=shape), key)
    x = jr.normal(jr.fold_in(key, 1), shape)
    y, log_det_f = rotation.transform_and_log_det(x)
    x_rec, log_det_i = rotation.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-5)


def test_lu_linear_permute_log_det_matches_jacobian(key):
    shape = (5,)
    rotation = _perturb_lu(LULinearPermute(shape=shape), key)
    L, U = rotation._build_lu()
    perm_matrix = jnp.eye(shape[0])[rotation.permutation]
    full = perm_matrix @ L @ U
    expected = jnp.log(jnp.abs(jnp.linalg.det(full)))
    x = jr.normal(jr.fold_in(key, 2), shape)
    _, log_det = rotation.transform_and_log_det(x)
    assert jnp.allclose(log_det, expected, atol=1e-5)


def test_lu_linear_permute_default_is_identity(key):
    shape = (4,)
    rotation = LULinearPermute(shape=shape)
    x = jr.normal(key, shape)
    y, log_det = rotation.transform_and_log_det(x)
    # L = U = I at init, so y is just the reversed x and log_det = 0
    assert jnp.allclose(y, x[::-1], atol=1e-6)
    assert jnp.allclose(log_det, 0.0, atol=1e-6)


def test_lu_linear_permute_custom_permutation(key):
    shape = (4,)
    perm = jnp.array([2, 0, 3, 1])
    rotation = LULinearPermute(shape=shape, permutation=perm)
    x = jr.normal(key, shape)
    y, _ = rotation.transform_and_log_det(x)
    assert jnp.allclose(y, x[perm], atol=1e-6)
    x_rec, _ = rotation.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-6)


def test_gaussianization_flow_accepts_lu_rotation(key):
    flow = gaussianization_flow(key, n_dims=3, n_layers=2, rotation="lu")
    samples = flow.sample(jr.fold_in(key, 3), (16,))
    log_probs = flow.log_prob(samples)
    assert samples.shape == (16, 3)
    assert log_probs.shape == (16,)
    assert jnp.all(jnp.isfinite(log_probs))


@pytest.mark.parametrize(
    "bad_perm",
    [
        jnp.array([0, 1, 1, 3]),  # duplicate
        jnp.array([0, 1, 2, 4]),  # out of range (high)
        jnp.array([-1, 0, 1, 2]),  # out of range (low)
    ],
    ids=["duplicate", "out_of_range_high", "out_of_range_low"],
)
def test_lu_linear_permute_rejects_invalid_permutation(bad_perm):
    with pytest.raises(ValueError, match="permutation must be a permutation"):
        LULinearPermute(shape=(4,), permutation=bad_perm)


def test_lu_linear_permute_rejects_wrong_shape_permutation():
    with pytest.raises(ValueError, match="permutation must have shape"):
        LULinearPermute(shape=(4,), permutation=jnp.array([0, 1, 2]))
