"""Tests for coupling transforms."""

import jax.numpy as jnp
import jax.random as jr

from gauss_flows import ActNorm1D, AffineCoupling, DeepSigmoidCoupling, RQSplineCoupling


def test_act_norm_1d_forward_inverse(key):
    shape = (4,)
    act_norm = ActNorm1D(shape)
    x = jr.normal(key, shape)
    y, log_det = act_norm.transform_and_log_det(x)
    x_rec, log_det_inv = act_norm.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det + log_det_inv, 0.0, atol=1e-5)


def test_act_norm_1d_shape(key):
    shape = (4,)
    act_norm = ActNorm1D(shape)
    x = jr.normal(key, shape)
    y, log_det = act_norm.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_affine_coupling_forward_inverse(key):
    shape = (4,)
    coupling = AffineCoupling(key, shape, nn_width=16, nn_depth=1)
    x = jr.normal(key, shape)
    y, log_det_f = coupling.transform_and_log_det(x)
    x_rec, log_det_i = coupling.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-5)


def test_affine_coupling_shape(key):
    shape = (4,)
    coupling = AffineCoupling(key, shape, nn_width=16, nn_depth=1)
    x = jr.normal(key, shape)
    y, log_det = coupling.transform_and_log_det(x)
    assert y.shape == shape
    assert log_det.shape == ()


def test_rqspline_coupling_forward_inverse(key):
    shape = (4,)
    coupling = RQSplineCoupling(key, shape, n_bins=4, nn_width=16, nn_depth=1)
    x = jr.normal(key, shape)
    y, log_det_f = coupling.transform_and_log_det(x)
    x_rec, log_det_i = coupling.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.allclose(log_det_f + log_det_i, 0.0, atol=1e-4)


def test_deep_sigmoid_coupling_forward_inverse(key):
    shape = (4,)
    coupling = DeepSigmoidCoupling(key, shape, nn_width=16, nn_depth=1)
    x = jr.normal(key, shape)
    y, _log_det_f = coupling.transform_and_log_det(x)
    x_rec, _log_det_i = coupling.inverse_and_log_det(y)
    assert jnp.allclose(x, x_rec, atol=1e-5)
