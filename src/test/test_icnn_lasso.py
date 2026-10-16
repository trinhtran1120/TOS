"""Check frozen atoms, query-only derivatives, training, and checkpoint inference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from icnn import (init_icnn_params, load_lasso_model, make_icnn,
                  make_lasso_dictionary, save_lasso_model)


def geometry():
    return np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([0.5])


def test_dictionary_values_and_derivatives():
    A, G, h = geometry()
    atom = make_lasso_dictionary(A, G, h, 2., 0.5, a_l1=[0.2, 0.3], a_eq=0.4, a_ineq=0.6)
    # q=[0.25, 1.5], x=[-0.25]: quadratic/linear Huber regions,
    # an equality projection of length 0.5 and a half-space violation of 1.
    packed = jnp.array([0.25, 1.5, -0.25])
    expected = 0.2 * 0.5 * (0.25 ** 2 / 2) + 0.3 * 0.5 * (1.5 - 0.5)
    expected += 0.4 * 0.5 ** 2 / 4 + 0.6 * 1. ** 2 / 4
    np.testing.assert_allclose(atom(packed), expected, rtol=1e-6)
    grad = jax.grad(atom)(packed)
    np.testing.assert_allclose(grad, [0.2 * 0.5 * 0.25 + 0.4 * 0.5 / 2,
                                      0.3 * 0.5 + 0.6 / 2, -0.4 * 0.5 / 2], rtol=1e-6)
    for i in range(3):
        step = jnp.eye(3)[i] * 1e-3
        numerical = (atom(packed + step) - atom(packed - step)) / 2e-3
        np.testing.assert_allclose(grad[i], numerical, rtol=1e-3, atol=1e-5)
    # Exactly on both Huber transitions and the half-space boundary.
    for point in ([1., 0.5, 0.], [-1., -1., 0.], [0., 0., 0.]):
        assert np.isfinite(atom(jnp.array(point)))
        assert np.isfinite(jax.grad(atom)(jnp.array(point))).all()


def test_empty_constraints_zero_l1_and_invalid_geometry():
    empty = np.empty((0, 2))
    atom = make_lasso_dictionary(empty, empty, np.empty(0), 2., 0., a_l1=1., a_eq=1., a_ineq=1.)
    assert atom(jnp.zeros(2)) == 0
    np.testing.assert_array_equal(jax.grad(atom)(jnp.zeros(2)), np.zeros(2))
    A, G, h = geometry()
    with pytest.raises(ValueError, match="nonnegative"):
        make_lasso_dictionary(A, G, h, 2., 1., a_l1=-1.)
    with pytest.raises(ValueError, match="zero-norm"):
        make_lasso_dictionary(A, G * 0, h, 2., 1.)
    with pytest.raises(ValueError, match="full row rank"):
        make_lasso_dictionary(np.vstack([A, A]), G, h, 2., 1.)


def test_conditional_gradient_convexity_and_fixed_dictionary():
    A, G, h = geometry()
    atom = make_lasso_dictionary(A, G, h, 2., 0.5, a_l1=0.2, a_eq=0.1, a_ineq=0.1)
    model = make_icnn("softplus", query_dim=2, dictionary=atom)
    baseline = make_icnn("softplus", query_dim=2)
    params = init_icnn_params(jax.random.PRNGKey(0), 3, [4, 4], init="fan_in")
    packed = jnp.array([0.25, 1.5, -0.25])
    g = model.grad_wrt_x(params, packed)
    assert g.shape == (2,)
    np.testing.assert_allclose(g, jax.grad(lambda q: model.icnn_forward(
        params, jnp.concatenate([q, packed[2:]])))(packed[:2]), atol=1e-6)
    np.testing.assert_allclose(g - baseline.grad_wrt_x(params, packed),
                               jax.grad(atom)(packed)[:2], atol=1e-6)
    # A zero dictionary reproduces the baseline exactly.
    zero_model = make_icnn("softplus", query_dim=2,
                           dictionary=make_lasso_dictionary(A, G, h, 2., 0.5))
    np.testing.assert_allclose(zero_model.icnn_forward(params, packed),
                               baseline.icnn_forward(params, packed), atol=1e-7)
    rng = np.random.default_rng(1)
    for _ in range(12):
        u, v = rng.normal(size=(2, 2))
        t = rng.uniform()
        forward = lambda q: model.icnn_forward(params, jnp.concatenate([jnp.asarray(q), packed[2:]]))
        assert float(forward(t*u + (1-t)*v)) <= float(t*forward(u) + (1-t)*forward(v)) + 1e-5


def test_training_and_checkpoint(tmp_path):
    A, G, h = geometry()
    weights = dict(a_l1=0.01, a_eq=0.01, a_ineq=0.01)
    atom = make_lasso_dictionary(A, G, h, 2., 0.5, **weights)
    model = make_icnn("softplus", query_dim=2, dictionary=atom)
    X = np.random.default_rng(3).normal(size=(9, 3)).astype(np.float32)
    y, g = np.zeros(9, np.float32), np.zeros((9, 2), np.float32)
    params = model.train_icnn(X, y, g, 3, widths=[4, 4], init="fan_in",
                              batch_size=4, epochs=8, lr=0.01, val_data=(X, y, g))
    initial = init_icnn_params(jax.random.PRNGKey(0), 3, [4, 4], init="fan_in")
    assert float(model.loss_fn(params, X, y, g)[0]) < float(model.loss_fn(initial, X, y, g)[0])
    settings = dict(weight_act="softplus", keep_best=True, value_weight=1., dictionary=weights)
    path = tmp_path / "model.npz"
    save_lasso_model(params, path, A, G, h, 2., 0.5, settings)
    restored, restored_params = load_lasso_model(path)
    value, grad, prox = restored.batched_predict(restored_params, X[:, :2], X[:, 2:])
    np.testing.assert_allclose(value, model.batched_forward(params, X), atol=1e-6)
    np.testing.assert_allclose(grad, model.batched_grad_wrt_x(params, X), atol=1e-6)
    reference_grad = np.ones_like(grad)
    reference_prox = X[:, :2] - 2 * reference_grad
    np.testing.assert_allclose(np.linalg.norm(prox - reference_prox, axis=1),
                               2 * np.linalg.norm(grad - reference_grad, axis=1), rtol=1e-6)


def test_legacy_input_gradient_and_value_weight():
    model = make_icnn(value_weight=0.)
    params = init_icnn_params(jax.random.PRNGKey(1), 2, [4])
    X = jnp.ones((3, 2))
    g = model.batched_grad_wrt_x(params, X)
    assert g.shape == X.shape
    loss, _ = model.loss_fn(params, X, jnp.zeros(3), g, 1.)
    np.testing.assert_allclose(loss, 0., atol=1e-6)
