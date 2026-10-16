"""Verify configured LASSO matrices and feasible JAX instance samples."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lasso.problem import generate_data, generate_problem, load_config


def test_exact_matrices_and_random_state() -> None:
    """Match the current matrix draws without changing NumPy's global RNG."""
    before = np.random.get_state()
    problem = generate_problem()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:], "Generation must preserve global RNG state"
    rng = np.random.default_rng(problem.config.problem_seed)
    rng.random(100)
    np.testing.assert_array_equal(problem.p, rng.random(100))
    np.testing.assert_array_equal(problem.a, rng.normal(size=(50, 100)))
    rng.uniform(-1, 1, size=(10000, 50))
    np.testing.assert_array_equal(problem.g, rng.normal(size=(50, 100)))
    np.testing.assert_allclose(
        problem.h, np.abs(problem.g @ np.linalg.pinv(problem.a)).sum(axis=1)
    )
    b = np.random.RandomState(problem.config.quadratic_seed).normal(size=(100, 100))
    expected = (
        b @ b.T / problem.config.quadratic_scale
        + problem.config.quadratic_ridge * np.eye(100)
    )
    np.testing.assert_array_equal(problem.q_matrix, (expected + expected.T) / 2)


@pytest.mark.parametrize("n_samples", [0, 1, 16])
def test_instance_generation(n_samples: int) -> None:
    """Return reproducible JAX parameter rows in the configured box."""
    problem = generate_problem()
    x = generate_data(problem, n_samples=n_samples, seed=2)
    assert x.shape == (n_samples, problem.n_eq), "Expected parameter rows"
    assert problem.n_var == 100, "Expected 100 decision variables"
    assert problem.n_eq == problem.n_ineq == 50, "Expected 50 constraints of each kind"
    arrays = (problem.p, problem.a, problem.g, problem.h, x)
    assert all(isinstance(value, jax.Array) for value in arrays), "Expected JAX arrays"
    assert all(value.dtype == jnp.float64 for value in arrays), (
        "Preserve reference precision"
    )
    np.testing.assert_array_equal(
        x, generate_data(problem, n_samples=n_samples, seed=2)
    )
    assert jnp.all((x >= -1) & (x < 1)), "Instances must lie in [-1,1)"


def test_json_controls_dimensions_and_feasibility(tmp_path: Path) -> None:
    """Apply a changed JSON setup and preserve feasibility across its x box."""
    source = Path(__file__).parents[1] / "lasso" / "configs.json"
    settings = json.loads(source.read_text(encoding="utf-8-sig"))
    settings.update(
        n_var=8,
        n_eq=3,
        n_ineq=4,
        n_samples=16,
        matrix_discard_samples=5,
        x_low=-3.0,
        x_high=2.0,
    )
    config_path = tmp_path / "configs.json"
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    problem = generate_problem(load_config(config_path))
    x = generate_data(problem)
    assert x.shape == (16, 3), "JSON must control sample count and instance dimension"
    assert problem.a.shape == (3, 8), "JSON must control decision dimension"
    assert problem.g.shape == (4, 8), "JSON must control inequality count"
    assert jnp.all((x >= -3) & (x < 2)), "JSON must control instance bounds"
    witness = np.asarray(x) @ np.linalg.pinv(problem.a).T
    np.testing.assert_allclose(witness @ problem.a.T, x, atol=1e-12)
    assert np.all(witness @ problem.g.T <= problem.h + 1e-12), (
        "Witnesses must be feasible"
    )


def test_invalid_split_is_rejected() -> None:
    """Reject split fractions that would lose or duplicate samples."""
    with pytest.raises(ValueError, match="split_ratio"):
        replace(load_config(), split_ratio=[0.8, 0.2, 0.1])
