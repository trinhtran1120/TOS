"""Verify constrained Moreau labels, gradients, and reproducible NPZ splits."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from lasso.gen_data import generate_dataset, get_moreau, print_minimizer_comparison
from lasso.problem import generate_data, generate_problem, load_config


def test_moreau_definition_constraints_and_gradient() -> None:
    """Validate labels and q gradients against feasibility and differences."""
    config = replace(
        load_config(),
        n_var=6,
        n_eq=2,
        n_ineq=3,
        n_samples=3,
        matrix_discard_samples=4,
        progress_every=0,
    )
    problem = generate_problem(config)
    x = generate_data(problem)
    q = np.random.default_rng(4).uniform(-1, 1, (3, config.n_var))
    data = get_moreau(problem, x, q)
    y = np.asarray(data.minimizers)
    np.testing.assert_allclose(y @ problem.a.T, x, atol=1e-7)
    assert np.all(y @ problem.g.T <= problem.h + 1e-7), "Minimizers must be feasible"
    expected = 0.5 * np.einsum("bi,ij,bj->b", y, problem.q_matrix, y)
    expected += y @ problem.p + config.l1_weight * np.abs(y).sum(axis=1)
    expected += np.sum((y - q) ** 2, axis=1) / (2 * config.rho)
    np.testing.assert_allclose(data.values, expected)
    np.testing.assert_allclose(data.gradients, (q - y) / config.rho)
    step = np.zeros_like(q)
    step[:, 0] = 1e-4
    plus = get_moreau(problem, x, q + step)
    minus = get_moreau(problem, x, q - step)
    np.testing.assert_allclose(
        (plus.values - minus.values) / 2e-4, data.gradients[:, 0], atol=2e-4
    )


def test_npz_splits_and_reproducibility(tmp_path: Path) -> None:
    """Save every instance once with aligned inputs, labels, and metadata."""
    config = replace(
        load_config(),
        n_var=6,
        n_eq=2,
        n_ineq=3,
        n_samples=20,
        matrix_discard_samples=4,
        progress_every=0,
        output_dir=str(tmp_path),
    )
    path = generate_dataset(config)
    with np.load(path, allow_pickle=False) as archive:
        first = {key: archive[key].copy() for key in archive.files}
    indices = []
    for name, count in (("train", 16), ("val", 2), ("test", 2)):
        assert first[f"x_{name}"].shape == (count, 2), "Wrong instance split"
        assert first[f"q_{name}"].shape == (count, 6), "Wrong evaluation-point split"
        assert first[f"moreau_{name}"].shape == (count,), "Wrong scalar label shape"
        assert first[f"gradient_{name}"].shape == (count, 6), "Wrong gradient shape"
        indices.append(first[f"indices_{name}"])
        np.testing.assert_allclose(
            first[f"gradient_{name}"],
            (first[f"q_{name}"] - first[f"minimizer_{name}"]) / config.rho,
        )
    np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(20))
    assert first["Q"].shape == (6, 6), "Save the problem matrices for reuse"
    assert first["gradient_variable"].item() == "q", "Record gradient semantics"
    assert first["envelope_convention"].item() == "1/(2*rho)", "Record rho convention"
    generate_dataset(config)
    with np.load(path, allow_pickle=False) as archive:
        for key, value in first.items():
            np.testing.assert_array_equal(archive[key], value)


def test_infeasible_instance_is_not_saved_as_label() -> None:
    """Fail clearly if no constrained proximal minimizer exists."""
    config = replace(
        load_config(),
        n_var=2,
        n_eq=2,
        n_ineq=2,
        n_samples=1,
        matrix_discard_samples=4,
        progress_every=0,
    )
    problem = generate_problem(config)
    y = np.linalg.solve(problem.g, np.asarray(problem.h) + 100)
    x = np.asarray(problem.a) @ y
    with pytest.raises(RuntimeError, match="instance 0"):
        get_moreau(problem, x[None, :], np.zeros((1, 2)))


def test_direct_lasso_solution_is_a_proximal_fixed_point() -> None:
    """Check the diagnostic solves distinct objectives and recovers a fixed point."""
    config = replace(
        load_config(),
        n_var=6,
        n_eq=2,
        n_ineq=3,
        n_samples=1,
        matrix_discard_samples=4,
        progress_every=0,
    )
    problem = generate_problem(config)
    x = generate_data(problem)[0]
    q = np.ones(config.n_var)
    moreau, original = print_minimizer_comparison(problem, x, q)

    def objective(y: NDArray[np.float64]) -> float:
        return float(
            0.5 * y @ problem.q_matrix @ y
            + problem.p @ y
            + config.l1_weight * np.abs(y).sum()
        )

    f_original = objective(np.asarray(original))
    f_moreau = objective(np.asarray(moreau))
    assert f_original <= f_moreau + 1e-7, "Original solve must minimize f"
    augmented_original = f_original + np.sum((original - q) ** 2) / (2 * config.rho)
    augmented_moreau = f_moreau + np.sum((moreau - q) ** 2) / (2 * config.rho)
    assert augmented_moreau <= augmented_original + 1e-7, (
        "Moreau solve must minimize the penalized objective"
    )
    fixed_point = get_moreau(problem, x[None, :], original[None, :])
    np.testing.assert_allclose(fixed_point.minimizers[0], original, atol=1e-5)
