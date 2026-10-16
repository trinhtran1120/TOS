"""Compute and save constrained Moreau-envelope labels using Clarabel.

For each (x, q), minimize f(y) + 1/(2*rho) ||y-q||^2 subject to Ay=x, Gy<=h,
where f(y) = 0.5 y.T Q y + p.T y + lambda ||y||_1. Values are in original
units and gradients are with respect to q: (q-y_star) / rho.

Run from the repository root with ``uv run python -m lasso.gen_data``.
All setup, sampling, solver, split, and output settings come from configs.json.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from cvxpy.error import SolverError
from jax.typing import ArrayLike

from lasso.problem import (
    LassoConfig,
    LassoProblem,
    generate_data,
    generate_problem,
    load_config,
)


@dataclass(frozen=True)
class MoreauData:
    """Paired instance inputs and constrained envelope labels as JAX arrays.

    Attributes:
        x: Equality right-hand sides, shape (n_samples, n_eq).
        q: Envelope evaluation points, shape (n_samples, n_var).
        values: Original-unit scalar envelope values, shape (n_samples,).
        gradients: Original-unit gradients with respect to q.
        minimizers: Constrained proximal minimizers, shape (n_samples, n_var).
    """

    x: jax.Array
    q: jax.Array
    values: jax.Array
    gradients: jax.Array
    minimizers: jax.Array


def get_moreau(problem: LassoProblem, x: ArrayLike, q: ArrayLike) -> MoreauData:
    """Compute E_rho(q;x) and its q gradient for paired rows of x and q.

    Reuse one parameterized CVXPY problem for the entire batch. Both
    constraint sets are included. Clarabel must report an optimal solution
    for every row; failed or inaccurate solves raise instead of emitting labels.
    This reference solver runs on the host and returns float64 JAX arrays.
    Also solve the original LASSO for each x and print the L_inf difference
    from the proximal minimizer at the sampled q.
    """
    config = problem.config
    x_host = np.asarray(x, dtype=np.float64)
    q_host = np.asarray(q, dtype=np.float64)
    quadratic = np.asarray(problem.q_matrix)
    linear = np.asarray(problem.p)
    y = cp.Variable(problem.n_var)
    x_param = cp.Parameter(problem.n_eq)
    q_param = cp.Parameter(problem.n_var)
    base_objective = (
        0.5 * cp.quad_form(y, cp.psd_wrap(quadratic))
        + linear @ y
        + problem.l1_weight * cp.norm1(y)
    )
    reference = cp.Problem(
        cp.Minimize(base_objective + cp.sum_squares(y - q_param) / (2 * config.rho)),
        [
            np.asarray(problem.a) @ y == x_param,
            np.asarray(problem.g) @ y <= np.asarray(problem.h),
        ],
    )
    original = cp.Problem(cp.Minimize(base_objective), reference.constraints)
    values = np.empty(len(x_host), dtype=np.float64)
    minimizers = np.empty_like(q_host)
    for index, (instance, point) in enumerate(zip(x_host, q_host, strict=True)):
        x_param.value = instance
        q_param.value = point
        try:
            _ = reference.solve(
                solver=config.solver,
                tol_gap_abs=config.solver_tol,
                tol_gap_rel=config.solver_tol,
                tol_feas=config.solver_tol,
                max_iter=config.solver_max_iter,
                warm_start=True,
            )
        except SolverError as error:
            raise RuntimeError(
                f"Clarabel failed for instance {index}: {error}"
            ) from error
        if reference.status != cp.OPTIMAL or y.value is None:
            raise RuntimeError(
                f"Clarabel failed for instance {index}: {reference.status}"
            )
        solution = np.asarray(y.value, dtype=np.float64)
        if not np.isfinite(solution).all():
            raise RuntimeError(f"Nonfinite minimizer for instance {index}")
        minimizers[index] = solution
        values[index] = (
            0.5 * solution @ quadratic @ solution
            + linear @ solution
            + problem.l1_weight * np.abs(solution).sum()
            + np.sum((solution - point) ** 2) / (2 * config.rho)
        )
        try:
            _ = original.solve(
                solver=config.solver,
                tol_gap_abs=config.solver_tol,
                tol_gap_rel=config.solver_tol,
                tol_feas=config.solver_tol,
                max_iter=config.solver_max_iter,
                warm_start=True,
            )
        except SolverError as error:
            raise RuntimeError(
                f"Original LASSO failed for instance {index}: {error}"
            ) from error
        if original.status != cp.OPTIMAL or y.value is None:
            raise RuntimeError(
                f"Original LASSO failed for instance {index}: {original.status}"
            )
        difference = np.linalg.norm(minimizers[index] - y.value, ord=np.inf)
        print(
            f"Instance {index + 1}/{len(x_host)}: "
            f"L_inf minimizer difference (proximal at sampled q vs LASSO) = {difference:.10g}",
            flush=True,
        )
    return MoreauData(
        jnp.array(x_host),
        jnp.array(q_host),
        jnp.array(values),
        jnp.array((q_host - minimizers) / config.rho),
        jnp.array(minimizers),
    )


def generate_dataset(config: LassoConfig | None = None) -> Path:
    """Generate paired inputs, compute labels, and save shuffled NPZ splits.

    Store x, q, moreau, gradient, minimizer, and original row indices for each
    of train/val/test. Shared matrices, rho, lambda, and the complete JSON
    config are included for reproduction. No pickle is needed to load files.
    Relative output directories resolve against the repository root.
    """
    config = load_config() if config is None else config
    problem = generate_problem(config)
    x = generate_data(problem)
    q = jax.random.uniform(
        jax.random.key(config.q_seed),
        (config.n_samples, config.n_var),
        dtype=jnp.float64,
        minval=config.q_low,
        maxval=config.q_high,
    )
    data = get_moreau(problem, x, q)
    if config.n_samples:
        print_minimizer_comparison(problem, x[0], q[0], data.minimizers[0])
    permutation = np.asarray(
        jax.random.permutation(jax.random.key(config.split_seed), config.n_samples)
    )
    n_train = int(config.n_samples * config.split_ratio[0])
    n_val = int(config.n_samples * config.split_ratio[1])
    split_indices = np.split(permutation, [n_train, n_train + n_val])
    arrays = {
        "x": np.asarray(data.x),
        "q": np.asarray(data.q),
        "moreau": np.asarray(data.values),
        "gradient": np.asarray(data.gradients),
        "minimizer": np.asarray(data.minimizers),
    }
    archive = {
        f"{key}_{split}": array[indices]
        for split, indices in zip(("train", "val", "test"), split_indices, strict=True)
        for key, array in arrays.items()
    }
    for split, indices in zip(("train", "val", "test"), split_indices, strict=True):
        archive[f"indices_{split}"] = indices
    archive.update(
        Q=np.asarray(problem.q_matrix),
        p=np.asarray(problem.p),
        A=np.asarray(problem.a),
        G=np.asarray(problem.g),
        h=np.asarray(problem.h),
        rho=np.asarray(config.rho),
        l1_weight=np.asarray(config.l1_weight),
        config_json=np.asarray(json.dumps(asdict(config), sort_keys=True)),
        gradient_variable=np.asarray("q"),
        normalized=np.asarray(False),
        envelope_convention=np.asarray("1/(2*rho)"),
    )
    root = Path(__file__).resolve().parents[2]
    output_dir = root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.output_file
    np.savez_compressed(output_path, allow_pickle=False, **archive)
    return output_path


def print_minimizer_comparison(
    problem: LassoProblem,
    x: ArrayLike,
    q: ArrayLike,
    moreau_minimizer: ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Print Moreau and direct LASSO minimizers for the same instance x.

    Return the Moreau and original LASSO minimizers, respectively. Reuse a
    supplied Moreau minimizer or compute it with get_moreau. The original
    LASSO solve excludes the 1/(2*rho) penalty and uses Clarabel directly.
    """
    config = problem.config
    x_host = np.asarray(x, dtype=np.float64)
    q_host = np.asarray(q, dtype=np.float64)
    if moreau_minimizer is None:
        data = get_moreau(problem, x_host[None, :], q_host[None, :])
        moreau_minimizer = data.minimizers[0]
    y_moreau = np.asarray(moreau_minimizer)
    quadratic = np.asarray(problem.q_matrix)
    linear = np.asarray(problem.p)
    a, g, h = np.asarray(problem.a), np.asarray(problem.g), np.asarray(problem.h)
    y = cp.Variable(problem.n_var)
    objective = (
        0.5 * cp.quad_form(y, cp.psd_wrap(quadratic))
        + linear @ y
        + problem.l1_weight * cp.norm1(y)
    )
    original = cp.Problem(cp.Minimize(objective), [a @ y == x_host, g @ y <= h])
    _ = original.solve(
        solver=config.solver,
        tol_gap_abs=config.solver_tol,
        tol_gap_rel=config.solver_tol,
        tol_feas=config.solver_tol,
        max_iter=config.solver_max_iter,
    )
    if original.status != cp.OPTIMAL or y.value is None:
        raise RuntimeError(f"Original LASSO solve failed: {original.status}")
    y_lasso = np.asarray(y.value, dtype=np.float64)
    print("Minimizer comparison for the same x (Clarabel):")
    print("The Moreau subproblem includes 1/(2*rho) * ||y-q||^2.")
    with np.printoptions(precision=8, threshold=max(problem.n_var, problem.n_eq)):
        print(f"x = {x_host}")
        print(f"q = {q_host}")
        print(f"Proximal minimizer at sampled q = {y_moreau}")
        print(f"Original LASSO minimizer = {y_lasso}")
    for name, solution in (("Moreau", y_moreau), ("Original LASSO", y_lasso)):
        value = float(
            0.5 * solution @ quadratic @ solution
            + linear @ solution
            + problem.l1_weight * np.abs(solution).sum()
        )
        augmented = value + float(np.sum((solution - q_host) ** 2)) / (2 * config.rho)
        equality_error = float(np.max(np.abs(a @ solution - x_host)))
        inequality_error = max(0.0, float(np.max(g @ solution - h)))
        print(
            f"{name}: f(y)={value:.10g}, penalized objective={augmented:.10g}, "
            f"equality residual={equality_error:.3e}, "
            f"inequality violation={inequality_error:.3e}"
        )
    print(
        "L_inf norm of minimizer difference (proximal at sampled q vs LASSO) = "
        f"{np.linalg.norm(y_moreau - y_lasso, ord=np.inf):.10g}"
    )
    return jnp.array(y_moreau), jnp.array(y_lasso)


def main() -> None:
    """Generate the configured dataset from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print minimizers for the first instance without generating a dataset.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check_only:
        if config.n_samples < 1:
            parser.error("--check-only requires n_samples >= 1")
        problem = generate_problem(config)
        x = generate_data(problem)
        q = jax.random.uniform(
            jax.random.key(config.q_seed),
            (config.n_samples, config.n_var),
            dtype=jnp.float64,
            minval=config.q_low,
            maxval=config.q_high,
        )
        print_minimizer_comparison(problem, x[0], q[0])
        return
    path = generate_dataset(config)
    print(f"Saved constrained Moreau dataset to {path}", flush=True)


if __name__ == "__main__":
    main()
