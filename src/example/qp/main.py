"""Compare vanilla TOS with a direct OSQP solve on DC3 QP samples."""

import jax

jax.config.update("jax_enable_x64", True)

from problem import generate_problem, objective
from solver import solve_qp
from tos_std import tos


def main() -> None:
    """Generate QP samples and print per-sample solver comparisons."""
    n_samples = 10
    eta = 1.5
    problem, b_batch = generate_problem(n_samples=n_samples)
    print("gap (%) = 100 * abs(objective(OSQP) - objective(TOS)) / abs(objective(OSQP))")
    print(
        "sample  gap (%)          OSQP (s)    TOS (s)  TOS proj (s)  "
        "TOS converged  eq residual  ineq residual"
    )

    osqp_total = 0.0
    tos_total = 0.0
    projection_total = 0.0
    for sample_index, b in enumerate(b_batch, start=1):
        reference, osqp_seconds = solve_qp(problem, b)
        estimate, diagnostics = tos(problem, b, eta=eta)
        objective_osqp = float(objective(problem, reference))
        objective_tos = float(objective(problem, estimate))
        gap = (
            100.0 * abs(objective_osqp - objective_tos) / abs(objective_osqp)
            if objective_osqp != 0.0
            else float("nan")
        )
        tos_seconds = float(diagnostics["total_seconds"])
        projection_seconds = float(diagnostics["inequality_projection_seconds"])
        osqp_total += osqp_seconds
        tos_total += tos_seconds
        projection_total += projection_seconds
        print(
            f"{sample_index:>6}  {gap:>+.6e}  {osqp_seconds:>10.4f}  "
            f"{tos_seconds:>10.4f}  {projection_seconds:>12.4f}  "
            f"{str(diagnostics['converged']):>13}  "
            f"{diagnostics['equality_residual']:>11.3e}  "
            f"{diagnostics['inequality_residual']:>13.3e}"
        )

    print(
        f"Total solving time: OSQP={osqp_total:.4f}s, TOS={tos_total:.4f}s, "
        f"TOS inequality projection={projection_total:.4f}s"
    )


if __name__ == "__main__":
    main()
