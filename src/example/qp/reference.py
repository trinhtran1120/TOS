"""Solve the DC3 quadratic program using CVXPY and OSQP."""

import time

import cvxpy as cp
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy import sparse

from example.qp.problem import QP


def solve_qp(problem: QP, b: jax.Array) -> tuple[jax.Array, float]:
    """Return the validated solution and total wall time for one QP."""
    start = time.perf_counter()
    q_diag = np.asarray(problem.Q, dtype=np.float64)
    p = np.asarray(problem.p, dtype=np.float64)
    a_mat = np.asarray(problem.A, dtype=np.float64)
    g_mat = np.asarray(problem.G, dtype=np.float64)
    h = np.asarray(problem.h, dtype=np.float64)
    b_host = np.asarray(b, dtype=np.float64)

    y = cp.Variable(p.size)
    equality = a_mat @ y == b_host
    inequality = g_mat @ y <= h
    qp = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(y, sparse.diags(q_diag)) + p @ y),
        [equality, inequality],
    )
    qp.solve(
        solver=cp.OSQP,
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=1000,
        polishing=True,
        verbose=False,
    )
    if qp.status != cp.OPTIMAL or y.value is None:
        raise RuntimeError(f"QP solve failed: {qp.status}")

    solution_host = np.asarray(y.value)
    nu = np.asarray(equality.dual_value)
    mu = np.asarray(inequality.dual_value)
    slack = g_mat @ solution_host - h
    residuals = (
        np.max(np.maximum(slack, 0.0)),
        np.max(np.abs(a_mat @ solution_host - b_host)),
        np.max(np.abs(q_diag * solution_host + p + g_mat.T @ mu + a_mat.T @ nu)),
        np.max(np.maximum(-mu, 0.0)),
        np.max(np.abs(mu * slack)),
    )
    if not np.all(np.isfinite(residuals)) or max(residuals) > 1e-6:
        raise RuntimeError(f"QP failed KKT validation: {residuals}")

    solution = jnp.asarray(solution_host)
    solution.block_until_ready()
    return solution, time.perf_counter() - start
