from collections.abc import Callable
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

import cvxpy as cp
import numpy as np

from qp.problem import QP, objective_grad


def projection_inequality(problem: QP, tol: float) -> Callable[[jax.Array], tuple[jax.Array, float]]:
    G = np.asarray(problem.G)
    h = np.asarray(problem.h)
    n_var = problem.p.shape[0]

    u = cp.Parameter(n_var)
    x = cp.Variable(n_var)
    constraint = [G @ x <= h]
    objective = cp.Minimize(0.5*cp.sum_squares(x - u))
    problem = cp.Problem(objective, constraint)

    def solve(z: jax.Array) -> tuple[jax.Array, float]:
        start = time.perf_counter()
        u.value = np.asarray(z)

        problem.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)

        if problem.status != cp.OPTIMAL or x.value is None:
            raise RuntimeError(f"Inequality projection failed: {problem.status}")

        viol = np.max(np.maximum(G @ x.value - h, 0.0))
        if viol > tol:
            raise RuntimeError(f"Inequality projection violation: {viol}")
        solution = jnp.asarray(x.value)
        solution.block_until_ready()
        return solution,problem.value, time.perf_counter() - start


    return solve


@jax.jit
def _tos_step(problem, b, z, x, AAt_factor, gamma, eta, eps_abs, eps_rel):
    v = 2.0 * x - z - gamma * objective_grad(problem, x)
    correction = cho_solve((AAt_factor, True), problem.A @ v - b)
    y = v - problem.A.T @ correction

    Ax = problem.A @ x
    Gx = problem.G @ x
    res = jnp.max(jnp.abs(y - x)) / gamma
    eq_viol = jnp.max(jnp.abs(Ax - b))
    ineq_viol = jnp.max(jnp.maximum(Gx - problem.h, 0.0))

    res_limit = eps_abs + eps_rel / gamma * jnp.maximum(
        jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
    )
    eq_limit = eps_abs + eps_rel * jnp.maximum(
        jnp.max(jnp.abs(Ax)), jnp.max(jnp.abs(b))
    )
    ineq_limit = eps_abs + eps_rel * jnp.maximum(
        jnp.max(jnp.abs(Gx)), jnp.max(jnp.abs(problem.h))
    )
    converged = (res <= res_limit) & (eq_viol <= eq_limit) & (ineq_viol <= ineq_limit)
    return z + eta * (y - x), eq_viol, ineq_viol, converged

def tos(
        problem: QP,
        b: jax.Array,
        gamma: float = 1.0,
        tol: float = 1e-6,
        eps_abs: float = 1e-6,
        eps_rel: float = 1e-6,
        max_iter: int = 1000,
        eta: float = 1.0
        ) -> tuple[jax.Array, dict[str, float | int | bool]]:
    start = time.perf_counter()
    projection_ineq = projection_inequality(problem, tol)
    projection_setup_time = time.perf_counter() - start
    AAt_factor, _ = cho_factor(problem.A @ problem.A.T, lower=True)
    z = jnp.zeros_like(problem.p)
    projection_time = 0.0

    for i in range(1, max_iter + 1):
        x,_,project_time = projection_ineq(z)
        projection_time += project_time
        z, eq_viol, ineq_viol, converged = _tos_step(
            problem, b, z, x, AAt_factor, gamma, eta, eps_abs, eps_rel
        )
        if bool(converged):
            print(f"Converge at iteration {i}")
            break

    x.block_until_ready()
    return x, {
        "converged": bool(converged),
        "gamma": gamma,
        "equality_residual": float(eq_viol),
        "inequality_residual": float(ineq_viol),
        "inequality_projection_seconds": projection_time,
        "inequality_projection_setup_seconds": projection_setup_time,
        "total_seconds": time.perf_counter() - start,
    }
