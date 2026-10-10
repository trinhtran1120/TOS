"""TOS with the learned Moreau inequality projection."""

import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor

from model.icnn import load_checkpoint, predict_batch

from experiments.qp.problem.problem import QP
from experiments.qp.problem.tos import _tos_step


def learned_projection(z, params, model):
    """Compute z - tau * grad M_hat_tau(z) from the trained ICNN."""
    values, gradients = predict_batch(
        params, z[None, :].astype(jnp.float32),
        model["input_mean"], model["input_std"],
        model["output_scale"], model["output_offset"],
        model["weight_act"], model["nonnegative_output"],
    )
    x = z - model["tau"] * gradients[0].astype(z.dtype)
    x.block_until_ready()
    value = float(values[0])
    return x, value


def learned_tos(
    problem: QP,
    b: jax.Array,
    params,
    model,
    gamma: float = 1.0,
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
    max_iter: int = 1000,
    eta: float = 1.5,
    record_trajectory: bool = False,
):
    """Run TOS using the learned projection without feasibility correction."""
    start = time.perf_counter()
    AAt_factor, _ = cho_factor(problem.A @ problem.A.T, lower=True)
    z = jnp.zeros_like(problem.p)
    projection_seconds = 0.0
    trajectory = [] if record_trajectory else None

    for iteration in range(1, max_iter + 1):
        projection_start = time.perf_counter()
        x, envelope_value = learned_projection(z, params, model)
        projection_seconds += time.perf_counter() - projection_start
        if trajectory is not None:
            trajectory.append({
                "z": np.asarray(z).copy(),
                "predicted_x": np.asarray(x).copy(),
                "envelope_value": envelope_value,
            })
        z, eq_viol, ineq_viol, converged = _tos_step(
            problem, b, z, x, AAt_factor, gamma, eta, eps_abs, eps_rel
        )
        if bool(converged):
            print(f"Converged after {iteration} iterations")
            break

    x.block_until_ready()
    diagnostics = {
        "converged": bool(converged),
        "iterations": iteration,
        "gamma": gamma,
        "equality_residual": float(eq_viol),
        "inequality_residual": float(ineq_viol),
        "inequality_projection_seconds": projection_seconds,
        "total_seconds": time.perf_counter() - start,
    }
    if trajectory is not None:
        diagnostics["trajectory"] = trajectory
    return x, diagnostics
