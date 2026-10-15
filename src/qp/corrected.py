"""Learned TOS with radius-aware inequality correction."""

from functools import partial
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor

from qp.correction import _round_records, correct_projection
from qp.learned import learned_projection, predict_batch
from qp.problem import QP
from qp.tos import _tos_step


def learn_and_correct_tos(
    problem: QP,
    b: jax.Array,
    params,
    model,
    gamma: float = 1.0,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    max_iter: int = 1000,
    eta: float = 1.5,
    record_trajectory: bool = False,
    correction_steps: int = 100,
    correction_cycles: int = 100,
    mu: float = 1.0,
    eps_feas: float = 1e-3,
    eps_geom: float = 1e-3,
):
    """Run learned TOS with feasibility and radius correction of the x-update."""
    start = time.perf_counter()
    AAt_factor, _ = cho_factor(problem.A @ problem.A.T, lower=True)
    z = jnp.zeros_like(problem.p)
    projection_seconds = 0.0
    correction_seconds = 0.0
    accepted_corrections = jnp.array(0)
    correction_attempts = jnp.array(0)
    retraction_cycles = jnp.array(0)
    correction_failures = jnp.array(0)
    trajectory = [] if record_trajectory else None

    for iteration in range(1, max_iter + 1):
        projection_start = time.perf_counter()
        x, envelope_value = learned_projection(z, params, model)
        radius = jnp.sqrt(2 * model["tau"] * envelope_value)
        radius.block_until_ready()
        projection_seconds += time.perf_counter() - projection_start
        correction_start = time.perf_counter()
        (x_next, corrected_count, attempted_count, violation,
         geom_residual, cycles, round_data) = correct_projection(
            z, x, radius, problem.G, problem.h,
            steps=correction_steps, mu=mu, eps_feas=eps_feas,
            eps_geom=eps_geom, max_cycles=correction_cycles,
        )
        x_next.block_until_ready()
        correction_seconds += time.perf_counter() - correction_start
        correction_attempts += attempted_count
        accepted_corrections += corrected_count
        retraction_cycles += cycles
        correction_ok = (
            (violation <= eps_feas) & (geom_residual <= eps_geom)
        )
        correction_failures += ~correction_ok
        if trajectory is not None:
            trajectory.append({
                "z": np.asarray(z).copy(),
                "predicted_x": np.asarray(x).copy(),
                "corrected_x": np.asarray(x_next).copy(),
                "envelope_value": envelope_value,
                "accepted_corrections": int(corrected_count),
                "corrected_violation": float(violation),
                "corrected_geometric_residual": float(geom_residual),
                "correction_rounds": _round_records(round_data, cycles),
            })
        z, eq_viol, ineq_viol, converged = _tos_step(
            problem, b, z, x_next, AAt_factor, gamma, eta, eps_abs, eps_rel
        )
        converged = bool(converged)
        if converged:
            print(f"Converged after {iteration} iterations")
            break

    x_next.block_until_ready()
    round_diagnostics = _round_records(round_data, cycles)
    diagnostics = {
        "converged": bool(converged),
        "iterations": iteration,
        "gamma": gamma,
        "equality_residual": float(eq_viol),
        "inequality_residual": float(ineq_viol),
        "inequality_projection_seconds": projection_seconds,
        "correction_seconds": correction_seconds,
        "correction_attempts": int(correction_attempts),
        "accepted_corrections": int(accepted_corrections),
        "correction_stalls": int(correction_attempts - accepted_corrections),
        "retraction_cycles": int(retraction_cycles),
        "correction_failures": int(correction_failures),
        "corrected_violation": float(violation),
        "corrected_geometric_residual": float(geom_residual),
        "correction_rounds": round_diagnostics,
        "mu": mu,
        "eps_feas": eps_feas,
        "eps_geom": eps_geom,
        "total_seconds": time.perf_counter() - start,
    }
    if trajectory is not None:
        diagnostics["trajectory"] = trajectory
    return x_next, diagnostics


@partial(jax.jit, static_argnames=("correction_steps", "correction_cycles"))
def _batched_step(problem, b_batch, z_batch, predicted, radii, AAt_factor,
                  gamma, eta, eps_abs, eps_rel, mu, eps_feas, eps_geom,
                  correction_steps, correction_cycles):
    def correct_one(z, x, radius):
        return correct_projection(
            z, x, radius, problem.G, problem.h,
            steps=correction_steps, mu=mu, eps_feas=eps_feas,
            eps_geom=eps_geom, max_cycles=correction_cycles,
        )

    corrected, accepted, attempted, violation, geom, cycles, _ = jax.vmap(
        correct_one
    )(z_batch, predicted, radii)
    next_z, eq_viol, ineq_viol, converged = jax.vmap(
        _tos_step, in_axes=(None, 0, 0, 0, None, None, None, None, None)
    )(
        problem, b_batch, z_batch, corrected, AAt_factor,
        gamma, eta, eps_abs, eps_rel,
    )
    return (corrected, next_z, eq_viol, ineq_viol, converged,
            accepted, attempted, violation, geom, cycles)


@partial(jax.jit, static_argnames=(
    "weight_act", "nonnegative_output", "correction_steps",
    "correction_cycles",
))
def _batched_solve(problem, b_batch, params, input_mean, input_std,
                   output_scale, output_offset, tau, weight_act,
                   nonnegative_output, AAt_factor, gamma, eta, eps_abs,
                   eps_rel, max_iter, correction_steps, correction_cycles,
                   mu, eps_feas, eps_geom):
    z_batch = jnp.zeros((len(b_batch), problem.p.size), dtype=problem.p.dtype)
    solutions = z_batch
    iterations = jnp.zeros(len(b_batch), dtype=jnp.int32)
    converged = jnp.zeros(len(b_batch), dtype=jnp.bool_)
    accepted_total = jnp.zeros_like(iterations)
    attempted_total = jnp.zeros_like(iterations)
    cycles_total = jnp.zeros_like(iterations)
    failures = jnp.zeros_like(iterations)
    eq_viol = jnp.zeros(len(b_batch), dtype=problem.p.dtype)
    ineq_viol = jnp.zeros_like(eq_viol)
    violation = jnp.zeros_like(eq_viol)
    geom = jnp.zeros_like(eq_viol)

    def continue_solve(carry):
        return (carry[0] < max_iter) & (~jnp.all(carry[3]))

    def solve_step(carry):
        (iteration, z_batch, solutions, converged, iterations,
         accepted_total, attempted_total, cycles_total, failures,
         eq_viol, ineq_viol, violation, geom) = carry
        values, gradients = predict_batch(
            params, z_batch.astype(jnp.float32),
            input_mean, input_std, output_scale, output_offset,
            weight_act, nonnegative_output,
        )
        predicted = z_batch - tau * gradients.astype(z_batch.dtype)
        radii = jnp.sqrt(2 * tau * values)
        (corrected, next_z, next_eq, next_ineq, next_converged,
         accepted, attempted, next_violation, next_geom, cycles) = _batched_step(
            problem, b_batch, z_batch, predicted, radii, AAt_factor,
            gamma, eta, eps_abs, eps_rel, mu, eps_feas, eps_geom,
            correction_steps, correction_cycles,
        )
        active = ~converged
        solutions = jnp.where(active[:, None], corrected, solutions)
        z_batch = jnp.where(active[:, None], next_z, z_batch)
        iterations = jnp.where(active, iteration + 1, iterations)
        eq_viol = jnp.where(active, next_eq, eq_viol)
        ineq_viol = jnp.where(active, next_ineq, ineq_viol)
        violation = jnp.where(active, next_violation, violation)
        geom = jnp.where(active, next_geom, geom)
        accepted_total += jnp.where(active, accepted, 0)
        attempted_total += jnp.where(active, attempted, 0)
        cycles_total += jnp.where(active, cycles, 0)
        failures += active & ((next_violation > eps_feas) | (next_geom > eps_geom))
        converged |= active & next_converged
        return (iteration + 1, z_batch, solutions, converged, iterations,
                accepted_total, attempted_total, cycles_total, failures,
                eq_viol, ineq_viol, violation, geom)

    result = jax.lax.while_loop(
        continue_solve, solve_step,
        (jnp.array(0), z_batch, solutions, converged, iterations,
         accepted_total, attempted_total, cycles_total, failures,
         eq_viol, ineq_viol, violation, geom),
    )
    return (result[2], result[4], result[3], result[5], result[6],
            result[7], result[8], result[9], result[10], result[11],
            result[12])


@partial(jax.jit, static_argnames=(
    "weight_act", "nonnegative_output", "correction_steps",
    "correction_cycles",
))
def _mapped_solve(problem, b_batch, params, input_mean, input_std,
                  output_scale, output_offset, tau, weight_act,
                  nonnegative_output, AAt_factor, gamma, eta, eps_abs,
                  eps_rel, max_iter, correction_steps, correction_cycles,
                  mu, eps_feas, eps_geom):
    def solve_one(b):
        result = _batched_solve(
            problem, b[None, :], params, input_mean, input_std,
            output_scale, output_offset, tau, weight_act,
            nonnegative_output, AAt_factor, gamma, eta, eps_abs,
            eps_rel, max_iter, correction_steps, correction_cycles,
            mu, eps_feas, eps_geom,
        )
        return tuple(value[0] for value in result)

    return jax.lax.map(solve_one, b_batch)


def batch_learn_and_correct_tos(
    problem, b_batch, params, model, gamma=1.0, eps_abs=1e-2,
    eps_rel=1e-2, max_iter=1000, eta=1.5, correction_steps=50,
    correction_cycles=50, mu=1.0, eps_feas=1e-3, eps_geom=1e-3,
    parallel_correction=None,
):
    """Solve independent QPs together while retaining per-instance stopping."""
    start = time.perf_counter()
    AAt_factor, _ = cho_factor(problem.A @ problem.A.T, lower=True)
    if parallel_correction is None:
        parallel_correction = jax.default_backend() != "cpu"
    solve = _batched_solve if parallel_correction else _mapped_solve
    (solutions, iterations, converged, accepted_total, attempted_total,
     cycles_total, failures, eq_viol, ineq_viol, violation, geom) = solve(
        problem, b_batch, params, model["input_mean"], model["input_std"],
        model["output_scale"], model["output_offset"], model["tau"],
        model["weight_act"], model["nonnegative_output"], AAt_factor,
        gamma, eta, eps_abs, eps_rel, max_iter, correction_steps,
        correction_cycles, mu, eps_feas, eps_geom,
    )

    solutions.block_until_ready()
    elapsed = time.perf_counter() - start
    diagnostics = {
        "converged": np.asarray(converged),
        "iterations": np.asarray(iterations),
        "equality_residual": np.asarray(eq_viol),
        "inequality_residual": np.asarray(ineq_viol),
        "correction_attempts": np.asarray(attempted_total),
        "accepted_corrections": np.asarray(accepted_total),
        "retraction_cycles": np.asarray(cycles_total),
        "correction_failures": np.asarray(failures),
        "corrected_violation": np.asarray(violation),
        "corrected_geometric_residual": np.asarray(geom),
        "total_seconds": elapsed,
    }
    return solutions, diagnostics
