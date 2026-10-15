"""L-BFGS feasibility seeking with intermediate Moreau-radius retraction."""

from functools import partial
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor

from learned_tos import learned_projection, predict_batch
from problem import QP
from tos_std import _tos_step


def _round_records(data, count):
    return [
        {name: (bool(values[index]) if name == "terminated_before_retraction"
                else int(values[index]) if name == "lbfgs_iterations"
                else float(values[index]))
         for name, values in data.items()}
        for index in range(int(count))
    ]


@partial(jax.jit, static_argnames=("steps", "max_cycles"))
def correct_projection(z, predicted_x, radius, G, h, steps=50, mu=0.1,
                       eps_feas=1e-3, eps_geom=1e-3, max_cycles=50):
    """Seek a feasible point near the learned sphere with L-BFGS.

    The learned radius may not intersect Gx <= h, so cycles have a finite cap
    and stop when neither criterion improves.
    """
    memory = max(1, min(steps, 30))
    alphas = 0.5 ** jnp.arange(20, dtype=z.dtype)
    guard = jnp.finfo(z.dtype).eps * jnp.maximum(1.0, radius)
    s_history = jnp.zeros((memory, z.size), dtype=z.dtype)
    y_history = jnp.zeros_like(s_history)
    inverse_curvature = jnp.zeros(memory, dtype=z.dtype)

    def evaluate(point):
        gaps = G @ point - h
        residual = jnp.maximum(gaps, 0.0)
        displacement = point - z
        distance = jnp.linalg.norm(displacement)
        radius_error = distance - radius
        feasibility_term = 0.5 * jnp.dot(residual, residual)
        radius_term = 0.5 * mu * radius_error ** 2
        grad = G.T @ residual + mu * radius_error * displacement / jnp.maximum(distance, guard)
        return (gaps, feasibility_term + radius_term, grad, jnp.max(residual),
                jnp.abs(radius_error), feasibility_term, radius_term)

    def direction_from_history(grad, s_hist, y_hist, rho_hist, head, size):
        def backward(q, index):
            slot = (head - 1 - index) % memory
            coefficient = rho_hist[slot] * jnp.dot(s_hist[slot], q)
            coefficient = jnp.where(index < size, coefficient, 0.0)
            return q - coefficient * y_hist[slot], coefficient

        q, coefficients = jax.lax.scan(backward, grad, jnp.arange(memory))
        newest = (head - 1) % memory
        s_last, y_last = s_hist[newest], y_hist[newest]
        scale = jnp.where(
            size > 0,
            jnp.dot(s_last, y_last) / jnp.maximum(jnp.dot(y_last, y_last), guard),
            1.0,
        )

        def forward(result, index):
            slot = (head - size + index) % memory
            coefficient = coefficients[jnp.maximum(size - 1 - index, 0)]
            beta = rho_hist[slot] * jnp.dot(y_hist[slot], result)
            updated = result + s_hist[slot] * (coefficient - beta)
            return jnp.where(index < size, updated, result), None

        result, _ = jax.lax.scan(forward, scale * q, jnp.arange(memory))
        return -result

    def step(carry):
        (u, gaps, merit, grad, _, s_hist, y_hist, rho_hist, head, size,
         attempts, accepted_count, _, _, _, _, _) = carry
        direction = direction_from_history(grad, s_hist, y_hist, rho_hist, head, size)
        direction = jnp.where(jnp.dot(grad, direction) < 0, direction, -grad)
        slope = jnp.dot(grad, direction)
        directional_gaps = G @ direction

        def trial(alpha):
            residual = jnp.maximum(gaps + alpha * directional_gaps, 0.0)
            distance = jnp.linalg.norm(u + alpha * direction - z)
            candidate_merit = (0.5 * jnp.dot(residual, residual)
                               + 0.5 * mu * (distance - radius) ** 2)
            armijo = candidate_merit <= merit + 1e-4 * alpha * slope
            accepted = (
                jnp.isfinite(candidate_merit) & armijo & (slope < 0)
                & jnp.all(jnp.isfinite(direction)) & (merit > 0)
            )
            return accepted

        accepted = jax.vmap(trial)(alphas)
        chosen = jnp.argmax(accepted)
        moved = jnp.any(accepted)
        new_u = jnp.where(moved, u + alphas[chosen] * direction, u)
        new_gaps, new_merit, new_grad, new_violation, new_geom, feasibility_term, radius_term = evaluate(new_u)

        displacement = new_u - u
        gradient_change = new_grad - grad
        curvature = jnp.dot(displacement, gradient_change)
        store_pair = moved & jnp.isfinite(curvature) & (
            curvature > guard * jnp.linalg.norm(displacement) * jnp.linalg.norm(gradient_change)
        )
        s_hist = s_hist.at[head].set(jnp.where(store_pair, displacement, s_hist[head]))
        y_hist = y_hist.at[head].set(jnp.where(store_pair, gradient_change, y_hist[head]))
        rho_hist = rho_hist.at[head].set(jnp.where(
            store_pair, 1.0 / jnp.maximum(curvature, guard), rho_hist[head]
        ))
        head = jnp.where(store_pair, (head + 1) % memory, head)
        size = jnp.where(store_pair, jnp.minimum(size + 1, memory), size)
        return (new_u, new_gaps, new_merit, new_grad, new_violation,
                s_hist, y_hist, rho_hist, head, size, attempts + 1,
                accepted_count + moved, ~moved, new_geom, feasibility_term,
                radius_term, jnp.where(moved, alphas[chosen], carry[16]))

    def continue_seeking(carry):
        return (carry[10] < steps) & ((carry[4] > eps_feas) | (carry[13] > eps_geom)) & (~carry[12])

    def seek(point, lbfgs_history):
        gaps, merit, grad, violation, geom, feasibility_term, radius_term = evaluate(point)
        s_hist, y_hist, rho_hist, head, size = lbfgs_history
        start = (point, gaps, merit, grad, violation, s_hist, y_hist,
                 rho_hist, head, size,
                 jnp.array(0), jnp.array(0), jnp.array(False), geom,
                 feasibility_term, radius_term, jnp.array(0.0, dtype=z.dtype))
        return jax.lax.while_loop(continue_seeking, step, start)

    def retract(point):
        displacement = point - z
        length = jnp.linalg.norm(displacement)
        fallback = predicted_x - z
        direction = jnp.where(length > guard, displacement, fallback)
        return z + radius * direction / jnp.maximum(jnp.linalg.norm(direction), guard)

    round_diagnostics = {
        "feasibility_residual": jnp.zeros(max_cycles, dtype=z.dtype),
        "geometric_residual": jnp.zeros(max_cycles, dtype=z.dtype),
        "feasibility_term": jnp.zeros(max_cycles, dtype=z.dtype),
        "radius_term": jnp.zeros(max_cycles, dtype=z.dtype),
        "objective": jnp.zeros(max_cycles, dtype=z.dtype),
        "accepted_step_size": jnp.zeros(max_cycles, dtype=z.dtype),
        "lbfgs_iterations": jnp.zeros(max_cycles, dtype=jnp.int32),
        "terminated_before_retraction": jnp.zeros(max_cycles, dtype=jnp.bool_),
    }

    def cycle(carry):
        current, best, cycles, accepted, attempts, violation, geom, _, history, lbfgs_history = carry
        result = seek(current, lbfgs_history)
        candidate = result[0]
        candidate_violation = result[4]
        candidate_geom = result[13]
        succeeded = (candidate_violation <= eps_feas) & (candidate_geom <= eps_geom)
        score = jnp.maximum(candidate_violation / eps_feas, candidate_geom / eps_geom)
        old_score = jnp.maximum(violation / eps_feas, geom / eps_geom)
        improved = ((cycles == 0) | ((candidate_violation <= eps_feas) & (violation > eps_feas))
                    | (((candidate_violation <= eps_feas) == (violation <= eps_feas))
                       & (score < old_score)))
        next_start = jax.lax.cond(succeeded, lambda: candidate, lambda: retract(candidate))
        history = {
            "feasibility_residual": history["feasibility_residual"].at[cycles].set(candidate_violation),
            "geometric_residual": history["geometric_residual"].at[cycles].set(candidate_geom),
            "feasibility_term": history["feasibility_term"].at[cycles].set(result[14]),
            "radius_term": history["radius_term"].at[cycles].set(result[15]),
            "objective": history["objective"].at[cycles].set(result[2]),
            "accepted_step_size": history["accepted_step_size"].at[cycles].set(result[16]),
            "lbfgs_iterations": history["lbfgs_iterations"].at[cycles].set(result[10]),
            "terminated_before_retraction": history["terminated_before_retraction"].at[cycles].set(succeeded),
        }
        return (jnp.where(improved, next_start, current),
                jnp.where(improved, candidate, best), cycles + 1,
                accepted + result[11], attempts + result[10],
                jnp.where(improved, candidate_violation, violation),
                jnp.where(improved, candidate_geom, geom), succeeded | ~improved,
                history, result[5:10])

    def continue_cycles(carry):
        return (carry[2] < max_cycles) & (~carry[7])

    start = (predicted_x, predicted_x, jnp.array(0), jnp.array(0), jnp.array(0),
             jnp.array(jnp.inf, dtype=z.dtype), jnp.array(jnp.inf, dtype=z.dtype),
             jnp.array(False), round_diagnostics,
             (s_history, y_history, inverse_curvature, jnp.array(0), jnp.array(0)))
    result = jax.lax.while_loop(continue_cycles, cycle, start)
    return result[1], result[3], result[4], result[5], result[6], result[2], result[8]


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
