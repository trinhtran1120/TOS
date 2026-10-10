"""L-BFGS correction of a learned QP inequality projection."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


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


