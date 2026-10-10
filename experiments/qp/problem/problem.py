"""Define the convex QP example from DC3 (Donti et.al., 2021)"""
from typing import NamedTuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


class QP(NamedTuple):
    Q: jax.Array
    p: jax.Array
    A: jax.Array
    G: jax.Array
    h: jax.Array

def generate_problem(seed: int = 17,
                     n_var: int = 100,
                     n_eq: int = 50,
                     n_ineq: int = 50,
                     n_samples: int = 10000,
        ) -> tuple[QP, jax.Array]:
    key = jax.random.PRNGKey(seed)
    key_q, key_p, key_a, key_g, key_b = jax.random.split(key, 5)
    
    Q = jax.random.uniform(key_q, (n_var, ))
    p = jax.random.uniform(key_p, (n_var, ))
    A = jax.random.normal(key_a, (n_eq, n_var))
    G = jax.random.normal(key_g, (n_ineq, n_var))
    b_instance = jax.random.uniform(key_b, (n_samples, n_eq), minval=-1.0, maxval=1.0)

    AAt = A @ A.T
    GA_plus = jnp.linalg.solve(AAt, (G @ A.T).T).T
    h = jnp.sum(jnp.abs(GA_plus), axis=1)

    problem = QP(
            Q = Q,
            p=p,
            A=A,
            G=G,
            h=h,
            )
    return problem, b_instance

def objective(problem: QP, y: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(y, problem.Q*y) + jnp.dot(problem.p, y)


def objective_grad(problem: QP, y: jax.Array) -> jax.Array:
    return problem.Q*y + problem.p


def equaility_res(problem: QP, y: jax.Array, b: jax.Array) -> jax.Array:
    return problem.A@y - b


def inequality_res(problem: QP, y: jax.Array) -> jax.Array:
    return problem.G@y - problem.h

def feasible_point(problem: QP, b: jax.Array) -> jax.Array:
    AAt = problem.A @ problem.A.T
    return problem.A.T @ jnp.linalg.solve(AAt, b)

