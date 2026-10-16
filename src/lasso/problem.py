"""Configuration-driven LASSO constraints and JAX instance parameters.

Stored matrices and generated x rows are JAX arrays. NumPy preserves the
current default_rng matrix draw order. Float64 preserves reference precision.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class LassoConfig:
    """Problem, sampling, solver, split, and output settings from configs.json."""

    n_var: int
    n_ineq: int
    n_eq: int
    n_samples: int
    seed: int
    problem_seed: int
    quadratic_seed: int
    quadratic_scale: float
    quadratic_ridge: float
    matrix_discard_samples: int
    x_low: float
    x_high: float
    l1_weight: float
    rho: float
    q_low: float
    q_high: float
    q_seed: int
    split_ratio: list[float]
    split_seed: int
    solver: str
    solver_tol: float
    solver_max_iter: int
    progress_every: int
    output_dir: str
    output_file: str
    training: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject configurations that cannot produce the requested dataset."""
        if self.n_var < 1 or not 1 <= self.n_eq <= self.n_var or self.n_ineq < 1:
            raise ValueError("Require n_var >= n_eq >= 1 and n_ineq >= 1")
        if self.n_samples < 0 or self.matrix_discard_samples < 0:
            raise ValueError("Sample counts must be nonnegative")
        for low, high in ((self.x_low, self.x_high), (self.q_low, self.q_high)):
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError("Sampling bounds must be finite with low < high")
        if not np.isfinite(self.l1_weight) or self.l1_weight < 0:
            raise ValueError("l1_weight must be finite and nonnegative")
        if not np.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be finite and positive")
        if (
            not np.isfinite([self.quadratic_scale, self.quadratic_ridge]).all()
            or self.quadratic_scale <= 0
            or self.quadratic_ridge <= 0
        ):
            raise ValueError("Quadratic scale and ridge must be finite and positive")
        if self.progress_every < 0:
            raise ValueError("progress_every must be nonnegative")
        ratios = np.asarray(self.split_ratio)
        if (
            ratios.shape != (3,)
            or np.any(ratios < 0)
            or not np.isclose(ratios.sum(), 1)
        ):
            raise ValueError(
                "split_ratio must contain three nonnegative fractions summing to 1"
            )
        if self.solver != "CLARABEL":
            raise ValueError("This generator supports the CLARABEL solver")
        if (
            not np.isfinite(self.solver_tol)
            or self.solver_tol <= 0
            or self.solver_max_iter < 1
        ):
            raise ValueError("Solver tolerance and iteration limit must be positive")


def load_config(path: str | Path | None = None) -> LassoConfig:
    """Load setup settings from the sibling configs.json or an explicit file."""
    config_path = (
        Path(path) if path is not None else Path(__file__).with_name("configs.json")
    )
    with config_path.open(encoding="utf-8-sig") as stream:
        return LassoConfig(**json.load(stream))


@dataclass(frozen=True)
class LassoProblem:
    """Fixed coefficients and constraints shared across LASSO instances.

    Attributes:
        q_matrix: Positive definite quadratic objective matrix Q.
        p: Linear coefficient with one entry per decision variable.
        a: Equality matrix in A y = x.
        g: Inequality matrix in G y <= h.
        h: Fixed inequality bound.
        l1_weight: L1 penalty coefficient, lambda.
        config: Settings used to generate this problem and its instances.
    """

    q_matrix: jax.Array
    p: jax.Array
    a: jax.Array
    g: jax.Array
    h: jax.Array
    l1_weight: float
    config: LassoConfig

    @property
    def n_var(self) -> int:
        """Return the number of decision variables."""
        return self.a.shape[1]

    @property
    def n_eq(self) -> int:
        """Return the dimension of each instance parameter x."""
        return self.a.shape[0]

    @property
    def n_ineq(self) -> int:
        """Return the number of inequality constraints."""
        return self.g.shape[0]


def generate_problem(config: LassoConfig | None = None) -> LassoProblem:
    """Generate fixed coefficients and constraints from the configuration.

    Preserve the current default_rng draw order, including discarded draws.
    The bound h = radius * abs(G pinv(A)) @ ones guarantees a feasible
    witness y = pinv(A) x throughout the configured x box.
    """
    config = load_config() if config is None else config
    rng = np.random.default_rng(config.problem_seed)
    rng.random(config.n_var)
    p = rng.random(config.n_var)
    a = rng.normal(size=(config.n_eq, config.n_var))
    rng.uniform(
        config.x_low, config.x_high, (config.matrix_discard_samples, config.n_eq)
    )
    g = rng.normal(size=(config.n_ineq, config.n_var))
    radius = max(abs(config.x_low), abs(config.x_high))
    h = radius * np.sum(np.abs(g @ np.linalg.pinv(a)), axis=1)
    b = np.random.RandomState(config.quadratic_seed).normal(
        size=(config.n_var, config.n_var)
    )
    q_matrix = b @ b.T / config.quadratic_scale + config.quadratic_ridge * np.eye(
        config.n_var
    )
    q_matrix = (q_matrix + q_matrix.T) / 2
    return LassoProblem(
        jnp.array(q_matrix),
        jnp.array(p),
        jnp.array(a),
        jnp.array(g),
        jnp.array(h),
        config.l1_weight,
        config,
    )


def generate_data(
    problem: LassoProblem,
    *,
    n_samples: int | None = None,
    seed: int | None = None,
) -> jax.Array:
    """Generate only x rows in A y = x using the configured sample box.

    Return shape (n_samples, problem.n_eq). Optional overrides are useful
    for small checks; normal dataset generation reads all settings from JSON.
    """
    config = problem.config
    n_samples = config.n_samples if n_samples is None else n_samples
    seed = config.seed if seed is None else seed
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative")
    return jax.random.uniform(
        jax.random.key(seed),
        (n_samples, problem.n_eq),
        dtype=jnp.float64,
        minval=config.x_low,
        maxval=config.x_high,
    )
