"""Generate Moreau-envelope labels from the inequality queries in TOS."""

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.linalg import cho_factor
import numpy as np

from example.qp.problem import generate_problem
from example.qp.tos import _tos_step, projection_inequality


def tos_gen_data(problem, b, max_iter, tau, record, gamma=1.0, eta=1.5):
    eps_abs = 1e-6
    eps_rel = 1e-6
    tol = 1e-6
    projection_ineq = projection_inequality(problem, tol)
    AAt_factor, _ = cho_factor(problem.A @ problem.A.T, lower=True)
    z = jnp.zeros_like(problem.p)

    for iteration in range(1, max_iter + 1):
        # x-update
        x, projection_value, _ = projection_ineq(z)
        # collect data
        ME_value = projection_value / tau
        grad_ME = (z - x) / tau
        record(iteration, z, x, ME_value, grad_ME)
        z, _, _, converged = _tos_step(
            problem, b, z, x, AAt_factor, gamma, eta, eps_abs, eps_rel
        )
        if bool(converged):
            print(f"Converge at the {iteration} iteration")
            break

    x.block_until_ready()
    return x, bool(converged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/data.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "assets/qp/datasets/moreau_data_qp.npz")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--limit-runs", type=int, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    count = config["dataset"]["num_samples"]
    ratios = config["dataset"]["split"]
    if args.tau <= 0 or args.gamma <= 0 or args.eta <= 0 or args.max_iter < 2 or count < 1:
        parser.error("tau, gamma, and eta must be positive; max-iter >= 2 and num_samples >= 1")
    if set(ratios) != {"train", "validation", "test"} or not np.isclose(
        sum(ratios.values()), 1.0
    ) or any(value < 0 for value in ratios.values()):
        parser.error("split must contain nonnegative train/validation/test ratios summing to 1")
    if args.limit_runs is not None and not 1 <= args.limit_runs <= count:
        parser.error("limit-runs must be between 1 and num_samples")
    n_train = round(count * ratios["train"])
    n_validation = round(count * ratios["validation"])
    if n_train + n_validation > count:
        parser.error("split rounding exceeds num_samples")
    split_by_run = np.full(count, "test", dtype="<U10")
    order = np.random.default_rng(args.seed + 1).permutation(count)
    split_by_run[order[:n_train]] = "train"
    split_by_run[order[n_train:n_train + n_validation]] = "validation"

    problem, b_batch = generate_problem(seed=args.seed, n_samples=count)
    G = np.asarray(problem.G)
    h = np.asarray(problem.h)
    rows = []
    final_x = []
    final_run_ids = []
    failures = []
    runs = count if args.limit_runs is None else args.limit_runs

    for run_id in range(runs):
        b = b_batch[run_id]
        first_row = len(rows)

        def record(iteration, z, x_star, ME_value, grad_ME):
            # Every run starts at z=0; saving it in all splits would leak an
            # identical label. The first b-dependent query is iteration 2.
            if iteration == 1:
                return
            z_host = np.asarray(z, dtype=np.float64).copy()
            x_host = np.asarray(x_star, dtype=np.float64).copy()
            ME_value = float(ME_value)
            grad_host = np.asarray(grad_ME, dtype=np.float64).copy()
            violation = float(np.max(np.maximum(G @ x_host - h, 0.0)))
            if not np.allclose(grad_host, (z_host - x_host) / args.tau,
                               rtol=1e-12, atol=1e-10) or violation > 1e-6:
                raise RuntimeError("Projection label failed validation")
            rows.append((run_id, iteration, z_host, ME_value, grad_host, violation))

        try:
            x, _ = tos_gen_data(problem, b, args.max_iter, args.tau, record, args.gamma, args.eta)
            final_x.append(np.asarray(x, dtype=np.float64).copy())
            final_run_ids.append(run_id)
        except (RuntimeError, ValueError) as error:
            del rows[first_row:]
            failures.append({"run_id": run_id, "error": str(error)})

    if not rows:
        raise RuntimeError(f"No validated labels generated; failures={failures[:3]}")

    row_run_ids = np.asarray([row[0] for row in rows])
    output = args.output.with_name(
        f"{args.output.stem}_eta{args.eta:g}_gamma{args.gamma:g}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        z=np.stack([row[2] for row in rows]),
        envelope=np.asarray([row[3] for row in rows]),
        gradient=np.stack([row[4] for row in rows]),
        inequality_residual=np.asarray([row[5] for row in rows]),
        final_x=np.stack(final_x),
        final_run_id=np.asarray(final_run_ids),
        run_id=row_run_ids,
        iteration=np.asarray([row[1] for row in rows]),
        split=split_by_run[row_run_ids],
        b=np.asarray(b_batch),
        G=G,
        h=h,
        tau=np.asarray(args.tau),
        gamma=np.asarray(args.gamma),
        eta=np.asarray(args.eta),
    )
    output.with_suffix(".json").write_text(json.dumps({
        "seed": args.seed,
        "num_b_samples": count,
        "runs_executed": runs,
        "num_labels": len(rows),
        "max_iter": args.max_iter,
        "tau": args.tau,
        "gamma": args.gamma,
        "eta": args.eta,
        "split_counts_by_run": {
            name: int(np.count_nonzero(split_by_run == name))
            for name in ("train", "validation", "test")
        },
        "failures": failures,
    }, indent=2) + "\n")
    print(f"Saved {len(rows)} labels from {runs} TOS runs to {output}")


if __name__ == "__main__":
    main()
