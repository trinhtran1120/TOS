"""Benchmark learned TOS against OSQP on held-out QP runs."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from experiments.qp.problem.learned import learned_projection, learned_tos, load_checkpoint
from experiments.qp.problem.problem import generate_problem, objective
from experiments.qp.problem.reference import solve_qp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_DATA = ROOT / "data/qp/moreau_data_qp_eta1.5_gamma1.npz"
DEFAULT_CHECKPOINT = ROOT / "src/qp/checkpoints/qp_icnn.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--run-ids", type=int, nargs="+", help="specific held-out run IDs")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=1.5)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = ROOT / "results/qp/generated" / run_id / "learned.json"

    metadata = json.loads(args.data.with_suffix(".json").read_text())
    with np.load(args.data) as data:
        held_out_ids = np.unique(data["run_id"][data["split"] == "test"])
        run_ids = np.asarray(args.run_ids) if args.run_ids is not None else held_out_ids[:args.samples]
        if len(np.unique(run_ids)) != len(run_ids) or not np.all(np.isin(run_ids, held_out_ids)):
            parser.error("--run-ids must contain distinct held-out run IDs")
        b_batch = np.asarray(data["b"][run_ids])
        stored_g = np.asarray(data["G"])
        stored_h = np.asarray(data["h"])
    if args.run_ids is None and len(run_ids) != args.samples:
        raise ValueError("Fewer held-out runs than requested samples")

    problem, _ = generate_problem(
        seed=metadata["seed"],
        n_var=stored_g.shape[1],
        n_eq=b_batch.shape[1],
        n_ineq=stored_g.shape[0],
        n_samples=1,
    )
    if not (
        np.allclose(np.asarray(problem.G), stored_g, rtol=0, atol=1e-10)
        and np.allclose(np.asarray(problem.h), stored_h, rtol=0, atol=1e-10)
    ):
        raise RuntimeError("Saved labels and benchmark use different inequality sets")

    setup_start = time.perf_counter()
    params, model = load_checkpoint(args.checkpoint)
    learned_projection(jnp.zeros_like(problem.p), params, model)
    learned_setup_seconds = time.perf_counter() - setup_start

    print(f"device={jax.devices()[0]} network_dtype=float32 tos_dtype={problem.p.dtype}")
    print(f"held-out run IDs={run_ids.tolist()} learned setup={learned_setup_seconds:.4f}s")
    print("gap (%) = 100 * abs(objective(OSQP) - objective(learned TOS)) / abs(objective(OSQP))")
    print(
        "run  gap (%)          OSQP (s)  learned (s)  learned proj (s)  "
        "learned converged  eq residual  ineq residual"
    )

    osqp_total = 0.0
    learned_total = 0.0
    projection_total = 0.0
    rows = []
    for run_id, b in zip(run_ids, b_batch):
        b = jnp.asarray(b)
        reference, osqp_seconds = solve_qp(problem, b)
        learned, diagnostics = learned_tos(
            problem, b, params, model, eta=args.eta, max_iter=args.max_iter
        )
        reference_value = float(objective(problem, reference))
        learned_value = float(objective(problem, learned))
        gap = (
            100.0 * abs(reference_value - learned_value) / abs(reference_value)
            if reference_value != 0.0
            else float("nan")
        )
        learned_seconds = float(diagnostics["total_seconds"])
        projection_seconds = float(diagnostics["inequality_projection_seconds"])
        osqp_total += osqp_seconds
        learned_total += learned_seconds
        projection_total += projection_seconds
        print(
            f"{run_id:>3}  {gap:>+.6e}  {osqp_seconds:>10.4f}  "
            f"{learned_seconds:>11.4f}  {projection_seconds:>16.4f}  "
            f"{str(diagnostics['converged']):>17}  "
            f"{diagnostics['equality_residual']:>11.3e}  "
            f"{diagnostics['inequality_residual']:>13.3e}"
        )
        rows.append({
            "run_id": int(run_id),
            "osqp_seconds": osqp_seconds,
            "reference_objective": reference_value,
            "learned_gap_percent": gap,
            "learned": diagnostics,
        })

    print(
        f"Total solving time: OSQP={osqp_total:.4f}s, learned TOS={learned_total:.4f}s, "
        f"learned inequality projection={projection_total:.4f}s"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as file:
        data_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    with args.checkpoint.open("rb") as file:
        checkpoint_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    args.output.write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "seed": metadata["seed"],
        "run_ids": run_ids.tolist(),
        "eta": args.eta,
        "max_iter": args.max_iter,
        "learned_setup_seconds": learned_setup_seconds,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
