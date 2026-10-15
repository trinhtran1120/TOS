"""Benchmark L-BFGS-corrected learned TOS on held-out QP runs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from qp.corrected import learn_and_correct_tos
from qp.learned import load_checkpoint
from qp.problem import generate_problem, objective
from qp.reference import solve_qp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_DATA = ROOT / "assets/qp/datasets/moreau_data_qp_eta1.5_gamma1.npz"
DEFAULT_CHECKPOINT = ROOT / "assets/qp/checkpoints/qp_icnn.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--correction-steps", type=int, default=100)
    parser.add_argument("--correction-cycles", type=int, default=100)
    parser.add_argument("--mu", type=float, default=200.0)
    parser.add_argument("--eps-abs", type=float, default=1e-4)
    parser.add_argument("--eps-rel", type=float, default=1e-4)
    parser.add_argument("--eps-feas", type=float, default=1e-5)
    parser.add_argument("--eps-geom", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.5)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = ROOT / "runs/qp/generated" / run_id / "corrected.json"

    metadata = json.loads(args.data.with_suffix(".json").read_text())
    with np.load(args.data) as data:
        held_out_ids = np.unique(data["run_id"][data["split"] == "test"])
        run_ids = held_out_ids[:args.samples]
        b_batch = np.asarray(data["b"][run_ids])
        stored_g = np.asarray(data["G"])
        stored_h = np.asarray(data["h"])
    if len(run_ids) != args.samples:
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

    params, model = load_checkpoint(args.checkpoint)
    learn_and_correct_tos(
        problem, jnp.asarray(b_batch[0]), params, model,
        gamma=args.gamma, eta=args.eta, max_iter=2, correction_steps=args.correction_steps,
        correction_cycles=args.correction_cycles, mu=args.mu,
        eps_abs=args.eps_abs, eps_rel=args.eps_rel,
        eps_feas=args.eps_feas, eps_geom=args.eps_geom,
    )
    print(f"device={jax.devices()[0]} correction=L-BFGS steps={args.correction_steps} cycles={args.correction_cycles}")
    print(f"held-out run IDs={run_ids.tolist()}")
    print("gap (%) = 100 * abs(objective(OSQP) - objective(corrected TOS)) / abs(objective(OSQP))")
    print(
        "run  gap (%)          OSQP (s)  learned TOS (s)  learned projection in TOS (s)  "
        "correction (s)  iterations  converged  eq residual  ineq residual"
    )
    osqp_total = 0.0
    learned_total = 0.0
    projection_total = 0.0
    correction_total = 0.0
    rows = []
    for run_id, b in zip(run_ids, b_batch):
        b = jnp.asarray(b)
        reference, osqp_seconds = solve_qp(problem, b)
        learned, info = learn_and_correct_tos(
            problem, b, params, model, gamma=args.gamma, eta=args.eta,
            max_iter=args.max_iter, correction_steps=args.correction_steps,
            correction_cycles=args.correction_cycles, mu=args.mu,
            eps_abs=args.eps_abs, eps_rel=args.eps_rel,
            eps_feas=args.eps_feas, eps_geom=args.eps_geom,
        )
        reference_value = float(objective(problem, reference))
        learned_value = float(objective(problem, learned))
        gap = (
            100.0 * abs(reference_value - learned_value) / abs(reference_value)
            if reference_value != 0.0 else float("nan")
        )
        learned_seconds = float(info["total_seconds"])
        projection_seconds = float(info["inequality_projection_seconds"])
        correction_seconds = float(info["correction_seconds"])
        osqp_total += osqp_seconds
        learned_total += learned_seconds
        projection_total += projection_seconds
        correction_total += correction_seconds
        print(
            f"{run_id:>3}  {gap:>+.6e}  {osqp_seconds:>10.4f}  "
            f"{learned_seconds:>11.4f}  {projection_seconds:>16.4f}  "
            f"{correction_seconds:>14.4f}  "
            f"{info['iterations']:>10}  {str(info['converged']):>9}  "
            f"{info['equality_residual']:>11.3e}  "
            f"{info['inequality_residual']:>13.3e}"
        )
        rows.append({
            "run_id": int(run_id),
            "reference_objective": reference_value,
            "learned_objective": learned_value,
            "gap_percent": gap,
            "osqp_seconds": osqp_seconds,
            "learned": info,
        })

    print(
        f"Total solving time: OSQP={osqp_total:.4f}s, "
        f"learned TOS={learned_total:.4f}s, "
        f"learned projection={projection_total:.4f}s, correction={correction_total:.4f}s"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": metadata["seed"],
        "gamma": args.gamma,
        "eta": args.eta,
        "max_iter": args.max_iter,
        "correction_steps": args.correction_steps,
        "correction_cycles": args.correction_cycles,
        "mu": args.mu,
        "eps_abs": args.eps_abs,
        "eps_rel": args.eps_rel,
        "eps_feas": args.eps_feas,
        "eps_geom": args.eps_geom,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
