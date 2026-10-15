# Corrected learned TOS: QP benchmark results

This report summarizes the archived run in [learned_tos_final.json](../../../runs/qp/legacy/learned_tos_final.json). It compares L-BFGS-corrected learned TOS with OSQP on 10 held-out instances of the same convex quadratic program. The reported learned solution is the corrected `x` point, so every residual and objective below is measured on that returned point.

## Main results

- **Convergence:** All 10 instances met the outer TOS stopping condition, in 22–24 iterations (23.3 on average). There were no recorded correction failures.
- **Accuracy:** The mean absolute objective gap from OSQP was **0.0121%**; the largest was **0.0272%**. The largest equality residual was **1.90 × 10⁻⁴**, and the largest inequality violation was **6.94 × 10⁻⁶**.
- **Time:** Across the 10 instances, learned TOS took **0.1139 s** and OSQP took **0.0732 s**. Learned TOS was **1.56×** the measured OSQP time in this run. Within learned TOS, learned projection took 0.0366 s and L-BFGS correction took 0.0469 s.

| Run ID | Gap (%) | OSQP (s) | Learned TOS (s) | Learned projection (s) | Correction (s) | TOS iterations | Equality residual | Inequality residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 0.0112 | 0.0099 | 0.0129 | 0.0042 | 0.0054 | 23 | 1.10e-04 | 2.72e-06 |
| 24 | 0.0048 | 0.0075 | 0.0123 | 0.0039 | 0.0050 | 23 | 1.59e-04 | 0.00e+00 |
| 30 | 0.0063 | 0.0071 | 0.0111 | 0.0038 | 0.0042 | 23 | 1.45e-04 | 0.00e+00 |
| 41 | 0.0182 | 0.0068 | 0.0100 | 0.0035 | 0.0037 | 24 | 1.65e-04 | 4.53e-07 |
| 43 | 0.0272 | 0.0065 | 0.0090 | 0.0033 | 0.0031 | 23 | 1.86e-04 | 2.84e-06 |
| 69 | 0.0085 | 0.0075 | 0.0129 | 0.0034 | 0.0064 | 22 | 1.75e-04 | 6.43e-06 |
| 72 | 0.0155 | 0.0068 | 0.0104 | 0.0037 | 0.0036 | 23 | 1.90e-04 | 1.50e-06 |
| 77 | 0.0050 | 0.0071 | 0.0124 | 0.0037 | 0.0054 | 24 | 7.97e-05 | 6.94e-06 |
| 96 | 0.0042 | 0.0070 | 0.0128 | 0.0035 | 0.0065 | 24 | 9.49e-05 | 5.90e-06 |
| 104 | 0.0198 | 0.0070 | 0.0100 | 0.0036 | 0.0036 | 24 | 1.27e-04 | 0.00e+00 |

## Experimental setup

The benchmark uses one fixed QP with 100 variables, 50 equality constraints, and 50 inequality constraints. It varies the equality right-hand side across the selected held-out run IDs. The problem seed is 17; the dataset and trained ICNN checkpoint are identified in the [source JSON](../../../runs/qp/legacy/learned_tos_final.json). That archived JSON records paths from before the move; the current files are in `assets/qp/`.

| Parameter | Value |
| --- | ---: |
| TOS step size `gamma` | 1.0 |
| TOS relaxation `eta` | 1.5 |
| Outer `eps_abs`, `eps_rel` | 1e-4 each |
| Maximum TOS iterations | 1,000 |
| L-BFGS radius weight `mu` | 200 |
| Maximum L-BFGS steps and correction rounds | 100 each |
| Inner `eps_feas`, `eps_geom` | 1e-5 each |

OSQP is the reference solver, called with `eps_abs=eps_rel=1e-8` and polishing enabled in [reference.py](../../../src/qp/reference.py). The objective gap is

```text
100 × |objective(OSQP) − objective(learned TOS)| / |objective(OSQP)|.
```

The equality residual is `max(abs(A @ x − b))`; the inequality residual is `max(max(G @ x − h, 0))`. The benchmark warms learned TOS for two iterations before timing the instances. Its learned times exclude checkpoint loading and warmup; the OSQP times include construction and solution of each CVXPY problem. The time columns therefore describe this benchmark's measured paths, rather than identical solver-only timing boundaries.

## Interpretation

The saved run shows small objective gaps and inequality violations for these 10 selected instances. Its returned `x` retains an equality residual around 10⁻⁴, so it is **not** a machine-precision equality solution. The reported convergence flag comes from the outer TOS stopping test; it does not imply exact feasibility. These results cover 10 of the dataset's 100 held-out runs and do not establish performance on the remaining runs.

To reproduce the benchmark with the parameters above, run:

```sh
uv run --locked python -m qp.commands.benchmark_corrected
```

The command writes a new JSON file under `runs/qp/generated/`. The archived [learned_tos_final.json](../../../runs/qp/legacy/learned_tos_final.json) remains unchanged.
