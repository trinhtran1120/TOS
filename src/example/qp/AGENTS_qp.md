# Instructions for Codex: projection QP research experiment

## Mission

Implement `guide.md` in Python/JAX. The first deliverable is a validated, reproducible large projection-QP benchmark with synthetic prediction errors and 0–3 feasibility corrections. Complete implementation and run appropriate checks; do not stop at a plan. Network training and outer TOS are deferred.

Read `guide.md` before coding. Inspect existing repository instructions and dependencies. Preserve existing structure and unrelated files. If the broader N2027 project note conflicts with the latest experiment, follow this guide for the current scope; flag material ambiguity rather than silently changing the method.

## Before implementation

State assumptions, a short implementation plan, and verification goals. Ask only when a missing decision materially blocks correctness. Use the specified defaults for routine choices. Do not add speculative features or abstractions. Every changed line must support this experiment.

## Mathematical rules

- QP: min 0.5*||x-z||^2 subject to Gx<=h, Ax=b. Use G,h for inequalities and A,b for equalities.
- Default n=1000, m=2000, p=100, seed=42; use the paired normalized Gaussian generator and strict feasible center in the guide.
- Validate reference feasibility, stationarity, multiplier signs, and complementarity independently of solver status.
- Initialize using controlled angular error and radius error from validated reference projections. Label this oracle-based diagnostic honestly.
- Residual includes positive-part inequalities and weighted equalities. At an inequality switch use residual derivative zero.
- Use ambient damped Gauss–Newton, then sphere normalization. Never silently substitute a tangent-constrained solve, adaptive radius, equality elimination, or another algorithm.
- Accept using the post-normalization residual. For negative curve derivative use Armijo; otherwise require strict decrease. Retain the point on failure and log the stall.
- Never form matrix inverses. Never assume ambient descent survives normalization.
- Keep radius fixed within a correction trajectory. Count zero radii, invalid directions, nonfinite trials, and failed references.
- Record J=0,1,2,3; compare no-retraction Gauss–Newton and sphere gradient corrections using matched data and budgets.

## Implementation rules

- Enable JAX float64 before creating arrays. Correction kernels must use JAX; CPU NumPy/SciPy/OSQP are allowed for references, I/O, and analysis.
- Make `matrix_free_cg` the default large-problem backend; keep `dense_cholesky` for validation and explicit comparisons. Both solve the same ambient damped Gauss–Newton system.
- Implement Jv and JT through matvecs, and H(v)=G.T@(D*(G@v))+rho*A.T@(A@v)+lambda*v. Do not construct a residual Jacobian, Gram matrix, or factorization in the matrix-free backend. Freeze D within each linear solve.
- Use diagonal-preconditioned CG, initial direction zero, rtol=1e-6, atol=1e-10, maxiter=200. Check the true linear residual and record iterations, convergence, cap hits, and breakdowns. Follow the guide's failure/acceptance rules; never silently fall back to dense solves.
- Matrix-free does not remove storage for dense G/A. Account for preconditioner construction, matvecs, residual checks, memory, and all CG iterations in timing. Avoid active-row boolean slicing inside jit.
- Use vmap for candidate steps and lax.scan for sequential corrections. Do not claim parallel-in-time execution.
- Use safe denominators and explicit masks for invalid candidates. Preserve strict acceptance conditions and the current iterate on failure.
- Bound query batches to available memory; do not vmap all dense factorizations without checking memory.
- Use explicit PRNG subkeys and save arrays, seeds, settings, package versions, backend, and device.
- Pin tested dependencies. Keep the repository small; avoid unrequested training frameworks, dashboards, or architecture changes.

## Verification and execution

Run mathematical tests and a small smoke experiment before the default large run. Tests must cover analytic references, KKT validation, direction equivalence, Jacobian adjoint identity, matrix-free/dense products and direction agreement, PCG termination/failure diagnostics, the normalized-curve derivative, radius invariance, accepted residual decrease, and stalls/degenerate cases. Do not write tests that assert the hypothesis is always true.

Synchronize JAX timing. Separate compilation, setup, transfers, and steady-state work; include required preprocessing and every candidate/solve in total latency. Report CPU reference timing and JAX device timing with their different hardware stated. Do not count synthetic reference-derived initialization as a real neural prediction.

Run the default benchmark if resources permit. If a resource or dependency prevents it, complete independent work, report the precise blocker and completed checks, and provide runnable commands; never fabricate results. Sensitivity and n=3000 runs are explicit later stages.

Preserve per-query results and failures. Report projection error, inequality and equality violation, sphere-radius drift, accepted steps, stalls, and timings. Summarize spread and worst cases as well as typical behavior. Keep a brief decision log of findings and next steps.

## Claims and completion

Do not infer convergence from monotone violation, ICNN convexity, or the contracting-dynamics paper. The exact-radius sphere is tangent to the convex set; correction can stall. Underestimated radii cannot become feasible through angular changes. Report failed regimes.

Completion requires runnable implementation, documented commands, passing relevant checks, saved experiment metadata/results for runs actually completed, and a concise evidence-based report. State whether the sphere mechanism improves the measured projection error and what remains uncertain. A negative outcome is a valid research result.
