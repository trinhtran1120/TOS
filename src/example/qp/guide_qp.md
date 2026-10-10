# Large projection QP: Python/JAX implementation guide

## Objective and scope

Implement a reproducible diagnostic of our learned-sphere projection approach. First simulate prediction errors using trusted reference projections; do not train a network or implement outer TOS yet. Test whether 1–3 damped Gauss–Newton corrections followed by sphere normalization improve projection accuracy at a reasonable total cost.

This guide follows the latest research discussion. The attached project specification describes a broader, earlier TOS plan; this experiment isolates its projection component. Sphere normalization is auxiliary search geometry, not a manifold constraint on the original QP. No contraction, convergence, or speedup is assumed.

## 1. Problem generation

Solve, for each query z:

```text
minimize_x  0.5 * ||x - z||_2^2
subject to G @ x <= h, A @ x = b.
```

Default dimensions: n=1000, m=2000, p=100. Seed=42. Generate one fixed feasible set and vary z. Optional later scale: n=3000, m=6000, p=300; do not run automatically before the default experiment passes validation.

1. Thin QR of a Gaussian (n,p) matrix gives Q. Set A=Q.T, so A @ A.T=I.
2. Generate Gaussian B with shape (n,n); normalize each row to unit Euclidean norm.
3. Sample x_ctr uniformly in [-0.5,0.5]^n and s_plus, s_minus uniformly in [0.5,1.5]^n.
4. Set G=concatenate([B,-B], axis=0), h=G @ x_ctr + concatenate([s_plus,s_minus]), b=A @ x_ctr.

The center is strictly inequality-feasible and equality-feasible. B is nonsingular almost surely, making the set bounded. Record B's condition number; boundedness does not imply good conditioning. Do not silently replace B with an orthogonal matrix or reject difficult instances.

Use independent, documented JAX PRNG subkeys for matrices, slacks, queries, and perturbations. Save the generated arrays so reruns do not depend on cross-version RNG behavior.

Generate 25 queries at each sigma in {0.5,1,2,4}:

```text
z = x_ctr + sigma * normal(shape=(n,))
```

### DC3/TOS envelope-data variant

For the current `problem.py` and `tos_std.py` workflow, keep one sampled DC3 problem (`Q`, `p`, `A`, `G`, `h`) fixed and vary the equality right-hand side `b` between TOS runs. Following the [DC3 QP construction](https://arxiv.org/abs/2104.12225), draw each component independently as `b_j ~ Uniform[-1,1]`. DC3 sets `h_i = sum_j |(G A^+)_{ij}|`, so `A^+ b` is feasible for every sampled `b` in this range when `A` has full row rank. Record the problem seed and the sampled `b` arrays.

Generate 10,000 independent `b` samples and assign 8,000/1,000/1,000 complete TOS runs to training/validation/test using a saved random split. This ratio is this experiment's configuration, not a claim about the split used in DC3. Keep every query from a run in its assigned split. Read sample count and ratios from `configs.json`.

For each run, record the internal `z` passed to `projection_inequality(z)` from iteration 2 onward. Iteration 1 always uses `z=0` in the current TOS code; omit that duplicate query so the same label does not appear in every split. These are the deployment-relevant queries for learning the envelope of the **fixed inequality set** `C={x: Gx<=h}`. Varying `b` changes the TOS trajectory but does not change `C` or its envelope. Do not substitute the DC3 optimization objective or the projection onto `{x: Gx<=h, Ax=b}` for this label. The Gaussian `z` queries above remain the separate large-projection diagnostic; they are not samples of `b`.

For each recorded `z`, use the validated inequality projection `x_star` to store `M_tau(z)=||z-x_star||^2/(2*tau)` and `grad M_tau(z)=(z-x_star)/tau`, with `tau>0`. If the CVXPY projection objective is `0.5*sum_squares(x-z)`, its optimal value divided by `tau` is `M_tau(z)`. Save `z`, `b`, `x_star`, both labels, run/iteration IDs, split, solver status, and reference residuals. Keep `tau` distinct from the TOS step size `gamma`, even if both initially equal 1.

## 2. Trusted references

Use CPU OSQP through its Python API, with NumPy/SciPy only for reference solving and serialization. Supply P=I, q=-z, C=stack([G,A]), lower=concatenate([-inf_m,b]), upper=concatenate([h,b]). The omitted constant 0.5*||z||^2 does not change the minimizer; restore it when reporting the projection objective.

Start with eps_abs=eps_rel=1e-8, polishing enabled, max_iter=100000. Record exact installed package versions and solver settings. Validate every result independently:

```text
ineq_res = max(max(G @ x_star - h, 0))
eq_res   = max(abs(A @ x_star - b))
stat_res = max(abs(x_star - z + G.T @ mu + A.T @ nu))
dual_res = max(max(-mu, 0))
comp_res = max(abs(mu * (G @ x_star - h)))
```

For OSQP's two-sided formulation, use the first m dual entries as upper-bound multipliers mu; equality multipliers nu are unrestricted. Require each absolute residual <=1e-6 for the initial benchmark. Save residuals and status; exclude failed references from accuracy summaries and count them explicitly. If precision is insufficient, refine or independently cross-check, documenting the action. Reference numerical uncertainty must be smaller than effects being compared.

Reuse the solver workspace/factorization across q updates. Time setup separately and label cold versus previous-query warm starts; do not merge their timings.

## 3. Controlled prediction, before learning

For each validated reference define r_star=||x_star-z|| and v_star=(x_star-z)/r_star. Draw omega and normalize its component orthogonal to v_star to obtain w. Initialize:

```text
r_hat = (1 + epsilon_r) * r_star
u0 = z + r_hat * (cos(theta) * v_star + sin(theta) * w)
```

Angles: 5,15,30 degrees, converted to radians. First use epsilon_r=0, then {-0.05,-0.01,0.01,0.05}. Reuse w across angles, radius errors, and competing methods for each query. Save perturbation seeds/directions.

These are oracle-based synthetic initializations, not neural outputs. No learned inference or training speed claim is supported by this phase. If a query is feasible, return z directly. Handle a numerically zero reference distance without division; record the bypass. Resample a degenerate orthogonal random vector deterministically.

## 4. Residual and structured direction

Use rho=1 and lambda=1e-3 by default; sensitivity lambdas={1e-4,1e-2} after the main run. G rows have unit norm and A rows are orthonormal, defining the initial constraint scaling.

```text
r(u) = concatenate([maximum(G @ u - h,0), sqrt(rho)*(A @ u-b)])
V(u) = 0.5 * dot(r(u),r(u))
D_i(u) = 1 if G_i @ u > h_i else 0
B_r(u) = stack([D[:,None]*G, sqrt(rho)*A])
grad_V = G.T @ maximum(G @ u-h,0) + rho*A.T @ (A @ u-b)
```

At exactly zero inequality residual choose derivative zero. V is continuously differentiable despite the positive-part residual's kink. Use analytic QP Jacobians, not automatic differentiation of every row.

Compute the ambient damped Gauss–Newton direction using a linear solve:

```text
d = -(B_r.T @ B_r + lambda*I)^(-1) @ (B_r.T @ r)
```

Never form an inverse. Implement two explicitly named backends: `matrix_free_cg` (default for the large experiment) and `dense_cholesky` (validation and timing baseline). Both solve the same ambient damped Gauss–Newton system; neither changes the sphere update.

### 4.1 Matrix-free Jacobian products

At each correction freeze the violated-row mask D at the current u. Provide products without assembling B_r, a diagonal D, or a Gram matrix:

```text
Jv(v) = concatenate([D * (G @ v), sqrt(rho) * (A @ v)])
JT(w) = G.T @ (D * w[:m]) + sqrt(rho) * A.T @ w[m:]
H(v)  = JT(Jv(v)) + lambda * v
      = G.T @ (D * (G @ v)) + rho*A.T @ (A @ v) + lambda*v
rhs   = -grad_V
```

Here D_i is binary, so D_i squared equals D_i. Solve H(d)=rhs using preconditioned conjugate gradients (PCG). Lambda>0 makes H symmetric positive definite, including rank-deficient constraint Jacobians. Do not form A.T @ A in this backend. Recompute the mask, right-hand side, and preconditioner at every correction; freeze them during its PCG solve.

For this QP, matrix-free means no materialized residual Jacobian, Gram matrix, or factorization. The dense input matrices G and A still consume O(m*n+p*n) memory; matvecs still cost O(m*n+p*n). Do not claim linear storage in n or constant-cost products.

Optional generator-specific storage optimization: keep the original B rather than duplicate G=[B;-B] in the JAX correction backend. Compute Gv=[Bv;-Bv] and G.T*w=B.T@(w[:n]-w[n:]). Evaluate inequalities through B@(u-x_ctr) and the two slack vectors. Verify equivalence with explicit G and count any setup cost. OSQP may still require explicit sparse-format constraint storage.

### 4.2 PCG specification

Use a diagonal (Jacobi) preconditioner, computed without a Gram matrix:

```text
diag_H = (G*G).T @ D + rho*sum(A*A,axis=0) + lambda
M_inv(v) = v / diag_H
```

Cache columnwise squared-sum information for A; optionally cache elementwise squares of G only if the extra matrix storage is explicitly measured. A rowwise reduction can avoid that persistent extra allocation. Lambda bounds the preconditioner diagonal away from zero.

Initialize d=0 for each solve, to avoid adding an unrequested warm-start mechanism. Defaults: cg_rtol=1e-6, cg_atol=1e-10, cg_maxiter=200. Stop when the freshly checked Euclidean linear residual satisfies:

```text
||H(d)-rhs||_2 <= max(cg_atol, cg_rtol*||rhs||_2)
```

A zero right-hand side returns d=0 with zero iterations. Implement a small instrumented PCG with `lax.while_loop` if needed to expose iteration counts, residuals, and termination status. Guard zero/nonfinite recurrence denominators and report numerical breakdown. Verify the true residual before marking a solve converged; recursive residual drift must not produce a false success.

For a finite, unconverged direction at the iteration cap, still evaluate the ordinary post-normalization acceptance rule, but mark `cg_converged=false`. For numerical breakdown/nonfinite direction retain u and log a linear-solve failure. Never silently switch to dense Cholesky on a failed CG solve or omit failed solves from timing. Limited CG gives an inexact Gauss–Newton direction; check its actual curve derivative rather than assuming descent.

After the main benchmark, sweep cg_rtol in {1e-4,1e-6,1e-8} to measure solve accuracy, iteration count, projection quality, and total cost. CG convergence is not guaranteed within the fixed cap; damping and constraint conditioning matter. No quadratic convergence or improved runtime is assumed.

### 4.3 Dense comparison and validation

For `dense_cholesky`, assemble H=G.T@(D[:,None]*G)+rho*A.T@A+lambda*I and use a Cholesky solve in JAX. The residual-space equivalent is -B_r.T@solve(B_r@B_r.T+lambda*I,r). Precompute the equality Gram matrix only for the dense backend.

Check H(v) against the explicit matrix product and PCG directions against Cholesky on small cases. The latter comparison must account for conditioning: a small linear residual alone does not imply a similarly small relative direction error. With this SPD system, ||d_cg-d_exact|| <= ||H(d_cg)-rhs||/lambda, apart from reference numerical error.

Do not boolean-index variable numbers of active rows inside jit. Optional residual-space optimization must use explicit shape buckets/padding or a documented host-side path. Benchmark that variant separately and include packing/transfer costs. Compare both main backends at the same initialization, precision, damping, and correction budget; report any accuracy differences from inexact CG.

## 5. Sphere normalization and acceptance

Keep z and r_hat fixed throughout all J corrections. For r_hat>0:

```text
N(v) = z + r_hat * (v-z)/||v-z||
s = (u-z)/r_hat
P_d = d - s*dot(s,d)
a = dot(grad_V, P_d)
u_trial(alpha) = N(u + alpha*d)
```

The direction is ambient Gauss–Newton followed by normalization. Do not replace it with tangent-constrained Gauss–Newton; that is a different method. Ambient descent does not guarantee a<0 after normalization.

Candidates: alpha={1,1/2,1/4,1/8,1/16}, largest first. Evaluate all candidates using vmap, including all constraints at each final normalized point. If a<0, accept the largest candidate satisfying:

```text
V(u_trial) <= V(u) + 1e-4 * alpha * a
```

If a>=0, accept only a strict decrease in V at the normalized point. If none qualifies retain u, label a stall, and continue to the fixed budget. Record numerical near-stalls separately; do not add an acceptance slack that allows increases.

Reject nonfinite candidates or undefined normalization. Use safe denominators before vectorized selection; a where expression alone must not expose division by zero. Record any infeasible zero-radius prediction as a failure; return the current point without angular correction. Use a dtype-aware denominator guard, document its value, and do not change r_hat to rescue a failed case.

Save states for J=0,1,2,3 from the same three-step trajectory. J=0 means normalized initialization.

## 6. Baselines and fair comparisons

Run identical queries, u0, rho, lambda, candidate sets, precision, and correction budgets:

- Ambient Gauss–Newton without normalization. Its Armijo derivative is dot(grad_V,d), evaluated on u+alpha*d.
- Ambient Gauss–Newton with sphere normalization: the main method.
- Sphere-normalized gradient correction d=-grad_V: tests whether the structured solve earns its cost.
- Validated OSQP reference: exact numerical projection baseline.

No-retraction corrections target feasibility rather than the projection objective. Measure projection error as well as feasibility for every method. Controlled initializations already encode reference information and cannot be used to claim a deployable replacement for OSQP.

An underestimated radius cannot reach the convex feasible set. An oversized sphere need not intersect a bounded set. Even with the exact radius, tangency and spherical stationary points can slow or stall correction. Preserve and report such cases.

## 7. JAX and resource requirements

Use Python, JAX/jax.numpy, and jax.scipy.linalg for correction kernels. Enable jax_enable_x64 before generating arrays. NumPy/SciPy/OSQP are permitted outside kernels. Pin the actually tested dependencies; do not guess current versions.

Use jit for pure fixed-shape kernels, vmap for trial candidates, and lax.scan for the three sequential corrections. Parallel candidate evaluation does not parallelize correction iterations. Initially process queries sequentially or in small batches: do not allocate 100 dense Cholesky systems concurrently. Matrix-free batches also require memory checks for inputs, candidate points, PCG vectors, and temporary matvec arrays. Make batch size an explicit experiment option and report memory failures.

Verify device float64 support and print backend/device/dtype. Synchronize outputs with block_until_ready before stopping timers. Separate compilation, transfer, reference setup, correction preprocessing, and steady-state execution. Also report a total online latency including required preprocessing. Count all candidate evaluations and linear algebra; do not time only dispatched asynchronous work.

Warm up each relevant compiled shape. Use at least five timed repeats for representative cases, reporting median and spread. Persist accuracy from a deterministic run; repeated timings do not create independent accuracy samples. No end-to-end learned timing is available until an actual network is implemented.

## 8. Minimal repository deliverables

Use the existing repository structure if present. Otherwise a small structure is sufficient:

```text
qp_problem.py       generation and OSQP reference validation
correction.py       JAX residual, direction, normalization, search
run_experiment.py   CLI and staged experiment driver
summarize.py        aggregate tables and plots
requirements.txt    tested dependency versions
tests/              mathematical and numerical checks
outputs/            generated arrays, JSON metadata, CSV results, plots
```

Avoid a framework or plugin system. Provide CLI options for dimensions, seed, query count, batch size, output directory, linear-solver backend, and CG tolerance/iteration cap. Document runnable commands after implementation, for example:

```bash
python -m pytest -q
python run_experiment.py --n 20 --m 40 --p 2 --queries-per-scale 2 --output outputs/smoke
python run_experiment.py --n 1000 --m 2000 --p 100 --queries-per-scale 25 --linear-solver matrix_free_cg --cg-rtol 1e-6 --cg-maxiter 200 --output outputs/main
python summarize.py --input outputs/main
```

Require m=2*n for this paired-row generator; reject incompatible input rather than silently changing dimensions. Require 0<=p<n. Expose sensitivity/scale stages explicitly so the main command does not unexpectedly run all expensive experiments.

## 9. Validation sequence and success criteria

1. Small analytic and numerical tests: validate generation, residuals, solver signs, and correction mathematics.
2. Smoke run at n=20: all reference checks pass; accepted V values never increase and normalized states preserve radius within floating-point tolerance.
3. Default n=1000 experiment: report full accuracy, failure, and timing results before sensitivity runs.
4. Radius and damping sensitivity: determine the accuracy requirements and failure regimes.
5. Optional n=3000 experiment: only after checking memory and documented results.

Meaningful tests include an analytic half-space projection, an equality-only projection with orthonormal A, primal/residual-space direction equivalence, matrix-free Jacobian adjoint identity, matrix-free/dense H products, PCG/dense direction agreement, zero-RHS and iteration-cap/breakdown diagnostics, finite-difference verification of the normalized curve derivative, and deterministic feasible/zero-radius/undefined-candidate handling. Use tolerances reflecting float64 and solver accuracy. Include a non-descent normalized direction case: verify the fallback/stall rule rather than assuming every step improves.

Per-query CSV fields: method, seed, query_id, sigma, theta, epsilon_r, lambda, J, absolute and relative projection errors, inequality/equality infinity residuals, V, radius drift, accepted alpha, a, violated-row count, stalls, invalid candidates, linear_solver, cg_iterations, cg_converged, cg_true_residual, cg_relative_residual, cg_breakdown, and timings. Define cg_relative_residual using max(||rhs||, cg_atol) as denominator. Relative error denominator=max(1,r_star). Store reference validation metadata and trajectories separately. Summaries need medians, upper quantiles, maxima, improvement fractions, and failure counts; do not rely on means alone.

Plots: projection error versus J, both feasibility residuals versus J, radius-error sensitivity, stall frequency, and error-versus-latency. A 2D diagnostic trajectory is optional for interpretation, not performance evidence.

Predeclare the empirical gate: sphere correction is promising if it lowers projection error versus matched no-retraction corrections for a substantial reported fraction of queries, while maintaining useful feasibility and competitive total latency. Report actual effect sizes; do not invent a guaranteed threshold or suppress negative results. If it fails with exact radii and small angular errors, investigate the correction before network training.

## 10. Deferred learning and outer solver

Only after this gate, learn the scalar envelope M_gamma(z)=dist(z,X)^2/(2*gamma), derive bar_x=z-gamma*grad_z(M_hat), and r_hat=sqrt(2*gamma*M_hat). Normalize bar_x-z to r_hat before correction. A negative predicted envelope or zero direction requires an explicit reported convention, not silent clipping disguised as mathematical validity.

Train on both value and input-gradient targets; convexity alone does not guarantee a valid envelope. Hold out queries for a fixed set and clearly distinguish this from generalization across constraint sets. Count network value/input-gradient evaluation in online latency. Outer TOS, adaptive steps, and nonconvex stationarity analysis are later tasks requiring their own specification and validation.
