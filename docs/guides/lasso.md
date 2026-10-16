# qp_l1: problem setup and ICNN learning

Instructions for Claude Code: reproduce the mathematical problem and learning structure below. Sources: `problems/qp_l1/problem.jl`, `python/configs/qp_l1.json`, `python/micnn.py`, and `python/train.py`.

## Problem setup

Solve the strongly convex quadratic-plus-L1 problem in the original decision-variable coordinates:

```text
minimize_y  0.5 y^T Q y + p^T y + lambda ||y||_1
subject to  A y = x
            G y <= h.
```

| Quantity | Dimension | Definition |
| --- | --- | --- |
| Decision variable `y` | 100 | Real-valued; no additional box bounds |
| Quadratic matrix `Q` | 100 x 100 | Dense SPD matrix `B B^T / 400 + 0.1 I_100` |
| Linear coefficient `p` | 100 | Independent Uniform `[0,1)` entries |
| Equality matrix `A` | 50 x 100 | Independent standard normal entries |
| Instance parameter `x` | 50 | Independent Uniform `[-1,1]` entries |
| Inequality matrix `G` | 50 x 100 | Independent standard normal entries |
| Inequality bound `h` | 50 | `h_i = sum_j abs((G pinv(A))_ij)` |
| L1 coefficient `lambda` | Scalar | `1.0` |
| Envelope penalty `rho` | Scalar | `10.0` |

The matrices `Q`, `p`, `A`, `G`, and `h` are fixed across instances; only `x` changes. Variable scaling is `1.0`. `B` has independent standard normal entries and uses a separate NumPy seed of 18. The `0.1 I` term ensures `Q` is positive definite. Source comments describe its eigenvalues as approximately in `[0.1,1.1]`; the actual eigenvalues are determined by the seeded draw.

### Exact matrix generation

Preserve NumPy's legacy random generator and the original draw order. The discarded draws below are necessary to obtain the same `p`, `A`, and `G` as the implementation.

```python
import numpy as np

np.random.seed(17)
np.random.random(100)                       # intentionally discarded
p = np.random.random(100)
A = np.random.normal(size=(50, 100))
X = np.random.uniform(-1, 1, size=(10000, 50))  # advances RNG before G
G = np.random.normal(size=(50, 100))
h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)

B = np.random.RandomState(18).normal(size=(100, 100))
Q = B @ B.T / 400 + 0.1 * np.eye(100)
Q = (Q + Q.T) / 2
```

`X` is an intermediate draw in the matrix-generation routine; it is not returned by that routine. New instances in the Julia problem constructor draw `x` from `Uniform(-1,1)` using Julia's current RNG; that constructor does not set its own seed.

The construction of `h` gives a feasible point `y = pinv(A) x` for every `x` in the specified box, provided `A` has full row rank, as this seeded Gaussian matrix does: `Ay=x`, and `Gy <= |G pinv(A)| 1 = h`.

## ICNN learning structure
Implementation Guide: Structure-Aware ICNN for the Moreau Envelope of Constrained LASSO
Purpose. Implement a minimal, attention-free, structure-aware input-convex neural network (ICNN) to approximate the full Moreau envelope and its gradient for a family of constrained LASSO problems. The analytic dictionary is evaluated in parallel with the ICNN; its coefficients are manually specified and frozen. The network predicts a scalar envelope value, and automatic differentiation of that scalar with respect to the query produces the predicted envelope gradient and proximal point.
Implementation language: Python + JAX (Flax optional); prefer float64 (`jax_enable_x64=True` or `JAX_ENABLE_X64=1`). Matrix-free computations where sensible; avoid unnecessary explicit Hessians. No attention, Transformers, learned dictionary coefficients, learned projections, or sequential training in this version.
1. Mathematical problem
For each instance indexed by (\xi), solve the convex constrained LASSO:
[
\begin{aligned}
\min_{x\in\mathbb R^n}\quad & f_\xi(x)=\tfrac12|A x-b|_2^2+\lambda|x|_1\
\mathrm{s.t.}\quad & Gx=h,\
& Cx\le d.
\end{aligned}
]
Dimensions:
(A\in\mathbb R^{m\times n}), (b\in\mathbb R^m);
(\lambda>0);
(G\in\mathbb R^{r\times n}), (h\in\mathbb R^r);
(C\in\mathbb R^{s\times n}), (d\in\mathbb R^s).
Assume the feasible set is nonempty:
[
\mathcal X_\xi={x:Gx=h,\ Cx\le d}.
]
Let (\Psi_\xi=f_\xi+I_{\mathcal X_\xi}). For fixed (\gamma>0), define the full Moreau envelope
[
\boxed{M_\gamma(z;\xi)=\min_{x\in\mathcal X_\xi}\left[\tfrac12|Ax-b|_2^2+\lambda|x|_1+\tfrac1{2\gamma}|x-z|2^2\right].}
]
The unique proximal point (p^*(z;\xi)) solves that problem and
[
\boxed{\nabla_zM\gamma(z;\xi)=\frac{z-p^*(z;\xi)}{\gamma}.}
]
Do not confuse (x) (inner optimization variable) with (z) (query at which envelope and its gradient are evaluated).
2. Network input and output: explicit contract
Inputs to the neural model
A single training/evaluation example is a problem–query pair ((\xi,z)):
Query `z`: shape `(n,)`, `float64` — the variable with respect to which the model output is differentiated.
Problem data `xi`:
`A`: `(m,n)`; `b`: `(m,)`; `lam`: scalar;
`G`: `(r,n)`; `h`: `(r,)`;
`C`: `(s,n)`; `d`: `(s,)`.
Envelope smoothing parameter `gamma`: positive scalar (keep fixed in version 1; document in configuration).
Manual dictionary weights `weights`: fixed nonnegative scalars / arrays, passed as model configuration (not optimized).
The model must condition on both `z` and `xi`: otherwise one shared model cannot learn envelopes for varying LASSO instances. For a simple first implementation with fixed dimensions, flatten and concatenate the numeric instance data into an instance feature vector `q_xi`; normalize features using training-set statistics only. Train and validation/test instances must be distinct. `gamma` may be appended as a static condition if supported, but begin with a fixed `gamma`.
Convexity requirement: The model's scalar output must be convex in the raw query `z` for each fixed `xi`. Do not feed nonlinear `z`-dependent dictionary outputs as unconstrained conditioning features into the ICNN. Instance features may parameterize query-affine terms or other ICNN mechanisms that preserve convexity in `z`; see section 4. Input standardization of `z` may use a fixed affine transform (convexity preserved); when computing `grad_z`, differentiate through this transform and use the gradient in original query coordinates.
Outputs from the neural model
Primary learned output:
`M_hat(z, xi)`: one scalar, shape `()` (approximate full Moreau-envelope value).
Derived outputs (not independent heads):
`grad_M_hat = jax.grad(M_hat, argnums_for_z)(z,xi)`: shape `(n,)`, derivative with respect to query `z` only;
`p_hat = z - gamma * grad_M_hat`: shape `(n,)`, approximate proximal point.
Important: Do not output the decision vector directly and do not add an independently trained gradient head. The approximate gradient must be the derivative of the same scalar output.
3. Fixed analytic dictionary (calculated in parallel)
Split the feasible set as
[
\mathcal X_\xi=\mathcal C_{\rm eq}\cap\bigcap_{j=1}^s\mathcal C_j,
\quad\mathcal C_{\rm eq}={x:Gx=h},\quad\mathcal C_j={x:c_j^\top x\le d_j},
]
where `c_j = C[j, :]`. Define the following fixed-form convex, (C^{1,1}) atoms. They represent the Moreau envelopes of isolated components, not the full composite envelope.
3.1 L1 regularizer: coordinatewise Huber atoms
Define
[
H_\delta(t)=\begin{cases}t^2/(2\delta),&|t|\le\delta,\ |t|-\delta/2,&|t|>\delta.\end{cases}
]
For each coordinate (i=1,\dots,n),
[
\boxed{\phi_i^{\ell_1}(z;\xi)=\lambda H_{\gamma\lambda}(z_i).}
]
This is exactly (M_{\gamma\lambda|\cdot|}(z_i)). For stability, evaluate Huber in a JAX-compatible, differentiable piecewise form. The derivative w.r.t. `z_i` is continuous at the transitions (z_i=\pm\gamma\lambda).
3.2 Affine equality-set atom
When `r > 0` and (\mathcal C_{\rm eq}) is nonempty,
[
\boxed{\phi^{\rm eq}(z;\xi)=\frac{1}{2\gamma}(Gz-h)^\top(GG^\top)^\dagger(Gz-h).}
]
Use a stable least-squares / QR / SVD-based operation, not an explicit dense matrix inverse. If `G` has full row rank, a well-conditioned linear solve is appropriate. If `r=0`, omit this atom. This atom has constant curvature (no kink), but encodes equality geometry.
3.3 Inequality half-space atoms
For every row with (|c_j|2>0),
[
\boxed{\phi_j^{\rm ineq}(z;\xi)=\frac{[c_j^\top z-d_j]+^2}{2\gamma|c_j|2^2},\quad[t]+=\max(t,0).}
]
These are the exact squared-distance envelopes of the individual half-space indicators. Their first derivatives are continuous; second derivatives may jump across the boundary. Reject or preprocess zero-norm rows (including feasibility checks); do not silently divide by zero.
3.4 Manual dictionary combination
All atom forms and geometry parameters remain fixed by the current problem instance. For version 1, no learned atom shifts, directions, scales, or attention scores.
[
\boxed{D(z;\xi)=\sum_{i=1}^{n} a^{\ell_1}i\phi_i^{\ell_1}(z;\xi)+a{\rm eq}\phi^{\rm eq}(z;\xi)+\sum_{j=1}^{s} a^{\rm ineq}_j\phi_j^{\rm ineq}(z;\xi).}
]
All coefficients are nonnegative, manually assigned, and frozen. Supply config values for `a_l1` (scalar broadcast or length-n vector), `a_eq` (scalar), and `a_ineq` (scalar broadcast or length-s vector). For the initial test, make each atom-family multiplier an easily swept hyperparameter, including the all-zero baseline. Do not assume setting all weights to 1 is mathematically correct.
For every query, compute the Huber, equality, and inequality atoms independently / in parallel, sum them, and add the sum to the ICNN output. There is no iterative projection onto the intersection in the dictionary branch.
4. Full structure-aware ICNN
[
\boxed{\widehat M_\theta(z;\xi)=F_\theta(z;\xi)+D(z;\xi).}
]
`F_theta` is a conditional ICNN convex in query `z`. An acceptable minimal implementation is a standard ICNN whose layer weights from previous hidden convex states are constrained nonnegative and whose activations are convex and nondecreasing (e.g., softplus); dependence on `xi` should enter through query-affine terms or additive, problem-conditioned biases. One possible layer pattern:
[
u_1=\operatorname{softplus}(U_1 z+V_1 q_\xi+b_1),
]
[
u_{k+1}=\operatorname{softplus}(W_{k,+}u_k+U_{k+1}z+V_{k+1}q_\xi+b_{k+1}),\quad W_{k,+}=\operatorname{softplus}(\widetilde W_k),
]
[
F_\theta(z;\xi)=w_+^\top u_L+a^\top z+c(q_\xi),\quad w_+=\operatorname{softplus}(\widetilde w).
]
Here `U_k`, `V_k`, `a` may have signed entries; the hidden-state and final hidden-output weights are constrained nonnegative. `q_xi` must be independent of `z`. For the simplest version, `q_xi` can be the flattened, normalized instance data; `c(q_xi)` is optional and may be a small affine mapping. Use smooth activations; in this model the dictionary atoms provide exact piecewise-curvature changes.
Critical limitation: Convexity plus (C^{1,1}) does not automatically enforce (\operatorname{Lip}(\nabla_z\widehat M)\le 1/\gamma). Do not state that the prediction is guaranteed to be an exact Moreau envelope. The nonnegative dictionary adds curvature, which the convex ICNN branch cannot cancel. Tune coefficients cautiously; optionally audit local Hessian eigenvalues / gradient Lipschitz behavior on samples. There is no requirement for `p_hat` to be feasible; downstream correction is separate and outside this implementation.
5. Exact training labels (full proximal problem)
For each instance (\xi_i), sample multiple queries (z_{i,1},\ldots,z_{i,N_q}). At each pair, solve the entire constrained proximal LASSO problem (section 1), using a high-accuracy reference convex solver (e.g., CVXPY with a reliable solver) and check its termination/constraint feasibility.
[
p^{ij}=\arg\min{Gx=h,\ Cx\le d}\left[\tfrac12|Ax-b|_2^2+\lambda|x|1+\tfrac1{2\gamma}|x-z{ij}|_2^2\right].
]
Build both labels:
[
\boxed{E^{ij}=\tfrac12|Ap^*{ij}-b|2^2+\lambda|p^*{ij}|1+\tfrac1{2\gamma}|p^*{ij}-z_{ij}|2^2,}
]
[
\boxed{g^*{ij}=(z_{ij}-p^*_{ij})/\gamma.}
]
Keep `p_star` too for evaluation. Reference label generation is offline: do not solve a proximal optimization problem inside the ICNN forward pass.
Suggested query sampling: a mixture of broadly distributed `z`, queries near optimal solutions, and queries close to likely active-set transitions; for the first implementation broad and near-solution sampling are enough. Validate on unseen problem instances, splitting at the instance level before sampling/assigning queries.
6. Training loss and optimization
The scalar network output and its input derivative are jointly supervised:
[
\boxed{\mathcal L(\theta)=\lambda_v\mathbb E_{(\xi,z)}|\widehat M_\theta(z;\xi)-E^|^2+\lambda_g\mathbb E_{(\xi,z)}|\nabla_z\widehat M_\theta(z;\xi)-g^|_2^2.}
]
`lambda_v`, `lambda_g`: manually configurable nonnegative loss weights; keep identical for all model comparisons.
Train only ICNN weights `theta`; dictionary weights are configuration, not trainable state.
Use `jax.grad` on the whole scalar `F_theta + D`, and use `jax.value_and_grad`/`jax.vmap`/`jax.jit` as appropriate.
Ensure reference `E_star` and `g_star` are in the same units and scaling as network outputs. If implementing output normalization, apply the chain rule and undo it consistently in inference.
A naive normalized envelope may need a learned instance-dependent additive constant; ensure the architecture can represent envelope offsets, because the gradient loss cannot identify constants.
7. Prediction, metrics, and tests
Inference API:
```python
M_hat = model.apply(params, z, xi, config)
g_hat = jax.grad(lambda zz: model.apply(params, zz, xi, config))(z)
p_hat = z - gamma * g_hat
```
Return at least `M_hat` (scalar), `g_hat` `(n,)`, and `p_hat` `(n,)`.
Report on unseen instances:
Envelope value absolute/relative error.
Gradient error `||g_hat - g_star||_2` and `||...||_inf`.
Proximal prediction error `||p_hat-p_star||_2` and `||...||_inf`.
Gradient error near known/empirically estimated transition regions versus elsewhere.
Runtime of one batched forward-and-gradient evaluation (with warm-up and synchronized JAX timing).
Optional constraint violations of `p_hat` as diagnostics only (feasibility is not the network's guarantee).
Mandatory correctness checks:
Compare analytic dictionary atom values and `jax.grad` against closed-form projections / finite differences at points away from transition boundaries.
Check gradients are taken with respect to `z`, not `x` or `xi`.
Verify gradients of manually fixed dictionary contributions pass through to `g_hat`.
Check shape consistency and support zero equality / inequality constraints.
Check finite output and gradients for test examples (including exactly on Huber and half-space transitions).
Numerically test Jensen convexity in `z` at fixed `xi` (diagnostic; not a formal proof).
Check the identity `||p_hat-p_star|| = gamma * ||g_hat-g_star||` numerically.
8. Minimal ablations (no attention)
Keep identical ICNN, train/validation/test instance splits, query sampling, labels, optimizer, and training budget across:
Experiment	Manually fixed dictionary weights	Purpose
`baseline`	All zero	Standard conditional ICNN
`l1_only`	Only Huber family nonzero	Isolate the effect of nonsmooth-objective atoms
`constraints_only`	Only equality/inequality atoms nonzero	Isolate constraint geometry
`combined`	L1 + constraint atoms nonzero	Test the proposed full dictionary
Use a small validation sweep of manually specified weights. No attention and no learnable coefficient generator.
9. Suggested project organization and deliverables
```text
src/
  problem.py          # Constrained LASSO definition + instance validation
  dictionary.py       # Closed-form atom values and derivatives
  icnn.py             # Conditional ICNN, convex in z
  model.py            # M_hat = ICNN + fixed weighted atoms
  reference.py        # Offline accurate proximal labels via CVXPY
  dataset.py          # Instances, query sampling, train/val/test splits
  train.py            # Joint scalar-value + input-gradient loss
  evaluate.py         # Error metrics, timing, ablations
  config.py           # Dimensions, gamma, fixed weights, seeds, loss weights
 tests/
  test_dictionary.py
  test_gradients.py
  test_shapes_convexity.py
README.md
```
Produce a runnable minimal experiment with deterministic seeds, a small synthetic feasible instance generator, training/evaluation commands, and a comparison report across the four ablations.
Do not implement in version 1
Attention / Transformers / query-dependent weights.
Learned atom locations, learned directions, or learned dictionary coefficients.
Sequential residual fitting, separate atom networks, or per-atom supervised training.
End-to-end differentiation through CVXPY/reference solver.
Feasibility-correction step (to be integrated later).
Assertions that the sum of component envelopes equals the envelope of the full problem.
Success criterion: Relative to the standard ICNN, the fixed analytic dictionary improves full-envelope gradient and proximal-point prediction accuracy on held-out constrained LASSO instances, especially near changes of curvature, without sacrificing numerical stability.