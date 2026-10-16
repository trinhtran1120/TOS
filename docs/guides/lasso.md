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

Loading older messages…
        For each problem instance, where should we evaluate the envelope? Its gradient is with respect to the query, so the decision variable alone does not determine the labels.

Exactly. This is an important distinction for training our attention-guided ICNN.

Each optimization problem instance defines a Moreau-envelope function, but we need multiple query points to learn that function and its gradient.

1. Distinguish problem instances from envelope queries
Consider a family of convex optimization problems:

Ψ
ξ
(
x
)
=
f
ξ
(
x
)
+
I
X
ξ
(
x
)
,
Ψ 
ξ
​
 (x)=f 
ξ
​
 (x)+I 
X 
ξ
​
 
​
 (x),
where 
ξ
ξ describes the objective and constraints.

For a fixed problem instance 
ξ
ξ, the Moreau envelope is a function of 
z
z:

M
γ
(
z
;
ξ
)
=
min
⁡
x
∈
X
ξ
{
f
ξ
(
x
)
+
1
2
γ
∥
x
−
z
∥
2
}
.
M 
γ
​
 (z;ξ)= 
x∈X 
ξ
​
 
min
​
 {f 
ξ
​
 (x)+ 
2γ
1
​
 ∥x−z∥ 
2
 }.
​
 
There are three distinct quantities:

x
x: the decision variable optimized inside the proximal problem.

ξ
ξ: the problem parameters that define 
f
ξ
f 
ξ
​
  and 
X
ξ
X 
ξ
​
 .

z
z: the query point at which we evaluate the Moreau envelope.

For each pair 
(
ξ
,
z
)
(ξ,z), we obtain a different training label.

2. How to generate training data
For each problem instance 
ξ
i
ξ 
i
​
 , sample multiple queries

z
i
,
1
,
z
i
,
2
,
…
,
z
i
,
N
q
.
z 
i,1
​
 ,z 
i,2
​
 ,…,z 
i,N 
q
​
 
​
 .
At each query, solve

p
i
,
j
⋆
=
arg
⁡
min
⁡
x
∈
X
ξ
i
{
f
ξ
i
(
x
)
+
1
2
γ
∥
x
−
z
i
,
j
∥
2
}
.
p 
i,j
⋆
​
 =arg 
x∈X 
ξ 
i
​
 
​
 
min
​
 {f 
ξ 
i
​
 
​
 (x)+ 
2γ
1
​
 ∥x−z 
i,j
​
 ∥ 
2
 }.
​
 
Then calculate the exact labels:

E
i
,
j
=
f
ξ
i
(
p
i
,
j
⋆
)
+
1
2
γ
∥
p
i
,
j
⋆
−
z
i
,
j
∥
2
,
G
i
,
j
=
z
i
,
j
−
p
i
,
j
⋆
γ
.
E 
i,j
​
 
G 
i,j
​
 
​
  
=f 
ξ 
i
​
 
​
 (p 
i,j
⋆
​
 )+ 
2γ
1
​
 ∥p 
i,j
⋆
​
 −z 
i,j
​
 ∥ 
2
 ,
= 
γ
z 
i,j
​
 −p 
i,j
⋆
​
 
​
 .
​
 
​
 
Our training dataset is therefore

D
t
r
a
i
n
=
{
(
ξ
i
,
z
i
,
j
,
E
i
,
j
,
G
i
,
j
)
}
i
,
j
.
D 
train
​
 ={(ξ 
i
​
 ,z 
i,j
​
 ,E 
i,j
​
 ,G 
i,j
​
 )} 
i,j
​
 .
​
 
The gradient label 
G
i
,
j
G 
i,j
​
  is taken with respect to the query 
z
i
,
j
z 
i,j
​
 , not the decision variable 
x
x.

3. Where should we sample the queries?
I would use three groups of query points:

Query distribution

Purpose

Broad samples from a bounded domain

Learn the global envelope shape

Samples near optimal solutions 
x
ξ
⋆
x 
ξ
⋆
​
 

Learn the region relevant to optimization

Samples near curvature transitions

Improve accuracy where the envelope gradient changes slope

The third group is particularly relevant to our Huber-like atom dictionary.

However, we should not assume that transitions of the full envelope coincide with the boundaries of individual constraints.

A useful practical strategy is to begin with broad samples, then use preliminary proximal solutions to identify regions where the proximal mapping changes its local behavior. We can oversample those regions.

For the second group, remember that if 
x
ξ
⋆
x 
ξ
⋆
​
  minimizes 
Ψ
ξ
Ψ 
ξ
​
 , then

prox
⁡
γ
Ψ
ξ
(
x
ξ
⋆
)
=
x
ξ
⋆
.
prox 
γΨ 
ξ
​
 
​
 (x 
ξ
⋆
​
 )=x 
ξ
⋆
​
 .
Therefore, 
x
ξ
⋆
x 
ξ
⋆
​
  is a useful center for local sampling.

But querying only near optimal solutions may be inadequate if the model will be evaluated far from them.

4. How this interacts with our attention architecture
Our architecture is

M
^
θ
(
z
;
ξ
)
=
F
θ
(
z
;
ξ
)
+
∑
k
=
1
K
α
k
(
ξ
)
ϕ
k
(
z
;
ξ
)
.
M
  
θ
​
 (z;ξ)=F 
θ
​
 (z;ξ)+ 
k=1
∑
K
​
 α 
k
​
 (ξ)ϕ 
k
​
 (z;ξ).
For each fixed instance 
ξ
i
ξ 
i
​
 , attention computes

α
(
ξ
i
)
α(ξ 
i
​
 )
from the problem description.

These weights remain the same for all queries 
z
i
,
j
z 
i,j
​
  associated with that instance.

The scalar output and gradient, however, change with 
z
i
,
j
z 
i,j
​
 :

∇
z
M
^
θ
(
z
i
,
j
;
ξ
i
)
.
∇ 
z
​
  
M
  
θ
​
 (z 
i,j
​
 ;ξ 
i
​
 ).
This is exactly what we want: attention selects the envelope-pattern dictionary at the problem level, while the ICNN learns the envelope over the query space.

5. One crucial implementation decision
We should train and test on different problem instances, not merely different queries from the same problems.

For example, suppose we generate 1,000 different constrained LASSO instances, each with 100 query points.

We should split the 1,000 instances into training, validation, and test groups before generating or assigning their queries.

Otherwise, the network may see the same optimization problems during training and testing, leading to an overly optimistic assessment of generalization.

The key takeaway
Our learning task is

(
ξ
,
z
)
⟼
(
M
γ
(
z
;
ξ
)
,
∇
z
M
γ
(
z
;
ξ
)
)
.
(ξ,z)⟼(M 
γ
​
 (z;ξ),∇ 
z
​
 M 
γ
​
 (z;ξ)).
​
 
One problem instance corresponds to one envelope function; multiple query points are needed to learn its shape and gradient.

For our first experiment, I would choose a query distribution based on the expected inputs to the downstream correction method, mixed with broad and transition-focused samples. That will tell us whether the learned envelope is accurate where we actually need it.

rewrite the problem formulation for lasso
Constrained LASSO: Problem Formulation
We consider a family of constrained LASSO problems parameterized by 
ξ
ξ:

min
⁡
x
∈
R
n
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
s
.
t
.
G
x
=
h
,
C
x
≤
d
,
(1)
x∈R 
n
 
min
​
 
s.t.
​
  
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 
Gx=h,
Cx≤d,
​
 
​
 (1)
where 
A
∈
R
m
×
n
A∈R 
m×n
  is the feature matrix, 
b
∈
R
m
b∈R 
m
  is the observation vector, 
λ
>
0
λ>0 is the regularization parameter, and 
G
,
C
,
h
,
d
G,C,h,d define the affine equality and inequality constraints.

The problem instance is specified by

ξ
=
(
A
,
b
,
λ
,
G
,
h
,
C
,
d
)
.
ξ=(A,b,λ,G,h,C,d).
1. Composite objective
Define the nonempty closed convex feasible set

X
ξ
=
{
x
∈
R
n
:
G
x
=
h
,
 
C
x
≤
d
}
.
(2)
X 
ξ
​
 ={x∈R 
n
 :Gx=h, Cx≤d}.
​
 (2)
Using its indicator function, the constrained LASSO problem becomes

min
⁡
x
∈
R
n
Ψ
ξ
(
x
)
,
(3)
x∈R 
n
 
min
​
 Ψ 
ξ
​
 (x),
​
 (3)
with

Ψ
ξ
(
x
)
=
1
2
∥
A
x
−
b
∥
2
2
⏟
Smooth data-fitting term
+
λ
∥
x
∥
1
⏟
Nonsmooth regularizer
+
I
X
ξ
(
x
)
⏟
Constraint indicator
.
Ψ 
ξ
​
 (x)= 
Smooth data-fitting term
2
1
​
 ∥Ax−b∥ 
2
2
​
 
​
 
​
 + 
Nonsmooth regularizer
λ∥x∥ 
1
​
 
​
 
​
 + 
Constraint indicator
I 
X 
ξ
​
 
​
 (x)
​
 
​
 .
2. Moreau envelope
For a query 
z
∈
R
n
z∈R 
n
  and parameter 
γ
>
0
γ>0, define

M
γ
(
z
;
ξ
)
=
min
⁡
x
∈
X
ξ
{
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
+
1
2
γ
∥
x
−
z
∥
2
2
}
.
(4)
M 
γ
​
 (z;ξ)= 
x∈X 
ξ
​
 
min
​
 { 
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 + 
2γ
1
​
 ∥x−z∥ 
2
2
​
 }.
​
 (4)
The corresponding proximal point is

p
γ
(
z
;
ξ
)
=
argmin
⁡
x
∈
X
ξ
{
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
+
1
2
γ
∥
x
−
z
∥
2
2
}
.
(5)
p 
γ
​
 (z;ξ)= 
x∈X 
ξ
​
 
argmin
​
 { 
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 + 
2γ
1
​
 ∥x−z∥ 
2
2
​
 }.
​
 (5)
Since 
Ψ
ξ
Ψ 
ξ
​
  is proper, closed, and convex, its envelope is convex and continuously differentiable, with

∇
z
M
γ
(
z
;
ξ
)
=
z
−
p
γ
(
z
;
ξ
)
γ
.
(6)
∇ 
z
​
 M 
γ
​
 (z;ξ)= 
γ
z−p 
γ
​
 (z;ξ)
​
 .
​
 (6)
Moreover,

M
γ
(
⋅
;
ξ
)
∈
C
1
,
1
,
Lip
⁡
(
∇
z
M
γ
)
≤
1
γ
.
M 
γ
​
 (⋅;ξ)∈C 
1,1
 ,Lip(∇ 
z
​
 M 
γ
​
 )≤ 
γ
1
​
 .
3. Learning problem
We seek a parameterized approximation

M
^
θ
(
z
;
ξ
)
≈
M
γ
(
z
;
ξ
)
(7)
M
  
θ
​
 (z;ξ)≈M 
γ
​
 (z;ξ)
​
 (7)
using our attention-guided, structure-aware ICNN:

M
^
θ
(
z
;
ξ
)
=
F
θ
(
z
;
ξ
)
+
∑
j
=
1
K
α
j
(
ξ
)
ϕ
j
(
z
;
ξ
)
.
(8)
M
  
θ
​
 (z;ξ)=F 
θ
​
 (z;ξ)+ 
j=1
∑
K
​
 α 
j
​
 (ξ)ϕ 
j
​
 (z;ξ).
​
 (8)
Here, 
F
θ
F 
θ
​
  is convex in 
z
z, 
ϕ
j
ϕ 
j
​
  are analytical 
C
1
,
1
C 
1,1
  convex atoms chosen to represent characteristic curvature transitions, and 
α
j
(
ξ
)
≥
0
α 
j
​
 (ξ)≥0 are problem-dependent attention weights.

For LASSO, Huber atoms are natural candidates because

M
γ
λ
∣
⋅
∣
(
t
)
=
λ
H
γ
λ
(
t
)
.
M 
γλ∣⋅∣
​
 (t)=λH 
γλ
​
 (t).
Other atoms may encode geometric patterns associated with the equality and inequality constraints.

4. Training labels and objective
For each problem instance 
ξ
i
ξ 
i
​
 , we sample multiple queries 
z
i
,
j
z 
i,j
​
  and solve (5) to obtain 
p
i
,
j
⋆
p 
i,j
⋆
​
 .

The envelope-value and gradient labels are

E
i
,
j
=
1
2
∥
A
p
i
,
j
⋆
−
b
∥
2
2
+
λ
∥
p
i
,
j
⋆
∥
1
+
1
2
γ
∥
p
i
,
j
⋆
−
z
i
,
j
∥
2
2
,
G
i
,
j
=
z
i
,
j
−
p
i
,
j
⋆
γ
.
(9)
E 
i,j
​
 
G 
i,j
​
 
​
  
= 
2
1
​
 ∥Ap 
i,j
⋆
​
 −b∥ 
2
2
​
 +λ∥p 
i,j
⋆
​
 ∥ 
1
​
 + 
2γ
1
​
 ∥p 
i,j
⋆
​
 −z 
i,j
​
 ∥ 
2
2
​
 ,
= 
γ
z 
i,j
​
 −p 
i,j
⋆
​
 
​
 .
​
 
​
 (9)
The network is trained by minimizing

L
(
θ
)
=
1
N
∑
i
,
j
[
λ
v
∣
M
^
θ
(
z
i
,
j
;
ξ
i
)
−
E
i
,
j
∣
2
+
λ
g
∥
∇
z
M
^
θ
(
z
i
,
j
;
ξ
i
)
−
G
i
,
j
∥
2
2
]
,
(10)
L(θ)= 
N
1
​
  
i,j
∑
​
 [λ 
v
​
 ∣ 
M
  
θ
​
 (z 
i,j
​
 ;ξ 
i
​
 )−E 
i,j
​
 ∣ 
2
 +λ 
g
​
 ∥∇ 
z
​
  
M
  
θ
​
 (z 
i,j
​
 ;ξ 
i
​
 )−G 
i,j
​
 ∥ 
2
2
​
 ],
​
 (10)
where 
N
N is the total number of training query–instance pairs.

5. Proximal prediction
After training, the approximate proximal point is obtained directly from the learned gradient:

p
^
θ
(
z
;
ξ
)
=
z
−
γ
∇
z
M
^
θ
(
z
;
ξ
)
.
(11)
p
​
  
θ
​
 (z;ξ)=z−γ∇ 
z
​
  
M
  
θ
​
 (z;ξ).
​
 (11)
The resulting prediction can serve as the initialization for our subsequent correction step.

The key distinction: The problem parameters 
ξ
ξ determine which Moreau-envelope patterns are emphasized by attention, while the query 
z
z determines where the learned envelope and its gradient are evaluated.

Not only the huber loss. We need to split X into subsets and add those Moreau envelope-approximated as well
Exactly. For constrained LASSO, our dictionary should capture both the nonsmooth structure of the 
ℓ
1
ℓ 
1
​
  regularizer and the geometric structure of the feasible set.

We can formulate this without introducing additional neural networks for individual components.

1. Constrained LASSO
Consider

min
⁡
x
∈
R
n
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
s
.
t
.
G
x
=
h
,
C
x
≤
d
.
(1)
x∈R 
n
 
min
​
 
s.t.
​
  
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 
Gx=h,
Cx≤d.
​
 
​
 (1)
Define

f
(
x
)
=
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
f(x)= 
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 
and the feasible set

X
=
{
x
:
G
x
=
h
,
 
C
x
≤
d
}
.
X={x:Gx=h, Cx≤d}.
We decompose the feasible set into elementary convex sets:

X
=
⋂
j
=
0
m
c
X
j
,
X= 
j=0
⋂
m 
c
​
 
​
 X 
j
​
 ,
​
 
where

X
0
=
{
x
:
G
x
=
h
}
,
X
j
=
{
x
:
c
j
⊤
x
≤
d
j
}
,
j
=
1
,
…
,
m
c
,
X 
0
​
 
X 
j
​
 
​
  
={x:Gx=h},
={x:c 
j
⊤
​
 x≤d 
j
​
 },j=1,…,m 
c
​
 ,
​
 
and 
c
j
⊤
c 
j
⊤
​
  is the 
j
j-th row of 
C
C.

2. Construct the analytical Moreau-envelope dictionary
We now have three types of functions to represent.

A. Nonsmooth regularizer: Huber atoms
Because

λ
∥
x
∥
1
=
∑
i
=
1
n
λ
∣
x
i
∣
,
λ∥x∥ 
1
​
 = 
i=1
∑
n
​
 λ∣x 
i
​
 ∣,
we have the exact envelope of the isolated regularizer:

ϕ
ℓ
1
(
z
)
=
λ
∑
i
=
1
n
H
γ
λ
(
z
i
)
.
ϕ 
ℓ 
1
​
 
​
 (z)=λ 
i=1
∑
n
​
 H 
γλ
​
 (z 
i
​
 ).
​
 
This encodes the characteristic curvature transitions of the 
ℓ
1
ℓ 
1
​
  term.

B. Equality constraints: Affine-set atom
The Moreau envelope of 
I
X
0
I 
X 
0
​
 
​
  is

ϕ
e
q
(
z
)
=
1
2
γ
(
G
z
−
h
)
⊤
(
G
G
⊤
)
†
(
G
z
−
h
)
.
ϕ 
eq
​
 (z)= 
2γ
1
​
 (Gz−h) 
⊤
 (GG 
⊤
 ) 
†
 (Gz−h).
​
 
This is exact when the affine set is nonempty.

C. Inequality constraints: Half-space atoms
For each inequality set 
X
j
X 
j
​
 ,

ϕ
j
(
z
)
=
[
c
j
⊤
z
−
d
j
]
+
2
2
γ
∥
c
j
∥
2
2
,
c
j
≠
0.
ϕ 
j
​
 (z)= 
2γ∥c 
j
​
 ∥ 
2
2
​
 
[c 
j
⊤
​
 z−d 
j
​
 ] 
+
2
​
 
​
 ,c 
j
​
 

=0.
​
 
This is the exact Moreau envelope of the individual constraint indicator 
I
X
j
I 
X 
j
​
 
​
 .

Its Hessian changes across the hyperplane

c
j
⊤
z
=
d
j
.
c 
j
⊤
​
 z=d 
j
​
 .
Thus, the dictionary includes analytical functions representing curvature patterns associated with both nonsmooth regularization and constraints.

3. Our attention-guided ICNN architecture
We can now write

M
^
θ
(
z
;
ξ
)
=
F
θ
(
z
;
ξ
)
+
α
ℓ
1
(
ξ
)
ϕ
ℓ
1
(
z
;
ξ
)
+
α
e
q
(
ξ
)
ϕ
e
q
(
z
;
ξ
)
+
∑
j
=
1
m
c
α
j
(
ξ
)
ϕ
j
(
z
;
ξ
)
.
(2)
M
  
θ
​
 (z;ξ)=
​
  
F 
θ
​
 (z;ξ)
+α 
ℓ 
1
​
 
​
 (ξ)ϕ 
ℓ 
1
​
 
​
 (z;ξ)
+α 
eq
​
 (ξ)ϕ 
eq
​
 (z;ξ)
+ 
j=1
∑
m 
c
​
 
​
 α 
j
​
 (ξ)ϕ 
j
​
 (z;ξ).
​
 
​
 (2)
Here,

ξ
=
(
A
,
b
,
λ
,
G
,
h
,
C
,
d
,
γ
)
ξ=(A,b,λ,G,h,C,d,γ)
describes the problem instance.

The attention mechanism computes nonnegative coefficients

α
(
ξ
)
=
Attention
⁡
ω
(
ξ
)
,
α
j
(
ξ
)
≥
0.
α(ξ)=Attention 
ω
​
 (ξ),α 
j
​
 (ξ)≥0.
Importantly, the coefficients depend on the problem instance, not the query 
z
z.

Problem instance 
ξ
ξ

Objective, equality constraints, inequality constraints

Problem-dependent attention

α
ℓ
1
,
α
e
q
,
α
1
,
…
,
α
m
c
α 
ℓ 
1
​
 
​
 ,α 
eq
​
 ,α 
1
​
 ,…,α 
m 
c
​
 
​
 

ICNN branch

F
θ
(
z
;
ξ
)
F 
θ
​
 (z;ξ)

Global envelope shape

Analytical dictionary

α
ℓ
1
ϕ
ℓ
1
+
α
e
q
ϕ
e
q
+
∑
j
α
j
ϕ
j
α 
ℓ 
1
​
 
​
 ϕ 
ℓ 
1
​
 
​
 +α 
eq
​
 ϕ 
eq
​
 +∑ 
j
​
 α 
j
​
 ϕ 
j
​
 

Known objective and constraint curvature patterns

Predicted Moreau envelope 
M
^
θ
(
z
;
ξ
)
M
  
θ
​
 (z;ξ)

4. Training and proximal prediction
The ground-truth envelope remains the envelope of the complete constrained LASSO problem:

M
γ
(
z
;
ξ
)
=
min
⁡
G
x
=
h
C
x
≤
d
{
1
2
∥
A
x
−
b
∥
2
+
λ
∥
x
∥
1
+
1
2
γ
∥
x
−
z
∥
2
}
.
M 
γ
​
 (z;ξ)= 
Gx=h
Cx≤d
​
 
min
​
 { 
2
1
​
 ∥Ax−b∥ 
2
 +λ∥x∥ 
1
​
 + 
2γ
1
​
 ∥x−z∥ 
2
 }.
​
 
We do not use the sum of individual atom values as the ground-truth label.

For each instance–query pair 
(
ξ
,
z
)
(ξ,z), solve this proximal problem to obtain 
p
γ
⋆
(
z
;
ξ
)
p 
γ
⋆
​
 (z;ξ). Then

E
⋆
=
M
γ
(
z
;
ξ
)
,
g
⋆
=
z
−
p
γ
⋆
(
z
;
ξ
)
γ
.
E 
⋆
 =M 
γ
​
 (z;ξ),g 
⋆
 = 
γ
z−p 
γ
⋆
​
 (z;ξ)
​
 .
Train using

L
(
θ
,
ω
)
=
λ
v
E
∣
M
^
θ
−
E
⋆
∣
2
+
λ
g
E
∥
∇
z
M
^
θ
−
g
⋆
∥
2
2
.
L(θ,ω)=λ 
v
​
 E∣ 
M
  
θ
​
 −E 
⋆
 ∣ 
2
 +λ 
g
​
 E∥∇ 
z
​
  
M
  
θ
​
 −g 
⋆
 ∥ 
2
2
​
 .
​
 
Finally,

p
^
θ
(
z
;
ξ
)
=
z
−
γ
∇
z
M
^
θ
(
z
;
ξ
)
.
p
​
  
θ
​
 (z;ξ)=z−γ∇ 
z
​
  
M
  
θ
​
 (z;ξ).
​
 
This is the initial proximal-point estimate supplied to your correction step.

5. Two mathematical details we must keep in mind
First, the dictionary does not give an exact decomposition.

Generally,

M
γ
(
f
+
I
X
)
≠
M
γ
f
+
∑
j
M
γ
I
X
j
.
M 
γ(f+I 
X
​
 )
​
 

=M 
γf
​
 + 
j
∑
​
 M 
γI 
X 
j
​
 
​
 
​
 .
The analytical atoms provide useful structural patterns; the ICNN learns the remaining behavior and interactions.

Second, the atom locations may not match the true envelope transitions.

For example, the half-space atom changes curvature at 
c
j
⊤
z
=
d
j
c 
j
⊤
​
 z=d 
j
​
 , whereas the envelope of the full constrained LASSO problem may transition elsewhere because of the quadratic objective and 
ℓ
1
ℓ 
1
​
  regularization.

This means fixed, geometry-derived atoms are a sensible starting point, but not necessarily a complete representation. It also means the convex ICNN residual may be unable to cancel excessive curvature introduced by an atom. We should therefore control atom weights and evaluate approximation errors near the true transition regions.

This is the formulation I suggest we adopt for constrained LASSO: one ICNN, a dictionary containing exact envelopes of the 
ℓ
1
ℓ 
1
​
  term and elementary constraint sets, and problem-dependent attention to weight those atoms.

It preserves the simplicity of your original Huber-augmented ICNN while extending the same principle to the complete feasible-set decomposition.

So now, we will construct our dictionary learning for that reformulated lasso
Let's construct a Moreau-envelope dictionary for constrained LASSO, using the structure of the objective and constraints to define analytical atoms, then learning how much each atom contributes through attention.

The central principle is:

We know the mathematical form of each dictionary atom. We learn its importance for each optimization problem instance, while the ICNN learns the remaining convex structure.

We do not train separate neural networks for the atoms.

1. Original optimization problem
Consider the parameterized constrained LASSO problem

min
⁡
x
∈
R
n
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
s
.
t
.
G
x
=
h
,
C
x
≤
d
.
(1)
x∈R 
n
 
min
​
 
s.t.
​
  
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 
Gx=h,
Cx≤d.
​
 
​
 (1)
Define the problem instance

ξ
=
(
A
,
b
,
λ
,
G
,
h
,
C
,
d
)
,
ξ=(A,b,λ,G,h,C,d),
and the nonempty feasible set

X
ξ
=
{
x
:
G
x
=
h
,
 
C
x
≤
d
}
.
X 
ξ
​
 ={x:Gx=h, Cx≤d}.
The composite objective becomes

Ψ
ξ
(
x
)
=
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
+
I
X
ξ
(
x
)
.
Ψ 
ξ
​
 (x)= 
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 +I 
X 
ξ
​
 
​
 (x).
We want to approximate its Moreau envelope:

M
γ
(
z
;
ξ
)
=
min
⁡
x
∈
X
ξ
{
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
+
1
2
γ
∥
x
−
z
∥
2
2
}
.
(2)
M 
γ
​
 (z;ξ)= 
x∈X 
ξ
​
 
min
​
 { 
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 + 
2γ
1
​
 ∥x−z∥ 
2
2
​
 }.
​
 (2)
Our goal is to learn both 
M
γ
(
z
;
ξ
)
M 
γ
​
 (z;ξ) and 
∇
z
M
γ
(
z
;
ξ
)
∇ 
z
​
 M 
γ
​
 (z;ξ).

2. Construct the dictionary from the problem structure
We decompose the nonsmooth regularizer and the feasible set into elementary components.

The 
ℓ
1
ℓ 
1
​
  regularizer is separable:

λ
∥
x
∥
1
=
∑
i
=
1
n
λ
∣
x
i
∣
.
λ∥x∥ 
1
​
 = 
i=1
∑
n
​
 λ∣x 
i
​
 ∣.
For the constraints, define

X
ξ
=
C
e
q
∩
⋂
j
=
1
m
c
C
j
,
X 
ξ
​
 =C 
eq
​
 ∩ 
j=1
⋂
m 
c
​
 
​
 C 
j
​
 ,
where

C
e
q
=
{
x
:
G
x
=
h
}
,
C
j
=
{
x
:
c
j
⊤
x
≤
d
j
}
.
C 
eq
​
 ={x:Gx=h},C 
j
​
 ={x:c 
j
⊤
​
 x≤d 
j
​
 }.
Here 
c
j
⊤
c 
j
⊤
​
  denotes the 
j
j-th row of 
C
C.

We now have three types of dictionary atoms.

Atom A: Huber functions for 
ℓ
1
ℓ 
1
​
 
For each coordinate 
i
i,

ϕ
i
ℓ
1
(
z
;
ξ
)
=
λ
H
γ
λ
(
z
i
)
.
ϕ 
i
ℓ 
1
​
 
​
 (z;ξ)=λH 
γλ
​
 (z 
i
​
 ).
​
 
This is the exact Moreau envelope of the isolated scalar regularizer 
λ
∣
x
i
∣
λ∣x 
i
​
 ∣.

Its gradient contribution is

∇
z
ϕ
i
ℓ
1
=
λ
H
γ
λ
′
(
z
i
)
e
i
.
∇ 
z
​
 ϕ 
i
ℓ 
1
​
 
​
 =λH 
γλ
′
​
 (z 
i
​
 )e 
i
​
 .
Thus, these atoms explicitly encode the piecewise-linear gradient associated with soft-thresholding.

Atom B: Equality-constraint envelope
For the affine set 
C
e
q
C 
eq
​
 , define

ϕ
e
q
(
z
;
ξ
)
=
1
2
γ
(
G
z
−
h
)
⊤
(
G
G
⊤
)
†
(
G
z
−
h
)
.
ϕ 
eq
 (z;ξ)= 
2γ
1
​
 (Gz−h) 
⊤
 (GG 
⊤
 ) 
†
 (Gz−h).
​
 
Its gradient is

∇
z
ϕ
e
q
=
1
γ
G
⊤
(
G
G
⊤
)
†
(
G
z
−
h
)
.
∇ 
z
​
 ϕ 
eq
 = 
γ
1
​
 G 
⊤
 (GG 
⊤
 ) 
†
 (Gz−h).
​
 
This is a quadratic atom representing the geometry of the affine equality set.

Atom C: Inequality-constraint envelopes
For each half-space 
C
j
C 
j
​
 , define

ϕ
j
i
n
e
q
(
z
;
ξ
)
=
[
c
j
⊤
z
−
d
j
]
+
2
2
γ
∥
c
j
∥
2
2
,
ϕ 
j
ineq
​
 (z;ξ)= 
2γ∥c 
j
​
 ∥ 
2
2
​
 
[c 
j
⊤
​
 z−d 
j
​
 ] 
+
2
​
 
​
 ,
​
 
assuming 
c
j
≠
0
c 
j
​
 

=0.

Its gradient is

∇
z
ϕ
j
i
n
e
q
=
[
c
j
⊤
z
−
d
j
]
+
γ
∥
c
j
∥
2
2
c
j
.
∇ 
z
​
 ϕ 
j
ineq
​
 = 
γ∥c 
j
​
 ∥ 
2
2
​
 
[c 
j
⊤
​
 z−d 
j
​
 ] 
+
​
 
​
 c 
j
​
 .
​
 
These atoms capture curvature changes associated with half-space projection geometry.

3. The complete analytical dictionary
Collecting these functions gives

D
ξ
=
{
ϕ
1
ℓ
1
,
…
,
ϕ
n
ℓ
1
,
ϕ
e
q
,
ϕ
1
i
n
e
q
,
…
,
ϕ
m
c
i
n
e
q
}
.
(3)
D 
ξ
​
 ={ϕ 
1
ℓ 
1
​
 
​
 ,…,ϕ 
n
ℓ 
1
​
 
​
 ,ϕ 
eq
 ,ϕ 
1
ineq
​
 ,…,ϕ 
m 
c
​
 
ineq
​
 }.
​
 (3)
The dictionary contains

K
=
n
+
1
+
m
c
K=n+1+m 
c
​
 
​
 
atoms when an equality block is present.

Each atom has a known convex 
C
1
,
1
C 
1,1
  form.

Notice that the dictionary is problem-informed. Its functions are determined directly by 
λ
,
G
,
h
,
C
,
d
,
γ
λ,G,h,C,d,γ, rather than chosen arbitrarily.

4. Attention-based dictionary learning
Now we introduce the attention mechanism.

Our objective is to learn which atoms are most useful for predicting the Moreau envelope of a particular problem instance.

Let 
e
ξ
e 
ξ
​
  represent the optimization problem and 
e
j
e 
j
​
  represent the description of dictionary atom 
j
j.

We compute attention scores:

q
ξ
=
W
Q
e
ξ
,
k
j
=
W
K
e
j
,
q 
ξ
​
 =W 
Q
​
 e 
ξ
​
 ,k 
j
​
 =W 
K
​
 e 
j
​
 ,
s
j
(
ξ
)
=
q
ξ
⊤
k
j
d
k
.
s 
j
​
 (ξ)= 
d 
k
​
 
​
 
q 
ξ
⊤
​
 k 
j
​
 
​
 .
Then define nonnegative coefficients

α
j
(
ξ
)
=
softplus
⁡
(
s
j
(
ξ
)
)
.
(4)
α 
j
​
 (ξ)=softplus(s 
j
​
 (ξ)).
​
 (4)
I propose using softplus rather than softmax initially, because softmax requires

∑
j
α
j
=
1
,
j
∑
​
 α 
j
​
 =1,
which is an unnecessary restriction for an additive envelope representation.

We only need

α
j
(
ξ
)
≥
0.
α 
j
​
 (ξ)≥0.
These weights are shared across all queries 
z
z for the same problem instance.

5. Our proposed ICNN architecture
The complete model becomes

M
^
θ
(
z
;
ξ
)
=
F
θ
(
z
;
ξ
)
+
∑
i
=
1
n
α
i
ℓ
1
(
ξ
)
ϕ
i
ℓ
1
(
z
;
ξ
)
+
α
e
q
(
ξ
)
ϕ
e
q
(
z
;
ξ
)
+
∑
j
=
1
m
c
α
j
i
n
e
q
(
ξ
)
ϕ
j
i
n
e
q
(
z
;
ξ
)
.
(5)
M
  
θ
​
 (z;ξ)=
​
  
F 
θ
​
 (z;ξ)
+ 
i=1
∑
n
​
 α 
i
ℓ 
1
​
 
​
 (ξ)ϕ 
i
ℓ 
1
​
 
​
 (z;ξ)
+α 
eq
 (ξ)ϕ 
eq
 (z;ξ)
+ 
j=1
∑
m 
c
​
 
​
 α 
j
ineq
​
 (ξ)ϕ 
j
ineq
​
 (z;ξ).
​
 
​
 (5)
Here 
F
θ
F 
θ
​
  is an ICNN that is convex and 
C
1
,
1
C 
1,1
  in 
z
z for fixed 
ξ
ξ.

Problem instance 
ξ
ξ

A
,
b
,
λ
,
G
,
h
,
C
,
d
A,b,λ,G,h,C,d

Analytical dictionary + attention

Huber, affine distance, half-space distance

α
j
(
ξ
)
≥
0
α 
j
​
 (ξ)≥0

ICNN branch

F
θ
(
z
;
ξ
)
F 
θ
​
 (z;ξ)

Global envelope approximation

Dictionary branch

∑
j
α
j
(
ξ
)
ϕ
j
(
z
;
ξ
)
∑ 
j
​
 α 
j
​
 (ξ)ϕ 
j
​
 (z;ξ)

Known curvature patterns

Predicted Moreau envelope

M
^
θ
(
z
;
ξ
)
M
  
θ
​
 (z;ξ)

Gradient-based proximal prediction

p
^
θ
=
z
−
γ
∇
z
M
^
θ
p
​
  
θ
​
 =z−γ∇ 
z
​
  
M
  
θ
​
 

Since the attention coefficients do not depend on 
z
z, and all dictionary atoms are convex, the complete function is convex in 
z
z.

It also belongs to 
C
1
,
1
C 
1,1
  if the base ICNN does. A separate curvature constraint would be needed to guarantee the exact Moreau smoothness bound 
1
/
γ
1/γ.

6. Training the dictionary and ICNN jointly
For each instance 
ξ
i
ξ 
i
​
 , sample queries 
z
i
,
j
z 
i,j
​
  and solve the complete proximal problem to obtain

p
i
,
j
⋆
=
prox
⁡
γ
Ψ
ξ
i
(
z
i
,
j
)
.
p 
i,j
⋆
​
 =prox 
γΨ 
ξ 
i
​
 
​
 
​
 (z 
i,j
​
 ).
Construct the labels

E
i
,
j
=
M
γ
(
z
i
,
j
;
ξ
i
)
,
g
i
,
j
=
z
i
,
j
−
p
i
,
j
⋆
γ
.
E 
i,j
​
 =M 
γ
​
 (z 
i,j
​
 ;ξ 
i
​
 ),g 
i,j
​
 = 
γ
z 
i,j
​
 −p 
i,j
⋆
​
 
​
 .
We train the ICNN and attention parameters jointly using

L
(
θ
)
=
λ
v
E
[
∣
M
^
θ
(
z
;
ξ
)
−
M
γ
(
z
;
ξ
)
∣
2
]
+
λ
g
E
[
∥
∇
z
M
^
θ
(
z
;
ξ
)
−
∇
z
M
γ
(
z
;
ξ
)
∥
2
2
]
.
(6)
L(θ)=
​
  
λ 
v
​
 E[∣ 
M
  
θ
​
 (z;ξ)−M 
γ
​
 (z;ξ)∣ 
2
 ]
+λ 
g
​
 E[∥∇ 
z
​
  
M
  
θ
​
 (z;ξ)−∇ 
z
​
 M 
γ
​
 (z;ξ)∥ 
2
2
​
 ].
​
 
​
 (6)
All gradients are taken through the entire scalar-output architecture, including the weighted analytical atoms.

We do not train on the component envelopes separately. The target is always the envelope of the complete constrained LASSO objective.

7. One issue we should take seriously
There is a structural limitation in this model.

The individual dictionary atoms are exact envelopes of isolated terms, but their weighted sum is generally not the envelope of the full problem.

Moreover, if an atom adds curvature in a region where the true envelope has less curvature, a convex ICNN cannot simply subtract that curvature.

Consequently, the attention coefficients may need to become small for some atoms, even if those atoms correspond to genuine terms or constraints in the original problem.

This is why learning the coefficients is important, rather than setting every coefficient to one.

It also motivates comparing the architecture against a standard ICNN and a fixed-weight dictionary ICNN.

8. What I recommend implementing first
For the first experiment, keep 
n
n, the number of equalities, and the number of inequalities fixed across problem instances. Vary their coefficients and the LASSO parameters.

This avoids introducing unnecessary variable-size architecture issues before we know whether the dictionary is useful.

We can compare:

Model

What it tests

Standard ICNN

Learning the full envelope without structural atoms

ICNN + fixed dictionary

Benefit of analytical curvature patterns

ICNN + attention-weighted dictionary

Benefit of adapting atom weights to each problem instance

The main evaluation metric should be the error in the predicted proximal point:

∥
p
^
θ
(
z
;
ξ
)
−
p
γ
⋆
(
z
;
ξ
)
∥
2
,
∥ 
p
​
  
θ
​
 (z;ξ)−p 
γ
⋆
​
 (z;ξ)∥ 
2
​
 ,
​
 
with special attention to regions where the envelope curvature changes.

This gives us a complete first version of the proposed dictionary-learning architecture for constrained LASSO: analytically defined atoms, problem-dependent attention weights, and one shared ICNN trained on the full Moreau envelope.

The central empirical question is whether learning how to combine those known envelope patterns gives us a more accurate gradient—and therefore a better proximal initialization—than learning the entire envelope with a conventional ICNN alone.

temporarily turn off the adaptive weight (attention mechanism). keep it manually. Now, should we learn sequential or parallel the dictionary for the LASSO
I recommend parallel dictionary construction and joint end-to-end training, rather than learning the dictionary atoms sequentially.

The distinction matters: our dictionary atoms are already known analytically. We are not learning each atom's functional form; we are learning how the ICNN uses their combined contribution to approximate the full Moreau envelope.

1. Our LASSO dictionary
Recall the constrained LASSO problem:

min
⁡
x
1
2
∥
A
x
−
b
∥
2
+
λ
∥
x
∥
1
s
.
t
.
G
x
=
h
,
C
x
≤
d
.
x
min
​
 
s.t.
​
  
2
1
​
 ∥Ax−b∥ 
2
 +λ∥x∥ 
1
​
 
Gx=h,Cx≤d.
​
 
Our dictionary contains:

D
ξ
(
z
)
=
{
λ
H
γ
λ
(
z
i
)
⏟
ℓ
1
 atoms
,
d
2
(
z
,
C
e
q
)
2
γ
⏟
equality atom
,
d
2
(
z
,
C
j
)
2
γ
⏟
inequality atoms
}
.
D 
ξ
​
 (z)= 
⎩
⎨
⎧
​
  
ℓ 
1
​
  atoms
λH 
γλ
​
 (z 
i
​
 )
​
 
​
 , 
equality atom
2γ
d 
2
 (z,C 
eq
​
 )
​
 
​
 
​
 , 
inequality atoms
2γ
d 
2
 (z,C 
j
​
 )
​
 
​
 
​
  
⎭
⎬
⎫
​
 .
For now, assign manually selected nonnegative weights 
a
j
a 
j
​
 .

The network becomes

M
^
θ
(
z
;
ξ
)
=
F
θ
(
z
;
ξ
)
+
∑
j
=
1
K
a
j
ϕ
j
(
z
;
ξ
)
.
M
  
θ
​
 (z;ξ)=F 
θ
​
 (z;ξ)+ 
j=1
∑
K
​
 a 
j
​
 ϕ 
j
​
 (z;ξ).
​
 
2. Sequential versus parallel
Option A — Sequential learning

Huber atoms
Constraint atoms
ICNN
We would fit one component, then another, and finally train the ICNN on the remaining error.

The problem is that the Moreau envelope is not additive:

M
γ
(
f
+
I
X
)
≠
M
γ
f
+
M
γ
I
X
.
M 
γ(f+I 
X
​
 )
​
 

=M 
γf
​
 +M 
γI 
X
​
 
​
 .
Consequently, a sequential approach can produce an intermediate residual that is not convex. An ICNN may then be unable to fit that residual.

Option B — Parallel construction and joint learning

Huber atoms

Constraint atoms

ICNN

M
^
θ
(
z
;
ξ
)
M
  
θ
​
 (z;ξ)

All analytical atoms are evaluated independently at the same query 
z
z. Their weighted outputs are added to the ICNN output.

We then supervise the complete scalar function and its gradient, not the atoms individually.

3. How I would train the model
For the first experiment, use fixed manual weights and train only the ICNN parameters 
θ
θ.

For every problem–query pair 
(
ξ
,
z
)
(ξ,z), the forward pass is

D
(
z
;
ξ
)
=
∑
j
=
1
K
a
j
ϕ
j
(
z
;
ξ
)
,
M
^
θ
(
z
;
ξ
)
=
F
θ
(
z
;
ξ
)
+
D
(
z
;
ξ
)
,
∇
z
M
^
θ
=
∇
z
F
θ
+
∇
z
D
.
D(z;ξ)
M
  
θ
​
 (z;ξ)
∇ 
z
​
  
M
  
θ
​
 
​
  
= 
j=1
∑
K
​
 a 
j
​
 ϕ 
j
​
 (z;ξ),
=F 
θ
​
 (z;ξ)+D(z;ξ),
=∇ 
z
​
 F 
θ
​
 +∇ 
z
​
 D.
​
 
The loss remains

L
(
θ
)
=
λ
v
E
∣
M
^
θ
−
M
γ
∣
2
+
λ
g
E
∥
∇
z
M
^
θ
−
∇
z
M
γ
∥
2
.
L(θ)=λ 
v
​
 E∣ 
M
  
θ
​
 −M 
γ
​
 ∣ 
2
 +λ 
g
​
 E∥∇ 
z
​
  
M
  
θ
​
 −∇ 
z
​
 M 
γ
​
 ∥ 
2
 .
​
 
Because the dictionary has no trainable parameters at this stage, there is no reason to train its atoms separately.

4. One critical caution about manual weights
We cannot arbitrarily assign large positive weights to the atoms.

Each convex atom adds nonnegative curvature. The ICNN cannot subtract that curvature while remaining convex.

For example, the LASSO Huber atom has a quadratic central region, and a constraint-distance atom also has positive curvature outside its corresponding set. Their sum may have greater curvature than the actual Moreau envelope.

Therefore, I recommend beginning with small nonnegative weights, including zero as a baseline.

For a fixed dictionary 
D
D, the decomposition is compatible with a convex residual only if

M
γ
−
D
M 
γ
​
 −D
is convex. We do not know that beforehand.

A practical experiment could compare three manual configurations:

Configuration

Dictionary contribution

Baseline

a
j
=
0
a 
j
​
 =0 for all atoms

Weak dictionary

Small positive weights

Stronger dictionary

Larger weights, with curvature checked

The weights should be selected using validation data, not assumed theoretically correct.

My recommendation: Evaluate all Huber, equality, and inequality atoms in parallel; sum their manually weighted contributions; and train one ICNN against the exact full-envelope labels.

Do not use sequential residual fitting yet. It adds complexity and introduces a convexity problem without a clear benefit.

The first result we need is simple: does adding a fixed analytical dictionary improve the gradient accuracy of the ICNN compared with the same ICNN without the dictionary?

ok. Now generate an .md file to describe our learning structure (no attention) so that Codex can implement our problem. Point out the input. output of the neural network

Worked for 1m 32s
I've created the implementation guide for Codex, using Python + JAX, with no attention mechanism and manually fixed dictionary weights.

Structure-Aware ICNN for Constrained LASSO

Markdown implementation guide · 15 KB

Download STRUCTURE_AWARE_ICNN_LASSO_GUIDE.md

Network input and output
Definition

Input 1

Query 
z
∈
R
n
z∈R 
n
 

Input 2

Problem parameters 
ξ
=
(
A
,
b
,
λ
,
G
,
h
,
C
,
d
)
ξ=(A,b,λ,G,h,C,d)

Input 3

Moreau parameter 
γ
>
0
γ>0, fixed initially

Output

Scalar predicted envelope 
M
^
θ
(
z
;
ξ
)
M
  
θ
​
 (z;ξ)

Derived gradient

∇
z
M
^
θ
(
z
;
ξ
)
∇ 
z
​
  
M
  
θ
​
 (z;ξ)

Predicted proximal point

p
^
θ
=
z
−
γ
∇
z
M
^
θ
p
​
  
θ
​
 =z−γ∇ 
z
​
  
M
  
θ
​
 

Learning structure
The guide specifies the architecture

M
^
θ
(
z
;
ξ
)
=
F
θ
(
z
;
ξ
)
⏟
ICNN
+
∑
i
a
i
ℓ
1
ϕ
i
ℓ
1
(
z
)
+
a
e
q
ϕ
e
q
(
z
)
+
∑
j
a
j
i
n
e
q
ϕ
j
i
n
e
q
(
z
)
⏟
Fixed analytical dictionary
M
  
θ
​
 (z;ξ)= 
ICNN
F 
θ
​
 (z;ξ)
​
 
​
 + 
Fixed analytical dictionary
i
∑
​
 a 
i
ℓ 
1
​
 
​
 ϕ 
i
ℓ 
1
​
 
​
 (z)+a 
eq
​
 ϕ 
eq
​
 (z)+ 
j
∑
​
 a 
j
ineq
​
 ϕ 
j
ineq
​
 (z)
​
 
​
 
​
 
The three dictionary components are Huber atoms for the 
ℓ
1
ℓ 
1
​
  regularizer, squared-distance atoms for affine equalities, and squared-hinge atoms for half-space inequalities.

All atoms are evaluated in parallel. Their weights are configured manually and remain frozen. Only the ICNN parameters are trained.

What Codex is instructed to implement
The guide includes exact proximal-label generation, value-and-gradient supervision, JAX automatic differentiation, validation tests, and four controlled experiments:

Standard ICNN without a dictionary.

ICNN with Huber atoms only.

ICNN with constraint atoms only.

ICNN with the complete dictionary.

The primary evaluation is proximal-point prediction error on unseen constrained LASSO instances, especially near curvature transitions.

You can give Codex this initial instruction:

Implement the constrained LASSO structure-aware ICNN described in STRUCTURE_AWARE_ICNN_LASSO_GUIDE.md. Follow the mathematical definitions and architecture exactly. Use Python and JAX, implement the fixed analytical dictionary in parallel with the ICNN, and train using full Moreau-envelope value and gradient labels. Do not implement attention or the downstream correction step. Start with the minimal experiment and unit tests.

Implementation Guide: Structure-Aware ICNN for the Moreau Envelope of Constrained LASSO
Purpose. Implement a minimal, attention-free, structure-aware input-convex neural network (ICNN) to approximate the full Moreau envelope and its gradient for a family of constrained LASSO problems. The analytic dictionary is evaluated in parallel with the ICNN; its coefficients are manually specified and frozen. The network predicts a scalar envelope value, and automatic differentiation of that scalar with respect to the query produces the predicted envelope gradient and proximal point.

Implementation language: Python + JAX (Flax optional); prefer float64 (jax_enable_x64=True or JAX_ENABLE_X64=1). Matrix-free computations where sensible; avoid unnecessary explicit Hessians. No attention, Transformers, learned dictionary coefficients, learned projections, or sequential training in this version.

1. Mathematical problem
For each instance indexed by 
ξ
ξ, solve the convex constrained LASSO:

min
⁡
x
∈
R
n
f
ξ
(
x
)
=
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
s
.
t
.
G
x
=
h
,
C
x
≤
d
.
x∈R 
n
 
min
​
 
s.t.
​
  
f 
ξ
​
 (x)= 
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 
Gx=h,
Cx≤d.
​
 
Dimensions:

A
∈
R
m
×
n
A∈R 
m×n
 , 
b
∈
R
m
b∈R 
m
 ;
λ
>
0
λ>0;
G
∈
R
r
×
n
G∈R 
r×n
 , 
h
∈
R
r
h∈R 
r
 ;
C
∈
R
s
×
n
C∈R 
s×n
 , 
d
∈
R
s
d∈R 
s
 .
Assume the feasible set is nonempty:

X
ξ
=
{
x
:
G
x
=
h
,
 
C
x
≤
d
}
.
X 
ξ
​
 ={x:Gx=h, Cx≤d}.
Let 
Ψ
ξ
=
f
ξ
+
I
X
ξ
Ψ 
ξ
​
 =f 
ξ
​
 +I 
X 
ξ
​
 
​
 . For fixed 
γ
>
0
γ>0, define the full Moreau envelope

M
γ
(
z
;
ξ
)
=
min
⁡
x
∈
X
ξ
[
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
+
1
2
γ
∥
x
−
z
∥
2
2
]
.
M 
γ
​
 (z;ξ)= 
x∈X 
ξ
​
 
min
​
 [ 
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 + 
2γ
1
​
 ∥x−z∥ 
2
2
​
 ].
​
 
The unique proximal point 
p
∗
(
z
;
ξ
)
p 
∗
 (z;ξ) solves that problem and

∇
z
M
γ
(
z
;
ξ
)
=
z
−
p
∗
(
z
;
ξ
)
γ
.
∇ 
z
​
 M 
γ
​
 (z;ξ)= 
γ
z−p 
∗
 (z;ξ)
​
 .
​
 
Do not confuse 
x
x (inner optimization variable) with 
z
z (query at which envelope and its gradient are evaluated).

2. Network input and output: explicit contract
Inputs to the neural model
A single training/evaluation example is a problem–query pair 
(
ξ
,
z
)
(ξ,z):

Query z: shape (n,), float64 — the variable with respect to which the model output is differentiated.
Problem data xi:
A: (m,n); b: (m,); lam: scalar;
G: (r,n); h: (r,);
C: (s,n); d: (s,).
Envelope smoothing parameter gamma: positive scalar (keep fixed in version 1; document in configuration).
Manual dictionary weights weights: fixed nonnegative scalars / arrays, passed as model configuration (not optimized).
The model must condition on both z and xi: otherwise one shared model cannot learn envelopes for varying LASSO instances. For a simple first implementation with fixed dimensions, flatten and concatenate the numeric instance data into an instance feature vector q_xi; normalize features using training-set statistics only. Train and validation/test instances must be distinct. gamma may be appended as a static condition if supported, but begin with a fixed gamma.

Convexity requirement: The model's scalar output must be convex in the raw query z for each fixed xi. Do not feed nonlinear z-dependent dictionary outputs as unconstrained conditioning features into the ICNN. Instance features may parameterize query-affine terms or other ICNN mechanisms that preserve convexity in z; see section 4. Input standardization of z may use a fixed affine transform (convexity preserved); when computing grad_z, differentiate through this transform and use the gradient in original query coordinates.

Outputs from the neural model
Primary learned output:

M_hat(z, xi): one scalar, shape () (approximate full Moreau-envelope value).
Derived outputs (not independent heads):

grad_M_hat = jax.grad(M_hat, argnums_for_z)(z,xi): shape (n,), derivative with respect to query z only;
p_hat = z - gamma * grad_M_hat: shape (n,), approximate proximal point.
Important: Do not output the decision vector directly and do not add an independently trained gradient head. The approximate gradient must be the derivative of the same scalar output.

3. Fixed analytic dictionary (calculated in parallel)
Split the feasible set as

X
ξ
=
C
e
q
∩
⋂
j
=
1
s
C
j
,
C
e
q
=
{
x
:
G
x
=
h
}
,
C
j
=
{
x
:
c
j
⊤
x
≤
d
j
}
,
X 
ξ
​
 =C 
eq
​
 ∩ 
j=1
⋂
s
​
 C 
j
​
 ,C 
eq
​
 ={x:Gx=h},C 
j
​
 ={x:c 
j
⊤
​
 x≤d 
j
​
 },
where c_j = C[j, :]. Define the following fixed-form convex, 
C
1
,
1
C 
1,1
  atoms. They represent the Moreau envelopes of isolated components, not the full composite envelope.

3.1 L1 regularizer: coordinatewise Huber atoms
Define

H
δ
(
t
)
=
{
t
2
/
(
2
δ
)
,
∣
t
∣
≤
δ
,
∣
t
∣
−
δ
/
2
,
∣
t
∣
>
δ
.
H 
δ
​
 (t)={ 
t 
2
 /(2δ),
∣t∣−δ/2,
​
  
∣t∣≤δ,
∣t∣>δ.
​
 
For each coordinate 
i
=
1
,
…
,
n
i=1,…,n,

ϕ
i
ℓ
1
(
z
;
ξ
)
=
λ
H
γ
λ
(
z
i
)
.
ϕ 
i
ℓ 
1
​
 
​
 (z;ξ)=λH 
γλ
​
 (z 
i
​
 ).
​
 
This is exactly 
M
γ
λ
∣
⋅
∣
(
z
i
)
M 
γλ∣⋅∣
​
 (z 
i
​
 ). For stability, evaluate Huber in a JAX-compatible, differentiable piecewise form. The derivative w.r.t. z_i is continuous at the transitions 
z
i
=
±
γ
λ
z 
i
​
 =±γλ.

3.2 Affine equality-set atom
When r > 0 and 
C
e
q
C 
eq
​
  is nonempty,

ϕ
e
q
(
z
;
ξ
)
=
1
2
γ
(
G
z
−
h
)
⊤
(
G
G
⊤
)
†
(
G
z
−
h
)
.
ϕ 
eq
 (z;ξ)= 
2γ
1
​
 (Gz−h) 
⊤
 (GG 
⊤
 ) 
†
 (Gz−h).
​
 
Use a stable least-squares / QR / SVD-based operation, not an explicit dense matrix inverse. If G has full row rank, a well-conditioned linear solve is appropriate. If r=0, omit this atom. This atom has constant curvature (no kink), but encodes equality geometry.

3.3 Inequality half-space atoms
For every row with 
∥
c
j
∥
2
>
0
∥c 
j
​
 ∥ 
2
​
 >0,

ϕ
j
i
n
e
q
(
z
;
ξ
)
=
[
c
j
⊤
z
−
d
j
]
+
2
2
γ
∥
c
j
∥
2
2
,
[
t
]
+
=
max
⁡
(
t
,
0
)
.
ϕ 
j
ineq
​
 (z;ξ)= 
2γ∥c 
j
​
 ∥ 
2
2
​
 
[c 
j
⊤
​
 z−d 
j
​
 ] 
+
2
​
 
​
 ,[t] 
+
​
 =max(t,0).
​
 
These are the exact squared-distance envelopes of the individual half-space indicators. Their first derivatives are continuous; second derivatives may jump across the boundary. Reject or preprocess zero-norm rows (including feasibility checks); do not silently divide by zero.

3.4 Manual dictionary combination
All atom forms and geometry parameters remain fixed by the current problem instance. For version 1, no learned atom shifts, directions, scales, or attention scores.

D
(
z
;
ξ
)
=
∑
i
=
1
n
a
i
ℓ
1
ϕ
i
ℓ
1
(
z
;
ξ
)
+
a
e
q
ϕ
e
q
(
z
;
ξ
)
+
∑
j
=
1
s
a
j
i
n
e
q
ϕ
j
i
n
e
q
(
z
;
ξ
)
.
D(z;ξ)= 
i=1
∑
n
​
 a 
i
ℓ 
1
​
 
​
 ϕ 
i
ℓ 
1
​
 
​
 (z;ξ)+a 
eq
​
 ϕ 
eq
 (z;ξ)+ 
j=1
∑
s
​
 a 
j
ineq
​
 ϕ 
j
ineq
​
 (z;ξ).
​
 
All coefficients are nonnegative, manually assigned, and frozen. Supply config values for a_l1 (scalar broadcast or length-n vector), a_eq (scalar), and a_ineq (scalar broadcast or length-s vector). For the initial test, make each atom-family multiplier an easily swept hyperparameter, including the all-zero baseline. Do not assume setting all weights to 1 is mathematically correct.

For every query, compute the Huber, equality, and inequality atoms independently / in parallel, sum them, and add the sum to the ICNN output. There is no iterative projection onto the intersection in the dictionary branch.

4. Full structure-aware ICNN
M
^
θ
(
z
;
ξ
)
=
F
θ
(
z
;
ξ
)
+
D
(
z
;
ξ
)
.
M
  
θ
​
 (z;ξ)=F 
θ
​
 (z;ξ)+D(z;ξ).
​
 
F_theta is a conditional ICNN convex in query z. An acceptable minimal implementation is a standard ICNN whose layer weights from previous hidden convex states are constrained nonnegative and whose activations are convex and nondecreasing (e.g., softplus); dependence on xi should enter through query-affine terms or additive, problem-conditioned biases. One possible layer pattern:

u
1
=
softplus
⁡
(
U
1
z
+
V
1
q
ξ
+
b
1
)
,
u 
1
​
 =softplus(U 
1
​
 z+V 
1
​
 q 
ξ
​
 +b 
1
​
 ),
u
k
+
1
=
softplus
⁡
(
W
k
,
+
u
k
+
U
k
+
1
z
+
V
k
+
1
q
ξ
+
b
k
+
1
)
,
W
k
,
+
=
softplus
⁡
(
W
~
k
)
,
u 
k+1
​
 =softplus(W 
k,+
​
 u 
k
​
 +U 
k+1
​
 z+V 
k+1
​
 q 
ξ
​
 +b 
k+1
​
 ),W 
k,+
​
 =softplus( 
W
  
k
​
 ),
F
θ
(
z
;
ξ
)
=
w
+
⊤
u
L
+
a
⊤
z
+
c
(
q
ξ
)
,
w
+
=
softplus
⁡
(
w
~
)
.
F 
θ
​
 (z;ξ)=w 
+
⊤
​
 u 
L
​
 +a 
⊤
 z+c(q 
ξ
​
 ),w 
+
​
 =softplus( 
w
 ).
Here U_k, V_k, a may have signed entries; the hidden-state and final hidden-output weights are constrained nonnegative. q_xi must be independent of z. For the simplest version, q_xi can be the flattened, normalized instance data; c(q_xi) is optional and may be a small affine mapping. Use smooth activations; in this model the dictionary atoms provide exact piecewise-curvature changes.

Critical limitation: Convexity plus 
C
1
,
1
C 
1,1
  does not automatically enforce 
Lip
⁡
(
∇
z
M
^
)
≤
1
/
γ
Lip(∇ 
z
​
  
M
 )≤1/γ. Do not state that the prediction is guaranteed to be an exact Moreau envelope. The nonnegative dictionary adds curvature, which the convex ICNN branch cannot cancel. Tune coefficients cautiously; optionally audit local Hessian eigenvalues / gradient Lipschitz behavior on samples. There is no requirement for p_hat to be feasible; downstream correction is separate and outside this implementation.

5. Exact training labels (full proximal problem)
For each instance 
ξ
i
ξ 
i
​
 , sample multiple queries 
z
i
,
1
,
…
,
z
i
,
N
q
z 
i,1
​
 ,…,z 
i,N 
q
​
 
​
 . At each pair, solve the entire constrained proximal LASSO problem (section 1), using a high-accuracy reference convex solver (e.g., CVXPY with a reliable solver) and check its termination/constraint feasibility.

p
i
j
∗
=
arg
⁡
min
⁡
G
x
=
h
,
 
C
x
≤
d
[
1
2
∥
A
x
−
b
∥
2
2
+
λ
∥
x
∥
1
+
1
2
γ
∥
x
−
z
i
j
∥
2
2
]
.
p 
ij
∗
​
 =arg 
Gx=h, Cx≤d
min
​
 [ 
2
1
​
 ∥Ax−b∥ 
2
2
​
 +λ∥x∥ 
1
​
 + 
2γ
1
​
 ∥x−z 
ij
​
 ∥ 
2
2
​
 ].
Build both labels:

E
i
j
∗
=
1
2
∥
A
p
i
j
∗
−
b
∥
2
2
+
λ
∥
p
i
j
∗
∥
1
+
1
2
γ
∥
p
i
j
∗
−
z
i
j
∥
2
2
,
E 
ij
∗
​
 = 
2
1
​
 ∥Ap 
ij
∗
​
 −b∥ 
2
2
​
 +λ∥p 
ij
∗
​
 ∥ 
1
​
 + 
2γ
1
​
 ∥p 
ij
∗
​
 −z 
ij
​
 ∥ 
2
2
​
 ,
​
 
g
i
j
∗
=
(
z
i
j
−
p
i
j
∗
)
/
γ
.
g 
ij
∗
​
 =(z 
ij
​
 −p 
ij
∗
​
 )/γ.
​
 
Keep p_star too for evaluation. Reference label generation is offline: do not solve a proximal optimization problem inside the ICNN forward pass.

Suggested query sampling: a mixture of broadly distributed z, queries near optimal solutions, and queries close to likely active-set transitions; for the first implementation broad and near-solution sampling are enough. Validate on unseen problem instances, splitting at the instance level before sampling/assigning queries.

6. Training loss and optimization
The scalar network output and its input derivative are jointly supervised:

L
(
θ
)
=
λ
v
E
(
ξ
,
z
)
∣
M
^
θ
(
z
;
ξ
)
−
E
∗
∣
2
+
λ
g
E
(
ξ
,
z
)
∥
∇
z
M
^
θ
(
z
;
ξ
)
−
g
∗
∥
2
2
.
L(θ)=λ 
v
​
 E 
(ξ,z)
​
 ∣ 
M
  
θ
​
 (z;ξ)−E 
∗
 ∣ 
2
 +λ 
g
​
 E 
(ξ,z)
​
 ∥∇ 
z
​
  
M
  
θ
​
 (z;ξ)−g 
∗
 ∥ 
2
2
​
 .
​
 
lambda_v, lambda_g: manually configurable nonnegative loss weights; keep identical for all model comparisons.
Train only ICNN weights theta; dictionary weights are configuration, not trainable state.
Use jax.grad on the whole scalar F_theta + D, and use jax.value_and_grad/jax.vmap/jax.jit as appropriate.
Ensure reference E_star and g_star are in the same units and scaling as network outputs. If implementing output normalization, apply the chain rule and undo it consistently in inference.
A naive normalized envelope may need a learned instance-dependent additive constant; ensure the architecture can represent envelope offsets, because the gradient loss cannot identify constants.
7. Prediction, metrics, and tests
Inference API:

M_hat = model.apply(params, z, xi, config)
g_hat = jax.grad(lambda zz: model.apply(params, zz, xi, config))(z)
p_hat = z - gamma * g_hat
Return at least M_hat (scalar), g_hat (n,), and p_hat (n,).

Report on unseen instances:

Envelope value absolute/relative error.
Gradient error ||g_hat - g_star||_2 and ||...||_inf.
Proximal prediction error ||p_hat-p_star||_2 and ||...||_inf.
Gradient error near known/empirically estimated transition regions versus elsewhere.
Runtime of one batched forward-and-gradient evaluation (with warm-up and synchronized JAX timing).
Optional constraint violations of p_hat as diagnostics only (feasibility is not the network's guarantee).
Mandatory correctness checks:

Compare analytic dictionary atom values and jax.grad against closed-form projections / finite differences at points away from transition boundaries.
Check gradients are taken with respect to z, not x or xi.
Verify gradients of manually fixed dictionary contributions pass through to g_hat.
Check shape consistency and support zero equality / inequality constraints.
Check finite output and gradients for test examples (including exactly on Huber and half-space transitions).
Numerically test Jensen convexity in z at fixed xi (diagnostic; not a formal proof).
Check the identity ||p_hat-p_star|| = gamma * ||g_hat-g_star|| numerically.
8. Minimal ablations (no attention)
Keep identical ICNN, train/validation/test instance splits, query sampling, labels, optimizer, and training budget across:

Experiment	Manually fixed dictionary weights	Purpose
baseline	All zero	Standard conditional ICNN
l1_only	Only Huber family nonzero	Isolate the effect of nonsmooth-objective atoms
constraints_only	Only equality/inequality atoms nonzero	Isolate constraint geometry
combined	L1 + constraint atoms nonzero	Test the proposed full dictionary
Use a small validation sweep of manually specified weights. No attention and no learnable coefficient generator.

9. Suggested project organization and deliverables
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
Produce a runnable minimal experiment with deterministic seeds, a small synthetic feasible instance generator, training/evaluation commands, and a comparison report across the four ablations.

Do not implement in version 1
Attention / Transformers / query-dependent weights.
Learned atom locations, learned directions, or learned dictionary coefficients.
Sequential residual fitting, separate atom networks, or per-atom supervised training.
End-to-end differentiation through CVXPY/reference solver.
Feasibility-correction step (to be integrated later).
Assertions that the sum of component envelopes equals the envelope of the full problem.
Success criterion: Relative to the standard ICNN, the fixed analytic dictionary improves full-envelope gradient and proximal-point prediction accuracy on held-out constrained LASSO instances, especially near changes of curvature, without sacrificing numerical stability.



Beta
0 / 0
used queries
1