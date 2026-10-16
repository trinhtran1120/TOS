# Input-Convex Neural Network with value + gradient supervision in JAX.
# Convexity is enforced by nonnegative weights (via a projection act_p) on the state
# connections and the last-layer z-weights, and by using a convex, nondecreasing activation.
#
# The problems differ in three training choices, selected with make_icnn / train_icnn:
#   weight_act  "relu" (entr_max, mpc) or "softplus" (power_grid, mvee): the projection act_p
#   keep_best   return the parameters with the best validation objective (entr_max, mpc)
#               or those of the last epoch (power_grid, mvee).  Given val_data = (X, y, g),
#               train_icnn scores that validation set at every learning-rate transition
#               (every lr_decay transition_steps, or every epochs/10 without lr_decay); otherwise
#               the objective is scored on the first 1024 training samples
#   acts        hidden-layer activations, one per layer (default softplus everywhere): "softplus",
#               "softplus<β>" = softplus(β t)/β, or "elu".  Layer 1 needs a convex activation, the
#               later layers convex and nondecreasing; all of these are both.  save_model writes them
#               to the .npz (act_kind, act_beta), which src/icnn.jl reads.
#   huber_terms k > 0 adds Σᵢ softplus(c̃ᵢ) H_δᵢ(Bᵢᵀx + dᵢ), δᵢ = softplus(δ̃ᵢ), to the ICNN: learned
#               Huber terms (H_δ = Moreau envelope of |t|) that give the gradient exact corners, for
#               an L1 term in the objective.  Still convex.  save_model writes huber_B, huber_d,
#               huber_c, huber_delta to the .npz, which src/icnn.jl reads.
#   f           lower-bound targets: adds penalty_weight * mean(relu(f_pred - f)) and an
#               explicit L2 term l2_reg * ||params||² to the loss (mvee)

import json
import os
import pickle
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

# -----------------------------
# Utils & parameter initialization
# -----------------------------
def glorot_uniform(key, shape):
    fan_in, fan_out = shape[1], shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

def init_icnn_params(
    key: jax.Array,
    n_in: int,
    widths: List[int],
    init: str = "default",
    huber_terms: int = 0,
) -> Dict[str, Any]:
    """
    ICNN parameters:
      For L hidden layers with widths = [w1, w2, ..., wL]:
        z0 = act(U0 x + b0)
        zk = act(act_p(Wk) zk-1 + Uk x + bk ),   k=1..L-1
      Output:
        f(x) = act_p(v)^T zL + a^T x + c

      Notes:
        - act is convex & nondecreasing; we use softplus for smooth gradients.
        - act_p(Wk) and act_p(v) ensure elementwise nonnegativity.
        - a^T x + c is linear (convex).
      init = "default": the raw W and v near 0, so act_p(W), act_p(v) ≈ softplus(0) = 0.69 and the
        output grows like width² (about 2e4 for 256x256).  init = "fan_in": raw values with
        softplus(raw) ≈ 1/fan_in (times a lognormal-like spread), so each unit averages its inputs
        and the output starts at O(1).
    """
    def nonneg_raw(k, shape, fan_in):   # softplus(raw) ≈ exp(0.5 ξ) / fan_in, ξ ~ N(0, 1)
        target = jnp.exp(0.5 * jax.random.normal(k, shape)) / fan_in
        return jnp.log(jnp.expm1(target))
    num_layers = len(widths)
    keys = jax.random.split(key, 3 * num_layers + 3)

    U, W, b = [], [], []
    ki = 0
    for i, w in enumerate(widths):
        # Input skip for every layer
        U.append(glorot_uniform(keys[ki], (w, n_in))); ki += 1

        if i == 0:
            # First layer has no state weight (depends only on x)
            W.append(jax.numpy.zeros((w, n_in)))
        else:
            # Unconstrained state weight; act_p applied in forward pass
            W.append(nonneg_raw(keys[ki], (w, widths[i - 1]), widths[i - 1]) if init == "fan_in"
                     else 0.05 * jax.random.normal(keys[ki], (w, widths[i - 1]))); ki += 1

        b.append(jnp.zeros((w,)))

    # Last-layer z-weights (constrained nonnegative via act_p), linear term, and bias
    v = (nonneg_raw(keys[-3], (widths[-1],), widths[-1]) if init == "fan_in"
         else 0.05 * jax.random.normal(keys[-3], (widths[-1],)))
    a = 0.01 * jax.random.normal(keys[-2], (n_in,))
    c = jnp.array(0.0)

    params = {"U": U, "W": W, "b": b, "v": v, "a": a, "c": c}
    if huber_terms > 0:   # B starts near the identity (one term per coordinate), c = δ = 0.1
        start = jnp.log(jnp.expm1(0.1))
        params.update(huber_B=jnp.eye(n_in, huber_terms) + 0.01 * jax.random.normal(jax.random.fold_in(key, 99), (n_in, huber_terms)),
                      huber_d=jnp.zeros(huber_terms), huber_c=jnp.full(huber_terms, start),
                      huber_delta=jnp.full(huber_terms, start))
    return params


def huber(t, delta):
    """H_δ(t) = t²/2δ on |t| ≤ δ, |t| - δ/2 outside: the Moreau envelope of |t|."""
    a = jnp.abs(t)
    return jnp.where(a <= delta, t * t / (2 * delta), a - delta / 2)


WEIGHT_ACTS = {"relu": jax.nn.relu, "softplus": jax.nn.softplus}

# Hidden-layer activations (convex, nondecreasing): name -> (kind code for src/icnn.jl, β).
ACT_KINDS = {"softplus": 0, "elu": 1}


def parse_act(name: str):
    """'softplus' (β = 1), 'softplus<β>' (softplus(β t)/β) or 'elu' -> (kind, β)."""
    if name == "elu":
        return "elu", 1.0
    if name.startswith("softplus"):
        return "softplus", float(name[len("softplus"):] or 1.0)
    raise ValueError(f"unknown activation {name!r}")


def act_fn(name: str):
    kind, beta = parse_act(name)
    if kind == "elu":
        return jax.nn.elu
    return jax.nn.softplus if beta == 1.0 else (lambda t: jax.nn.softplus(beta * t) / beta)


def learning_rate(lr: float, lr_decay: Optional[Dict[str, Any]] = None, steps_per_epoch: int = 1):
    """Constant lr, or lr · decay_rate^(t / transition_steps) when lr_decay =
    {"transition_steps": .., "decay_rate": .., "unit": "step" | "epoch", "staircase": false};
    t counts optimizer steps, or epochs with unit = "epoch" (default "step")."""
    if not lr_decay:
        return lr
    per = steps_per_epoch if lr_decay.get("unit", "step") == "epoch" else 1
    return optax.exponential_decay(init_value=lr, transition_steps=lr_decay["transition_steps"] * per,
                                   decay_rate=lr_decay["decay_rate"],
                                   staircase=lr_decay.get("staircase", False))


def make_lasso_dictionary(A, G, h, rho: float, l1_weight: float,
                          a_l1=0.0, a_eq: float = 0.0, a_ineq=0.0):
    """Fixed analytic atoms for [q, x], where Ay=x and Gy<=h.

    These are envelopes of isolated components, not an additive formula for the
    full envelope. All weights are manual, nonnegative, and outside params.
    """
    A, G, h = (np.asarray(a, dtype=np.float64) for a in (A, G, h))
    if (A.ndim != 2 or G.ndim != 2 or A.shape[1] != G.shape[1]
            or h.shape != (G.shape[0],) or A.shape[1] < 1
            or not all(np.isfinite(a).all() for a in (A, G, h))):
        raise ValueError("Invalid dictionary constraint matrices")
    if not np.isfinite([rho, l1_weight]).all() or rho <= 0 or l1_weight < 0:
        raise ValueError("Require rho > 0 and l1_weight >= 0")
    n = A.shape[1]

    def weights(value, size):
        result = np.broadcast_to(np.asarray(value, dtype=np.float32), (size,))
        if not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError("Dictionary weights must be finite and nonnegative")
        return jnp.asarray(result)

    l1_weights, ineq_weights = weights(a_l1, n), weights(a_ineq, G.shape[0])
    eq_weight = float(a_eq)
    if not np.isfinite(eq_weight) or eq_weight < 0:
        raise ValueError("a_eq must be finite and nonnegative")
    # A full-row-rank equality system is feasible for every sampled x. The
    # pseudoinverse gives the Euclidean projection displacement without an inverse.
    if A.shape[0] and np.linalg.matrix_rank(A) != A.shape[0]:
        raise ValueError("Equality matrix must have full row rank")
    row_norms = np.sum(G * G, axis=1)
    if np.any(row_norms == 0):
        raise ValueError("Remove zero-norm inequality rows before constructing atoms")
    Aj, Gj, hj, pinv, norms = (
        jnp.asarray(a, dtype=jnp.float32)
        for a in (A, G, h, np.linalg.pinv(A), row_norms)
    )

    @jax.jit
    def dictionary(packed):
        q, x = packed[:n], packed[n:]
        if q.shape != (n,) or x.shape != (A.shape[0],):
            raise ValueError("Dictionary input must pack [q, x]")
        l1 = (jnp.dot(l1_weights, l1_weight * huber(q, rho * l1_weight))
              if l1_weight > 0 else jnp.asarray(0.0, dtype=q.dtype))
        displacement = pinv @ (Aj @ q - x)
        equality = eq_weight * jnp.sum(displacement ** 2) / (2 * rho)
        violation = jax.nn.relu(Gj @ q - hj)
        inequality = jnp.sum(ineq_weights * violation ** 2 / (2 * rho * norms))
        return l1 + equality + inequality

    return dictionary


def make_icnn(weight_act: str = "relu", keep_best: bool = True, acts: Optional[List[str]] = None,
              query_dim: Optional[int] = None, dictionary=None, value_weight: float = 1.0) -> SimpleNamespace:
    """The jitted ICNN functions for one choice of weight projection and hidden-layer
    activations (see the file header; acts = None: softplus in every layer).
    Conditional inputs pack [q, x]; query_dim restricts supervision to d/dq.
    dictionary is a fixed scalar function of the same packed input."""
    if not np.isfinite(value_weight) or value_weight < 0:
        raise ValueError("value_weight must be finite and nonnegative")
    act_p = WEIGHT_ACTS[weight_act]
    layer_act = lambda i: jax.nn.softplus if acts is None else act_fn(acts[i])

    # -----------------------------
    # Forward pass
    # -----------------------------
    @jax.jit
    def icnn_forward(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        """
        f(x): R^n -> R (scalar)
        x shape: (n,)
        Returns: scalar (0-d array)
        """
        U, W, b = params["U"], params["W"], params["b"]
        if acts is not None and len(acts) != len(U):
            raise ValueError(f"{len(acts)} activations for {len(U)} layers")

        z = layer_act(0)(jnp.dot(U[0], x) + b[0])  # first layer (no state W)
        for i in range(1, len(U)):
            # state contribution is constrained nonnegative via act_p(Wi)
            z = layer_act(i)(jnp.dot(act_p(W[i]), z) + jnp.dot(U[i], x) + b[i])

        v_nonneg = act_p(params["v"])  # nonnegative combination of convex features
        f = jnp.dot(v_nonneg, z) + jnp.dot(params["a"], x) + params["c"]
        if "huber_B" in params:   # + Σ softplus(c̃) H_softplus(δ̃)(Bᵀx + d)
            t = jnp.dot(params["huber_B"].T, x) + params["huber_d"]
            f = f + jnp.dot(jax.nn.softplus(params["huber_c"]), huber(t, jax.nn.softplus(params["huber_delta"])))
        if dictionary is not None:
            f = f + dictionary(x)
        return f

    batched_forward = jax.vmap(icnn_forward, in_axes=(None, 0))

    # -----------------------------
    # Gradient wrt input
    # -----------------------------
    @jax.jit
    def grad_wrt_x(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        if query_dim is None:
            return jax.grad(lambda xx: icnn_forward(params, xx))(x)
        return jax.grad(lambda q: icnn_forward(params, jnp.concatenate((q, x[query_dim:]))))(x[:query_dim])

    batched_grad_wrt_x = jax.jit(jax.vmap(grad_wrt_x, in_axes=(None, 0)))

    # -----------------------------
    # Loss: value + gradient supervision
    # -----------------------------
    @jax.jit
    def loss_fn(params, xb, yb, gb, grad_weight: float = 1.0):
        """Returns (total_loss, (value_mse, grad_mse))"""
        y_pred = batched_forward(params, xb)             # (B,)
        g_pred = batched_grad_wrt_x(params, xb)          # (B, n)
        value_mse = jnp.mean((y_pred - yb) ** 2)
        grad_mse = jnp.mean(jnp.sum((g_pred - gb) ** 2, axis=1))
        total = value_weight * value_mse + grad_weight * grad_mse
        return total, (value_mse, grad_mse)

    @jax.jit
    def loss_fn_bounded(params, xb, yb, gb, fb, grad_weight: float = 1.0,
                        penalty_weight: float = 1.0, l2_reg: float = 0.0):
        """Returns (total_loss, (value_mse, grad_mse, lower_bound_penalty))"""
        y_pred = batched_forward(params, xb)
        g_pred = batched_grad_wrt_x(params, xb)
        value_mse = jnp.mean((y_pred - yb) ** 2)
        grad_mse = jnp.mean(jnp.sum((g_pred - gb) ** 2, axis=1))
        lower_bound_penalty = jnp.mean(jax.nn.relu(y_pred - fb))
        l2_penalty = l2_reg * jnp.sum(jnp.array([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)]))
        total = value_weight * value_mse + grad_weight * grad_mse + penalty_weight * lower_bound_penalty + l2_penalty
        return total, (value_mse, grad_mse, lower_bound_penalty)

    def make_train_step(optimizer, bounded: bool):
        if bounded:
            @jax.jit
            def train_step(params, opt_state, xb, yb, gb, fb, grad_weight, penalty_weight, l2_reg):
                (loss, aux), grads = jax.value_and_grad(loss_fn_bounded, has_aux=True)(
                    params, xb, yb, gb, fb, grad_weight, penalty_weight, l2_reg
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return optax.apply_updates(params, updates), opt_state, loss, aux
        else:
            @jax.jit
            def train_step(params, opt_state, xb, yb, gb, grad_weight):
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, xb, yb, gb, grad_weight
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return optax.apply_updates(params, updates), opt_state, loss, aux
        return train_step

    def make_train_epoch(train_step, batch_size: int):
        """One epoch as a single compiled call: the batches of batch_iterator's shuffle (full
        batches by lax.scan, then the remainder) instead of a Python loop of small GPU calls.
        Returns params, opt_state and the summed (value MSE, gradient MSE) over the batches."""
        @jax.jit
        def train_epoch(params, opt_state, key, arrays, extra):
            N = arrays[0].shape[0]
            perm = jax.random.permutation(key, N)
            nb = N // batch_size

            def body(carry, j):
                p, s = carry
                p, s, _, aux = train_step(p, s, *(a[j] for a in arrays), *extra)
                return (p, s), jnp.stack(aux[:2])

            (params, opt_state), auxs = jax.lax.scan(
                body, (params, opt_state), perm[:nb * batch_size].reshape(nb, batch_size))
            sums = auxs.sum(axis=0)
            if N % batch_size:
                rest = perm[nb * batch_size:]
                params, opt_state, _, aux = train_step(params, opt_state, *(a[rest] for a in arrays), *extra)
                sums = sums + jnp.stack(aux[:2])
            return params, opt_state, sums
        return train_epoch

    def train_icnn(
        X: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
        n_in: int,
        widths: List[int] = [64, 64, 64],
        lr: float = 1e-3,
        grad_weight: float = 1.0,
        l2_reg: float = 0,
        batch_size: int = 128,
        epochs: int = 200,
        seed: int = 0,
        f: Optional[np.ndarray] = None,
        penalty_weight: float = 1.0,
        lr_decay: Optional[Dict[str, Any]] = None,
        val_data: Optional[tuple] = None,
        init: str = "default",
        huber_terms: int = 0,
        save_at: List[int] = (),
        on_save=None,
    ) -> Dict[str, Any]:
        """save_at: epochs at which on_save(epoch, params, best_epoch) receives the model kept
        so far (the best validated one with keep_best), as a run stopped at that epoch returns."""
        bounded = f is not None
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != n_in:
            raise ValueError("X must be a nonempty (samples, n_in) array")
        if batch_size < 1 or epochs < 1 or not widths or any(w < 1 for w in widths):
            raise ValueError("batch_size, epochs and hidden widths must be positive")
        if query_dim is not None and not 1 <= query_dim <= n_in:
            raise ValueError("query_dim must be between 1 and n_in")
        if dictionary is not None and huber_terms:
            raise ValueError("Fixed dictionary training must not add learned Huber atoms")
        if not np.isfinite(grad_weight) or grad_weight < 0 or value_weight + grad_weight == 0:
            raise ValueError("Loss weights must be nonnegative and not both zero")
        for data in ((X, y, g),) + (() if val_data is None else (val_data[:3],)):
            xx, yy, gg = data
            if (xx.ndim != 2 or xx.shape[1] != n_in or len(xx) == 0
                    or yy.shape != (len(xx),) or gg.shape != (len(xx), query_dim or n_in)
                    or not all(np.isfinite(a).all() for a in data)):
                raise ValueError("Invalid input/value/query-gradient shapes or nonfinite data")
        key = jax.random.PRNGKey(seed)
        arrays = [jnp.asarray(a, dtype=jnp.float32) for a in ((X, y, g, f) if bounded else (X, y, g))]
        val_every = max(1, epochs // 10)
        if val_data is not None:   # separate validation set (X, y, g), scored at each lr transition
            val_arrays = [jnp.asarray(a, dtype=jnp.float32) for a in val_data]
            if lr_decay:
                steps_per_epoch = -(-X.shape[0] // batch_size)
                per = 1 if lr_decay.get("unit", "step") == "epoch" else steps_per_epoch
                val_every = max(1, -(-lr_decay["transition_steps"] // per))
            print(f"training on {X.shape[0]} samples, validating on {val_arrays[0].shape[0]} samples every {val_every} epochs")
        else:
            val_arrays = [a[:1024] for a in arrays]
        Xj, yj, gj = arrays[:3]
        Xv, yv, gv = val_arrays[:3]

        params = init_icnn_params(key, n_in=n_in, widths=widths, init=init, huber_terms=huber_terms)
        params = jax.tree_util.tree_map(lambda p: p.astype(jnp.float32), params)
        best_params = params
        best_val, best_ep = jnp.inf, 0

        optimizer = optax.adamw(learning_rate=learning_rate(lr, lr_decay, -(-X.shape[0] // batch_size)), weight_decay=l2_reg)
        opt_state = optimizer.init(params)
        train_epoch = make_train_epoch(make_train_step(optimizer, bounded), batch_size)
        extra = (grad_weight, penalty_weight, l2_reg) if bounded else (grad_weight,)
        n_batches = -(-X.shape[0] // batch_size)

        for ep in range(1, epochs + 1):
            it_key, key = jax.random.split(key)
            # train_sums: summed training value / gradient MSE over the epoch's batches
            params, opt_state, train_sums = train_epoch(params, opt_state, it_key, tuple(arrays), extra)

            if ep % val_every == 0 or ep == 1 or ep == epochs:
                yp = batched_forward(params, Xv)
                gp = batched_grad_wrt_x(params, Xv)
                vm = jnp.mean((yp - yv) ** 2)
                gm = jnp.mean(jnp.sum((gp - gv) ** 2, axis=1))
                val_obj = value_weight * vm + grad_weight * gm
                if not np.isfinite(float(val_obj)):
                    raise FloatingPointError(f"Nonfinite validation loss at epoch {ep}")
                msg = f"Epoch {ep:4d} | val MSE: {vm:.4e} | grad MSE: {gm:.4e}"
                if val_data is not None:
                    tv, tg = (float(t) / n_batches for t in train_sums)
                    msg += f" | train MSE: {tv:.4e} | train grad MSE: {tg:.4e}"
                if bounded:
                    pm = jnp.mean(jax.nn.relu(yp - val_arrays[3]))
                    val_obj = val_obj + penalty_weight * pm
                    msg += f" | penalty MSE: {pm:.4e}"
                if keep_best and val_obj < best_val:
                    best_val, best_ep = val_obj, ep
                    best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
                if val_data is not None or ep % max(1, epochs // 10) == 0 or ep == 1:
                    print(msg)   # every validation with val_data, else every epochs/10

            if ep in save_at and ep != epochs and on_save is not None:
                on_save(ep, jax.tree_util.tree_map(lambda x: x.copy(), best_params if keep_best else params),
                        best_ep if keep_best else ep)

        if keep_best:
            print(f"best validation objective {float(best_val):.4e} at epoch {best_ep}")
        return best_params if keep_best else params

    def train_icnn_selfsup(
        X: np.ndarray,
        n_in: int,
        objective,
        rho: float,
        widths: List[int] = [64, 64, 64],
        lr: float = 1e-3,
        l2_reg: float = 0,
        batch_size: int = 128,
        epochs: int = 200,
        seed: int = 0,
        y: Optional[np.ndarray] = None,
        label_weight: float = 0.0,
        lr_decay: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Training without prox solves or gradient labels.  With p = x - ∇f_θ(x)/rho, the prox
        given by ∇ME = rho (x - prox(x)), the loss is
            mean[ objective(p) + rho/2 |p - x|² ]          (minimized at p = prox(x))
          + label_weight · mean[ (f_θ(x) - y)² ]           (given value labels y = ME(x))
        keep_best uses this loss."""

        def losses(params, xb, yb):
            p = xb - batched_grad_wrt_x(params, xb) / rho
            prox_obj = objective(p) + rho / 2 * jnp.sum((p - xb) ** 2, axis=1)
            label_gap = 0.0 if yb is None else jnp.mean((batched_forward(params, xb) - yb) ** 2)
            return jnp.mean(prox_obj), label_gap

        def loss(params, xb, yb):
            po, lg = losses(params, xb, yb)
            return po + label_weight * lg

        key = jax.random.PRNGKey(seed)
        Xj = jnp.asarray(X, dtype=jnp.float32)
        yj = None if y is None else jnp.asarray(y, dtype=jnp.float32)
        arrays = [Xj] if y is None else [Xj, yj]
        params = init_icnn_params(key, n_in=n_in, widths=widths)
        best_params, best_val = params, jnp.inf
        optimizer = optax.adamw(learning_rate=learning_rate(lr, lr_decay, -(-X.shape[0] // batch_size)), weight_decay=l2_reg)
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state, xb, yb=None):
            grads = jax.grad(loss)(params, xb, yb)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state

        for ep in range(1, epochs + 1):
            it_key, key = jax.random.split(key)
            for batch in batch_iterator(arrays, batch_size, it_key):
                params, opt_state = train_step(params, opt_state, *batch)

            if ep % max(1, epochs // 10) == 0 or ep == 1:
                po, lg = losses(params, Xj, yj)
                val_obj = po + label_weight * lg
                if keep_best and val_obj < best_val:
                    best_val = val_obj
                    best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
                print(f"Epoch {ep:4d} | prox objective: {po:.6e} | label gap: {lg:.4e}")

        return best_params if keep_best else params

    return SimpleNamespace(act_p=act_p, icnn_forward=icnn_forward, batched_forward=batched_forward,
                           grad_wrt_x=grad_wrt_x, batched_grad_wrt_x=batched_grad_wrt_x,
                           loss_fn=loss_fn, loss_fn_bounded=loss_fn_bounded, train_icnn=train_icnn,
                           train_icnn_selfsup=train_icnn_selfsup)


# -----------------------------
# Objectives for training without prox solves (train_icnn_selfsup): f summed over the input, per sample
# -----------------------------
def entr_max_objective(S0: float, eps: float = 1e-9):
    """x log(x/S0), the entr_max prox function (problems/entr_max/admm.jl), for x >= eps;
    a steep quadratic below eps, where the prox never lies."""
    def f(p):
        pc = jnp.maximum(p, eps)
        return jnp.sum(pc * jnp.log(pc / S0) + 1e3 * jax.nn.relu(eps - p) ** 2, axis=1)
    return f


def qp_objective(data: str):
    """1/2 x'Qx + p'x with the fixed diagonal Q and p of the QP family (problems/qp/problem.jl),
    read from a training data file (path relative to the repository root)."""
    d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", data))
    q, p = jnp.asarray(np.diag(d["Q"])), jnp.asarray(d["p"])
    def f(x):
        return jnp.sum(0.5 * q * x ** 2 + p * x, axis=1)
    return f


OBJECTIVES = {"entr_max": entr_max_objective, "qp": qp_objective}


# -----------------------------
# Batch setups
# -----------------------------
def batch_iterator(arrays, batch_size: int, shuffle_key: jax.Array):
    """
    Yields shuffled mini-batches of every array in `arrays` using pure JAX shuffling.
    """
    N = arrays[0].shape[0]
    perm = jax.random.permutation(shuffle_key, N)  # shape (N,)
    for s in range(0, N, batch_size):
        j = perm[s : s + batch_size]
        yield tuple(a[j] for a in arrays)


# -----------------------------
# Evaluation and export (used by train.py)
# -----------------------------
def save_lasso_model(params, path, A, G, h, rho, l1_weight, settings):
    """Save raw parameters and frozen geometry/settings for Python inference.

    Separate from the legacy Julia export: this model requires [q, x] and the
    fixed dictionary. load_lasso_model reconstructs the complete scalar model.
    """
    if not str(path).endswith(".npz"):
        raise ValueError("LASSO checkpoints must use the .npz extension")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {k: np.asarray(params[k]) for k in ("v", "a", "c")}
    for key in ("U", "W", "b"):
        arrays.update({f"{key}{i + 1}": np.asarray(x) for i, x in enumerate(params[key])})
    np.savez_compressed(path, **arrays, A=np.asarray(A), G=np.asarray(G), h=np.asarray(h),
                        rho=rho, l1_weight=l1_weight, layers=len(params["U"]),
                        settings_json=json.dumps(settings), format_version=1)


def load_lasso_model(path):
    """Return (model, params); model.predict(params, q, x) returns value, grad, prox.

    Construct inside jax.default_device(device) to place the model on a GPU.
    The predicted prox is q - rho * grad, with no feasibility guarantee.
    """
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != 1:
            raise ValueError("Unsupported LASSO checkpoint version")
        settings = json.loads(str(data["settings_json"]))
        rho, n = float(data["rho"]), data["A"].shape[1]
        dictionary = make_lasso_dictionary(data["A"], data["G"], data["h"], rho,
                                            float(data["l1_weight"]), **settings["dictionary"])
        model = make_icnn(settings["weight_act"], settings["keep_best"], settings.get("acts"),
                          query_dim=n, dictionary=dictionary, value_weight=settings["value_weight"])
        params = {k: jnp.asarray(data[k]) for k in ("v", "a", "c")}
        params.update({k: [jnp.asarray(data[f"{k}{i + 1}"]) for i in range(int(data["layers"]))]
                       for k in ("U", "W", "b")})

    @jax.jit
    def predict(params, q, x):
        packed = jnp.concatenate((q, x))
        value = model.icnn_forward(params, packed)
        grad = model.grad_wrt_x(params, packed)
        return value, grad, q - rho * grad

    model.predict = predict
    model.batched_predict = jax.jit(jax.vmap(predict, in_axes=(None, 0, 0)))
    return model, params


def report_test(icnn: SimpleNamespace, params, Xva, yva, gva, scale: float = 1.0) -> None:
    """Test errors of scale · ICNN against the labels (scale = ρ for a model trained on ME/ρ)."""
    y_pred = scale * icnn.batched_forward(params,    jnp.asarray(Xva))
    g_pred = scale * icnn.batched_grad_wrt_x(params, jnp.asarray(Xva))
    val_mse = jnp.mean((y_pred - jnp.asarray(yva)) ** 2)
    grad_mse = jnp.mean(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1))
    grad_max = jnp.sqrt(jnp.max(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1)))
    print(f"[TEST] value MSE: {val_mse:.4e} | grad MSE: {grad_mse:.4e} | grad MAX: {grad_max:.4e}")


def save_model(params, rho: float, path_to_save: str, export_act, out_scale: float = 1.0,
               acts: Optional[List[str]] = None) -> None:
    """Writes <path>.pkl and <path>.json (read by load_model in src/icnn.jl) with the
    projection export_act applied to v and the state weights W[1:], as the Julia ICNN
    uses the weights as they are.  A path ending in .npz writes that one file instead:
    U1.., W1.., b1.. (one per layer), v, a, c and rho, in float64.  out_scale multiplies the
    output layer (v, a, c), so the saved model is out_scale · ICNN (still convex for out_scale > 0)."""
    os.makedirs(os.path.dirname(os.path.abspath(path_to_save)), exist_ok=True)
    params["rho"] = rho
    params["v"]   = out_scale * export_act(params["v"])
    params["a"]   = out_scale * params["a"]
    params["c"]   = out_scale * params["c"]
    for i in range(1, len(params["W"])):
        params["W"][i] = export_act(params["W"][i])
    if path_to_save.endswith(".npz"):
        arrays = {"v": params["v"], "a": params["a"], "c": params["c"], "rho": rho}
        if "huber_B" in params:   # c scaled with the output; c and δ with the softplus applied
            arrays.update(huber_B=params["huber_B"], huber_d=params["huber_d"],
                          huber_c=out_scale * jax.nn.softplus(params["huber_c"]),
                          huber_delta=jax.nn.softplus(params["huber_delta"]))
        if acts is not None:   # per-layer activations for src/icnn.jl (absent: softplus everywhere)
            parsed = [parse_act(a) for a in acts]
            arrays["act_kind"] = np.array([ACT_KINDS[k] for k, _ in parsed])
            arrays["act_beta"] = np.array([beta for _, beta in parsed])
        for key in ("U", "W", "b"):
            arrays.update({f"{key}{i + 1}": x for i, x in enumerate(params[key])})
        np.savez(path_to_save, **{k: np.asarray(x, dtype=np.float64) for k, x in arrays.items()})
        return
    if acts is not None or "huber_B" in params:
        raise ValueError("non-default activations and Huber terms are saved only to .npz (src/icnn.jl reads them there)")
    with open(path_to_save + ".pkl", "wb") as f:
        pickle.dump(params, f)
    with open(path_to_save + ".json", 'w') as f:
        json.dump(to_serializable(params), f)


def to_serializable(obj):
    """Recursively convert JAX arrays (or numpy) to lists for JSON."""
    if isinstance(obj, (jax.Array, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    else:
        return obj  # numbers, strings, etc. stay the same
