"""Reusable smooth input-convex neural network (ICNN)."""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def init_icnn_params(key, input_dim, widths, weight_act="softplus"):
    """Initialize input skips, projected state weights, and scalar output."""
    keys = iter(jax.random.split(key, 2 * len(widths) + 2))
    input_weights, state_weights, biases = [], [], []
    raw_center = -3.0 if weight_act == "softplus" else 0.0

    for index, width in enumerate(widths):
        limit = jnp.sqrt(6.0 / (input_dim + width))
        input_weights.append(jax.random.uniform(
            next(keys), (width, input_dim), minval=-limit, maxval=limit
        ))
        if index:
            state_weights.append(
                raw_center + 0.05 * jax.random.normal(
                    next(keys), (width, widths[index - 1])
                )
            )
        biases.append(jnp.zeros(width))

    return {
        "U": tuple(input_weights),
        "W": tuple(state_weights),
        "b": tuple(biases),
        "v": raw_center + 0.05 * jax.random.normal(next(keys), (widths[-1],)),
        "a": 0.01 * jax.random.normal(next(keys), (input_dim,)),
        "c": jnp.array(0.0),
    }


def _positive_weight(raw, weight_act):
    if weight_act == "softplus":
        return jax.nn.softplus(raw)
    if weight_act == "relu":
        return jax.nn.relu(raw)
    raise ValueError(f"Unknown weight activation: {weight_act}")


def icnn_value(params, x, input_mean, input_std, output_scale,
               output_offset, weight_act="softplus", nonnegative_output=False):
    """Evaluate a convex scalar function in the original input coordinates.

    Hidden softplus activations are convex and nondecreasing. Nonnegative
    state and output feature weights preserve convexity. Affine input scaling
    and a positive output scale do too. The optional output softplus also
    enforces nonnegative values.
    """
    x = (x - input_mean) / input_std
    state = jax.nn.softplus(params["U"][0] @ x + params["b"][0])
    for index in range(1, len(params["U"])):
        weights = _positive_weight(params["W"][index - 1], weight_act)
        state = jax.nn.softplus(
            (weights / jnp.sqrt(state.shape[0])) @ state
            + params["U"][index] @ x + params["b"][index]
        )
    output_weights = _positive_weight(params["v"], weight_act)
    raw_value = (
        (output_weights / jnp.sqrt(state.shape[0])) @ state
        + params["a"] @ x + params["c"]
    )
    if nonnegative_output:
        return output_scale * jax.nn.softplus(raw_value)
    return output_offset + output_scale * raw_value


_value_and_gradient = jax.value_and_grad(icnn_value, argnums=1)


@partial(jax.jit, static_argnames=("weight_act", "nonnegative_output"))
def predict_batch(params, x_batch, input_mean, input_std, output_scale,
                  output_offset, weight_act="softplus", nonnegative_output=False):
    """Return values and input gradients in the original coordinates."""
    return jax.vmap(
        _value_and_gradient,
        in_axes=(None, 0, None, None, None, None, None, None),
    )(params, x_batch, input_mean, input_std, output_scale,
      output_offset, weight_act, nonnegative_output)


def save_checkpoint(path, params, model):
    """Save parameters and transformations needed for inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves, _ = jax.tree.flatten(params)
    arrays = {
        "input_mean": np.asarray(model["input_mean"]),
        "input_std": np.asarray(model["input_std"]),
        "output_scale": np.asarray(model["output_scale"]),
        "output_offset": np.asarray(model["output_offset"]),
        "input_dim": np.asarray(model["input_dim"]),
        "widths": np.asarray(model["widths"]),
        "weight_act": np.asarray(model["weight_act"]),
        "nonnegative_output": np.asarray(model["nonnegative_output"]),
    }
    if "tau" in model:
        arrays["tau"] = np.asarray(model["tau"])
    arrays.update({f"parameter_{i}": np.asarray(p) for i, p in enumerate(leaves)})
    np.savez_compressed(path, **arrays)


def load_checkpoint(path):
    """Load parameters and model metadata for predict_batch."""
    with np.load(path) as saved:
        widths = saved["widths"].tolist()
        weight_act = str(saved["weight_act"].item())
        template = init_icnn_params(
            jax.random.PRNGKey(0), int(saved["input_dim"]), widths, weight_act
        )
        _, structure = jax.tree.flatten(template)
        params = jax.tree.unflatten(
            structure,
            [jnp.asarray(saved[f"parameter_{i}"]) for i in range(structure.num_leaves)],
        )
        model = {
            "input_mean": jnp.asarray(saved["input_mean"]),
            "input_std": jnp.asarray(saved["input_std"]),
            "output_scale": jnp.asarray(saved["output_scale"]),
            "output_offset": jnp.asarray(saved["output_offset"]),
            "input_dim": int(saved["input_dim"]),
            "widths": widths,
            "weight_act": weight_act,
            "nonnegative_output": bool(saved["nonnegative_output"]),
        }
        if "tau" in saved:
            model["tau"] = float(saved["tau"])
    return params, model
