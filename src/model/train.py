"""Train an ICNN on value and input-gradient labels.

Run: uv run --locked python -m model.train --config src/qp/configs.json
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from model.icnn import init_icnn_params, predict_batch, save_checkpoint


def make_train_step(model, gradient_scale, gradient_weight, learning_rate):
    """Compile an Adam step through the input-gradient supervision."""
    def loss_fn(params, x, values, gradients):
        predicted_values, predicted_gradients = predict_batch(
            params, x, model["input_mean"], model["input_std"],
            model["output_scale"], model["output_offset"],
            model["weight_act"], model["nonnegative_output"],
        )
        value_mse = jnp.mean((predicted_values - values) ** 2)
        gradient_mse = jnp.mean(jnp.sum(
            (predicted_gradients - gradients) ** 2, axis=1
        ))
        loss = (
            value_mse / model["output_scale"] ** 2
            + gradient_weight * gradient_mse / gradient_scale**2
        )
        return loss, (value_mse, gradient_mse)

    @jax.jit
    def train_step(params, first, second, step, x, values, gradients):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, values, gradients
        )
        first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, first, grads)
        second = jax.tree.map(
            lambda v, g: 0.999 * v + 0.001 * g * g, second, grads
        )
        step += 1
        params = jax.tree.map(
            lambda p, m, v: p - learning_rate * (m / (1 - 0.9**step))
            / (jnp.sqrt(v / (1 - 0.999**step)) + 1e-8),
            params, first, second,
        )
        return params, first, second, step, jnp.stack(metrics)

    return train_step


def evaluate(params, model, x, values, gradients, batch_size):
    value_error = 0.0
    gradient_error = 0.0
    identity_error = 0.0
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        predicted_values, predicted_gradients = predict_batch(
            params, jnp.asarray(x[start:stop]),
            model["input_mean"], model["input_std"],
            model["output_scale"], model["output_offset"],
            model["weight_act"], model["nonnegative_output"],
        )
        value_error += float(jnp.sum(
            (predicted_values - jnp.asarray(values[start:stop])) ** 2
        ))
        gradient_error += float(jnp.sum(jnp.sum(
            (predicted_gradients - jnp.asarray(gradients[start:stop])) ** 2,
            axis=1,
        )))
        if "tau" in model:
            tau = model["tau"]
            identity_error += float(jnp.sum(
                (tau**2 * jnp.sum(predicted_gradients**2, axis=1)
                 - 2 * tau * predicted_values) ** 2
            ))
    result = {
        "value_rmse": np.sqrt(value_error / len(x)),
        "gradient_rmse": np.sqrt(gradient_error / len(x)),
    }
    if "tau" in model:
        result["projection_rmse"] = model["tau"] * result["gradient_rmse"]
        result["envelope_identity_rmse"] = np.sqrt(identity_error / len(x))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "qp/configs.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    data_config = config["data"]
    model_config = config["model"]
    training = config["training"]

    data_path = (args.data or config_path.parent / data_config["path"]).resolve()
    checkpoint_path = (args.checkpoint or config_path.parent / config["checkpoint"]).resolve()
    with np.load(data_path) as data:
        x = data[data_config["input_key"]].astype(np.float32)
        values = data[data_config["value_key"]].astype(np.float32)
        gradients = data[data_config["gradient_key"]].astype(np.float32)
        split = data[data_config["split_key"]]
        tau = float(data["tau"]) if "tau" in data else None

    train_rows = split == data_config["train_label"]
    validation_rows = split == data_config["validation_label"]
    test_rows = split == data_config["test_label"]
    x_train = x[train_rows]
    values_train = values[train_rows]
    gradients_train = gradients[train_rows]
    input_std = x_train.std(axis=0)
    input_std = np.where(input_std > 0, input_std, 1.0).astype(np.float32)
    nonnegative_output = model_config["nonnegative_output"]
    output_offset = 0.0 if nonnegative_output else float(values_train.mean())
    output_scale = float(np.sqrt(np.mean((values_train - output_offset) ** 2)))
    gradient_scale = jnp.asarray(np.sqrt(np.mean(
        np.sum(gradients_train**2, axis=1)
    )))
    model = {
        "input_mean": jnp.asarray(x_train.mean(axis=0)),
        "input_std": jnp.asarray(input_std),
        "output_scale": jnp.asarray(output_scale),
        "output_offset": jnp.asarray(output_offset),
        "input_dim": x.shape[1],
        "widths": model_config["widths"],
        "weight_act": model_config["weight_act"],
        "nonnegative_output": nonnegative_output,
    }
    if tau is not None:
        model["tau"] = tau

    x_train = jnp.asarray(x_train)
    values_train = jnp.asarray(values_train)
    gradients_train = jnp.asarray(gradients_train)
    key = jax.random.PRNGKey(training["seed"])
    key, init_key = jax.random.split(key)
    params = init_icnn_params(
        init_key, model["input_dim"], model["widths"], model["weight_act"]
    )
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    step = jnp.array(0)
    train_step = make_train_step(
        model, gradient_scale, training["gradient_weight"],
        training["learning_rate"],
    )
    batch_size = training["batch_size"]
    best_score = float("inf")
    best_params = params
    best_epoch = 0
    log_every = max(1, training["epochs"] // 10)
    print(
        f"training on {len(x_train)} samples, validating on "
        f"{int(validation_rows.sum())} samples every {log_every} epochs"
    )

    for epoch in range(1, training["epochs"] + 1):
        key, shuffle_key = jax.random.split(key)
        order = jax.random.permutation(shuffle_key, len(x_train))
        epoch_errors = jnp.zeros(2)
        for start in range(0, len(x_train), batch_size):
            indices = order[start:start + batch_size]
            params, first, second, step, batch_errors = train_step(
                params, first, second, step,
                x_train[indices], values_train[indices], gradients_train[indices],
            )
            epoch_errors = epoch_errors + len(indices) * batch_errors

        validation = evaluate(
            params, model, x[validation_rows], values[validation_rows],
            gradients[validation_rows], batch_size,
        )
        score = (
            validation["value_rmse"] ** 2 / output_scale**2
            + training["gradient_weight"]
            * validation["gradient_rmse"] ** 2 / float(gradient_scale) ** 2
        )
        if score < best_score:
            best_score, best_params, best_epoch = score, params, epoch
        if epoch == 1 or epoch % log_every == 0 or epoch == training["epochs"]:
            train_value_mse, train_gradient_mse = (
                np.asarray(epoch_errors) / len(x_train)
            )
            print(
                f"Epoch {epoch:4d} | val MSE: "
                f"{validation['value_rmse'] ** 2:.4e} | grad MSE: "
                f"{validation['gradient_rmse'] ** 2:.4e} | train MSE: "
                f"{train_value_mse:.4e} | train grad MSE: "
                f"{train_gradient_mse:.4e}",
                flush=True,
            )

    selected = best_params if model_config["keep_best"] else params
    selected_epoch = best_epoch if model_config["keep_best"] else training["epochs"]
    test = evaluate(
        selected, model, x[test_rows], values[test_rows],
        gradients[test_rows], batch_size,
    )
    save_checkpoint(checkpoint_path, selected, model)
    if model_config["keep_best"]:
        print(f"best validation objective {best_score:.4e} at epoch {best_epoch}")
    print(f"selected_epoch={selected_epoch} test={test}")
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
