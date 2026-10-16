"""Train the conditional LASSO Moreau-envelope ICNN.

    uv run python src/train.py lasso                 # src/lasso/configs.json; GPU required
    uv run python src/train.py lasso --device cpu --epochs 1  # CPU smoke check

LASSO reads src/lasso/configs.json (or --config). Its training section controls
the backend, fixed dictionary weights, loss weights, optimizer and checkpoint.
Inputs are [q, x], with fixed matrices from the dataset; outputs are a scalar
envelope and its q derivative. Only the ICNN parameters are optimized. Validation
selects the best epoch; test labels are used only for final metrics. The NPZ
checkpoint reloads with icnn.load_lasso_model, including frozen atom geometry.
Native Windows JAX cannot use NVIDIA CUDA; run GPU training in Linux/WSL2 with
CUDA-enabled JAX. --device cpu is an explicit opt-in for CPU execution.

Paths in a config are relative to the repository root.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from icnn import make_icnn, make_lasso_dictionary, save_lasso_model

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def select_device(platform):
    """Require the selected backend, rather than silently training on CPU."""
    try:
        devices = jax.devices(platform)
    except RuntimeError as error:
        raise RuntimeError(
            f"JAX {platform!r} backend unavailable. NVIDIA GPU training requires "
            "CUDA-enabled JAX on Linux/WSL2. Use --device cpu for an explicit CPU run."
        ) from error
    if not devices:
        raise RuntimeError(f"No JAX {platform} devices found")
    return devices[0]


def train_lasso(args):
    """Train with fixed shared matrices, instance x and query q from the NPZ."""
    from lasso.problem import load_config

    config = load_config(args.config)
    settings = dict(config.training)
    required = {"train", "device", "output", "dictionary", "weight_act", "keep_best", "value_weight"}
    missing = required - settings.keys()
    if missing:
        raise ValueError(f"Missing training configuration keys: {', '.join(sorted(missing))}")
    train_args = dict(settings["train"])
    if args.epochs is not None:
        train_args["epochs"] = args.epochs
    if args.init is not None:
        train_args["init"] = args.init
    settings["train"] = train_args
    settings["device"] = args.device or settings["device"]
    if settings["device"] not in ("gpu", "cpu"):
        raise ValueError("training.device must be 'gpu' or 'cpu'")
    device = select_device(settings["device"])
    print(f"Training device: {device}")
    path = os.path.join(REPO, config.output_dir, config.output_file)
    output = os.path.join(REPO, args.output or settings["output"])
    settings["output"] = output
    if not output.endswith(".npz"):
        raise ValueError("LASSO output must end in .npz")
    if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(output)):
        raise ValueError("Model output must differ from the input dataset path")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. Generate it with "
            "uv run python -m lasso.gen_data"
            + (f" --config {args.config}" if args.config else "")
        )
    print(f"Loading dataset: {path}")
    with np.load(path, allow_pickle=False) as archive:
        A, G, h = (archive[k] for k in ("A", "G", "h"))
        rho, l1_weight = float(archive["rho"]), float(archive["l1_weight"])
        if (str(archive["gradient_variable"]) != "q" or bool(archive["normalized"])
                or str(archive["envelope_convention"]) != "1/(2*rho)"):
            raise ValueError("Expected original-unit envelope labels and gradients w.r.t. q")
        if (A.shape != (config.n_eq, config.n_var)
                or G.shape != (config.n_ineq, config.n_var) or h.shape != (config.n_ineq,)
                or rho != config.rho or l1_weight != config.l1_weight):
            raise ValueError("Dataset dimensions/rho/l1_weight disagree with configuration")
        splits = {}
        instances = []
        for split in ("train", "val", "test"):
            q, x, values, gradients, prox = (
                archive[f"{k}_{split}"] for k in ("q", "x", "moreau", "gradient", "minimizer"))
            N = len(q)
            if (N == 0 or q.shape != (N, config.n_var) or x.shape != (N, config.n_eq)
                    or values.shape != (N,) or gradients.shape != q.shape or prox.shape != q.shape
                    or not all(np.isfinite(a).all() for a in (q, x, values, gradients, prox))):
                raise ValueError(f"Invalid or empty {split} data")
            if not np.allclose(gradients, (q - prox) / rho, rtol=1e-7, atol=1e-8):
                raise ValueError(f"{split} gradient labels disagree with (q - minimizer) / rho")
            splits[split] = (np.concatenate((q, x), axis=1), values, gradients)
            instances.append({tuple(row) for row in x})
        if any(instances[i] & instances[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Instance x appears in multiple dataset splits")
        test_prox = archive["minimizer_test"]
    print("Samples: " + ", ".join(f"{name}={len(data[0])}" for name, data in splits.items()))

    # Compile the existing scan-based epoch on this device. Inputs and all
    # initialized parameters/optimizer state stay on the same device in float32.
    with jax.default_device(device):
        dictionary = make_lasso_dictionary(A, G, h, rho, l1_weight, **settings["dictionary"])
        model = make_icnn(settings["weight_act"], settings["keep_best"], settings.get("acts"),
                          query_dim=config.n_var, dictionary=dictionary,
                          value_weight=settings["value_weight"])
        params = model.train_icnn(*splits["train"], n_in=config.n_var + config.n_eq,
                                  val_data=splits["val"], **train_args)
        X, values, gradients = (jnp.asarray(a, dtype=jnp.float32) for a in splits["test"])

        @jax.jit
        def evaluate(p, packed):
            return model.batched_forward(p, packed), model.batched_grad_wrt_x(p, packed)

        jax.block_until_ready(evaluate(params, X))  # warm-up before synchronized timing
        start = time.perf_counter()
        predicted_values, predicted_gradients = jax.block_until_ready(evaluate(params, X))
        elapsed = time.perf_counter() - start
        value_error = np.asarray(predicted_values) - np.asarray(values)
        grad_error = np.asarray(predicted_gradients) - np.asarray(gradients)
        predicted_prox = np.asarray(X[:, :config.n_var]) - rho * np.asarray(predicted_gradients)
        prox_error = predicted_prox - test_prox
        metrics = {
            "value_mae": float(np.mean(np.abs(value_error))),
            "value_relative_l2": float(np.linalg.norm(value_error) / max(np.linalg.norm(values), 1e-12)),
            "gradient_mean_l2": float(np.mean(np.linalg.norm(grad_error, axis=1))),
            "gradient_max_linf": float(np.max(np.abs(grad_error))),
            "prox_mean_l2": float(np.mean(np.linalg.norm(prox_error, axis=1))),
            "prox_max_linf": float(np.max(np.abs(prox_error))),
            "batch_seconds": elapsed,
        }
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Nonfinite test predictions")
        print("[TEST] " + json.dumps(metrics, indent=2))
        save_lasso_model(params, output, A, G, h, rho, l1_weight, settings)
    with open(output + ".metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(f"saved {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", nargs="?", default="lasso", choices=("lasso",),
                    help="problem name (default: lasso)")
    ap.add_argument("--config", help="LASSO configuration JSON (default: src/lasso/configs.json)")
    ap.add_argument("--device", choices=("gpu", "cpu"), help="override LASSO training backend")
    ap.add_argument("--epochs", type=int, help="override the number of epochs")
    ap.add_argument("--init", choices=("default", "fan_in"), help="override the ICNN initialisation (see init_icnn_params)")
    ap.add_argument("--output", help="override the model checkpoint path (must end in .npz)")
    a = ap.parse_args()
    if a.epochs is not None and a.epochs < 1:
        ap.error("--epochs must be positive")
    try:
        train_lasso(a)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        ap.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
