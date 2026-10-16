"""Train a conditional LASSO Moreau-envelope ICNN, or a legacy problem model.

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

Legacy problems: entr_max, mpc, power_grid, mvee. The config sets data and output
(<output>.json is read by load_model in src/icnn.jl, <output>.pkl keeps the JAX
parameters; an output ending in .npz is written as that single file, which load_model
also reads), the training choices of make_icnn (weight_act, keep_best), the projection
applied to the exported weights (export_act), whether the loss uses the lower bounds
`org_f` of the data (lower_bound), and the arguments of train_icnn (train).  A config with
"self_supervised": {"objective": <name in icnn.OBJECTIVES>, ...its arguments} trains without
prox solves or gradient labels instead (train_icnn_selfsup, whose arguments are then in train);
a "label_weight" > 0 in train adds the ME labels as a supervised value term.  The labels of the
test data are still used by report_test.
Paths in a config are relative to the repository root.
"""

import argparse
import json
import os
import time

import jax
import numpy as np

import jax.numpy as jnp

from icnn import (OBJECTIVES, WEIGHT_ACTS, make_icnn, make_lasso_dictionary,
                  report_test, save_model, save_lasso_model)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


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
    train_args = dict(settings["train"])
    if args.epochs is not None:
        train_args["epochs"] = args.epochs
    if args.init is not None:
        train_args["init"] = args.init
    settings["train"] = train_args
    settings["device"] = args.device or settings["device"]
    device = select_device(settings["device"])
    print(f"Training device: {device}")
    path = os.path.join(REPO, config.output_dir, config.output_file)
    output = os.path.join(REPO, args.output or settings["output"])
    if not output.endswith(".npz"):
        raise ValueError("LASSO output must end in .npz")
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
            np.testing.assert_allclose(gradients, (q - prox) / rho, rtol=1e-7, atol=1e-8)
            splits[split] = (np.concatenate((q, x), axis=1), values, gradients)
            instances.append({tuple(row) for row in x})
        if any(instances[i] & instances[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Instance x appears in multiple dataset splits")
        test_prox = archive["minimizer_test"]

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
    ap.add_argument("problem", nargs="?", default="lasso", help="problem name (default: lasso)")
    ap.add_argument("--config", help="LASSO configuration JSON (default: src/lasso/configs.json)")
    ap.add_argument("--device", choices=("gpu", "cpu"), help="override LASSO training backend")
    ap.add_argument("--epochs", type=int, help="override the number of epochs")
    ap.add_argument("--init", choices=("default", "fan_in"), help="override the ICNN initialisation (see init_icnn_params)")
    ap.add_argument("--output", help="override the output path (without extension)")
    a = ap.parse_args()
    if a.problem == "lasso":
        train_lasso(a)
        return

    with open(os.path.join(CONFIG_DIR, f"{a.problem}.json")) as fh:
        cfg = json.load(fh)
    train_args = dict(cfg["train"])
    if a.epochs is not None:
        train_args["epochs"] = a.epochs
    if a.init is not None:
        train_args["init"] = a.init
    output = a.output or os.path.join(REPO, cfg["output"])

    print(jax.devices())
    data_train = np.load(os.path.join(REPO, cfg["train_data"]))
    data_test = np.load(os.path.join(REPO, cfg["test_data"]))
    Xtr, ytr, gtr = data_train["input"].T, data_train["enve"], data_train["grad"].T
    Xva, yva, gva = data_test["input"].T, data_test["enve"], data_test["grad"].T
    n, N = Xtr.shape[1], Xtr.shape[0]
    labels_test = (yva, gva)   # original units, for the final reports
    # scale_by_rho: train on ME/ρ and ∇ME/ρ (the envelope of f/ρ at ρ = 1, gradient q - prox(q));
    # the saved model has its output layer multiplied by ρ, so it still computes ME.
    out_scale = float(data_train["rho"]) if cfg.get("scale_by_rho", False) else 1.0
    if out_scale != 1.0:
        ytr, gtr, yva, gva = ytr / out_scale, gtr / out_scale, yva / out_scale, gva / out_scale
        print(f"training on ME/rho and grad ME/rho (rho = {out_scale:g}); logged MSEs are in these units (x rho^2 = original)")
    print(f"Number of data: {N}")

    acts = cfg.get("acts")   # per-layer hidden activations (default: softplus everywhere)
    icnn = make_icnn(cfg["weight_act"], keep_best=cfg["keep_best"], acts=acts)
    if "self_supervised" in cfg:   # no gradient labels; the ME labels only with label_weight > 0
        ss = dict(cfg["self_supervised"])
        objective = OBJECTIVES[ss.pop("objective")](**ss)
        use_labels = train_args.get("label_weight", 0) > 0   # the ME labels (not the gradient labels)
        params = icnn.train_icnn_selfsup(Xtr, n_in=n, objective=objective, rho=data_train["rho"].item(),
                                         y=ytr if use_labels else None, **train_args)
    else:
        if cfg["lower_bound"]:
            train_args["f"] = data_train["org_f"]
        if cfg.get("validate_on_test", False):   # select the best epoch on the test data
            train_args["val_data"] = (Xva, yva, gva)
        if train_args.get("save_at"):   # <output>-<epoch>ep.npz at each save_at epoch and at the end
            base = output[:-len(".npz")] if output.endswith(".npz") else output
            output = f"{base}-{train_args['epochs']}ep.npz"

            def on_save(ep, snapshot, best_ep):
                path = f"{base}-{ep}ep.npz"
                print(f"epoch {ep}: model from epoch {best_ep}")
                report_test(icnn, snapshot, Xva, *labels_test, scale=out_scale)
                save_model(snapshot, data_train["rho"].item(), path, export_act=WEIGHT_ACTS[cfg["export_act"]],
                           out_scale=out_scale, acts=acts)
                print(f"saved {path}")
            train_args["on_save"] = on_save
        params = icnn.train_icnn(Xtr, ytr, gtr, n_in=n, **train_args)

    report_test(icnn, params, Xva, *labels_test, scale=out_scale)
    save_model(params, data_train["rho"].item(), output, export_act=WEIGHT_ACTS[cfg["export_act"]],
               out_scale=out_scale, acts=acts)
    print(f"saved {output}" if output.endswith(".npz") else f"saved {output}.json and {output}.pkl")


if __name__ == "__main__":
    main()
