# Quadratic-program experiment

Run these commands from the repository root after `uv sync --locked`.

| Location | Purpose |
| --- | --- |
| `src/qp/problem.py` | QP construction and objective |
| `src/qp/reference.py` | OSQP reference solve |
| `src/qp/tos.py` | Standard TOS and exact inequality projection |
| `src/qp/learned.py` | Learned projection and TOS |
| `src/qp/correction.py` | L-BFGS correction of learned projections |
| `src/qp/corrected.py` | Corrected TOS and batch execution |
| `src/qp/icnn.py` | ICNN model and checkpoint format |
| `src/qp/commands/` | Data generation, training, and benchmark commands |
| `src/qp/configs/` | Data generation and training settings |
| `assets/qp/datasets/` | Saved Moreau labels and metadata |
| `assets/qp/checkpoints/` | Saved ICNN checkpoint |
| `runs/qp/legacy/` | Archived benchmark outputs |
| `runs/qp/generated/` | New benchmark outputs, ignored by Git |

## Reproduce the saved benchmark

The included dataset and checkpoint allow a benchmark without retraining:

```sh
uv run --locked python -m qp.commands.benchmark_corrected
```

This selects the first 10 held-out run IDs, compares corrected learned TOS against OSQP, prints per-instance timings and residuals, and saves a new JSON result under `runs/qp/generated/<UTC timestamp>/corrected.json`. See the [archived report](reports/legacy.md) for the earlier run. Timings depend on the machine.

Other comparisons:

```sh
uv run --locked python -m qp.commands.benchmark_learned
uv run --locked python -m qp.commands.benchmark_standard
```

## Generate data and train

The full data configuration requests 1,000 QP instances and training requests 5,000 epochs; both can take substantial time. To create a new dataset, checkpoint, and benchmark result in one directory:

```sh
mkdir -p runs/qp/generated/my_run
uv run --locked python -m qp.commands.generate_data --output runs/qp/generated/my_run/moreau_data_qp.npz
uv run --locked python -m qp.commands.train \
  --data runs/qp/generated/my_run/moreau_data_qp_eta1.5_gamma1.npz \
  --checkpoint runs/qp/generated/my_run/qp_icnn.npz
uv run --locked python -m qp.commands.benchmark_corrected \
  --data runs/qp/generated/my_run/moreau_data_qp_eta1.5_gamma1.npz \
  --checkpoint runs/qp/generated/my_run/qp_icnn.npz \
  --output runs/qp/generated/my_run/corrected.json
```

`generate_data.py` writes an `.npz` dataset and a neighboring `.json` metadata file. The benchmark reads both. The training command accepts `--config` to choose another training configuration; `--data` and `--checkpoint` override its paths. The benchmark flags expose solver tolerances and correction settings (`--help` lists them). Use `--output` for a stable path when comparing parameter choices.

Once a dataset and checkpoint are chosen for shared experiments, copy them and the dataset metadata into `assets/qp/`; keep exploratory runs under `runs/qp/generated/`.

The archived [research guide](docs/guide_qp.md) and [earlier notes](docs/AGENTS_qp.md) are preserved for context; their historical command examples are not the current entry points.
