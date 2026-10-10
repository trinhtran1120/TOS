# Quadratic-program example

Run commands from the repository root after `uv sync --locked`.

| Location | Purpose |
| --- | --- |
| `src/model/icnn.py`, `train.py` | ICNN implementation and training command |
| `src/qp/configs.json` | Training settings and input paths |
| `src/qp/checkpoints/` | Saved ICNN checkpoint |
| `data/qp/` | Saved Moreau labels and metadata |
| `results/qp/legacy/` | Archived benchmark outputs and [report](../../results/qp/legacy/report.md) |
| `results/qp/generated/` | New benchmark output, ignored by Git |
| `experiments/qp/benchmark/` | Standard, learned, and corrected benchmarks |
| `experiments/qp/problem/` | QP definition, reference solver, TOS, and correction |
| `experiments/qp/data_gen/` | Data generator and its settings |

## Reproduce the saved benchmark

The included dataset and checkpoint allow a benchmark without retraining:

```sh
uv run --locked python -m experiments.qp.benchmark.benchmark_corrected
```

This compares corrected learned TOS against OSQP on the first 10 held-out IDs and writes JSON under `results/qp/generated/<UTC timestamp>/corrected.json`. Timings depend on the machine. The archived [report](../../results/qp/legacy/report.md) covers an earlier run.

To debug one held-out instance:

```sh
uv run --locked python -m experiments.qp.benchmark.benchmark_corrected \
  --run-ids 69 --output results/qp/generated/debug-69.json
```

The JSON records solver settings, correction diagnostics, selected IDs, and SHA-256 hashes of the dataset and checkpoint. Both learned benchmarks accept `--run-ids`; `--samples` selects the first held-out IDs when no IDs are specified.

Other comparisons:

```sh
uv run --locked python -m experiments.qp.benchmark.benchmark_learned
uv run --locked python -m experiments.qp.benchmark.benchmark_standard
```

## Generate data and train

The full configuration requests 1,000 QP instances and 5,000 training epochs. For a separate run:

```sh
uv run --locked python -m experiments.qp.data_gen.gen_data \
  --output data/qp/generated/my_run/moreau_data_qp.npz
uv run --locked python -m model.train \
  --data data/qp/generated/my_run/moreau_data_qp_eta1.5_gamma1.npz \
  --checkpoint results/qp/generated/my_run/qp_icnn.npz
uv run --locked python -m experiments.qp.benchmark.benchmark_corrected \
  --data data/qp/generated/my_run/moreau_data_qp_eta1.5_gamma1.npz \
  --checkpoint results/qp/generated/my_run/qp_icnn.npz \
  --output results/qp/generated/my_run/corrected.json
```

The generator writes `.npz` data and neighboring `.json` metadata. Training uses `src/qp/configs.json` by default and accepts `--config`, `--data`, and `--checkpoint` overrides. Promote a validated checkpoint into `src/qp/checkpoints/`; keep trial results under `results/qp/generated/`.

The archived [research guide](docs/guide_qp.md) and [earlier notes](docs/AGENTS_qp.md) provide historical context; their old command examples are not current entry points.
