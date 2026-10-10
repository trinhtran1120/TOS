# Three-operator splitting (TOS)

The [QP example](experiments/qp/README.md) is the runnable example in this repository. Run commands from the repository root.

```sh
uv sync --locked
uv run --locked python -m experiments.qp.benchmark.benchmark_corrected
```

## Repository layout

| Path | Role |
| --- | --- |
| `src/model/` | ICNN definition and training command |
| `src/qp/` | QP training config and saved ICNN checkpoint |
| `data/qp/` | QP dataset and metadata |
| `results/qp/` | Archived results and new benchmark output |
| `experiments/qp/benchmark/` | OSQP, standard TOS, and learned TOS comparisons |
| `experiments/qp/problem/` | QP definition, solvers, and correction |
| `experiments/qp/data_gen/` | Moreau-label generation and its config |

For another example, add a sibling to `experiments/qp/` and use corresponding subfolders in `data/` and `results/`. The QP README shows the generate → train → benchmark commands and how to inspect a single run. Python folders are namespace packages; there are no `__init__.py` files.
