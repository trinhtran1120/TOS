# QP experiment layout

- Keep the ICNN and training command in [`src/model/`](../../src/model/).
- Keep QP training settings and the saved checkpoint in [`src/qp/`](../../src/qp/).
- Keep the dataset in [`data/qp/`](../../data/qp/) and benchmark output in [`results/qp/`](../../results/qp/).
- Keep benchmark commands in [`experiments/qp/benchmark/`](../../experiments/qp/benchmark/), problem solvers in [`experiments/qp/problem/`](../../experiments/qp/problem/), and data generation in [`experiments/qp/data_gen/`](../../experiments/qp/data_gen/).

The [QP README](../../experiments/qp/README.md) lists the current commands. The [research guide](../../experiments/qp/docs/guide_qp.md) and [earlier notes](../../experiments/qp/docs/AGENTS_qp.md) are archived context.

Place future examples beside `qp/` under `experiments/`, `data/`, and `results/`.
