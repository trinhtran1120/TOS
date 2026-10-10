# Projection QP experiment layout

- Keep the projection QP implementation, tests, and problem configuration JSON files in `src/qp/`.
- Define problem dimensions and matrices in a JSON configuration file.
- Save generated problem and query data as `.npz` files in `src/dataset/`.
- Save per-query results as `.csv` files in `src/results/qp/`.
- Save the result report as a `.md` file beside the CSV in `src/results/qp/`.

Follow [`guide_qp.md`](../../src/qp/guide_qp.md) and [`AGENTS_qp.md`](../../src/qp/AGENTS_qp.md) for the experiment method and validation requirements.
