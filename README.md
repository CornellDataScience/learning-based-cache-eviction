# Adaptive Cache

A learning-based cache eviction simulator, dataset pipeline, and model training stack.

## Repo layout

- `src/`: Rust simulator, policies, workloads, and dataset builders
- `pytorch_model/`: Python training, sweep, and evaluation scripts
- `analysis_outputs/`: demo CLI and presentation/demo helpers
- `artifacts/models/`: offline checkpoints and normalization metadata
- `artifacts/datasets/`: generated train/validation/test CSVs
- `artifacts/traces/`: sampled real-world traces used by the simulator and builders
- `scripts/`: one-off data preparation helpers

## Common commands

Generate training data:

```bash
cargo run --bin pairwise_training_dataset_builder -- artifacts/traces/wiki_sampled_train_trace.tsv
```

Generate validation and test data:

```bash
cargo run --bin pairwise_evaluation_datasets_builder -- val:artifacts/traces/wiki_sampled_val_trace.tsv test:artifacts/traces/wiki_sampled_test_trace.tsv
```

Train the offline model:

```bash
python3 pytorch_model/model.py
```

Run the simulator demo CLI:

```bash
python3 analysis_outputs/demo_cli.py race --workload mixed --mixed-mode interleaved --cache-capacity 64 --include-learned
```
