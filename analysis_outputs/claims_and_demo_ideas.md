# Cache Eviction Demo CLI

Run commands from the repo root.

## 1. Trace summary

```bash
python3 analysis_outputs/demo_cli.py trace --format wiki --trace-path wiki_sampled_test_trace.tsv
```

Shows request count, number of unique keys, repeat rate, and top keys.

## 2. Policy race on a synthetic workload

```bash
python3 analysis_outputs/demo_cli.py race --workload zipf --cache-capacity 32 --key-space 256 --include-learned
```

Compares FIFO, LRU, learned, and optimal.

## 3. Policy race on the Wiki trace

```bash
python3 analysis_outputs/demo_cli.py race --workload wiki --trace-path wiki_sampled_test_trace.tsv --cache-capacity 256 --include-learned
```

This can take a while because the learned policy may retrain online.

## 4. Capacity sweep with a plot

```bash
python3 analysis_outputs/demo_cli.py sweep --workload wiki --trace-path wiki_sampled_test_trace.tsv --capacities 64,128,256,512,1024 --plot
```

Writes:

```text
analysis_outputs/demo_sweep_wiki.csv
analysis_outputs/demo_sweep_wiki.png
```

## Notes

- `key_space` is the number of distinct keys in a synthetic workload.
- `optimal` is Belady's oracle policy. It uses future knowledge and is only an upper bound.
- The learned policy is the system demo target. It uses observed history, an LRU shortlist, and pairwise model votes.
