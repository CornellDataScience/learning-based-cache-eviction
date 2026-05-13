#!/usr/bin/env python3
"""Demo console for the cache-eviction simulator.

Examples:
  python3 analysis_outputs/demo_cli.py trace --format wiki --trace-path artifacts/traces/wiki_sampled_test_trace.tsv
  python3 analysis_outputs/demo_cli.py race --workload looping --cache-capacity 64 --key-space 128
  python3 analysis_outputs/demo_cli.py race --workload wiki --trace-path artifacts/traces/wiki_sampled_test_trace.tsv --cache-capacity 256
  python3 analysis_outputs/demo_cli.py sweep --workload wiki --trace-path artifacts/traces/wiki_sampled_test_trace.tsv --capacities 32,64,128,256 --plot
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lbce_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/lbce_xdg_cache")

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "debug" / "lbce"
OUT = ROOT / "analysis_outputs"

DEFAULT_POLICIES = ["fifo", "lru", "optimal"]
POLICY_COLORS = {
    "fifo": "#c44536",
    "lru": "#1f77b4",
    "learned": "#f4a261",
    "optimal": "#2a9d8f",
}


def build_if_needed() -> None:
    if not BIN.exists():
        subprocess.run(["cargo", "build", "--bin", "lbce"], cwd=ROOT, check=True)


def parse_metric(text: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}:\s+([0-9.]+)", text)
    if not match:
        raise ValueError(f"Could not parse {name!r} from simulator output:\n{text}")
    return float(match.group(1))


def simulator_args(args: argparse.Namespace) -> list[str]:
    extra = [
        "--workload",
        args.workload,
        "--cache-capacity",
        str(args.cache_capacity),
    ]
    if args.trace_path:
        extra += ["--trace-path", args.trace_path]
    if args.workload in {"looping", "zipf", "mixed"}:
        extra += ["--key-space", str(args.key_space)]
    if args.workload in {"looping", "zipf", "mixed", "phase"}:
        extra += ["--total-requests", str(args.total_requests)]
    if args.workload in {"zipf", "mixed"}:
        extra += ["--zipf-skew", str(args.zipf_skew), "--zipf-seed", str(args.zipf_seed)]
    if args.workload in {"bursty", "mixed"}:
        extra += [
            "--bursty-cycles",
            str(args.bursty_cycles),
            "--bursty-quiet",
            str(args.bursty_quiet),
            "--bursty-burst",
            str(args.bursty_burst),
            "--background-keys",
            str(args.background_keys),
        ]
    if args.workload in {"phase", "mixed"}:
        extra += ["--phase-count", str(args.phase_count), "--keys-per-phase", str(args.keys_per_phase)]
    if args.workload == "mixed":
        extra += ["--mixed-mode", args.mixed_mode]
    return extra


def run_policy(policy: str, args: argparse.Namespace) -> dict[str, object]:
    command = [str(BIN), "--policy", policy, *simulator_args(args)]
    if policy == "learned":
        command += ["--model", args.model, "--shortlist-k", str(args.shortlist_k)]
        if args.debug_learned:
            command.append("--debug-learned")
    live_logs = policy == "learned" and args.show_learned_logs
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None if live_logs else subprocess.PIPE,
        check=True,
    )
    return {
        "policy": policy,
        "hit_rate": parse_metric(completed.stdout, "Hit Rate"),
        "miss_rate": parse_metric(completed.stdout, "Miss Rate"),
        "hit_count": int(parse_metric(completed.stdout, "Hit count")),
        "eviction_count": int(parse_metric(completed.stdout, "Eviction Count")),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def bar(value: float, width: int = 34) -> str:
    filled = round(value * width)
    return "#" * filled + "-" * (width - filled)


def policies(args: argparse.Namespace) -> list[str]:
    chosen = list(DEFAULT_POLICIES)
    if args.include_learned:
        chosen.insert(2, "learned")
    return chosen


def print_race(rows: list[dict[str, object]], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for row in sorted(rows, key=lambda r: float(r["hit_rate"]), reverse=True):
        print(
            f"{str(row['policy']).upper():7} "
            f"{float(row['hit_rate']):.3f}  "
            f"{bar(float(row['hit_rate']))}  "
            f"hits={row['hit_count']} evictions={row['eviction_count']}"
        )

    lru = next((r for r in rows if r["policy"] == "lru"), None)
    learned = next((r for r in rows if r["policy"] == "learned"), None)
    optimal = next((r for r in rows if r["policy"] == "optimal"), None)
    if lru and learned:
        learned_delta = float(learned["hit_rate"]) - float(lru["hit_rate"])
        direction = "beats" if learned_delta >= 0 else "trails"
        print(
            f"\nClaim hook: LEARNED {direction} LRU by {abs(learned_delta):.3f} "
            "hit-rate points using no future oracle."
        )
        if optimal:
            oracle_gap = float(optimal["hit_rate"]) - float(learned["hit_rate"])
            print(f"Oracle context: OPTIMAL is still {oracle_gap:.3f} above LEARNED, so there is headroom.")
    elif lru and optimal:
        oracle_gap = float(optimal["hit_rate"]) - float(lru["hit_rate"])
        print(f"\nOracle context: OPTIMAL is {oracle_gap:.3f} above LRU, showing available headroom.")

def command_race(args: argparse.Namespace) -> None:
    build_if_needed()
    rows = [run_policy(policy, args) for policy in policies(args)]
    print_race(
        rows,
        f"Policy race: workload={args.workload}, cache_capacity={args.cache_capacity}",
    )
    if args.show_learned_logs:
        learned = next((r for r in rows if r["policy"] == "learned"), None)
        if learned and learned["stderr"]:
            print("\nLearned-policy logs")
            print(str(learned["stderr"]).strip())


def command_sweep(args: argparse.Namespace) -> None:
    build_if_needed()
    capacities = [int(item) for item in args.capacities.split(",") if item]
    rows: list[dict[str, object]] = []
    original_capacity = args.cache_capacity
    for capacity in capacities:
        args.cache_capacity = capacity
        for policy in policies(args):
            row = run_policy(policy, args)
            row["cache_capacity"] = capacity
            rows.append(row)
    args.cache_capacity = original_capacity

    csv_path = OUT / f"demo_sweep_{args.workload}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["policy", "cache_capacity", "hit_rate", "miss_rate", "hit_count", "eviction_count"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCapacity sweep: workload={args.workload}")
    print(f"CSV: {csv_path}")
    for capacity in capacities:
        subset = [r for r in rows if r["cache_capacity"] == capacity]
        print_race(subset, f"capacity={capacity}")

    if args.plot:
        render_sweep(rows, capacities, args)


def render_sweep(rows: list[dict[str, object]], capacities: list[int], args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy in policies(args):
        subset = [r for r in rows if r["policy"] == policy]
        ax.plot(
            [r["cache_capacity"] for r in subset],
            [r["hit_rate"] for r in subset],
            marker="o",
            linewidth=2,
            label=policy.upper(),
            color=POLICY_COLORS[policy],
        )
    ax.set_title(f"Hit rate sweep: {args.workload}")
    ax.set_xlabel("Cache capacity")
    ax.set_ylabel("Hit rate")
    max_hit = max(float(r["hit_rate"]) for r in rows)
    ax.set_ylim(0, min(1.02, max(0.25, max_hit * 1.25)))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = OUT / f"demo_sweep_{args.workload}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    print(f"Plot: {path}")


def command_trace(args: argparse.Namespace) -> None:
    keys, sizes = load_trace_keys(args.trace_path, args.format)
    counts = Counter(keys)
    total = len(keys)
    unique = len(counts)
    top = counts.most_common(8)
    repeat_rate = 1.0 - unique / total if total else 0.0
    print(f"\nTrace: {args.trace_path}")
    print(f"format={args.format} requests={total} unique_keys={unique} repeat_rate={repeat_rate:.3f}")
    if sizes:
        total_bytes = sum(sizes.values())
        print(f"unique_object_bytes={total_bytes}")
    print("\nTop keys")
    for key, count in top:
        print(f"{str(key)[:18]:>18}  {count:6}  {bar(count / total if total else 0, 24)}")


def load_trace_keys(path: str, trace_format: str) -> tuple[list[str], dict[str, int]]:
    keys: list[str] = []
    sizes: dict[str, int] = {}
    trace_path = Path(path)
    if not trace_path.exists():
        sibling_tsv = trace_path.with_suffix(".tsv")
        hint = f" Did you mean {sibling_tsv}?" if sibling_tsv.exists() else ""
        raise SystemExit(f"Trace file not found: {trace_path}.{hint}")

    with trace_path.open(newline="") as fh:
        if trace_format == "wiki":
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                key = row["hashed_host_path_query"]
                keys.append(key)
                sizes.setdefault(key, int(float(row.get("response_size") or 0)))
        elif trace_format == "custom":
            reader = csv.DictReader(fh)
            for row in reader:
                keys.append(row["key"])
        else:
            raise ValueError(f"unknown trace format: {trace_format}")
    return keys, sizes


def add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workload", default="looping", choices=["looping", "zipf", "bursty", "phase", "mixed", "custom", "wiki"])
    parser.add_argument("--cache-capacity", type=int, default=64)
    parser.add_argument("--total-requests", type=int, default=10000)
    parser.add_argument("--key-space", type=int, default=128)
    parser.add_argument("--trace-path", default="")
    parser.add_argument("--zipf-skew", type=float, default=1.2)
    parser.add_argument("--zipf-seed", type=int, default=7)
    parser.add_argument("--bursty-cycles", type=int, default=50)
    parser.add_argument("--bursty-quiet", type=int, default=32)
    parser.add_argument("--bursty-burst", type=int, default=16)
    parser.add_argument("--background-keys", type=int, default=32)
    parser.add_argument("--phase-count", type=int, default=4)
    parser.add_argument("--keys-per-phase", type=int, default=32)
    parser.add_argument("--mixed-mode", default="interleaved", choices=["concat", "interleaved"])
    parser.add_argument("--include-learned", action="store_true", help="include learned policy; slower because it may retrain online")
    parser.add_argument("--model", default="artifacts/models/eviction_mlp.pt")
    parser.add_argument("--shortlist-k", type=int, default=4)
    parser.add_argument("--debug-learned", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo CLI for cache-eviction simulator experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    race = subparsers.add_parser("race", help="compare policies for one config")
    add_common_flags(race)
    race.add_argument("--show-learned-logs", action="store_true")
    race.set_defaults(func=command_race)

    sweep = subparsers.add_parser("sweep", help="sweep cache capacities and optionally render a plot")
    add_common_flags(sweep)
    sweep.add_argument("--capacities", default="8,16,32,64,96")
    sweep.add_argument("--plot", action="store_true")
    sweep.set_defaults(func=command_sweep)

    trace = subparsers.add_parser("trace", help="summarize a real/custom trace file")
    trace.add_argument("--trace-path", required=True)
    trace.add_argument("--format", default="wiki", choices=["wiki", "custom"])
    trace.set_defaults(func=command_trace)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
