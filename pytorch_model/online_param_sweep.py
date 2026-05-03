import argparse
import re
import shutil
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEARNED_POLICY_SRC = REPO_ROOT / "src" / "policies" / "learnedpolicy.rs"
WIKI_TRACE = REPO_ROOT / "wiki_sampled_test_trace.tsv"

RETRAIN_EVERY_VALUES = [25, 50, 100, 200, 500, 1000]

# starts at 500 bc thats the min buffer size and caps at 1000 bc thats the replay buffer capacity
RETRAIN_SAMPLE_SIZE_VALUES = [500, 1000, 2000, 5000, 10000]

BASE_WORKLOADS = [
    (
        "looping",
        ["--workload", "looping", "--total-requests", "30000", "--key-space", "256"],
    ),
    (
        "zipf",
        [
            "--workload",
            "zipf",
            "--total-requests",
            "30000",
            "--key-space",
            "256",
            "--zipf-skew",
            "1.2",
        ],
    ),
    ("bursty", ["--workload", "bursty", "--bursty-cycles", "60"]),
    (
        "phase",
        ["--workload", "phase", "--total-requests", "30000", "--phase-count", "4"],
    ),
    (
        "mixed",
        [
            "--workload",
            "mixed",
            "--total-requests",
            "30000",
            "--mixed-mode",
            "interleaved",
        ],
    ),
]

WIKI_WORKLOAD = ("wiki", ["--workload", "wiki", "--trace-path", str(WIKI_TRACE)])

HIT_RATE_RE = re.compile(r"hit[_\s]?rate\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RETRAIN_EVERY_RE = re.compile(r"const RETRAIN_EVERY:\s*u64\s*=\s*\d+\s*;")
RETRAIN_SAMPLE_RE = re.compile(r"const RETRAIN_SAMPLE_SIZE:\s*usize\s*=\s*\d+\s*;")


def patch_source(retrain_every: int, retrain_sample_size: int) -> str:
    original = LEARNED_POLICY_SRC.read_text()

    patched = RETRAIN_EVERY_RE.sub(
        f"const RETRAIN_EVERY: u64 = {retrain_every};", original
    )
    patched = RETRAIN_SAMPLE_RE.sub(
        f"const RETRAIN_SAMPLE_SIZE: usize = {retrain_sample_size};", patched
    )

    if patched == original:
        print(
            "source file unchanged, check that the constant names match",
            file=sys.stderr,
        )

    LEARNED_POLICY_SRC.write_text(patched)
    return original


def restore_source(original_text: str) -> None:
    LEARNED_POLICY_SRC.write_text(original_text)


def build(dry_run: bool) -> bool:
    cmd = ["cargo", "build", "--release"]
    if dry_run:
        print(f"dry run would run: {' '.join(cmd)}")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"cargo build failed ({elapsed:.1f}s):", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        return False

    print(f"build ok ({elapsed:.1f}s)")
    return True


def run_workload(
    binary: Path,
    model_path: str,
    workload_args: list[str],
    cache_capacity: int,
    dry_run: bool,
    verbose: bool = False,
) -> float | None:
    cmd = [
        str(binary),
        "--policy",
        "learned",
        "--model",
        model_path,
        "--cache-capacity",
        str(cache_capacity),
    ] + workload_args

    if dry_run:
        print(f"dry run would run: {' '.join(cmd)}")
        return None

    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300
    )

    if result.returncode != 0:
        print(f"binary failed (rc={result.returncode})", file=sys.stderr)
        print("stderr:", result.stderr[-500:], file=sys.stderr)
        return None

    if verbose and result.stderr:
        retrain_lines = [
            line for line in result.stderr.split("\n") if "online-learning" in line
        ]
        if retrain_lines:
            for line in retrain_lines:
                print(f"    {line}")

    m = HIT_RATE_RE.search(result.stdout)
    if m:
        return float(m.group(1))

    print("could not parse hit_rate from output:", file=sys.stderr)
    print(result.stdout[:400], file=sys.stderr)
    return None


def fmt_hit(val: float | None) -> str:
    return f"{val:.4f}" if val is not None else "  n/a "


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep RETRAIN_EVERY × RETRAIN_SAMPLE_SIZE"
    )
    parser.add_argument(
        "--binary",
        default=str(REPO_ROOT / "target" / "release" / "lbce"),
        help="Path to compiled binary",
    )
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "eviction_mlp.pt"),
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--cache-capacity",
        type=int,
        default=64,
        help="Cache capacity for all workloads (default: 64)",
    )
    parser.add_argument(
        "--skip-wiki",
        action="store_true",
        help="Skip wiki workload (e.g. if TSV is unavailable)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands but do not execute"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show retrain messages and other debug output",
    )
    args = parser.parse_args()

    binary = Path(args.binary)

    workloads = list(BASE_WORKLOADS)
    if not args.skip_wiki:
        if WIKI_TRACE.exists():
            workloads.append(WIKI_WORKLOAD)
        else:
            print(
                f"wiki trace not found at {WIKI_TRACE}, skipping"
            )

    workload_names = [name for name, _ in workloads]
    configs = list(product(RETRAIN_EVERY_VALUES, RETRAIN_SAMPLE_SIZE_VALUES))
    total_builds = len(configs)
    total_runs = total_builds * len(workloads)

    print(
        f"Sweep: {len(RETRAIN_EVERY_VALUES)} × {len(RETRAIN_SAMPLE_SIZE_VALUES)} = {total_builds} configs"
    )
    print(f"Workloads ({len(workloads)}): {', '.join(workload_names)}")
    print(f"Total runs: {total_runs}  (each config requires a recompile)")
    print(f"Binary: {binary}")
    print(f"Model:  {args.model}")
    print(f"Cache capacity: {args.cache_capacity}")
    print()

    results: dict[tuple[int, int], dict[str, float | None]] = {}

    original_src: str | None = None
    try:
        for i, (retrain_every, retrain_sample_size) in enumerate(configs, 1):
            label = f"RETRAIN_EVERY={retrain_every:5d}  RETRAIN_SAMPLE_SIZE={retrain_sample_size:5d}"
            print(f"[{i:2d}/{total_builds}] {label}")

            original_src = patch_source(retrain_every, retrain_sample_size)

            ok = build(dry_run=args.dry_run)
            if not ok:
                print(f"skip build failed, skipping workloads for this config")
                results[(retrain_every, retrain_sample_size)] = {
                    name: None for name in workload_names
                }
                restore_source(original_src)
                original_src = None
                continue

            hit_rates: dict[str, float | None] = {}
            for wl_name, wl_args in workloads:
                hr = run_workload(
                    binary=binary,
                    model_path=args.model,
                    workload_args=wl_args,
                    cache_capacity=args.cache_capacity,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )
                hit_rates[wl_name] = hr
                print(f"    {wl_name:20s} hit_rate={fmt_hit(hr)}")

            results[(retrain_every, retrain_sample_size)] = hit_rates

            restore_source(original_src)
            original_src = None
            print()

    except KeyboardInterrupt:
        print("\n[interrupted] restoring source file...")
    finally:
        if original_src is not None:
            restore_source(original_src)
            print("restored learnedpolicy.rs restored to original")

    if args.dry_run:
        print("dry run done, no results to report")
        return

    col_w = 10
    header_parts = [f"{'retrain_every':>13}", f"{'sample_size':>11}"]
    for wl in workload_names:
        header_parts.append(f"{wl:>{col_w}}")
    header_parts.append(f"{'mean_hr':>{col_w}}")
    print(
        "="
        * (13 + 11 + col_w * (len(workload_names) + 1) + 4 * (len(workload_names) + 2))
    )
    print("RESULTS")
    print(
        "="
        * (13 + 11 + col_w * (len(workload_names) + 1) + 4 * (len(workload_names) + 2))
    )
    print("  ".join(header_parts))
    print(
        "-"
        * (13 + 11 + col_w * (len(workload_names) + 1) + 4 * (len(workload_names) + 2))
    )

    ranked: list[tuple[float, tuple[int, int]]] = []

    for (re_val, rs_val), hit_rates in results.items():
        values = [hit_rates.get(wl) for wl in workload_names]
        valid = [v for v in values if v is not None]
        mean_hr = sum(valid) / len(valid) if valid else None

        row = [f"{re_val:>13d}", f"{rs_val:>11d}"]
        for v in values:
            row.append(f"{fmt_hit(v):>{col_w}}")
        row.append(f"{fmt_hit(mean_hr):>{col_w}}")
        print("  ".join(row))

        if mean_hr is not None:
            ranked.append((mean_hr, (re_val, rs_val)))

    print()
    print("=" * 60)
    print("TOP CONFIGURATIONS (by mean hit rate across workloads)")
    print("=" * 60)
    for mean_hr, (re_val, rs_val) in sorted(ranked, reverse=True)[:10]:
        print(
            f"mean_hit_rate={mean_hr:.4f}  RETRAIN_EVERY={re_val}  RETRAIN_SAMPLE_SIZE={rs_val}"
        )

    if ranked:
        best_mean, (best_re, best_rs) = sorted(ranked, reverse=True)[0]
        print()
        print(
            f"Best config: RETRAIN_EVERY={best_re}  RETRAIN_SAMPLE_SIZE={best_rs}  "
            f"(mean hit rate={best_mean:.4f})"
        )
        print()
        print("To apply permanently, update learnedpolicy.rs:")
        print(f"const RETRAIN_EVERY: u64 = {best_re};")
        print(f"const RETRAIN_SAMPLE_SIZE: usize = {best_rs};")


if __name__ == "__main__":
    main()
