import csv
import random
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("wiki_trace.tsv")
OUTPUT_PATH = Path("wiki_sampled_trace.tsv")

WINDOW_COUNT = 5
WINDOW_SIZE = 10_000
SEED = 44

KEY_COLUMN = "hashed_host_path_query"
EXPECTED_COLUMNS = [
    "relative_unix",
    "hashed_host_path_query",
    "response_size",
    "time_firstbyte",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Input file is missing a header row.")
        missing = [c for c in EXPECTED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input file missing expected columns: {missing}")
        return list(reader)


def filter_rows_by_key_frequency(rows: list[dict[str, str]], min_freq: int = 2) -> list[dict[str, str]]:
    counts = Counter(row[KEY_COLUMN].strip() for row in rows if row.get(KEY_COLUMN))
    return [row for row in rows if counts[row[KEY_COLUMN].strip()] >= min_freq]


def choose_window_starts(length: int, window_count: int, window_size: int, seed: int) -> list[int]:
    if length < window_size:
        raise ValueError(
            f"Filtered trace too small: need at least {window_size} rows, got {length}"
        )

    rng = random.Random(seed)
    max_start = length - window_size

    if window_count == 1:
        return [rng.randint(0, max_start)]

    segment = max(1, max_start // (window_count - 1))
    starts = []
    for i in range(window_count):
        base = min(i * segment, max_start)
        lo = max(0, base - segment // 4)
        hi = min(max_start, base + segment // 4)
        starts.append(rng.randint(lo, hi))
    return starts


def sample_windows(
    rows: list[dict[str, str]],
    window_count: int,
    window_size: int,
    seed: int,
) -> list[dict[str, str]]:
    filtered = filter_rows_by_key_frequency(rows, min_freq=2)
    starts = choose_window_starts(len(filtered), window_count, window_size, seed)

    sampled = []
    for start in starts:
        sampled.extend(filtered[start:start + window_size])
    return sampled


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in EXPECTED_COLUMNS})


def main() -> None:
    rows = load_rows(INPUT_PATH)
    sampled_rows = sample_windows(rows, WINDOW_COUNT, WINDOW_SIZE, SEED)
    write_rows(OUTPUT_PATH, sampled_rows)
    print(f"Wrote {len(sampled_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

