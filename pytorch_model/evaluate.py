import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    FEATURE_COLS,
    PairwiseDataset,
    EvictionMLP,
    default_checkpoint_path,
    default_metadata_path,
    datasets_dir,
)


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        metadata = json.load(f)

    feature_cols = metadata.get("feature_cols")
    if feature_cols != FEATURE_COLS:
        raise ValueError(
            f"metadata feature columns do not match expected schema: {feature_cols}"
        )

    return metadata


def load_model(checkpoint_path: Path) -> EvictionMLP:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Training stores norm tensors in the same checkpoint, but the model
    # itself only needs the learnable weights.
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("norm.")
    }

    model = EvictionMLP(input_dim=len(FEATURE_COLS), dropout=0.0, activation="relu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_csv(
    model: EvictionMLP,
    csv_path: Path,
    metadata: dict,
    batch_size: int,
) -> tuple[float, float, int]:
    dataset = PairwiseDataset(
        str(csv_path),
        transform=metadata["transform"],
        norm=metadata["normalization"],
        norm_params={"mean": metadata["mean"], "std": metadata["std"]},
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss_sum += criterion(logits, y).item() * len(y)
            preds = (logits > 0).float()
            correct += (preds == y).sum().item()
            total += len(y)

    return loss_sum / total, correct / total, total


def resolve_csv_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    raise FileNotFoundError(f"dataset path does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(default_checkpoint_path()),
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--metadata",
        default=str(default_metadata_path()),
        help="Path to model metadata JSON",
    )
    parser.add_argument(
        "--data",
        default=str(datasets_dir() / "pairwise_test_datasets"),
        help="CSV file or directory of CSV files to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    checkpoint_path = Path(args.model)
    metadata_path = Path(args.metadata)
    dataset_path = Path(args.data)

    metadata = load_metadata(metadata_path)
    model = load_model(checkpoint_path)
    csv_paths = resolve_csv_paths(dataset_path)

    if not csv_paths:
        raise FileNotFoundError(f"no CSV files found under {dataset_path}")

    print(f"model={checkpoint_path}")
    print(f"metadata={metadata_path}")
    print(f"datasets={len(csv_paths)}")

    for csv_path in csv_paths:
        loss, acc, total = evaluate_csv(model, csv_path, metadata, args.batch_size)
        print(
            f"{csv_path.name}: "
            f"rows={total} "
            f"loss={loss:.4f} "
            f"acc={acc:.4f}"
        )


if __name__ == "__main__":
    main()
