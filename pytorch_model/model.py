import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    FEATURE_COLS,
    PairwiseDataset,
    EvictionMLP,
    default_checkpoint_path,
    default_metadata_path,
    default_train_csv,
    default_val_csv,
)


def train(
    train_csv: str,
    val_csv: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    init_checkpoint: str | None = None
):
    train_set = PairwiseDataset(train_csv, transform="log1p", norm="zscore")
    val_set = PairwiseDataset(
        val_csv,
        transform="log1p",
        norm="zscore",
        norm_params=train_set.norm_params,
    )
    n_train = len(train_set)
    n_val = len(val_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = EvictionMLP(input_dim=len(FEATURE_COLS), dropout=0.0, activation="relu")

    if init_checkpoint != None:
        checkpoint = torch.load(init_checkpoint, map_location="cpu")

        state_dict = checpoint["state_dict"]

        model_weights = {
            k: v for k, v in state_dict.items()
            if not k.startswith("norm.")
        }

        model.load_state_dict(model_weights, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                preds = (logits > 0).float()
                correct += (preds == y).sum().item()
                total += len(y)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss/n_train:.4f} | "
            f"val_loss={val_loss/n_val:.4f} | "
            f"val_acc={correct/total:.4f}"
        )

    return model, train_set.norm_params["mean"], train_set.norm_params["std"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(default_train_csv()))
    parser.add_argument("--val-csv", default=str(default_val_csv()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default=str(default_checkpoint_path()))
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()

    model, mean, std = train(
        args.train_csv,
        args.val_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        init_checkpoint=args.init_checkpoint,
    )
    state_dict = model.state_dict()
    state_dict["norm.mean"] = torch.tensor(mean, dtype=torch.float32)
    state_dict["norm.std"] = torch.tensor(std, dtype=torch.float32)
    torch.save({"state_dict": state_dict}, args.output)
    metadata_path = default_metadata_path()
    metadata_path.write_text(
        json.dumps(
            {
                "transform": "log1p",
                "normalization": "zscore",
                "feature_cols": FEATURE_COLS,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            indent=2,
        )
    )
    print(f"Saved model to {args.output}")
    print(f"Saved metadata to {metadata_path}")
