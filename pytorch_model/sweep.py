"""
Hyperparameter sweep over the fixed EvictionMLP (input_dim -> 64 -> 32 -> 1).
Varies: normalization, feature transforms, batch size, learning rate, optimizer, dropout.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

FEATURE_COLS = [
    "resident_age_diff",
    "resident_time_since_last_diff",
    "resident_access_count_diff",
    "resident_frequency_diff",
    "global_age_since_first_request_diff",
    "global_time_since_last_request_diff",
    "global_total_request_count_diff",
    "last_interarrival_diff",
    "avg_interarrival_diff",
    "gap_count_diff",
    "decay_0_diff",
    "decay_1_diff",
    "decay_2_diff",
]
LABEL_COL = "y"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def apply_transform(x: np.ndarray, transform: str) -> np.ndarray:
    """Element-wise feature transform applied before normalization."""
    if transform == "none":
        return x
    elif transform == "log1p":
        # sign-preserving log1p for signed diffs
        return np.sign(x) * np.log1p(np.abs(x))
    elif transform == "sqrt":
        return np.sign(x) * np.sqrt(np.abs(x))
    elif transform == "tanh":
        # soft clipping — reduces outlier influence
        return np.tanh(x / (np.std(x, axis=0) + 1e-8))
    raise ValueError(transform)


def normalize(x: np.ndarray, norm: str, mean=None, std=None, q01=None, q99=None):
    """Normalize features. Returns (x_normed, params_dict)."""
    if norm == "none":
        return x, {}
    elif norm == "zscore":
        if mean is None:
            mean = x.mean(axis=0)
            std  = x.std(axis=0) + 1e-8
        return (x - mean) / std, {"mean": mean, "std": std}
    elif norm == "minmax":
        if q01 is None:
            q01 = x.min(axis=0)
            q99 = x.max(axis=0)
        denom = (q99 - q01) + 1e-8
        return (x - q01) / denom, {"q01": q01, "q99": q99}
    elif norm == "robust":
        # scale by IQR, center at median
        if mean is None:
            mean = np.median(x, axis=0)
            std  = (np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)) + 1e-8
        return (x - mean) / std, {"mean": mean, "std": std}
    raise ValueError(norm)


class PairwiseDataset(Dataset):
    def __init__(self, csv_path, transform="none", norm="zscore", norm_params=None):
        df = pd.read_csv(csv_path)
        x = df[FEATURE_COLS].values.astype(np.float32)
        x = apply_transform(x, transform)
        x, params = normalize(x, norm, **(norm_params or {}))
        self.norm_params = params
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(df[LABEL_COL].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class EvictionMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "leakyrelu": nn.LeakyReLU}[activation]
        layers = [
            nn.Linear(input_dim, 64), act(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(64, 32), act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Curated configs ───────────────────────────────────────────────────────────
# 60 configs chosen to cover the axes most likely to matter:
#   - transform + norm: biggest impact (log1p tames skewed diffs; robust handles outliers)
#   - lr: critical; sweep 1e-2 / 1e-3 / 3e-4 / 1e-4
#   - batch size: affects gradient noise; 64 / 256 / 1024
#   - optimizer: adamw generally better; include adam for comparison
#   - dropout: small model unlikely to overfit, but 0.1 worth trying
#   - activation: relu baseline vs gelu
# Each tuple: (transform, norm, batch_size, lr, optimizer, dropout, activation)

CONFIGS = [
    # ── Group A: core lr × transform × norm (adamw, bs=256, no dropout, relu) ──
    # Best-guess anchor: log1p + zscore is the expected winner
    ("log1p",  "zscore",  256,  1e-3, "adamw", 0.0, "relu"),
    ("log1p",  "zscore",  256,  3e-4, "adamw", 0.0, "relu"),
    ("log1p",  "zscore",  256,  1e-4, "adamw", 0.0, "relu"),
    ("log1p",  "zscore",  256,  1e-2, "adamw", 0.0, "relu"),
    ("log1p",  "robust",  256,  1e-3, "adamw", 0.0, "relu"),
    ("log1p",  "robust",  256,  3e-4, "adamw", 0.0, "relu"),
    ("log1p",  "robust",  256,  1e-4, "adamw", 0.0, "relu"),
    ("none",   "zscore",  256,  1e-3, "adamw", 0.0, "relu"),
    ("none",   "zscore",  256,  3e-4, "adamw", 0.0, "relu"),
    ("none",   "zscore",  256,  1e-4, "adamw", 0.0, "relu"),
    ("none",   "robust",  256,  1e-3, "adamw", 0.0, "relu"),
    ("none",   "robust",  256,  3e-4, "adamw", 0.0, "relu"),
    ("tanh",   "zscore",  256,  1e-3, "adamw", 0.0, "relu"),
    ("tanh",   "zscore",  256,  3e-4, "adamw", 0.0, "relu"),
    ("tanh",   "robust",  256,  1e-3, "adamw", 0.0, "relu"),

    # ── Group B: batch size sensitivity (log1p+zscore, best lrs) ──────────────
    ("log1p",  "zscore",   64,  1e-3, "adamw", 0.0, "relu"),
    ("log1p",  "zscore",   64,  3e-4, "adamw", 0.0, "relu"),
    ("log1p",  "zscore",  128,  1e-3, "adamw", 0.0, "relu"),
    ("log1p",  "zscore",  128,  3e-4, "adamw", 0.0, "relu"),
    ("log1p",  "zscore",  512,  1e-3, "adamw", 0.0, "relu"),
    ("log1p",  "zscore",  512,  3e-4, "adamw", 0.0, "relu"),
    ("log1p",  "zscore", 1024,  1e-3, "adamw", 0.0, "relu"),
    ("log1p",  "zscore", 1024,  3e-4, "adamw", 0.0, "relu"),
    ("none",   "zscore",   64,  1e-3, "adamw", 0.0, "relu"),
    ("none",   "zscore",  128,  1e-3, "adamw", 0.0, "relu"),
    ("none",   "zscore", 1024,  1e-3, "adamw", 0.0, "relu"),

    # ── Group C: adam vs adamw comparison ─────────────────────────────────────
    ("log1p",  "zscore",  256,  1e-3, "adam",  0.0, "relu"),
    ("log1p",  "zscore",  256,  3e-4, "adam",  0.0, "relu"),
    ("log1p",  "robust",  256,  1e-3, "adam",  0.0, "relu"),
    ("none",   "zscore",  256,  1e-3, "adam",  0.0, "relu"),
    ("none",   "zscore",  256,  3e-4, "adam",  0.0, "relu"),

    # ── Group D: dropout (log1p+zscore, best lrs) ─────────────────────────────
    ("log1p",  "zscore",  256,  1e-3, "adamw", 0.1, "relu"),
    ("log1p",  "zscore",  256,  3e-4, "adamw", 0.1, "relu"),
    ("log1p",  "zscore",  256,  1e-3, "adamw", 0.2, "relu"),
    ("log1p",  "zscore",  256,  3e-4, "adamw", 0.2, "relu"),
    ("none",   "zscore",  256,  1e-3, "adamw", 0.1, "relu"),
    ("log1p",  "robust",  256,  1e-3, "adamw", 0.1, "relu"),

    # ── Group E: activation (relu vs gelu) ────────────────────────────────────
    ("log1p",  "zscore",  256,  1e-3, "adamw", 0.0, "gelu"),
    ("log1p",  "zscore",  256,  3e-4, "adamw", 0.0, "gelu"),
    ("log1p",  "robust",  256,  1e-3, "adamw", 0.0, "gelu"),
    ("none",   "zscore",  256,  1e-3, "adamw", 0.0, "gelu"),
    ("none",   "zscore",  256,  3e-4, "adamw", 0.0, "gelu"),
    ("tanh",   "zscore",  256,  1e-3, "adamw", 0.0, "gelu"),

    # ── Group F: combined promising variations ────────────────────────────────
    ("log1p",  "zscore",  128,  3e-4, "adamw", 0.1, "gelu"),
    ("log1p",  "zscore",   64,  3e-4, "adamw", 0.1, "gelu"),
    ("log1p",  "robust",  128,  3e-4, "adamw", 0.0, "gelu"),
    ("log1p",  "robust",   64,  1e-3, "adamw", 0.1, "relu"),
    ("none",   "robust",  256,  3e-4, "adamw", 0.0, "gelu"),
    ("tanh",   "robust",  256,  1e-3, "adamw", 0.0, "relu"),
    ("tanh",   "robust",  256,  3e-4, "adamw", 0.0, "gelu"),
    ("log1p",  "zscore",  256,  1e-3, "adamw", 0.1, "gelu"),
    ("log1p",  "robust",  256,  3e-4, "adamw", 0.1, "gelu"),
    ("none",   "zscore",   64,  3e-4, "adamw", 0.1, "gelu"),
    ("log1p",  "zscore",  512,  3e-4, "adamw", 0.1, "gelu"),
    ("none",   "robust",   64,  3e-4, "adamw", 0.0, "relu"),
]

EPOCHS = 15
TRAIN_CSV = "pairwise_training_dataset.csv"
VAL_CSV   = "pairwise_validation_dataset.csv"


def run_config(cfg):
    transform, norm, batch_size, lr, optimizer, dropout, activation = cfg

    train_set = PairwiseDataset(TRAIN_CSV, transform=transform, norm=norm)
    val_set   = PairwiseDataset(VAL_CSV,   transform=transform, norm=norm,
                                norm_params=train_set.norm_params)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)

    model     = EvictionMLP(input_dim=len(FEATURE_COLS), dropout=dropout, activation=activation)
    criterion = nn.BCEWithLogitsLoss()

    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = (model(x) > 0).float()
                correct += (preds == y).sum().item()
                total   += len(y)
        best_acc = max(best_acc, correct / total)

    return best_acc


def main():
    print(f"Running {len(CONFIGS)} configs × {EPOCHS} epochs each\n")
    results = []
    for i, cfg in enumerate(CONFIGS):
        transform, norm, batch_size, lr, optimizer, dropout, activation = cfg
        label = (f"tr={transform:6s} norm={norm:7s} bs={batch_size:4d} lr={lr:.0e} "
                 f"opt={optimizer:5s} drop={dropout} act={activation}")
        acc = run_config(cfg)
        results.append((acc, label))
        print(f"[{i+1:2d}/60]  acc={acc:.4f}  {label}")

    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)
    for acc, label in sorted(results, reverse=True)[:10]:
        print(f"  acc={acc:.4f}  {label}")


if __name__ == "__main__":
    main()
