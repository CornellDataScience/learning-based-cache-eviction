from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_train_csv() -> Path:
    return repo_root() / "pairwise_training_dataset.csv"


def default_val_csv() -> Path:
    return repo_root() / "pairwise_validation_dataset.csv"


def default_checkpoint_path() -> Path:
    return repo_root() / "eviction_mlp.pt"


def default_metadata_path() -> Path:
    return repo_root() / "eviction_mlp.meta.json"


def fit_zscore(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_transform(x: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return x
    if transform == "log1p":
        return np.sign(x) * np.log1p(np.abs(x))
    if transform == "sqrt":
        return np.sign(x) * np.sqrt(np.abs(x))
    if transform == "tanh":
        return np.tanh(x / (np.std(x, axis=0) + 1e-8))
    raise ValueError(f"unsupported transform: {transform}")


def normalize(x: np.ndarray, norm: str, mean=None, std=None, q01=None, q99=None):
    if norm == "none":
        return x, {}
    if norm == "zscore":
        if mean is None:
            mean = x.mean(axis=0)
            std = x.std(axis=0) + 1e-8
        return (x - mean) / std, {"mean": mean, "std": std}
    if norm == "minmax":
        if q01 is None:
            q01 = x.min(axis=0)
            q99 = x.max(axis=0)
        denom = (q99 - q01) + 1e-8
        return (x - q01) / denom, {"q01": q01, "q99": q99}
    if norm == "robust":
        if mean is None:
            mean = np.median(x, axis=0)
            std = (np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)) + 1e-8
        return (x - mean) / std, {"mean": mean, "std": std}
    raise ValueError(f"unsupported normalization: {norm}")


class PairwiseDataset(Dataset):
    def __init__(self, csv_path, transform="log1p", norm="zscore", norm_params=None):
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


class EvictionMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "leakyrelu": nn.LeakyReLU}[activation]
        layers = [nn.Linear(input_dim, 64), act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(64, 32), act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
