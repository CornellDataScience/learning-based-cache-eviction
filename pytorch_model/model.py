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


def fit_zscore(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class PairwiseDataset(Dataset):
    def __init__(self, csv_path: str, mean=None, std=None):
        df = pd.read_csv(csv_path)
        x = df[FEATURE_COLS].values.astype(np.float32)
        x = np.sign(x) * np.log1p(np.abs(x))
        if mean is None or std is None:
            mean, std = fit_zscore(x)
        x = (x - mean) / std
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(df[LABEL_COL].values, dtype=torch.float32)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EvictionMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(train_csv: str, val_csv: str, epochs: int = 20, batch_size: int = 64, lr: float = 3e-4):
    train_set = PairwiseDataset(train_csv)
    val_set = PairwiseDataset(val_csv, mean=train_set.mean, std=train_set.std)
    n_train = len(train_set)
    n_val = len(val_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = EvictionMLP(input_dim=len(FEATURE_COLS), dropout=0.0)
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

    return model, train_set.mean, train_set.std


if __name__ == "__main__":
    import sys
    train_csv = sys.argv[1] if len(sys.argv) > 1 else "pairwise_training_dataset.csv"
    val_csv = sys.argv[2] if len(sys.argv) > 2 else "pairwise_validation_dataset.csv"
    model, mean, std = train(train_csv, val_csv)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": torch.tensor(mean, dtype=torch.float32),
        "std": torch.tensor(std, dtype=torch.float32),
    }, "eviction_mlp.pt")
    print("Saved model to eviction_mlp.pt")
