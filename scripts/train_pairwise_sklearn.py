#!/usr/bin/env python3
"""
Train a lightweight pairwise MLP using scikit-learn.

This is a simpler alternative to the PyTorch script. It keeps the same
dataset assumptions and preprocessing ideas:
1. Load train / validation / test CSVs.
2. Use only the numeric pairwise-difference features.
3. Fit preprocessing statistics on the training split only.
4. Train a small MLPClassifier.
5. Pick the best configuration using validation accuracy.
6. Evaluate once on the held-out test sets.
7. Save plots, preprocessing stats, and test metrics.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier


TRAIN_CSV = REPO_ROOT / "pairwise_training_dataset.csv"
VAL_CSV = REPO_ROOT / "pairwise_validation_dataset.csv"
TEST_DIR = REPO_ROOT / "pairwise_test_datasets"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

METADATA_COLUMNS = [
    "trace_name",
    "cache_size",
    "request_index",
    "tick",
    "key0",
    "key1",
]
LABEL_COLUMN = "y"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class PreprocessConfig:
    use_signed_log: bool = True
    use_clipping: bool = True
    clip_low_q: float = 0.005
    clip_high_q: float = 0.995


@dataclass
class TrainConfig:
    seed: int = 42
    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    learning_rate_init: float = 1e-3
    alpha: float = 1e-5
    batch_size: int = 512
    max_iter: int = 15


def load_split(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")
    return pd.read_csv(csv_path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLUMNS + [LABEL_COLUMN]]


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    positive_rate = float(df[LABEL_COLUMN].mean())
    print(f"\n{name}")
    print(f"rows: {len(df)}")
    print(f"positive_rate(y=1): {positive_rate:.4f}")

    by_trace = (
        df.groupby("trace_name")[LABEL_COLUMN]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
    )
    print("top trace balances:")
    print(by_trace.head(10).to_string())


def fit_preprocessor(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    config: PreprocessConfig,
) -> dict:
    x_train = train_df[feature_columns].to_numpy(dtype=np.float32)
    # select only the rows in which the trace_name is mixed_interleaved
    

    if config.use_signed_log:
        x_train = signed_log1p(x_train)

    if config.use_clipping:
        clip_low = np.quantile(x_train, config.clip_low_q, axis=0)
        clip_high = np.quantile(x_train, config.clip_high_q, axis=0)
        x_train = np.clip(x_train, clip_low, clip_high)
    else:
        clip_low = None
        clip_high = None

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    return {
        "feature_columns": feature_columns,
        "use_signed_log": config.use_signed_log,
        "use_clipping": config.use_clipping,
        "clip_low_q": config.clip_low_q,
        "clip_high_q": config.clip_high_q,
        "clip_low": None if clip_low is None else clip_low.tolist(),
        "clip_high": None if clip_high is None else clip_high.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def apply_preprocessor(df: pd.DataFrame, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    x = df[stats["feature_columns"]].to_numpy(dtype=np.float32)

    if stats["use_signed_log"]:
        x = signed_log1p(x)

    if stats["use_clipping"]:
        clip_low = np.asarray(stats["clip_low"], dtype=np.float32)
        clip_high = np.asarray(stats["clip_high"], dtype=np.float32)
        x = np.clip(x, clip_low, clip_high)

    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    x = (x - mean) / std
    y = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
    return x, y


def evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> dict:
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "accuracy": accuracy_score(y, preds),
        "loss": log_loss(y, probs),
    }


def save_loss_plot(loss_curve: list[float], output_path: Path) -> None:
    epochs = range(1, len(loss_curve) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss_curve, label="train_loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_experiment(
    preprocess_config: PreprocessConfig,
    train_config: TrainConfig,
) -> tuple[dict, MLPClassifier, dict]:
    train_df = load_split(TRAIN_CSV)
    val_df = load_split(VAL_CSV)
    feature_columns = get_feature_columns(train_df)

    print_split_summary("train", train_df)
    print_split_summary("validation", val_df)
    print(f"\nfeature_columns ({len(feature_columns)}): {feature_columns}")

    # use_columns = ["mixed_interleaved"]
    use_columns = "all"
    if type(use_columns) == list:
        train_df = train_df[train_df["trace_name"].isin(use_columns)]
    elif use_columns != "all":
        train_df = train_df[train_df["trace_name"] == use_columns]
    
    # print('train df size before oversampling', train_df.shape[0])

    # oversample zipf by x times relative to other traces during training
    # target_zipf_count = int(train_df[train_df["trace_name"] != "zipf"].shape[0]*0.5)
    # zipf_rows = train_df[train_df["trace_name"] == "zipf"].sample(target_zipf_count, replace=True, random_state=train_config.seed)
    # train_df = pd.concat([train_df, zipf_rows], ignore_index=True)
    # print("total size of train_df after oversampling", train_df.shape[0])
    

    stats = fit_preprocessor(train_df, feature_columns, preprocess_config)
    x_train, y_train = apply_preprocessor(train_df, stats)
    x_val, y_val = apply_preprocessor(val_df, stats)

    model = MLPClassifier(
        hidden_layer_sizes=train_config.hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=train_config.alpha,
        batch_size=train_config.batch_size,
        learning_rate_init=train_config.learning_rate_init,
        max_iter=train_config.max_iter,
        early_stopping=False,
        random_state=train_config.seed,
        verbose=True,
    )

    # try random forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    
    # model = RandomForestClassifier(
    #     n_estimators=200,
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features="sqrt",
    #     n_jobs=-1,
    #     random_state=train_config.seed,
    #     verbose=1,
    # )

    # model = XGBClassifier(
    #     n_estimators=300,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     reg_lambda=1.0,
    #     reg_alpha=0.0,
    #     objective="binary:logistic",
    #     eval_metric="logloss",
    #     use_label_encoder=False,
    #     n_jobs=-1,
    #     random_state=train_config.seed,
    #     verbosity=1,
    # )

    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2
        ),
        n_estimators=200,
        learning_rate=0.5,
        random_state=train_config.seed
    )

    
    model.fit(x_train, y_train)

    val_metrics = evaluate(model, x_val, y_val)

    result = {
        "preprocess_config": asdict(preprocess_config),
        "train_config": {
            **asdict(train_config),
            "hidden_layer_sizes": list(train_config.hidden_layer_sizes),
        },
        "val_metrics": val_metrics,
        # "loss_curve": [float(x) for x in model.loss_curve_],
    }
    return result, model, stats


def evaluate_all_tests(model: MLPClassifier, stats: dict) -> list[dict]:
    rows = []

    for csv_path in sorted(TEST_DIR.glob("*.csv")):
        df = load_split(csv_path)
        x_test, y_test = apply_preprocessor(df, stats)
        metrics = evaluate(model, x_test, y_test)
        rows.append(
            {
                "dataset": csv_path.stem,
                "rows": len(df),
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "positive_rate": float(df[LABEL_COLUMN].mean()),
            }
        )

    return rows


def save_test_metrics(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "rows", "accuracy", "loss", "positive_rate"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_seed(42)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    preprocess_options = [
        PreprocessConfig(use_signed_log=False, use_clipping=False),
        PreprocessConfig(use_signed_log=True, use_clipping=True),
    ]
    train_options = [
        TrainConfig(hidden_layer_sizes=(32,), learning_rate_init=1e-3),
        TrainConfig(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3),
        TrainConfig(hidden_layer_sizes=(128, 64), learning_rate_init=3e-4),
    ]

    best_result = None
    best_model = None
    best_stats = None
    # best_val_acc = -1.0

    # for preprocess_config in preprocess_options:
    #     for train_config in train_options:
    #         print("\n==============================")
    #         print("running configuration:")
    #         print(asdict(preprocess_config))
    #         print({**asdict(train_config), "hidden_layer_sizes": list(train_config.hidden_layer_sizes)})

    #         result, model, stats = run_experiment(preprocess_config, train_config)
    #         val_acc = result["val_metrics"]["accuracy"]

    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             best_result = result
    #             best_model = model
    #             best_stats = stats

    # only run one test using the first configuration of both
    preprocess_config = preprocess_options[1]
    train_config = train_options[1]
    print("\n==============================")
    print("running configuration:")
    print(asdict(preprocess_config))
    print({**asdict(train_config), "hidden_layer_sizes": list(train_config.hidden_layer_sizes)})

    result, model, stats = run_experiment(preprocess_config, train_config)

    best_result = result
    best_model = model
    best_stats = stats

    if best_result is None or best_model is None or best_stats is None:
        raise RuntimeError("No experiment completed successfully.")

    print("\nBest validation result:")
    print(json.dumps(best_result["train_config"], indent=2))
    print(json.dumps(best_result["preprocess_config"], indent=2))
    print(json.dumps(best_result["val_metrics"], indent=2))

    model_path = ARTIFACT_DIR / "pairwise_sklearn_model.joblib"
    try:
        import joblib

        joblib.dump(best_model, model_path)
    except Exception:
        print("joblib not available; skipping model serialization.")

    with (ARTIFACT_DIR / "preprocessing_stats_sklearn.json").open("w") as f:
        json.dump(best_stats, f, indent=2)

    # save_loss_plot(best_result["loss_curve"], ARTIFACT_DIR / "sklearn_loss_curve.png")

    test_rows = evaluate_all_tests(best_model, best_stats)
    save_test_metrics(test_rows, ARTIFACT_DIR / "test_metrics_sklearn.csv")

    print("\nTest results:")
    for row in test_rows:
        print(
            f"{row['dataset']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"loss={row['loss']:.4f}, "
            f"rows={row['rows']}"
        )


if __name__ == "__main__":
    main()
