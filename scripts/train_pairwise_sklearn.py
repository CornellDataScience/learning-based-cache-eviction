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

# silence annoying invalid value encountered in matmul, which I'm pretty sure isn't a problem?
np.seterr(invalid = "ignore", over="ignore", divide="ignore")

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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

    x = (x - mean) / std # normalize after clipping
    y = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
    return x, y


def evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> dict:
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "accuracy": accuracy_score(y, preds),
        "loss": log_loss(y, probs),
    }


def evaluate_by_trace(
    model: MLPClassifier,
    x: np.ndarray,
    y: np.ndarray,
    trace_names: np.ndarray,
) -> list[dict]:
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= 0.5).astype(np.int64)

    rows: list[dict] = []
    for trace in sorted(pd.unique(trace_names)):
        mask = trace_names == trace
        y_t = y[mask]
        probs_t = probs[mask]
        preds_t = preds[mask]
        rows.append(
            {
                "trace_name": str(trace),
                "rows": int(mask.sum()),
                "accuracy": float(accuracy_score(y_t, preds_t)),
                "loss": float(log_loss(y_t, probs_t)),
                "positive_rate": float(y_t.mean()),
            }
        )

    return rows


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


def compute_feature_correlation(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    x = train_df[feature_columns].copy()
    return x.corr(method="spearman")


def save_correlation_outputs(
    corr_df: pd.DataFrame,
    csv_output_path: Path,
    plot_output_path: Path,
) -> None:
    corr_df.to_csv(csv_output_path)

    n_features = corr_df.shape[0]
    fig_size = min(24, max(8, 0.28 * n_features))

    plt.figure(figsize=(fig_size, fig_size))
    im = plt.imshow(corr_df.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Spearman correlation")

    text_size = max(3, min(8, int(220 / max(n_features, 1))))
    for i in range(n_features):
        for j in range(n_features):
            value = float(corr_df.iat[i, j])
            text_color = "white" if abs(value) >= 0.5 else "black"
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=text_size,
            )

    if n_features <= 50:
        ticks = np.arange(n_features)
        plt.xticks(ticks, corr_df.columns, rotation=90, fontsize=7)
        plt.yticks(ticks, corr_df.index, fontsize=7)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.title("Feature Correlation Matrix (Train Split)")
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=150)
    plt.close()


def run_experiment(
    preprocess_config: PreprocessConfig,
    train_config: TrainConfig,
) -> tuple[dict, MLPClassifier, dict, pd.DataFrame]:
    train_df = load_split(TRAIN_CSV)
    val_df = load_split(VAL_CSV)
    feature_columns = get_feature_columns(train_df)

    print_split_summary("train", train_df)
    print_split_summary("validation", val_df)

    exclude_columns = ["resident_time_since_last_diff", "resident_frequency_diff", "decay_0_diff", "decay_1_diff", "gap_count_diff"]
    print("excluding columns", exclude_columns)
    if exclude_columns:
        train_df = train_df.drop(columns=exclude_columns, errors="ignore")
        val_df = val_df.drop(columns=exclude_columns, errors="ignore")
        feature_columns = [c for c in feature_columns if c not in exclude_columns]

    print("feature columns after excluding", len(feature_columns))
    corr_df = compute_feature_correlation(train_df, feature_columns)

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
        early_stopping=True,
        random_state=train_config.seed,
        verbose=True,
    )
    
    # model = AdaBoostClassifier(
    #     estimator=DecisionTreeClassifier(
    #         max_depth=2
    #     ),
    #     n_estimators=100,
    #     learning_rate=0.5,
    #     random_state=train_config.seed
    # )

    
    model.fit(x_train, y_train)

    val_metrics = evaluate(model, x_val, y_val)
    val_breakdown = evaluate_by_trace(
        model,
        x_val,
        y_val,
        val_df["trace_name"].to_numpy(),
    )

    result = {
        "preprocess_config": asdict(preprocess_config),
        "train_config": {
            **asdict(train_config),
            "hidden_layer_sizes": list(train_config.hidden_layer_sizes),
        },
        "val_metrics": val_metrics,
        "val_breakdown": val_breakdown,
        # "loss_curve": [float(x) for x in model.loss_curve_],
    }
    return result, model, stats, corr_df


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

    result, model, stats, corr_df = run_experiment(preprocess_config, train_config)

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

    save_correlation_outputs(
        corr_df,
        ARTIFACT_DIR / "feature_correlation_matrix_spearman.csv",
        ARTIFACT_DIR / "feature_correlation_matrix_spearman.png",
    )

    # save_loss_plot(best_result["loss_curve"], ARTIFACT_DIR / "sklearn_loss_curve.png")

    test_rows = evaluate_all_tests(best_model, best_stats)
    save_test_metrics(test_rows, ARTIFACT_DIR / "test_metrics_sklearn.csv")

    print("\nValidation breakdown by trace:")
    for row in best_result["val_breakdown"]:
        print(
            f"{row['trace_name']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"loss={row['loss']:.4f}, "
            f"rows={row['rows']}"
        )


if __name__ == "__main__":
    main()
