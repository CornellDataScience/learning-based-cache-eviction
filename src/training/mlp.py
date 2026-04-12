"""
From-Scratch MLP for Pairwise Cache Eviction
=============================================

This script trains a small neural network from scratch using NumPy.
It loads CSVs, trains the model, evaluates performance, and exports weights.

"""

import numpy as np
import os
import json
from pathlib import Path


# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = "../../pairwise_test_datasets"         #folder containing Rust-generated CSVs
SEED = 42
LEARNING_RATE = 0.01
EPOCHS = 200
BATCH_SIZE = 256
HIDDEN_SIZE = 32             # single hidden layer of 32 neurons
VAL_RATIO = 0.15
TEST_RATIO = 0.15
PATIENCE = 15               

# These are the metadata columns from your pairwise_csv_writer.rs
# Everything else (except 'y') is a feature
METADATA_COLS = ["trace_name", "cache_size", "request_index", "tick", "key0", "key1", "y"]

# Feature names from your Rust code (pairwise_csv_writer.rs)
# These should match exactly what's in your CSV headers
FEATURE_NAMES = [
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

np.random.seed(SEED)


# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def load_data(data_dir):
    """
    Load all CSVs from data_dir.
    Returns: features (N x 13), labels (N,), trace_names (N,)
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSVs found in {data_dir}/")
        print("Put your Rust-generated CSVs there and re-run.")
        raise SystemExit(1)

    all_X = []
    all_y = []
    all_traces = []

    for csv_path in csv_files:
        print(f"  Loading {csv_path.name}...")

        # Read header to find column indices
        with open(csv_path, 'r') as f:
            header = f.readline().strip().split(',')

        # Load full file, skip header
        raw = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=str)

        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if len(raw) == 0:
            print(f"    Skipping {csv_path.name} (empty)")
            continue

        # Find column indices
        trace_idx = header.index("trace_name")
        y_idx = header.index("y")

        # Feature columns = everything not in METADATA_COLS
        feature_indices = [i for i, h in enumerate(header) if h not in METADATA_COLS]

        traces = raw[:, trace_idx]
        y = raw[:, y_idx].astype(float)
        X = raw[:, feature_indices].astype(float)

        all_X.append(X)
        all_y.append(y)
        all_traces.append(traces)

        print(f"    → {len(X)} rows, {X.shape[1]} features")
        print(f"    → labels: {int(y.sum())} positive, {int(len(y) - y.sum())} negative")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    traces = np.concatenate(all_traces)

    print(f"\n  Total: {len(X)} samples, {X.shape[1]} features")
    return X, y, traces


# ============================================================================
# PART 2: PREPROCESSING
#
# Why normalize? Your features have different scales:
#   - resident_age_diff might range from -10000 to 10000
#   - access_count_diff might range from -50 to 50
#   - decay counters range roughly -10 to 10
#
# Without normalization, the large-scale features dominate gradient updates.
# The model spends all its effort adjusting weights for the big features
# and basically ignores the small ones.
#
# StandardScaler: subtract mean, divide by std. Computed on TRAINING SET ONLY.
# We apply those same training statistics to val/test — no data leakage.
# ============================================================================

def normalize(X_train, X_val, X_test):
    """
    Normalize using training set statistics only.
    Returns normalized arrays + the mean/std for export to Rust.
    """
    mean = X_train.mean(axis=0)          # shape: (13,)
    std = X_train.std(axis=0) + 1e-8     # add epsilon to avoid division by zero

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    # Print feature scales before/after
    print("\n  Feature scales (training set):")
    print(f"  {'Feature':<40} {'Raw Range':>20} {'Normalized Range':>20}")
    print(f"  {'-'*40} {'-'*20} {'-'*20}")

    feature_names = FEATURE_NAMES if len(FEATURE_NAMES) == X_train.shape[1] else \
                    [f"feature_{i}" for i in range(X_train.shape[1])]

    for i, name in enumerate(feature_names):
        raw_range = f"[{X_train[:, i].min():.1f}, {X_train[:, i].max():.1f}]"
        norm_range = f"[{X_train_norm[:, i].min():.2f}, {X_train_norm[:, i].max():.2f}]"
        print(f"  {name:<40} {raw_range:>20} {norm_range:>20}")

    return X_train_norm, X_val_norm, X_test_norm, mean, std


def check_balance(y, traces, split_name="Full dataset"):
    """Show label balance overall and per trace/dataset."""
    pos = y.sum()
    neg = len(y) - pos
    print(f"\n  {split_name}: {len(y)} samples, "
          f"{int(pos)} positive ({pos/len(y)*100:.1f}%), "
          f"{int(neg)} negative ({neg/len(y)*100:.1f}%)")

    unique_traces = np.unique(traces)
    if len(unique_traces) > 1:
        for t in sorted(unique_traces):
            mask = traces == t
            t_pos = y[mask].sum()
            t_total = mask.sum()
            print(f"    {t}: {int(t_total)} samples, "
                  f"{int(t_pos)} positive ({t_pos/t_total*100:.1f}%)")


def train_val_test_split(X, y, traces):
    """
    Stratified split by trace name so each split has proportional
    representation of all workload types.
    """
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y, traces = X[indices], y[indices], traces[indices]

    # Split per trace to maintain proportions
    train_idx, val_idx, test_idx = [], [], []

    for t in np.unique(traces):
        mask = np.where(traces == t)[0]
        n = len(mask)
        n_test = max(1, int(n * TEST_RATIO))
        n_val = max(1, int(n * VAL_RATIO))

        test_idx.extend(mask[:n_test])
        val_idx.extend(mask[n_test:n_test + n_val])
        train_idx.extend(mask[n_test + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    # Shuffle each split
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    print(f"\n  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return (X[train_idx], y[train_idx], traces[train_idx],
            X[val_idx], y[val_idx], traces[val_idx],
            X[test_idx], y[test_idx], traces[test_idx])


# ============================================================================
# PART 3: THE NEURAL NETWORK (from scratch)
#
# Architecture: Input(13) → Dense(32) → ReLU → Dense(1) → Sigmoid
#
# This is the smallest useful MLP. One hidden layer with 32 neurons.
# Total parameters: (13 * 32) + 32 + (32 * 1) + 1 = 449 learnable numbers.
#
# What each piece does:
#   - W1 (13x32): transforms 13 inputs into 32 hidden activations
#   - b1 (32,):   bias for hidden layer
#   - ReLU:       max(0, x) — introduces nonlinearity so the network can
#                 learn patterns that aren't just straight lines
#   - W2 (32x1):  transforms 32 hidden activations into 1 output
#   - b2 (1,):    bias for output
#   - Sigmoid:    squashes output to [0, 1] → probability of "evict key0"
# ============================================================================

class MLP:
    """A tiny MLP built from scratch. No frameworks, just NumPy."""

    def __init__(self, input_dim, hidden_dim=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Xavier initialization — prevents activations from exploding or
        # vanishing in early training. Scale weights by 1/sqrt(fan_in).
        scale1 = np.sqrt(2.0 / input_dim)       # He init for ReLU layers
        scale2 = np.sqrt(1.0 / hidden_dim)       # Xavier for sigmoid output

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * scale2
        self.b2 = np.zeros(1)

    def forward(self, X):
        """
        Forward pass: compute prediction from input.

        X: (batch_size, 13)
        Returns: probabilities (batch_size, 1)

        This is the ONLY part you need in Rust for inference.
        """
        # Layer 1: linear transform
        # z1 = X @ W1 + b1  →  shape: (batch, 32)
        self.z1 = X @ self.W1 + self.b1

        # ReLU activation: replace negatives with 0
        # This is what lets the network learn nonlinear patterns.
        # Without it, stacking linear layers is just one big linear layer.
        self.a1 = np.maximum(0, self.z1)

        # Layer 2: linear transform → single output
        # z2 = a1 @ W2 + b2  →  shape: (batch, 1)
        self.z2 = self.a1 @ self.W2 + self.b2

        # Sigmoid: squash to [0, 1]
        # Clamp z2 to avoid overflow in exp()
        z2_clamp = np.clip(self.z2, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-z2_clamp))

        # Save input for backprop
        self.X = X

        return self.out

    def backward(self, y_true, lr):
        """
        Backpropagation: compute gradients and update weights.

        This is the learning step. We work backwards from the output:
        1. How wrong was our prediction? (output gradient)
        2. How much did each hidden neuron contribute? (hidden gradient)
        3. How much should each weight change? (weight gradients)
        4. Update weights in the direction that reduces the loss.

        y_true: (batch_size, 1) — true labels
        lr: learning rate
        """
        batch_size = y_true.shape[0]

        # Output layer gradient
        # For sigmoid + binary cross-entropy, the gradient simplifies to:
        # dL/dz2 = prediction - true_label
        dz2 = self.out - y_true                    # (batch, 1)

        # Gradients for W2 and b2
        dW2 = (self.a1.T @ dz2) / batch_size      # (32, 1)
        db2 = dz2.mean(axis=0)                     # (1,)

        # ---- Hidden layer gradient ----
        # Propagate gradient back through W2
        da1 = dz2 @ self.W2.T                      # (batch, 32)

        # Propagate through ReLU: gradient is 0 where input was negative
        dz1 = da1 * (self.z1 > 0).astype(float)   # (batch, 32)

        # Gradients for W1 and b1
        dW1 = (self.X.T @ dz1) / batch_size       # (13, 32)
        db1 = dz1.mean(axis=0)                     # (32,)

        # ---- Update weights (gradient descent) ----
        # Move each weight slightly in the direction that reduces the loss.
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, X):
        """Predict class labels (0 or 1)."""
        probs = self.forward(X)
        return (probs >= 0.5).astype(float)

    def get_weights(self):
        """Export weights for Rust integration."""
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }


# ============================================================================
# PART 4: LOSS FUNCTION
#
# Binary Cross-Entropy: -[y * log(p) + (1-y) * log(1-p)]
#
# Intuition: if y=1 and p=0.99, loss is tiny (correct and confident).
# If y=1 and p=0.01, loss is huge (wrong and confident — maximum penalty).
# The loss punishes confident wrong answers more than uncertain ones.
# ============================================================================

def binary_cross_entropy(y_true, y_pred):
    """Compute BCE loss. Clip predictions to avoid log(0)."""
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def accuracy(y_true, y_pred_probs):
    """Compute classification accuracy."""
    preds = (y_pred_probs >= 0.5).astype(float).flatten()
    return np.mean(preds == y_true.flatten())


# ============================================================================
# PART 5: TRAINING LOOP
#
# For each epoch:
#   1. Shuffle training data for variety
#   2. Process in mini-batches for stable gradients
#   3. Forward pass → compute loss → backward pass → update weights
#   4. Check validation accuracy
#   5. Stop if validation accuracy does not improve for PATIENCE epochs
#
# Early stopping limits overfitting by using the validation set as a signal.
# ============================================================================

def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr, patience):
    """Train the MLP with mini-batch gradient descent and early stopping."""

    # Reshape y to column vectors for matrix math
    y_train = y_train.reshape(-1, 1)
    y_val_col = y_val.reshape(-1, 1)

    best_val_acc = 0
    best_weights = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    n = len(X_train)

    for epoch in range(epochs):
        # Shuffle training data each epoch
        perm = np.random.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        # Mini-batch training
        epoch_losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]

            # Forward
            pred = model.forward(X_batch)
            loss = binary_cross_entropy(y_batch, pred)
            epoch_losses.append(loss)

            # Backward + update
            model.backward(y_batch, lr)

        # Epoch metrics
        train_loss = np.mean(epoch_losses)
        train_pred = model.forward(X_train)
        train_acc = accuracy(y_train, train_pred)

        val_pred = model.forward(X_val)
        val_loss = binary_cross_entropy(y_val_col, val_pred)
        val_acc = accuracy(y_val_col, val_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {
                "W1": model.W1.copy(), "b1": model.b1.copy(),
                "W2": model.W2.copy(), "b2": model.b2.copy(),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if epoch % 10 == 0 or patience_counter >= patience:
            print(f"  Epoch {epoch:>3d}  |  "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
                  f"{'← best' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_weights:
        model.W1 = best_weights["W1"]
        model.b1 = best_weights["b1"]
        model.W2 = best_weights["W2"]
        model.b2 = best_weights["b2"]

    print(f"\n  Best validation accuracy: {best_val_acc:.4f}")
    return history


# ============================================================================
# PART 6: EVALUATION
# ============================================================================

def evaluate(model, X, y, traces, label="Test"):
    """Evaluate model, with per-dataset breakdown."""
    y_col = y.reshape(-1, 1)
    pred_probs = model.forward(X)
    preds = (pred_probs >= 0.5).astype(float).flatten()

    overall_acc = np.mean(preds == y)
    print(f"\n  {label} Accuracy: {overall_acc:.4f}")

    # Per-dataset breakdown
    unique_traces = np.unique(traces)
    if len(unique_traces) > 1:
        print(f"\n  Per-dataset {label.lower()} accuracy:")
        results = {}
        for t in sorted(unique_traces):
            mask = traces == t
            t_acc = np.mean(preds[mask] == y[mask])
            n = mask.sum()
            results[t] = t_acc
            status = "⚠️" if t_acc < 0.80 else "✓"
            print(f"    {status} {t}: {t_acc:.4f}  (n={n})")
        return overall_acc, results

    return overall_acc, {}


# ============================================================================
# PART 7: FEATURE IMPORTANCE ANALYSIS
#
# Three methods, each tells you something different:
#
# 1. Weight magnitude analysis
#    Look at W1 (the first layer weights). Each column of W1 corresponds to
#    one input feature. If a feature's weights are all near zero, the network
#    is ignoring it. Simple but crude.
#
# 2. Ablation study (drop-one-feature)
#    Train the model 13 times, each time zeroing out one feature.
#    Measure how much accuracy drops. Big drop = important feature.
#    This is the most honest test but expensive.
#
# 3. Permutation importance
#    Shuffle one feature column randomly, measure accuracy drop.
#    Repeat for each feature. No retraining needed — fast and reliable.
#    This is what sklearn does under the hood.
# ============================================================================

def weight_magnitude_analysis(model, feature_names):
    """
    Look at how much the network's first layer attends to each feature.
    Sum of absolute weights per input feature across all hidden neurons.
    """
    W1_abs = np.abs(model.W1)                    # (13, 32)
    feature_importance = W1_abs.sum(axis=1)       # (13,) — sum across hidden neurons
    feature_importance /= feature_importance.sum() # normalize to sum to 1

    print("\n  Weight Magnitude Analysis:")
    print(f"  {'Feature':<40} {'Importance':>12} {'Visual':>30}")
    print(f"  {'-'*40} {'-'*12} {'-'*30}")

    # Sort by importance
    order = np.argsort(feature_importance)[::-1]
    for idx in order:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        imp = feature_importance[idx]
        bar = '█' * int(imp / feature_importance.max() * 25)
        print(f"  {name:<40} {imp:>12.4f} {bar:>30}")

    return feature_importance


def permutation_importance_analysis(model, X_val, y_val, feature_names, n_repeats=10):
    """
    For each feature:
      1. Shuffle that feature's values randomly
      2. Measure how much accuracy drops
      3. Repeat n_repeats times for stability

    Big accuracy drop = model relies heavily on this feature.
    No drop = feature is redundant or uninformative.
    """
    baseline_acc = accuracy(y_val.reshape(-1, 1), model.forward(X_val))

    print(f"\n  Permutation Importance (baseline acc: {baseline_acc:.4f}):")
    print(f"  {'Feature':<40} {'Acc Drop':>10} {'± Std':>10} {'Visual':>25}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*25}")

    importances = {}

    for i in range(X_val.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_val.copy()
            X_shuffled[:, i] = np.random.permutation(X_shuffled[:, i])
            shuffled_acc = accuracy(y_val.reshape(-1, 1), model.forward(X_shuffled))
            drops.append(baseline_acc - shuffled_acc)

        mean_drop = np.mean(drops)
        std_drop = np.std(drops)
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        importances[name] = {"mean": mean_drop, "std": std_drop, "index": i}

    # Sort by importance
    sorted_feats = sorted(importances.items(), key=lambda x: -x[1]["mean"])
    for name, vals in sorted_feats:
        bar = '█' * max(0, int(vals["mean"] / max(sorted_feats[0][1]["mean"], 1e-6) * 20))
        print(f"  {name:<40} {vals['mean']:>10.4f} {vals['std']:>10.4f} {bar:>25}")

    return importances


def ablation_study(X_train, y_train, X_val, y_val, feature_names):
    """
    Train a separate model with each feature removed.
    This is the gold standard for feature importance — it shows what
    happens when the model genuinely can't see a feature (not just
    when values are shuffled).
    """
    print("\n  Ablation Study (train without each feature):")
    print("  This takes a while — training 13+ models...")

    # Baseline: full model
    baseline_model = MLP(X_train.shape[1], HIDDEN_SIZE)
    train(baseline_model, X_train, y_train.flatten(), X_val, y_val.flatten(),
          epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, patience=PATIENCE)
    baseline_acc = accuracy(y_val.reshape(-1, 1), baseline_model.forward(X_val))

    print(f"\n  Baseline (all features): {baseline_acc:.4f}")
    print(f"\n  {'Dropped Feature':<40} {'Val Acc':>10} {'Δ Acc':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")

    results = {}
    for i in range(X_train.shape[1]):
        # Remove feature i
        X_train_dropped = np.delete(X_train, i, axis=1)
        X_val_dropped = np.delete(X_val, i, axis=1)

        dropped_model = MLP(X_train_dropped.shape[1], HIDDEN_SIZE)

        # Train quietly
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        train(dropped_model, X_train_dropped, y_train.flatten(),
              X_val_dropped, y_val.flatten(),
              epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, patience=PATIENCE)
        sys.stdout = old_stdout

        dropped_acc = accuracy(y_val.reshape(-1, 1), dropped_model.forward(X_val_dropped))
        delta = baseline_acc - dropped_acc

        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        results[name] = {"acc_without": dropped_acc, "delta": delta}

        sign = "↓" if delta > 0.001 else ("↑" if delta < -0.001 else "≈")
        print(f"  {name:<40} {dropped_acc:>10.4f} {delta:>+10.4f} {sign}")

    # Summary
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["delta"])
    print(f"\n  Features ranked by impact when removed:")
    for rank, (name, vals) in enumerate(sorted_results, 1):
        print(f"    {rank}. {name}: accuracy drops by {vals['delta']:+.4f}")

    # Identify droppable features
    droppable = [name for name, vals in sorted_results if vals["delta"] < 0.005]
    if droppable:
        print(f"\n  Candidates for removal (accuracy drop < 0.5%):")
        for name in droppable:
            print(f"    - {name}")

    return results


# ============================================================================
# PART 8: FEATURE CORRELATION ANALYSIS
#
# If two features are highly correlated, they carry the same information.
# Dropping one won't hurt the model. This helps you identify redundancy
# among your 13 features.
# ============================================================================

def correlation_analysis(X, feature_names):
    """Find highly correlated feature pairs."""
    corr = np.corrcoef(X.T)  # (13, 13) correlation matrix

    print("\n  Highly correlated feature pairs (|r| > 0.7):")
    pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = corr[i, j]
            if abs(r) > 0.7:
                pairs.append((feature_names[i], feature_names[j], r))
                print(f"    {feature_names[i]} ↔ {feature_names[j]}: r = {r:.3f}")

    if not pairs:
        print("    None found.")

    return pairs


# ============================================================================
# PART 9: LEAN MODEL EXPERIMENT
#
# After identifying top features, train a model with only those features.
# Compare against the full model. If accuracy barely drops, the reduced
# feature set is likely adequate.
# ============================================================================

def lean_model_experiment(X_train, y_train, X_val, y_val, traces_val,
                          feature_names, top_k=5, top_indices=None):
    """Train model with only the top-k features."""
    if top_indices is None:
        print(f"\n  (Using first {top_k} features by default — "
              f"pass top_indices from importance analysis for better results)")
        top_indices = list(range(top_k))

    top_names = [feature_names[i] for i in top_indices[:top_k]]
    print(f"\n  Lean Model: training with {top_k} features:")
    for name in top_names:
        print(f"    - {name}")

    X_train_lean = X_train[:, top_indices[:top_k]]
    X_val_lean = X_val[:, top_indices[:top_k]]

    lean = MLP(top_k, HIDDEN_SIZE)
    train(lean, X_train_lean, y_train.flatten(), X_val_lean, y_val.flatten(),
          epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, patience=PATIENCE)

    evaluate(lean, X_val_lean, y_val, traces_val, label="Lean Model Val")
    return lean


# ============================================================================
# PART 10: EXPORT FOR RUST
#
# Once you have good weights, export them. In Rust, you only need the
# forward pass — about 20 lines of code:
#
#   fn predict(x: &[f32; 13], w1: &[[f32; 32]; 13], b1: &[f32; 32],
#              w2: &[f32; 32], b2: f32) -> f32 {
#       let mut hidden = [0.0f32; 32];
#       for j in 0..32 {
#           for i in 0..13 {
#               hidden[j] += x[i] * w1[i][j];
#           }
#           hidden[j] = (hidden[j] + b1[j]).max(0.0);  // ReLU
#       }
#       let mut out = b2;
#       for j in 0..32 {
#           out += hidden[j] * w2[j];
#       }
#       1.0 / (1.0 + (-out).exp())  // sigmoid
#   }
# ============================================================================

def export_weights(model, mean, std, feature_names, path="model_weights.json"):
    """
    Export model weights + normalization stats to JSON.
    Load this in Rust to run inference without Python.
    """
    export = {
        "architecture": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "activation": "relu",
            "output_activation": "sigmoid",
        },
        "feature_names": feature_names,
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "weights": model.get_weights(),
    }

    with open(path, 'w') as f:
        json.dump(export, f, indent=2)

    print(f"\n  Weights exported to {path}")
    print(f"  Load in Rust: normalize input with mean/std, then run forward pass.")


# ============================================================================
# MAIN — run everything
# ============================================================================

def main():
    print("=" * 65)
    print("  FROM-SCRATCH MLP — PAIRWISE CACHE EVICTION")
    print("=" * 65)

    # --- Load data ---
    print("\n[1/8] Loading data...")
    X, y, traces = load_data(DATA_DIR)
    feature_names = FEATURE_NAMES[:X.shape[1]]  # handle mismatched dims gracefully

    # --- Check balance ---
    print("\n[2/8] Checking dataset balance...")
    check_balance(y, traces)

    # --- Correlation analysis (on raw features) ---
    print("\n[3/8] Feature correlation analysis...")
    correlated_pairs = correlation_analysis(X, feature_names)

    # --- Split ---
    print("\n[4/8] Train/val/test split...")
    (X_train, y_train, traces_train,
     X_val, y_val, traces_val,
     X_test, y_test, traces_test) = train_val_test_split(X, y, traces)

    # --- Normalize ---
    print("\n[5/8] Normalizing (using training stats only)...")
    X_train, X_val, X_test, norm_mean, norm_std = normalize(X_train, X_val, X_test)

    # --- Train ---
    print("\n[6/8] Training MLP from scratch...")
    model = MLP(X_train.shape[1], HIDDEN_SIZE)
    history = train(model, X_train, y_train, X_val, y_val,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    lr=LEARNING_RATE, patience=PATIENCE)

    # --- Evaluate on validation ---
    print("\n[7/8] Evaluation...")
    val_acc, val_per_dataset = evaluate(model, X_val, y_val, traces_val, label="Validation")

    # --- Feature importance ---
    print("\n[8/8] Feature importance analysis...")

    # Method 1: Weight magnitudes (fast, rough)
    weight_imp = weight_magnitude_analysis(model, feature_names)

    # Method 2: Permutation importance (fast, reliable)
    perm_imp = permutation_importance_analysis(model, X_val, y_val, feature_names)

    # Get top features from permutation importance
    sorted_by_perm = sorted(perm_imp.items(), key=lambda x: -x[1]["mean"])
    top_indices = [item[1]["index"] for item in sorted_by_perm]

    # Method 3: Ablation study (slow, gold standard)
    # Uncomment to run — takes a while since it trains 13 separate models
    # ablation_results = ablation_study(X_train, y_train, X_val, y_val, feature_names)

    # --- Lean model experiment ---
    print("\n" + "=" * 65)
    print("  LEAN MODEL EXPERIMENT (top 5 features only)")
    print("=" * 65)
    lean = lean_model_experiment(
        X_train, y_train, X_val, y_val, traces_val,
        feature_names, top_k=5, top_indices=top_indices
    )

    # --- Final test evaluation ---
    print("\n" + "=" * 65)
    print("  FINAL TEST SET EVALUATION (run once!)")
    print("=" * 65)
    test_acc, test_per_dataset = evaluate(model, X_test, y_test, traces_test, label="Test (Full)")
    evaluate(lean, X_test[:, top_indices[:5]], y_test, traces_test, label="Test (Lean)")

    # --- Export weights ---
    print("\n" + "=" * 65)
    print("  EXPORT")
    print("=" * 65)
    export_weights(model, norm_mean, norm_std, feature_names, "model_weights.json")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("\n  1. Compare full and lean model accuracy.")
    print("  2. Review feature importance and model behavior.")
    print("  3. Check per-dataset performance for different workloads.")
    print("  4. Confirm Rust integration path with exported weights.")
    print("  5. Decide whether a smaller feature set is acceptable for production.")


if __name__ == "__main__":
    main()