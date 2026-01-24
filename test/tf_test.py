"""
tf_test.py

TensorFlow/Keras environment test:
1) Build a TRUST-compatible MLP (Dense + ReLU, linear output)
2) Train quickly on synthetic regression data
3) Export weights as a dict keyed by layer name:
       weights[layer_name] = {"W": W, "b": b}

Run:
    python -m test.tf_test
or:
    python test/tf_test.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class TFTestConfig:
    input_dim: int = 24
    hidden_sizes: Tuple[int, ...] = (10, 10)
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 15
    seed: int = 42
    verbose: int = 0  # 0 silent, 1 progress bar, 2 one line per epoch


# ----------------------------
# Reproducibility helpers
# ----------------------------

def set_global_seed(seed: int) -> None:
    """
    Sets seeds for reproducibility across NumPy and TensorFlow.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tf_runtime() -> None:
    """
    Optional runtime configuration.
    Keeps it simple and robust for a test script.
    """
    # Reduce TF verbosity if desired
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ----------------------------
# Model definition (TRUST-compatible)
# ----------------------------

def build_trust_mlp(
    input_dim: int,
    hidden_sizes: List[int] = [10],
    output_dim: int = 1,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build a fully-connected MLP compatible with the TRUST formulation.

    - Input:  flattened vector, shape (input_dim,)
    - Hidden: Dense layers with ReLU, named explicitly for TRUST export
    - Output: Dense layer with linear activation (RUL regression)
    - No dropout, batch norm, etc. (MILP-friendly)

    Returns a compiled Keras model.
    """
    inputs = layers.Input(shape=(input_dim,), name="mlp_input")

    x = inputs
    for i, units in enumerate(hidden_sizes, start=1):
        x = layers.Dense(
            units,
            activation="relu",
            name=f"hidden_l{i}",
        )(x)

    outputs = layers.Dense(
        output_dim,
        activation="linear",
        name="rul_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="trust_mlp")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ----------------------------
# Synthetic data
# ----------------------------

def make_synthetic_regression_data(
    n_samples: int,
    input_dim: int,
    noise_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a small regression dataset so the MLP can train fast.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, input_dim)).astype(np.float32)

    w_true = rng.normal(size=(input_dim,)).astype(np.float32)
    y = (X @ w_true) + 0.25 * np.sin(X[:, 0]) + rng.normal(scale=noise_std, size=(n_samples,)).astype(np.float32)

    return X, y.reshape(-1, 1).astype(np.float32)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Splits into train/val/test using simple slicing (deterministic).
    """
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


# ----------------------------
# Training + export (TRUST-ready weights dict)
# ----------------------------

def train_and_collect_weights(
    input_dim: int,
    hidden_sizes: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 128,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Trains the TRUST-compatible MLP and exports weights as:
        weights[layer_name] = {"W": W, "b": b}
    """
    model = build_trust_mlp(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=1,
        learning_rate=learning_rate,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Collect final metrics
    final_metrics = {
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "train_mae": float(history.history["mae"][-1]),
        "val_mae": float(history.history["val_mae"][-1]),
    }

    # Export weights by layer name (Dense layers only)
    layer_weights: Dict[str, Dict[str, np.ndarray]] = {}
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            W, b = weights
            layer_weights[layer.name] = {"W": W, "b": b}

    # Optional test evaluation
    test_results = None
    if X_test is not None and y_test is not None:
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        test_pred = model.predict(X_test, verbose=0).reshape(-1)
        test_results = {
            "test_loss": float(test_loss),
            "test_mae": float(test_mae),
            "test_predictions": test_pred,
        }

    return {
        "hidden_sizes": hidden_sizes,
        "model": model,
        "history": history.history,
        "weights": layer_weights,            # <-- TRUST / MILP input artifact
        "weights_list": model.get_weights(), # <-- optional raw list
        "final_metrics": final_metrics,
        "test_results": test_results,
    }


# ----------------------------
# Validation helpers
# ----------------------------

def summarize_exported_weights(weights: Dict[str, Dict[str, np.ndarray]]) -> None:
    """
    Prints shapes and sanity info for exported weights.
    """
    print("\n=== EXPORTED WEIGHTS (TRUST FORMAT) ===")
    if not weights:
        print("No weights exported.")
        return

    for layer_name, params in weights.items():
        W = params["W"]
        b = params["b"]
        print(f"- {layer_name}: W{tuple(W.shape)}, b{tuple(b.shape)}")


def run_tf_test(cfg: TFTestConfig) -> int:
    """
    Runs the TF/Keras MLP test and returns an exit code:
        0 if OK, 1 otherwise.
    """
    configure_tf_runtime()
    set_global_seed(cfg.seed)

    print("=== TensorFlow/Keras MLP Test (TRUST-compatible) ===")
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", bool(tf.config.list_physical_devices("GPU")))

    # Data
    X, y = make_synthetic_regression_data(
        n_samples=5000,
        input_dim=cfg.input_dim,
        noise_std=0.1,
        seed=cfg.seed,
    )
    split = train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.1)

    # Train + export
    info = train_and_collect_weights(
        input_dim=cfg.input_dim,
        hidden_sizes=list(cfg.hidden_sizes),
        X_train=split["X_train"],
        y_train=split["y_train"],
        X_val=split["X_val"],
        y_val=split["y_val"],
        X_test=split["X_test"],
        y_test=split["y_test"],
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
    )

    # Print summary
    print("\n=== FINAL METRICS ===")
    for k, v in info["final_metrics"].items():
        print(f"{k}: {v:.6f}")

    if info["test_results"] is not None:
        print("\n=== TEST METRICS ===")
        print(f"test_loss: {info['test_results']['test_loss']:.6f}")
        print(f"test_mae : {info['test_results']['test_mae']:.6f}")

    summarize_exported_weights(info["weights"])

    # Basic checks for a healthy export (must have hidden + output layers)
    expected_layers = [f"hidden_l{i+1}" for i in range(len(cfg.hidden_sizes))] + ["rul_output"]
    missing = [name for name in expected_layers if name not in info["weights"]]

    if missing:
        print("\n[FAIL] Missing expected layers in exported weights:", missing)
        return 1

    print("\n[PASS] TensorFlow/Keras MLP training and weights export succeeded.")
    return 0


def main() -> None:
    cfg = TFTestConfig()
    code = run_tf_test(cfg)
    sys.exit(code)


if __name__ == "__main__":
    main()