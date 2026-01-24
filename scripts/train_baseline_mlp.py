# scripts/train_baseline_mlp.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.mlp_utils import (
    MLPTrainConfig,
    build_trust_mlp,
    compute_final_metrics,
    ensure_2d_targets,
    export_layer_weights_dict,
    hidden_sizes_to_tag,
    set_global_seed,
)

# Root = directory where script is executed
DEFAULT_OUTPUT_ROOT = Path.cwd() / "mlp_baselines"


def train_baseline_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dataset_name: str,
    window_tag: str,   
    hidden_sizes: Tuple[int, ...],
    config: Optional[MLPTrainConfig] = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, Any]:
    """
    Train a TRUST-compatible baseline MLP and store artifacts under:

        mlp_baselines/<dataset_name>/<window_tag>/<hidden_sizes_tag>/
    """
    if config is None:
        config = MLPTrainConfig()

    set_global_seed(config.seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = ensure_2d_targets(y_train)
    y_val = ensure_2d_targets(y_val)

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Input dimension mismatch between train and val sets")

    input_dim = X_train.shape[1]
    hs_tag = hidden_sizes_to_tag(hidden_sizes)

    model = build_trust_mlp(
        input_dim=input_dim,
        hidden_sizes=list(hidden_sizes),
        learning_rate=config.learning_rate,
    )

    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=config.verbose,
    )

    history = history_obj.history
    final_metrics = compute_final_metrics(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    weights = export_layer_weights_dict(model)

    # -------------------------
    # Folder structure
    # -------------------------
    out_dir = output_root / dataset_name / window_tag / hs_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir / "model.keras")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    meta = {
        "dataset_name": dataset_name,
        "window_tag": window_tag,  # <-- NEW
        "hidden_sizes": list(hidden_sizes),
        "hidden_sizes_tag": hs_tag,
        "input_dim": int(input_dim),
        "final_metrics": final_metrics,
    }

    payload = {"meta_json": np.array(json.dumps(meta), dtype=object)}
    for layer, params in weights.items():
        payload[f"{layer}__W"] = params["W"]
        payload[f"{layer}__b"] = params["b"]

    np.savez_compressed(out_dir / "baseline_mlp_artifacts.npz", **payload)

    return {
        "dataset_name": dataset_name,
        "window_tag": window_tag,  # <-- NEW
        "hidden_sizes": list(hidden_sizes),
        "weights": weights,
        "final_metrics": final_metrics,
        "paths": {
            "out_dir": str(out_dir),
        },
    }