# scripts/run_trust_pipeline.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ------------------------------------------------------------
# Project path bootstrap
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Imports (adjust if your package layout differs)
# ------------------------------------------------------------
from trust.pipeline import execute_trust_algorithm
from utils.milp_utils import (
    TRUST_MODE_FULL,
    TRUST_MODE_SENSORS,
    TRUST_MODE_WINDOWS,
    MLP_REBUILD,
    MLP_TRANSFER,
    BASELINE_MODE_NONE,
    BASELINE_MODE_CAPPED,
)
from utils.trust_utils import mlp_forward_numpy  # optional, not strictly needed
from utils.mlp_utils import RANDOM_SEED

# =========================
# Small helpers
# =========================

def _window_tag(seq_len: int, step: int) -> str:
    return f"W{int(seq_len)}_step{int(step)}"


def _hidden_tag(hidden_layers: List[int]) -> str:
    # Must match hidden_sizes_to_tag() behavior used in baseline training.
    # Your baseline uses hidden_sizes_to_tag((10,10)) -> probably "h10_10".
    return "h" + "_".join(str(int(x)) for x in hidden_layers)


def _load_processed_dataset(processed_root: Path, dataset_name: str, window_tag: str) -> Dict[str, np.ndarray]:
    """
    Load dataset.npz from:
        datasets/processed/<dataset_name>/<window_tag>/dataset.npz

    Required keys:
        - X_train_mw, y_train
        - X_val_mw,   y_val
    """
    ds_dir = processed_root / dataset_name / window_tag
    npz_path = ds_dir / "dataset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset.npz: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    arrays = {k: data[k] for k in data.files}

    required = ["X_train_mw", "y_train", "X_val_mw", "y_val"]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise KeyError(f"dataset.npz missing keys {missing} in {npz_path}")

    return arrays


def _load_baseline_artifacts(
    baselines_root: Path,
    dataset_name: str,
    window_tag: str,
    hidden_layers: List[int],
) -> Dict[str, Any]:
    hs_tag = "h" + "_".join(str(x) for x in hidden_layers)
    art_path = baselines_root / dataset_name / window_tag / hs_tag / "baseline_mlp_artifacts.npz"
    if not art_path.exists():
        raise FileNotFoundError(f"Baseline artifact not found: {art_path}")

    data = np.load(art_path, allow_pickle=True)

    # ---- Parse meta_json ----
    if "meta_json" not in data.files:
        raise KeyError(f"baseline_mlp_artifacts.npz missing 'meta_json': {art_path}")

    meta_raw = data["meta_json"]
    # meta_raw is typically a 0-d array of dtype object
    meta_str = str(meta_raw.item())
    meta = json.loads(meta_str)

    # ---- Rebuild weights dict (layer -> {W,b}) ----
    weights: Dict[str, Dict[str, np.ndarray]] = {}
    for k in data.files:
        if k == "meta_json":
            continue
        if k.endswith("__W"):
            layer = k[:-3]
            weights.setdefault(layer, {})["W"] = data[k]
        elif k.endswith("__b"):
            layer = k[:-3]
            weights.setdefault(layer, {})["b"] = data[k]

    # Safety check (optional)
    if "final_metrics" not in meta:
        # still return something predictable
        meta["final_metrics"] = {}

    return {
        "artifact_path": str(art_path),
        "meta": meta,
        "weights": weights,
        "final_metrics": meta.get("final_metrics", {}),
        "input_dim": meta.get("input_dim", None),
        "hidden_sizes": meta.get("hidden_sizes", None),
        "window_tag": meta.get("window_tag", None),
        "dataset_name": meta.get("dataset_name", None),
    }


def _baseline_weights_to_trust_format(weights: Dict[str, Dict[str, np.ndarray]], hidden_layers: List[int]):
    """
    Convert baseline weights dict into TRUST format lists:
        hidden_Ws = [W1, W2, ..., W_out_vector]
        hidden_bs = [b1, b2, ..., b_out_scalar_like]
    """
    hidden_Ws: List[np.ndarray] = []
    hidden_bs: List[np.ndarray] = []

    for i in range(1, len(hidden_layers) + 1):
        lk = f"hidden_l{i}"
        if lk not in weights:
            raise KeyError(f"Missing '{lk}' in baseline weights.")
        hidden_Ws.append(np.asarray(weights[lk]["W"], dtype=float))
        hidden_bs.append(np.asarray(weights[lk]["b"], dtype=float).reshape(-1))

    W_out = np.asarray(weights["rul_output"]["W"], dtype=float).reshape(-1)
    b_out = np.asarray(weights["rul_output"]["b"], dtype=float).reshape(-1)
    hidden_Ws.append(W_out)
    hidden_bs.append(b_out)

    return hidden_Ws, hidden_bs

def _weights_dict_to_weights_list(weights: Dict[str, Dict[str, np.ndarray]], hidden_layers: List[int]) -> List[np.ndarray]:
    weights_list: List[np.ndarray] = []

    # Hidden layers
    for i in range(1, len(hidden_layers) + 1):
        k = f"hidden_l{i}"
        weights_list.append(np.asarray(weights[k]["W"]))
        weights_list.append(np.asarray(weights[k]["b"]))

    # Output
    weights_list.append(np.asarray(weights["rul_output"]["W"]))
    weights_list.append(np.asarray(weights["rul_output"]["b"]))

    return weights_list


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TRUST pipeline on a preprocessed dataset using a fixed baseline MLP."
    )

    # Data selection
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., FD001).")
    parser.add_argument("--processed-root", type=str, default="datasets/processed", help="Processed datasets root.")
    parser.add_argument("--seq-len", type=int, required=True, help="Window length W used during preprocessing.")
    parser.add_argument("--step", type=int, required=True, help="Window step used during preprocessing.")

    # Baseline selection
    parser.add_argument(
        "--baselines-root",
        type=str,
        default="mlp_baselines",
        help="Baselines root folder.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        required=True,
        help="Hidden layer sizes used for baseline and surrogate MLP (e.g., --hidden 10 10).",
    )

    # TRUST configuration
    parser.add_argument(
        "--mode",
        type=str,
        default=TRUST_MODE_SENSORS,
        choices=[TRUST_MODE_FULL, TRUST_MODE_SENSORS, TRUST_MODE_WINDOWS],
        help="TRUST selection mode.",
    )
    parser.add_argument(
        "--mlp-mode",
        type=str,
        default=MLP_REBUILD,
        choices=[MLP_REBUILD, MLP_TRANSFER],
        help="Surrogate MLP training mode.",
    )
    parser.add_argument("--C", type=int, default=50, help="Number of KMeans centroids.")
    parser.add_argument("--beta", type=float, default=0.5, help="ReLU bounds expansion factor.")
    parser.add_argument("--milp-time-cap", type=int, default=60, help="MILP-2 time limit (seconds).")

    # Baseline cap
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default=BASELINE_MODE_CAPPED,
        choices=[BASELINE_MODE_NONE, BASELINE_MODE_CAPPED],
        help="Whether to cap surrogate MAE by baseline cluster MAE.",
    )
    parser.add_argument(
        "--baseline-slack",
        type=float,
        default=0.0,
        help="Allowed improvement fraction over baseline (0.10 means allow 10% better).",
    )

    # Surrogate training hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    # Output / run control
    parser.add_argument("--results-root", type=str, default=None, help="Override results root folder.")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if present.")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")

    args = parser.parse_args()

    dataset_name = str(args.dataset)
    window_tag = _window_tag(args.seq_len, args.step)
    hidden_layers = [int(x) for x in args.hidden]

    processed_root = Path(args.processed_root)
    baselines_root = Path(args.baselines_root)
    results_root = Path(args.results_root) if args.results_root is not None else None

    # ------------------------------------------------------------
    # 1) Load processed dataset (mw arrays)
    # ------------------------------------------------------------
    arrays = _load_processed_dataset(
        processed_root=processed_root,
        dataset_name=dataset_name,
        window_tag=window_tag,
    )

    X_train = np.asarray(arrays["X_train_mw"], dtype=np.float32)
    y_train = np.asarray(arrays["y_train"], dtype=np.float32).reshape(-1)
    X_val = np.asarray(arrays["X_val_mw"], dtype=np.float32)
    y_val = np.asarray(arrays["y_val"], dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------
    # 2) Load baseline artifacts and convert to TRUST format
    # ------------------------------------------------------------
    baseline_art = _load_baseline_artifacts(
        baselines_root=baselines_root,
        dataset_name=dataset_name,
        window_tag=window_tag,
        hidden_layers=hidden_layers,
    )
    baseline_hidden_Ws, baseline_hidden_bs = _baseline_weights_to_trust_format(
        weights=baseline_art["weights"],
        hidden_layers=hidden_layers,
    )
    baseline_metrics = baseline_art.get("final_metrics", None)
    baseline_weights_list = _weights_dict_to_weights_list(
        weights=baseline_art["weights"],
        hidden_layers=hidden_layers,
    )


    if args.verbose:
        print("\n=== INPUTS ===")
        print("Dataset:", dataset_name)
        print("Window tag:", window_tag)
        print("X_train shape:", tuple(X_train.shape))
        print("X_val shape:", tuple(X_val.shape))
        print("Baseline artifacts:", baseline_art["artifact_path"])
        print("Hidden layers:", hidden_layers)

    # ------------------------------------------------------------
    # 3) Run TRUST
    # ------------------------------------------------------------
    result = execute_trust_algorithm(
        dataset_name=dataset_name,
        window_tag=window_tag,
        results_root=results_root,

        hidden_layers=tuple(hidden_layers),
        mode=str(args.mode),
        C=int(args.C),
        beta=float(args.beta),

        mlp_mode=str(args.mlp_mode),
        milp_time_cap=int(args.milp_time_cap),

        learning_rate=float(args.lr),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),

        baseline_mode=str(args.baseline_mode),
        baseline_slack=float(args.baseline_slack),

        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,

        # Baseline
        baseline_hidden_Ws=baseline_hidden_Ws,
        baseline_hidden_bs=baseline_hidden_bs,

        baseline_metrics=baseline_metrics,
        baseline_weights_list=baseline_weights_list,

        seed=int(args.seed),
        verbose=bool(args.verbose),
        resume=bool(args.resume),  # enable once your pipeline implements resume
    )

    # ------------------------------------------------------------
    # 4) Print final path
    # ------------------------------------------------------------
    out_dir = result.get("out_dir", None)
    if out_dir is not None:
        print("\n[OK] TRUST finished. Results saved to:", out_dir)
    else:
        print("\n[OK] TRUST finished.")


if __name__ == "__main__":
    main()