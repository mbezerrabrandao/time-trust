from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

from scripts.train_baseline_mlp import train_baseline_mlp
from utils.mlp_utils import MLPTrainConfig


# =========================
# Dataset loading (generic)
# =========================

def load_processed_npz(dataset_dir: Path) -> Dict[str, np.ndarray]:
    """
    Loads a processed dataset folder that contains dataset.npz.
    Expected keys (at minimum):
        - X_train_flat
        - y_train
        - X_val_flat
        - y_val
    """
    npz_path = dataset_dir / "dataset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset.npz in: {dataset_dir}")

    data = np.load(npz_path, allow_pickle=True)
    arrays = {k: data[k] for k in data.files}

    required = ["X_train_flat", "y_train", "X_val_flat", "y_val"]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise KeyError(f"dataset.npz missing keys {missing} in {dataset_dir}")

    return arrays


def resolve_dataset_dirs(processed_root: Path, dataset_names: List[str], window_tag: str) -> List[Tuple[str, Path]]:
    """
    Maps each dataset name to its processed folder:
        processed_root/<dataset_name>/<window_tag>/
    Example:
        datasets/processed/FD001/W30_step1/
    """
    out: List[Tuple[str, Path]] = []
    for name in dataset_names:
        d = processed_root / name / window_tag
        if not d.exists():
            raise FileNotFoundError(f"Processed dataset folder not found: {d}")
        out.append((name, d))
    return out


# =========================
# Baseline runner
# =========================

def run_baselines_for_dataset(
    dataset_name: str,
    dataset_dir: Path,
    window_tag: str, 
    hidden_grid: List[Tuple[int, ...]],
    config: MLPTrainConfig,
) -> Dict[str, Any]:
    """
    Runs baseline training for a single dataset across a list of hidden_sizes configs.
    Returns a summary dictionary (final metrics per architecture).
    """
    arrays = load_processed_npz(dataset_dir)

    X_train = arrays["X_train_flat"]
    y_train = arrays["y_train"]
    X_val = arrays["X_val_flat"]
    y_val = arrays["y_val"]

    results: Dict[str, Any] = {"dataset_name": dataset_name, "runs": []}

    for hs in hidden_grid:
        info = train_baseline_mlp(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            dataset_name=dataset_name,
            window_tag=window_tag,
            hidden_sizes=hs,
            config=config,
        )
        results["runs"].append(
            {
                "hidden_sizes": list(hs),
                "final_metrics": info["final_metrics"],
                "out_dir": info["paths"]["out_dir"],
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline TRUST-compatible MLPs for all processed datasets."
    )

    parser.add_argument(
        "--processed-root",
        type=str,
        default="datasets/processed",
        help="Root folder containing processed datasets (default: datasets/processed).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["FD001", "FD002", "FD003", "FD004"],
        help="Dataset names to run (default: FD001 FD002 FD003 FD004).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=30,
        help="Window length W used during preprocessing (default: 30).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Window step used during preprocessing (default: 1).",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--verbose", type=int, default=0)

    # Output summary
    parser.add_argument(
        "--summary-out",
        type=str,
        default="mlp_baselines/cmapss_baseline_summary.json",
        help="Where to save the JSON summary (default: mlp_baselines/cmapss_baseline_summary.json).",
    )

    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    window_tag = f"W{args.seq_len}_step{args.step}"

    dataset_dirs = resolve_dataset_dirs(processed_root, args.datasets, window_tag)

    hidden_grid: List[Tuple[int, ...]] = [
        (10,),
        (10, 10),
        (10, 10, 10),
    ]

    config = MLPTrainConfig(
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=args.verbose,
    )

    all_results: Dict[str, Any] = {
        "window_tag": window_tag,
        "datasets": args.datasets,
        "hidden_grid": [list(h) for h in hidden_grid],
        "train_config": {
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "verbose": config.verbose,
        },
        "results": [],
    }

    for dataset_name, dataset_dir in dataset_dirs:
        print(f"\n=== Dataset: {dataset_name} | {dataset_dir} ===")
        res = run_baselines_for_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            window_tag=window_tag,
            hidden_grid=hidden_grid,
            config=config,
        )
        all_results["results"].append(res)

    # Save summary
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== DONE ===")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()