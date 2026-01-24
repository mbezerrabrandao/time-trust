"""
inspect_mlp_baselines.py

Utility script to inspect trained baseline MLPs:
- Loads model.keras to print architecture
- Loads baseline_mlp_artifacts.npz to show final metrics

Run from project root:
    python scripts/inspect_mlp_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf


# -------------------------
# Fix imports when run from scripts/
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------
# Helpers
# -------------------------

def load_meta_from_npz(npz_path: Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    if "meta_json" not in data:
        raise KeyError(f"meta_json not found in {npz_path}")
    return json.loads(data["meta_json"].item())


def print_model_architecture(model: tf.keras.Model) -> None:
    print("  Architecture:")
    for i, layer in enumerate(model.layers):
        name = layer.name
        cls = layer.__class__.__name__

        # Robust shape extraction
        if hasattr(layer, "output"):
            try:
                shape = tuple(layer.output.shape)
            except Exception:
                shape = "?"
        else:
            shape = "?"

        print(f"    [{i:02d}] {name:<15} | {cls:<12} | output_shape={shape}")


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("  Final metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<12}: {v:.4f}")
        else:
            print(f"    {k:<12}: {v}")


# -------------------------
# Main inspection logic
# -------------------------

def inspect_all_baselines(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Baseline root not found: {root}")

    print("=== Inspecting MLP baselines ===")
    print("Root:", root)
    print()

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        print(f"\n=== DATASET: {dataset_dir.name} ===")

        for arch_dir in sorted(dataset_dir.iterdir()):
            if not arch_dir.is_dir():
                continue

            print(f"\n- Architecture folder: {arch_dir.name}")

            model_path = arch_dir / "model.keras"
            npz_path = arch_dir / "baseline_mlp_artifacts.npz"

            if not model_path.exists() or not npz_path.exists():
                print("  [SKIP] Missing model.keras or baseline_mlp_artifacts.npz")
                continue

            # Load model
            model = tf.keras.models.load_model(model_path, compile=False)

            # Load metadata
            meta = load_meta_from_npz(npz_path)

            # Print info
            print_model_architecture(model)

            if "final_metrics" in meta:
                print_metrics(meta["final_metrics"])
            else:
                print("  [WARN] final_metrics not found in meta")

            print(f"  Input dim       : {meta.get('input_dim')}")
            print(f"  Hidden sizes    : {meta.get('hidden_sizes')}")
            print(f"  Dataset name    : {meta.get('dataset_name')}")


def main() -> None:
    baseline_root = PROJECT_ROOT / "mlp_baselines"
    inspect_all_baselines(baseline_root)


if __name__ == "__main__":
    main()