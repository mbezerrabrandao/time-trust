# scripts/baseline_rankings.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from utils.ranking_utils import (
    build_group_ranking_from_mlp_first_layer,
    build_group_ranking_random_average,
)


# -------------------------
# Small helpers
# -------------------------

def _window_tag(seq_len: int, step: int) -> str:
    return f"W{int(seq_len)}_step{int(step)}"


def _hidden_sizes_to_tag(hidden: Sequence[int]) -> str:
    # Matches your existing convention: (10,10) -> "h10_10"
    return "h" + "_".join(str(int(x)) for x in hidden)


def _load_processed_dataset(processed_root: Path, dataset_name: str, window_tag: str) -> Dict[str, np.ndarray]:
    ds_dir = processed_root / dataset_name / window_tag
    npz_path = ds_dir / "dataset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset.npz at: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    arrays = {k: data[k] for k in data.files}

    required = ["X_train_mw", "y_train", "X_val_mw", "y_val"]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise KeyError(f"dataset.npz missing keys={missing}. Available keys={list(arrays.keys())}")

    return arrays


def _load_baseline_artifacts(
    baselines_root: Path,
    dataset_name: str,
    window_tag: str,
    hidden_layers: Sequence[int],
) -> Dict[str, Any]:
    hs_tag = _hidden_sizes_to_tag(hidden_layers)
    art_path = baselines_root / dataset_name / window_tag / hs_tag / "baseline_mlp_artifacts.npz"
    if not art_path.exists():
        raise FileNotFoundError(f"Missing baseline artifacts at: {art_path}")

    npz = np.load(art_path, allow_pickle=True)
    files = list(npz.files)

    if "meta_json" not in files:
        raise KeyError(f"baseline_mlp_artifacts.npz missing meta_json. Files={files}")

    meta = json.loads(str(npz["meta_json"].item()))

    # First Dense layer weights saved under:
    # "hidden_l1__W", "hidden_l1__b"
    w_key = "hidden_l1__W"
    b_key = "hidden_l1__b"
    if w_key not in files or b_key not in files:
        raise KeyError(f"Missing {w_key}/{b_key} in baseline artifacts. Files={files}")

    W1 = np.asarray(npz[w_key], dtype=float)  # shape: (input_dim, hidden_1)
    b1 = np.asarray(npz[b_key], dtype=float).reshape(-1)

    return {
        "artifact_path": str(art_path),
        "meta": meta,
        "W1": W1,
        "b1": b1,
        "final_metrics": meta.get("final_metrics", None),
    }


# -------------------------
# Main execution
# -------------------------

def build_and_save_rankings(
    *,
    processed_root: Path,
    baselines_root: Path,
    dataset_name: str,
    seq_len: int,
    step: int,
    hidden_layers: Sequence[int],
    n_random_trials: int,
    seed: int,
    random_mode: str,
    verbose: bool,
) -> Dict[str, Any]:
    window_tag = _window_tag(seq_len, step)
    hs_tag = _hidden_sizes_to_tag(hidden_layers)

    arrays = _load_processed_dataset(processed_root, dataset_name, window_tag)
    X_train_mw = np.asarray(arrays["X_train_mw"], dtype=np.float32)
    X_val_mw = np.asarray(arrays["X_val_mw"], dtype=np.float32)

    if X_train_mw.ndim != 3:
        raise ValueError(f"X_train_mw must be 3D [N,M,W], got {X_train_mw.shape}")

    # Authoritative M and W from processed dataset
    M0 = int(X_train_mw.shape[1])
    W0 = int(X_train_mw.shape[2])

    baseline = _load_baseline_artifacts(baselines_root, dataset_name, window_tag, hidden_layers)
    W1 = baseline["W1"]

    input_dim = int(W1.shape[0])
    if input_dim != M0 * W0:
        raise ValueError(
            "Baseline input_dim does not match processed dataset. "
            f"baseline input_dim={input_dim}, dataset M0*W0={M0*W0}, dataset M0={M0}, W0={W0}"
        )

    out_dir = baselines_root / dataset_name / window_tag / hs_tag / "rankings"
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_meta = {
        "dataset_name": dataset_name,
        "window_tag": window_tag,
        "hidden_layers": [int(x) for x in hidden_layers],
        "M": M0,
        "W": W0,
        "baseline_artifact_path": baseline["artifact_path"],
        "baseline_final_metrics": baseline.get("final_metrics", None),
        "random_avg": {"n_trials": int(n_random_trials), "seed": int(seed), "random_mode": str(random_mode)},
    }

    # Utils expect a list of hidden weight matrices, where [0] is W1
    baseline_hidden_Ws = [W1]

    # 1) Weight-based grouped ranking from first layer
    sensors_w_path = out_dir / "ranking_sensors_weights.json"
    windows_w_path = out_dir / "ranking_windows_weights.json"

    res_sensors_w = build_group_ranking_from_mlp_first_layer(
        baseline_hidden_Ws=baseline_hidden_Ws,
        M0=M0,
        W0=W0,
        group_mode="sensors",
        out_json=sensors_w_path,
        extra_meta={**extra_meta, "method": "mlp_first_layer_grouped"},
    )

    res_windows_w = build_group_ranking_from_mlp_first_layer(
        baseline_hidden_Ws=baseline_hidden_Ws,
        M0=M0,
        W0=W0,
        group_mode="windows",
        out_json=windows_w_path,
        extra_meta={**extra_meta, "method": "mlp_first_layer_grouped"},
    )

    # 2) Random-average grouped ranking
    sensors_r_path = out_dir / "ranking_sensors_randomavg.json"
    windows_r_path = out_dir / "ranking_windows_randomavg.json"

    res_sensors_r = build_group_ranking_random_average(
        M0=M0,
        W0=W0,
        group_mode="sensors",
        n_runs=int(n_random_trials),
        seed=int(seed),
        random_mode=str(random_mode),
        out_json=sensors_r_path,
        extra_meta={**extra_meta, "method": "random_average_grouped"},
    )

    res_windows_r = build_group_ranking_random_average(
        M0=M0,
        W0=W0,
        group_mode="windows",
        n_runs=int(n_random_trials),
        seed=int(seed) + 1,
        random_mode=str(random_mode),
        out_json=windows_r_path,
        extra_meta={**extra_meta, "method": "random_average_grouped"},
    )

    saved = [str(sensors_w_path), str(windows_w_path), str(sensors_r_path), str(windows_r_path)]

    if verbose:
        print("\n=== BASELINE RANKINGS ===")
        print("Dataset:", dataset_name)
        print("Window tag:", window_tag)
        print("Hidden layers:", list(hidden_layers))
        print("Shapes:", "X_train_mw", tuple(X_train_mw.shape), "| X_val_mw", tuple(X_val_mw.shape))
        print("M0:", M0, "W0:", W0)
        print("Random trials:", int(n_random_trials), "| random_mode:", str(random_mode), "| seed:", int(seed))
        print("Saved:")
        for p in saved:
            print(" -", p)
        print("\nTop-5 sensors (weights):", [it["group_name"] for it in res_sensors_w.items[:5]])
        print("Top-5 windows (weights):", [it["group_name"] for it in res_windows_w.items[:5]])

    return {"out_dir": str(out_dir), "saved": saved, "meta": extra_meta}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build baseline group rankings (sensors/windows) using shared utils (MLP first layer + random-average)."
    )

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., FD001).")
    parser.add_argument("--processed-root", type=str, default="datasets/processed", help="Processed datasets root.")
    parser.add_argument("--baselines-root", type=str, default="mlp_baselines", help="Baselines root folder.")

    parser.add_argument("--seq-len", type=int, required=True, help="Window length used during preprocessing.")
    parser.add_argument("--step", type=int, required=True, help="Window step used during preprocessing.")
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        required=True,
        help="Hidden layer sizes used in the baseline MLP (e.g., --hidden 10 10 10).",
    )

    parser.add_argument("--random-trials", type=int, default=200, help="Number of random runs to average.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed base.")
    parser.add_argument(
        "--random-mode",
        type=str,
        default="per_group",
        choices=["per_feature", "per_group"],
        help="Random baseline mode: per_feature (aggregate random per-feature) or per_group (random per-group).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")

    args = parser.parse_args()

    build_and_save_rankings(
        processed_root=Path(args.processed_root),
        baselines_root=Path(args.baselines_root),
        dataset_name=str(args.dataset),
        seq_len=int(args.seq_len),
        step=int(args.step),
        hidden_layers=[int(x) for x in args.hidden],
        n_random_trials=int(args.random_trials),
        seed=int(args.seed),
        random_mode=str(args.random_mode),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()