# scripts/rankings_metrics_evolution.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.mlp_utils import (
    MLPTrainConfig,
    build_trust_mlp,
    compute_final_metrics,
    ensure_2d_targets,
    set_global_seed,
)

# -------------------------
# IO helpers
# -------------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    _safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


# -------------------------
# Naming / parsing helpers
# -------------------------

def _hidden_tag_to_tuple(h_tag: str) -> Tuple[int, ...]:
    # h10_10 -> (10,10)
    if not h_tag.startswith("h"):
        raise ValueError(f"Invalid hidden tag: {h_tag}")
    parts = h_tag[1:].split("_")
    if any(p.strip() == "" for p in parts):
        raise ValueError(f"Invalid hidden tag: {h_tag}")
    return tuple(int(p) for p in parts)


def _is_hidden_tag(name: str) -> bool:
    return name.startswith("h") and any(ch.isdigit() for ch in name)


def _flatten_mw(X_mw: np.ndarray) -> np.ndarray:
    """
    X_mw: (N, M, W) -> (N, M*W) with sensor-major flatten (s*W + w).
    In NumPy default C-order, reshape does exactly that for (M, W).
    """
    X_mw = np.asarray(X_mw, dtype=np.float32)
    if X_mw.ndim != 3:
        raise ValueError(f"Expected X_mw to be 3D (N,M,W), got shape={X_mw.shape}")
    N, M, W = X_mw.shape
    return X_mw.reshape(N, M * W)


# -------------------------
# Ranking extraction
# -------------------------

def _extract_ranked_group_indices(ranking_json: Dict[str, Any]) -> List[int]:
    """
    Supports:
      - baseline rankings: items in "ranking" list with keys: index (1-based) or id like "s3"
      - timetrust selection ranking: items in "items" list with group_index_0based
    Returns list of 0-based group indices in the ranking order (best first).
    """
    if "items" in ranking_json and isinstance(ranking_json["items"], list):
        # timetrust format
        items = ranking_json["items"]
        out = []
        for it in items:
            if "group_index_0based" not in it:
                raise KeyError("timetrust ranking item missing group_index_0based")
            out.append(int(it["group_index_0based"]))
        return out

    if "ranking" in ranking_json and isinstance(ranking_json["ranking"], list):
        items = ranking_json["ranking"]
        out = []
        for it in items:
            if "index" in it:
                # baseline rankings store 1-based index
                out.append(int(it["index"]) - 1)
            elif "id" in it:
                # id like "s12" or "w5"
                s = str(it["id"])
                num = "".join([c for c in s if c.isdigit()])
                if num == "":
                    raise ValueError(f"Cannot parse id={s}")
                out.append(int(num) - 1)
            else:
                raise KeyError("baseline ranking item missing index/id")
        return out

    raise KeyError("Unrecognized ranking JSON format (expected 'items' or 'ranking').")


def _reverse_for_elimination(best_first: Sequence[int]) -> List[int]:
    """
    You want to 'apagar' following the ranking, where the first to turn off is last place.
    So elimination order is worst -> best, i.e., reverse(best_first).
    """
    return list(reversed([int(x) for x in best_first]))


# -------------------------
# Column removal mapping
# -------------------------

def _group_to_cols_sensors(sensor_idx_0: int, M: int, W: int) -> np.ndarray:
    # flat index: s*W + w
    s = int(sensor_idx_0)
    start = s * W
    return np.arange(start, start + W, dtype=int)


def _group_to_cols_windows(window_idx_0: int, M: int, W: int) -> np.ndarray:
    # flat index: s*W + w for all sensors s at fixed w
    w = int(window_idx_0)
    return (np.arange(M, dtype=int) * W + w).astype(int)


# -------------------------
# Training (from scratch)
# -------------------------

def _train_mlp_from_scratch(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: Tuple[int, ...],
    config: MLPTrainConfig,
) -> Dict[str, Any]:
    set_global_seed(config.seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = ensure_2d_targets(y_train)
    y_val = ensure_2d_targets(y_val)

    input_dim = int(X_train.shape[1])
    model = build_trust_mlp(
        input_dim=input_dim,
        hidden_sizes=list(hidden_sizes),
        learning_rate=config.learning_rate,
    )

    hist_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=config.verbose,
    )

    final_metrics = compute_final_metrics(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    return {
        "input_dim": int(input_dim),
        "final_metrics": final_metrics,
    }


# -------------------------
# One baseline folder (dataset/window/h_tag)
# -------------------------

def _find_rankings_folder(baselines_root: Path, dataset: str, window_tag: str, h_tag: str) -> Path:
    return baselines_root / dataset / window_tag / h_tag / "rankings"


def _rankings_to_process(rankings_dir: Path) -> Dict[str, Path]:
    """
    We expect these files if present:
      sensors: weights, randomavg, timetrust_selection
      windows: weights, randomavg, timetrust_selection
    We will only run those that exist.
    """
    candidates = {
        "sensors.weights": rankings_dir / "ranking_sensors_weights.json",
        "sensors.randomavg": rankings_dir / "ranking_sensors_randomavg.json",
        "sensors.timetrust": rankings_dir / "ranking_sensors_timetrust_selection.json",
        "windows.weights": rankings_dir / "ranking_windows_weights.json",
        "windows.randomavg": rankings_dir / "ranking_windows_randomavg.json",
        "windows.timetrust": rankings_dir / "ranking_windows_timetrust_selection.json",
    }
    return {k: p for k, p in candidates.items() if p.exists()}


def build_metrics_evolution_for_one_folder(
    *,
    baselines_root: Path,
    processed_root: Path,
    dataset: str,
    window_tag: str,
    h_tag: str,
    config: MLPTrainConfig,
    verbose: bool,
) -> Optional[Path]:
    rankings_dir = _find_rankings_folder(baselines_root, dataset, window_tag, h_tag)
    if not rankings_dir.exists():
        return None

    ranking_files = _rankings_to_process(rankings_dir)
    if len(ranking_files) == 0:
        return None

    # Load processed dataset once
    arrays = _load_processed_dataset(processed_root, dataset, window_tag)
    X_train_flat = _flatten_mw(arrays["X_train_mw"])
    X_val_flat = _flatten_mw(arrays["X_val_mw"])
    y_train = arrays["y_train"]
    y_val = arrays["y_val"]

    # Infer M, W
    M = int(arrays["X_train_mw"].shape[1])
    W = int(arrays["X_train_mw"].shape[2])

    hidden_sizes = _hidden_tag_to_tuple(h_tag)

    # Prepare output payload
    payload: Dict[str, Any] = {
        "type": "rankings_metrics_evolution",
        "dataset_name": dataset,
        "window_tag": window_tag,
        "hidden_tag": h_tag,
        "M": M,
        "W": W,
        "config": asdict(config),
        "definition": {
            "vector_position_0": "baseline MLP trained from scratch with all columns",
            "vector_position_k": "MLP trained from scratch after removing k groups (worst to best) according to ranking",
            "removal": "columns are removed from X_train/X_val, so input_dim changes each step",
            "flatten_order": "sensor-major: flat = s*W + w",
        },
        "results": {},  # filled below
    }

    # For each ranking type, train stepwise
    for key, rpath in ranking_files.items():
        group_mode, rank_src = key.split(".")  # sensors/windows + weights/randomavg/timetrust
        rjson = _load_json(rpath)
        best_first = _extract_ranked_group_indices(rjson)
        elim_order = _reverse_for_elimination(best_first)  # worst -> best

        n_groups = M if group_mode == "sensors" else W
        if len(best_first) != n_groups:
            raise ValueError(
                f"Ranking length mismatch for {key}: got {len(best_first)} expected {n_groups} "
                f"(dataset={dataset} window_tag={window_tag} h_tag={h_tag})"
            )

        # Start with all columns kept
        total_cols = M * W
        kept = np.ones((total_cols,), dtype=bool)

        metrics_vec: List[Dict[str, Any]] = []

        # Step 0: baseline (no removals)
        if verbose:
            print(f"[{dataset}/{window_tag}/{h_tag}] {key}: step 0 / {n_groups} (baseline)")
        train_out = _train_mlp_from_scratch(
            X_train=X_train_flat[:, kept],
            y_train=y_train,
            X_val=X_val_flat[:, kept],
            y_val=y_val,
            hidden_sizes=hidden_sizes,
            config=config,
        )
        metrics_vec.append(
            {
                "step": 0,
                "removed_groups_count": 0,
                "removed_group": None,
                "input_dim": int(train_out["input_dim"]),
                "final_metrics": train_out["final_metrics"],
            }
        )

        # Steps 1..n_groups: remove one more group each time
        removed_groups: List[int] = []
        for step_i, g0 in enumerate(elim_order, start=1):

            if group_mode == "sensors":
                cols = _group_to_cols_sensors(sensor_idx_0=g0, M=M, W=W)
            else:
                cols = _group_to_cols_windows(window_idx_0=g0, M=M, W=W)

            # Remove group
            kept[cols] = False
            removed_groups.append(int(g0))

            n_kept = int(np.sum(kept))

            if verbose:
                print(
                    f"[{dataset}/{window_tag}/{h_tag}] {key}: "
                    f"step {step_i}/{n_groups} remove g={g0} -> kept_cols={n_kept}"
                )

            # Bug Prevention: if no features left, stop
            if n_kept < 1:
                if verbose:
                    print("    -> no features left, stopping")
                break

            # Train (includes the case n_kept == 1)
            train_out = _train_mlp_from_scratch(
                X_train=X_train_flat[:, kept],
                y_train=y_train,
                X_val=X_val_flat[:, kept],
                y_val=y_val,
                hidden_sizes=hidden_sizes,
                config=config,
            )

            metrics_vec.append(
                {
                    "step": int(step_i),
                    "removed_groups_count": int(len(removed_groups)),
                    "removed_group": {
                        "group_index_0based": int(g0),
                        "group_index_1based": int(g0 + 1),
                        "group_name": f"{'s' if group_mode=='sensors' else 'w'}{g0+1}",
                    },
                    "input_dim": int(train_out["input_dim"]),
                    "final_metrics": train_out["final_metrics"],
                }
            )

            # If only 1 feature left, stop
            if n_kept == 1:
                if verbose:
                    print("    -> only 1 feature left, stopping")
                break

        payload["results"][key] = {
            "group_mode": group_mode,
            "ranking_source": rank_src,
            "ranking_file": str(rpath),
            "elimination_order_0based": elim_order,
            "elimination_order_rule": "worst_to_best (reverse of ranking list, where rank 1 is best)",
            "metrics_vector": metrics_vec,
        }

    out_path = rankings_dir / "rankings_metrics_evolution.json"
    _json_dump(out_path, payload)
    return out_path


# -------------------------
# Traverse baselines tree
# -------------------------

def _find_baseline_folders(baselines_root: Path) -> List[Tuple[str, str, str]]:
    """
    Finds tuples (dataset, window_tag, h_tag) where:
      mlp_baselines/<dataset>/<window_tag>/<h_tag>/rankings exists
    """
    out: List[Tuple[str, str, str]] = []
    if not baselines_root.exists():
        raise FileNotFoundError(f"baselines_root does not exist: {baselines_root}")

    for ds_dir in sorted([p for p in baselines_root.iterdir() if p.is_dir()]):
        for win_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()]):
            for h_dir in sorted([p for p in win_dir.iterdir() if p.is_dir() and _is_hidden_tag(p.name)]):
                rankings_dir = h_dir / "rankings"
                if rankings_dir.exists():
                    out.append((ds_dir.name, win_dir.name, h_dir.name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-train baseline MLP from scratch while removing groups according to rankings. "
                    "Writes rankings_metrics_evolution.json inside each rankings/ folder."
    )
    parser.add_argument("--baselines-root", type=str, default="mlp_baselines", help="Root of baseline MLP folders.")
    parser.add_argument("--processed-root", type=str, default="datasets/processed", help="Root of processed datasets.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    config = MLPTrainConfig(
        learning_rate=float(args.learning_rate),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        verbose=0 if not args.verbose else 0,  # keep TF quiet; change if you want per-epoch logs
    )

    baselines_root = Path(args.baselines_root)
    processed_root = Path(args.processed_root)

    targets = _find_baseline_folders(baselines_root)

    saved: List[str] = []
    for dataset, window_tag, h_tag in targets:
        out_path = build_metrics_evolution_for_one_folder(
            baselines_root=baselines_root,
            processed_root=processed_root,
            dataset=dataset,
            window_tag=window_tag,
            h_tag=h_tag,
            config=config,
            verbose=bool(args.verbose),
        )
        if out_path is not None:
            saved.append(str(out_path))

    if args.verbose:
        print("\n=== SUMMARY ===")
        print("Folders processed:", len(targets))
        print("Saved:", len(saved))
        for p in saved[:10]:
            print(" -", p)


if __name__ == "__main__":
    main()