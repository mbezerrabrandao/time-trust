# scripts/fulltrust_aggregate_rankings.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


GroupMode = Literal["sensors", "windows"]


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


def _normalize_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= 0:
        return x.copy()
    return x / s


def _coerce_matrix(x: Any, name: str, dtype: Any = float) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return arr


# -------------------------
# Processed metadata helpers
# -------------------------

def _read_processed_metadata(
    processed_root: Path,
    dataset: str,
    window_tag: str,
) -> Tuple[int, int, List[str], Dict[str, Any]]:
    """
    Returns:
      M0: number of sensors after preprocessing
      W0: window length
      kept_sensors: labels for the sensor axis
      meta: raw/partial metadata used for reproducibility

    Primary source is metadata.json. If it is missing, fall back to dataset.npz
    shape and default names s1..sM.
    """
    ds_dir = processed_root / dataset / window_tag
    meta_path = ds_dir / "metadata.json"

    if meta_path.exists():
        meta = _load_json(meta_path)

        if "structure" in meta:
            M0 = int(meta["structure"]["n_sensors"])
            W0 = int(meta["structure"]["window_length"])
        else:
            M0 = int(meta["windowing"]["X_train_mw_shape"][1])
            W0 = int(meta["windowing"]["X_train_mw_shape"][2])

        kept_sensors = meta.get("features", {}).get("kept_sensors", None)
        if kept_sensors is None:
            kept_sensors = [f"s{i+1}" for i in range(M0)]
        else:
            kept_sensors = list(kept_sensors)

        if len(kept_sensors) != M0:
            raise ValueError(
                f"kept_sensors length mismatch: len={len(kept_sensors)} but M0={M0} "
                f"(meta_path={meta_path})"
            )

        return M0, W0, kept_sensors, {
            "metadata_path": str(meta_path),
            "metadata_source": "metadata.json",
            "features": {"kept_sensors": kept_sensors},
        }

    # Fallback: infer from dataset.npz.
    npz_path = ds_dir / "dataset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing metadata.json and dataset.npz for {dataset}/{window_tag}: {ds_dir}"
        )

    data = np.load(npz_path, allow_pickle=True)
    if "X_train_mw" not in data.files:
        raise KeyError(f"dataset.npz missing X_train_mw. Files={list(data.files)}")
    X_train_mw = np.asarray(data["X_train_mw"])
    if X_train_mw.ndim != 3:
        raise ValueError(f"X_train_mw must be 3D [N,M,W], got shape={X_train_mw.shape}")

    M0 = int(X_train_mw.shape[1])
    W0 = int(X_train_mw.shape[2])
    kept_sensors = [f"s{i+1}" for i in range(M0)]

    return M0, W0, kept_sensors, {
        "metadata_path": str(npz_path),
        "metadata_source": "dataset.npz_shape_fallback",
        "features": {"kept_sensors": kept_sensors},
    }


# -------------------------
# Results traversal: FULL only
# -------------------------

def _is_full_mode_dir(name: str) -> bool:
    return name.startswith("full_")


def _find_full_experiment_dirs(
    results_root: Path,
    *,
    full_mode_contains: str = "",
) -> List[Path]:
    """
    Expected layout:
      results/<dataset>/<window_tag>/<h_tag>/<full_mode_dir>/vectors_selection_importance.json

    full_mode_dir must start with full_.
    """
    exp_dirs: List[Path] = []
    if not results_root.exists():
        raise FileNotFoundError(f"results_root does not exist: {results_root}")

    full_mode_contains = str(full_mode_contains or "")

    for ds_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        for win_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()]):
            for h_dir in sorted([p for p in win_dir.iterdir() if p.is_dir()]):
                for mode_dir in sorted([p for p in h_dir.iterdir() if p.is_dir()]):
                    if not _is_full_mode_dir(mode_dir.name):
                        continue
                    if full_mode_contains and full_mode_contains not in mode_dir.name:
                        continue
                    vpath = mode_dir / "vectors_selection_importance.json"
                    if vpath.exists():
                        exp_dirs.append(mode_dir)

    return exp_dirs


# -------------------------
# Ranking logic: FULL selection AUC aggregated to groups
# -------------------------

def _default_group_names(group_mode: GroupMode, n: int) -> List[str]:
    if group_mode == "sensors":
        return [f"s{i+1}" for i in range(n)]
    return [f"w{i+1}" for i in range(n)]


def _selection_unique_values_summary(sel_mat: np.ndarray, max_values: int = 10) -> Dict[str, Any]:
    vals = np.unique(sel_mat)
    out: Dict[str, Any] = {
        "n_unique": int(vals.size),
        "min": float(np.min(vals)) if vals.size else None,
        "max": float(np.max(vals)) if vals.size else None,
    }
    if vals.size <= int(max_values):
        out["values"] = [float(v) for v in vals.tolist()]
    return out


def _aggregate_full_selection_scores(
    *,
    sel_mat: np.ndarray,
    M0: int,
    W0: int,
    group_mode: GroupMode,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FULL selection matrix is feature-level: (T, M0*W0).

    We convert each row to (M0, W0), threshold selection > 0, then compute:

      sensors: score_s = sum_i mean_w selection_i[s,w]
      windows: score_w = sum_i mean_s selection_i[s,w]

    Returns:
      scores: (n_groups,) selection AUC / survival mass over full iterations
      activity_mat: (T,n_groups) group activity per full iteration in [0,1]
    """
    M0 = int(M0)
    W0 = int(W0)
    total_features = M0 * W0

    sel_mat = _coerce_matrix(sel_mat, "selection_vector", dtype=float)
    if int(sel_mat.shape[1]) != total_features:
        raise ValueError(
            f"FULL selection_vector width mismatch: got {sel_mat.shape[1]}, "
            f"expected M0*W0={total_features} (M0={M0}, W0={W0})"
        )

    sel_bin = (sel_mat > 0).astype(float)
    sel_3d = sel_bin.reshape(sel_bin.shape[0], M0, W0)

    if group_mode == "sensors":
        # (T,M0): fraction of windows still active inside each sensor.
        activity_mat = sel_3d.mean(axis=2)
    elif group_mode == "windows":
        # (T,W0): fraction of sensors still active inside each time window.
        activity_mat = sel_3d.mean(axis=1)
    else:
        raise ValueError(f"Invalid group_mode={group_mode}")

    scores = activity_mat.sum(axis=0).astype(float)
    return scores, activity_mat.astype(float)


def _scores_to_items(
    *,
    scores: np.ndarray,
    group_mode: GroupMode,
    group_names: Optional[Sequence[str]] = None,
    n_iters: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    scores_norm = _normalize_nonneg(scores)

    # Stable tie-break: larger score first, then smaller group index.
    idx = np.arange(scores.shape[0], dtype=int)
    ranking = np.lexsort((idx, -scores_norm))

    if group_names is None:
        group_names = _default_group_names(group_mode, scores.shape[0])
    group_names = list(group_names)

    if len(group_names) != scores.shape[0]:
        raise ValueError(
            f"group_names length {len(group_names)} != n_groups {scores.shape[0]} "
            f"for group_mode={group_mode}"
        )

    denom = float(n_iters) if n_iters is not None and int(n_iters) > 0 else None

    items: List[Dict[str, Any]] = []
    for rank_pos, g in enumerate(ranking.tolist(), start=1):
        item: Dict[str, Any] = {
            "rank": int(rank_pos),
            "group_index_0based": int(g),
            "group_index_1based": int(g + 1),
            "group_name": str(group_names[g]),
            "score": float(scores[g]),
            "score_norm": float(scores_norm[g]),
            "selection_auc": float(scores[g]),
        }
        if denom is not None:
            item["mean_activity_over_iters"] = float(scores[g] / denom)
        items.append(item)

    return scores_norm, ranking.astype(int), items


def build_fulltrust_aggregated_ranking_from_selection_json(
    *,
    vselimp_path: Path,
    M0: int,
    W0: int,
    group_mode: GroupMode,
    group_names: Optional[Sequence[str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    out_json: Optional[Path] = None,
) -> Dict[str, Any]:
    data = _load_json(vselimp_path)

    required = ["iter_idx", "n0_value", "selection_vector"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{vselimp_path} missing keys={missing}. Available keys={list(data.keys())}")

    iter_idx = np.asarray(data["iter_idx"], dtype=int).reshape(-1)
    n0_value = np.asarray(data["n0_value"], dtype=int).reshape(-1)
    sel_mat = _coerce_matrix(data["selection_vector"], "selection_vector", dtype=float)

    T = int(sel_mat.shape[0])
    total_features = int(M0) * int(W0)

    if iter_idx.shape[0] != T or n0_value.shape[0] != T:
        raise ValueError(
            f"iter_idx/n0_value length mismatch with selection_vector: "
            f"len(iter_idx)={iter_idx.shape[0]}, len(n0_value)={n0_value.shape[0]}, T={T}"
        )

    scores, activity_mat = _aggregate_full_selection_scores(
        sel_mat=sel_mat,
        M0=int(M0),
        W0=int(W0),
        group_mode=group_mode,
    )

    scores_norm, ranking, items = _scores_to_items(
        scores=scores,
        group_mode=group_mode,
        group_names=group_names,
        n_iters=T,
    )

    n_groups = int(M0) if group_mode == "sensors" else int(W0)

    payload: Dict[str, Any] = {
        "type": "fulltrust_aggregated_group_ranking_selection_based",
        "group_mode": group_mode,
        "n_sensors": int(M0),
        "n_windows": int(W0),
        "n_groups": int(n_groups),
        "n_iters": int(T),
        "input_dim": int(total_features),
        "score_name": "selection_auc",
        "definition": {
            "ranking_rule": "Rank groups by aggregated FULL TRUST selection survival mass; larger selection_auc is more important.",
            "score_sensors": "score_s = sum_i mean_w 1[selection_i[s,w] > 0]",
            "score_windows": "score_w = sum_i mean_s 1[selection_i[s,w] > 0]",
            "score_normalization": "score_norm = score / sum(score)",
            "flatten_order": "sensor-major: flat = s*W0 + w",
            "tie_break": "stable by group index",
        },
        "diagnostics": {
            "selection_vector_shape": [int(x) for x in sel_mat.shape],
            "activity_matrix_shape": [int(x) for x in activity_mat.shape],
            "selection_unique_values": _selection_unique_values_summary(sel_mat),
            "score_sum": float(np.sum(scores)),
            "score_norm_sum": float(np.sum(scores_norm)),
        },
        "trace_meta": {
            "iter_idx": iter_idx.tolist(),
            "n0_value": n0_value.tolist(),
        },
        "items": items,
    }

    if extra_meta:
        payload["meta"] = dict(extra_meta)

    if out_json is not None:
        _json_dump(out_json, payload)

    return payload


# -------------------------
# Batch build: read results/full_*, write into mlp_baselines/.../rankings
# -------------------------

def build_rankings_for_one_full_dir(
    *,
    mode_dir: Path,
    processed_root: Path,
    baselines_root: Path,
    overwrite: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    # results/<ds>/<window>/<h>/<mode_dir>/
    h_dir = mode_dir.parent
    win_dir = h_dir.parent
    ds_dir = win_dir.parent

    dataset_name = ds_dir.name
    window_tag = win_dir.name
    h_tag = h_dir.name
    mode_name = mode_dir.name

    if not _is_full_mode_dir(mode_name):
        raise ValueError(f"Expected full_* mode_dir, got: {mode_name}")

    vselimp_path = mode_dir / "vectors_selection_importance.json"
    if not vselimp_path.exists():
        raise FileNotFoundError(f"Missing vectors_selection_importance.json at: {vselimp_path}")

    M0, W0, kept_sensors, metadata_info = _read_processed_metadata(
        processed_root=processed_root,
        dataset=dataset_name,
        window_tag=window_tag,
    )

    out_dir = baselines_root / dataset_name / window_tag / h_tag / "rankings"
    _safe_mkdir(out_dir)

    sensors_out = out_dir / "ranking_sensors_fulltrust_agg_selection.json"
    windows_out = out_dir / "ranking_windows_fulltrust_agg_selection.json"

    saved: List[str] = []
    skipped: List[str] = []

    extra_meta = {
        "dataset_name": dataset_name,
        "window_tag": window_tag,
        "hidden_tag": h_tag,
        "mode_dir": mode_name,
        "source_json": str(vselimp_path),
        "results_root": str(mode_dir.parents[3]) if len(mode_dir.parents) >= 4 else None,
        "baselines_root": str(baselines_root),
        "processed_root": str(processed_root),
        "metadata": metadata_info,
        "method": "fulltrust_aggregated_selection_auc",
    }

    if overwrite or not sensors_out.exists():
        payload_s = build_fulltrust_aggregated_ranking_from_selection_json(
            vselimp_path=vselimp_path,
            M0=M0,
            W0=W0,
            group_mode="sensors",
            group_names=kept_sensors,
            extra_meta=extra_meta,
            out_json=sensors_out,
        )
        saved.append(str(sensors_out))
    else:
        payload_s = None
        skipped.append(str(sensors_out))

    if overwrite or not windows_out.exists():
        payload_w = build_fulltrust_aggregated_ranking_from_selection_json(
            vselimp_path=vselimp_path,
            M0=M0,
            W0=W0,
            group_mode="windows",
            group_names=[f"w{i+1}" for i in range(W0)],
            extra_meta=extra_meta,
            out_json=windows_out,
        )
        saved.append(str(windows_out))
    else:
        payload_w = None
        skipped.append(str(windows_out))

    if verbose:
        print(f"[OK] {dataset_name}/{window_tag}/{h_tag}/{mode_name}")
        print(f"     M0={M0} W0={W0} input_dim={M0*W0}")
        for p in saved:
            print(f"     saved: {p}")
        for p in skipped:
            print(f"     skipped existing: {p}")

        if payload_s is not None:
            top_s = [it["group_name"] for it in payload_s["items"][:5]]
            print(f"     top-5 sensors: {top_s}")
        if payload_w is not None:
            top_w = [it["group_name"] for it in payload_w["items"][:5]]
            print(f"     top-5 windows: {top_w}")

    return {
        "dataset_name": dataset_name,
        "window_tag": window_tag,
        "hidden_tag": h_tag,
        "mode_dir": mode_name,
        "M0": int(M0),
        "W0": int(W0),
        "saved": saved,
        "skipped": skipped,
    }


def build_rankings_for_results_tree(
    *,
    results_root: Path,
    processed_root: Path,
    baselines_root: Path,
    full_mode_contains: str = "",
    overwrite: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    exp_dirs = _find_full_experiment_dirs(
        results_root=results_root,
        full_mode_contains=full_mode_contains,
    )

    saved: List[str] = []
    skipped: List[str] = []
    errors: List[Dict[str, str]] = []

    for mode_dir in exp_dirs:
        try:
            out = build_rankings_for_one_full_dir(
                mode_dir=mode_dir,
                processed_root=processed_root,
                baselines_root=baselines_root,
                overwrite=overwrite,
                verbose=verbose,
            )
            saved.extend(out["saved"])
            skipped.extend(out["skipped"])
        except Exception as e:
            errors.append({"mode_dir": str(mode_dir), "error": str(e)})
            if verbose:
                print(f"[ERR] {mode_dir}: {e}")

    return {
        "n_full_dirs_found": int(len(exp_dirs)),
        "saved": saved,
        "n_saved": int(len(saved)),
        "skipped": skipped,
        "n_skipped": int(len(skipped)),
        "errors": errors,
        "n_errors": int(len(errors)),
    }


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build group-level rankings from FULL TRUST feature-level selection traces. "
            "Reads results/.../full_*/vectors_selection_importance.json and writes "
            "ranking_sensors_fulltrust_agg_selection.json / ranking_windows_fulltrust_agg_selection.json "
            "inside mlp_baselines/.../rankings/."
        )
    )
    parser.add_argument("--results-root", type=str, default="results", help="Root folder with TRUST/Time-TRUST runs.")
    parser.add_argument("--processed-root", type=str, default="datasets/processed", help="Root of processed datasets.")
    parser.add_argument("--baselines-root", type=str, default="mlp_baselines", help="Baseline MLP root folder.")
    parser.add_argument(
        "--full-mode-contains",
        type=str,
        default="",
        help="Optional substring filter for full_* mode dirs, e.g. mlp_rebuild__capped or slack0.000.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fulltrust aggregated ranking JSONs.")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")

    args = parser.parse_args()

    report = build_rankings_for_results_tree(
        results_root=Path(args.results_root),
        processed_root=Path(args.processed_root),
        baselines_root=Path(args.baselines_root),
        full_mode_contains=str(args.full_mode_contains),
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
    )

    if args.verbose:
        print("\n=== SUMMARY ===")
        print("Full dirs found:", report["n_full_dirs_found"])
        print("Saved:", report["n_saved"])
        print("Skipped:", report["n_skipped"])
        print("Errors:", report["n_errors"])
        if report["n_errors"] > 0:
            for er in report["errors"][:10]:
                print(" -", er["mode_dir"], "->", er["error"])


if __name__ == "__main__":
    main()
