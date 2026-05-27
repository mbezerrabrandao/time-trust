# scripts/ranking_retention_masking.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

GroupMode = Literal["sensors", "windows"]
AblationSplit = Literal["train", "val"]

PARTIAL_AUC_LO = 0.20
PARTIAL_AUC_HI = 0.80


# -------------------------
# Small IO helpers
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


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _safe_mkdir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = list(rows[0].keys())
    seen = set(fieldnames)
    for r in rows[1:]:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -------------------------
# Random seed helper
# -------------------------

def _stable_seed(base_seed: int, *parts: str) -> int:
    key = "|".join([str(base_seed), *[str(p) for p in parts]])
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# -------------------------
# Naming / traversal helpers
# -------------------------

def _is_hidden_tag(name: str) -> bool:
    return name.startswith("h") and any(ch.isdigit() for ch in name)


def _find_baseline_folders(
    baselines_root: Path,
    *,
    only_dataset: Optional[str] = None,
    only_window_tag: Optional[str] = None,
    only_hidden: Optional[str] = None,
) -> List[Tuple[str, str, str, Path]]:
    """
    Finds folders of the form:
      mlp_baselines/<dataset>/<window_tag>/<h_tag>/

    Returns:
      (dataset, window_tag, hidden_tag, folder_path)
    """
    if not baselines_root.exists():
        raise FileNotFoundError(f"baselines_root does not exist: {baselines_root}")

    out: List[Tuple[str, str, str, Path]] = []
    for ds_dir in sorted([p for p in baselines_root.iterdir() if p.is_dir()]):
        if only_dataset is not None and ds_dir.name != only_dataset:
            continue

        for win_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()]):
            if only_window_tag is not None and win_dir.name != only_window_tag:
                continue

            for h_dir in sorted([p for p in win_dir.iterdir() if p.is_dir() and _is_hidden_tag(p.name)]):
                if only_hidden is not None and h_dir.name != only_hidden:
                    continue
                out.append((ds_dir.name, win_dir.name, h_dir.name, h_dir))

    return out


def _ranking_path(rankings_dir: Path, group_mode: GroupMode, method: str) -> Path:
    if method == "fulltrust_agg_selection":
        return rankings_dir / f"ranking_{group_mode}_fulltrust_agg_selection.json"
    if method == "timetrust_selection":
        return rankings_dir / f"ranking_{group_mode}_timetrust_selection.json"
    if method == "weights":
        return rankings_dir / f"ranking_{group_mode}_weights.json"
    raise ValueError(f"Unknown ranking method: {method}")


# -------------------------
# Dataset/model loading
# -------------------------

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
        raise KeyError(
            f"dataset.npz missing keys={missing}. "
            f"Available keys={list(arrays.keys())}"
        )

    return arrays


def _load_keras_model(model_path: Path):
    """
    Lazy import TensorFlow/Keras so the script can still be inspected/compiled
    in environments where TensorFlow is not installed.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.keras at: {model_path}")

    try:
        from tensorflow import keras  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import tensorflow.keras. Install TensorFlow in the environment "
            "where this script is executed."
        ) from e

    return keras.models.load_model(model_path, compile=False)


def _flatten_mw(X_mw: np.ndarray) -> np.ndarray:
    """
    X_mw: (N, M, W) -> (N, M*W) with sensor-major flatten.
    For C-order NumPy arrays, reshape does flat = s*W + w.
    """
    X_mw = np.asarray(X_mw, dtype=np.float32)
    if X_mw.ndim != 3:
        raise ValueError(f"Expected X_mw to be 3D (N,M,W), got shape={X_mw.shape}")
    N, M, W = X_mw.shape
    return X_mw.reshape(N, M * W)


def _ensure_1d_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    return y.reshape(-1)


def _subsample_for_ablation(
    *,
    X_mw: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Optional deterministic subsampling for group-ablation ranking.
    max_samples <= 0 means use full split.
    """
    X_mw = np.asarray(X_mw, dtype=np.float32)
    y = _ensure_1d_y(y)

    n = int(X_mw.shape[0])
    max_samples = int(max_samples)

    if max_samples <= 0 or max_samples >= n:
        return X_mw, y, {
            "subsampled": False,
            "n_available": n,
            "n_used": n,
            "max_samples": max_samples,
        }

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=max_samples, replace=False)
    idx = np.sort(idx)

    return X_mw[idx], y[idx], {
        "subsampled": True,
        "n_available": n,
        "n_used": int(max_samples),
        "max_samples": max_samples,
        "seed": int(seed),
    }


# -------------------------
# Ranking extraction
# -------------------------

def _extract_ranked_group_indices(ranking_json: Dict[str, Any]) -> List[int]:
    """
    Supports JSON formats used in this project:
      - {"items": [{"group_index_0based": ...}, ...]}
      - {"ranking": [{"index": 1-based, ...}, ...]}
      - {"ranking": [{"id": "s3" or "w5", ...}, ...]}

    Returns 0-based group indices in best-first order.
    """
    if "items" in ranking_json and isinstance(ranking_json["items"], list):
        items = ranking_json["items"]
        out: List[int] = []
        for it in items:
            if "group_index_0based" not in it:
                raise KeyError("ranking item missing group_index_0based")
            out.append(int(it["group_index_0based"]))
        return out

    if "ranking" in ranking_json and isinstance(ranking_json["ranking"], list):
        items = ranking_json["ranking"]
        out = []
        for it in items:
            if "index" in it:
                out.append(int(it["index"]) - 1)
            elif "id" in it:
                s = str(it["id"])
                num = "".join(c for c in s if c.isdigit())
                if not num:
                    raise ValueError(f"Cannot parse group id={s}")
                out.append(int(num) - 1)
            else:
                raise KeyError("ranking item missing index/id")
        return out

    raise KeyError("Unrecognized ranking JSON format; expected 'items' or 'ranking'.")


def _validate_ranking(ranking: Sequence[int], n_groups: int, *, name: str) -> List[int]:
    arr = [int(x) for x in ranking]
    if len(arr) != int(n_groups):
        raise ValueError(f"{name}: ranking length={len(arr)} but expected n_groups={n_groups}")
    if sorted(arr) != list(range(int(n_groups))):
        raise ValueError(f"{name}: ranking is not a permutation of 0..{n_groups-1}: {arr}")
    return arr


def _load_ranking_file(path: Path, n_groups: int, *, name: str) -> List[int]:
    j = _load_json(path)
    ranking = _extract_ranked_group_indices(j)
    return _validate_ranking(ranking, n_groups=n_groups, name=name)


# -------------------------
# Metrics and prediction
# -------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = _ensure_1d_y(y_true)
    yp = _ensure_1d_y(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true/y_pred length mismatch: {yt.shape[0]} vs {yp.shape[0]}")

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse}


def _predict_flat(model: Any, X_flat: np.ndarray, *, batch_size: int) -> np.ndarray:
    pred = model.predict(
        np.asarray(X_flat, dtype=np.float32),
        batch_size=int(batch_size),
        verbose=0,
    )
    return np.asarray(pred, dtype=np.float32)


def _evaluate_full_metrics(
    *,
    model: Any,
    X_mw: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Dict[str, float]:
    pred = _predict_flat(model, _flatten_mw(X_mw), batch_size=batch_size)
    return _compute_metrics(y, pred)


def _mask_keep_groups(
    X_mw: np.ndarray,
    *,
    keep_groups: Sequence[int],
    group_mode: GroupMode,
    mask_value: float,
) -> np.ndarray:
    """
    Keep only the given groups and mask all other groups.
    """
    X = np.asarray(X_mw, dtype=np.float32).copy()
    _, M, W = X.shape

    if group_mode == "sensors":
        n_groups = M
        keep = np.asarray(list(keep_groups), dtype=int)
        remove = np.setdiff1d(np.arange(n_groups, dtype=int), keep, assume_unique=False)
        X[:, remove, :] = float(mask_value)
        return X

    if group_mode == "windows":
        n_groups = W
        keep = np.asarray(list(keep_groups), dtype=int)
        remove = np.setdiff1d(np.arange(n_groups, dtype=int), keep, assume_unique=False)
        X[:, :, remove] = float(mask_value)
        return X

    raise ValueError(f"Invalid group_mode={group_mode}")


def _mask_single_group(
    X_mw: np.ndarray,
    *,
    group_idx: int,
    group_mode: GroupMode,
    mask_value: float,
) -> np.ndarray:
    """
    Mask only one group; used for group-ablation ranking.
    """
    X = np.asarray(X_mw, dtype=np.float32).copy()

    if group_mode == "sensors":
        X[:, int(group_idx), :] = float(mask_value)
    elif group_mode == "windows":
        X[:, :, int(group_idx)] = float(mask_value)
    else:
        raise ValueError(f"Invalid group_mode={group_mode}")

    return X


def _predict_masked_stack_metrics(
    model: Any,
    X_stack_mw: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
) -> List[Dict[str, float]]:
    """
    Predict a stack of masked datasets in a single Keras predict call.

    X_stack_mw: (K, N, M, W)
    Returns K metric dicts.
    """
    X_stack_mw = np.asarray(X_stack_mw, dtype=np.float32)
    if X_stack_mw.ndim != 4:
        raise ValueError(f"Expected X_stack_mw to be 4D (K,N,M,W), got {X_stack_mw.shape}")

    K, N, M, W = X_stack_mw.shape
    X_flat = X_stack_mw.reshape(K * N, M * W)
    pred = _predict_flat(model, X_flat, batch_size=batch_size)
    pred = pred.reshape(K, N, -1)

    return [_compute_metrics(y, pred[i]) for i in range(K)]


# -------------------------
# AUC helpers
# -------------------------

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _interp_at(x: np.ndarray, y: np.ndarray, q: float) -> float:
    if not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.interp(float(q), x, y))


def _partial_auc_stats(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lo: float = PARTIAL_AUC_LO,
    hi: float = PARTIAL_AUC_HI,
) -> Dict[str, float]:
    """
    Returns:
      y_at_lo, y_at_hi, auc_raw, auc_norm

    auc_norm = auc_raw / (hi - lo), so it is comparable to the scale of y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if lo < float(np.min(x)) or hi > float(np.max(x)) or hi <= lo:
        return {
            "at_lo": float("nan"),
            "at_hi": float("nan"),
            "auc_raw": float("nan"),
            "auc_norm": float("nan"),
        }

    if not np.all(np.isfinite(y)):
        return {
            "at_lo": float("nan"),
            "at_hi": float("nan"),
            "auc_raw": float("nan"),
            "auc_norm": float("nan"),
        }

    y_lo = _interp_at(x, y, lo)
    y_hi = _interp_at(x, y, hi)

    mask = (x > lo) & (x < hi)

    x_part = np.concatenate([[lo], x[mask], [hi]]).astype(float)
    y_part = np.concatenate([[y_lo], y[mask], [y_hi]]).astype(float)

    auc_raw = _trapz(y_part, x_part)
    auc_norm = auc_raw / float(hi - lo)

    return {
        "at_lo": float(y_lo),
        "at_hi": float(y_hi),
        "auc_raw": float(auc_raw),
        "auc_norm": float(auc_norm),
    }


# -------------------------
# Ranking construction baselines
# -------------------------

def _build_group_ablation_ranking(
    *,
    model: Any,
    X_rank_mw: np.ndarray,
    y_rank: np.ndarray,
    group_mode: GroupMode,
    mae_full_rank: float,
    mask_value: float,
    batch_size: int,
    ablation_group_batch_size: int,
) -> Tuple[List[int], np.ndarray]:
    """
    Score_g = MAE(mask only group g on the ranking split) - MAE(full input on the ranking split).

    Larger score = more important.

    The ranking split is normally train, while retention is evaluated on validation.
    """
    _, M, W = X_rank_mw.shape
    n_groups = M if group_mode == "sensors" else W

    scores = np.zeros(n_groups, dtype=float)
    group_batch = max(1, int(ablation_group_batch_size))

    for start in range(0, n_groups, group_batch):
        end = min(n_groups, start + group_batch)
        groups = list(range(start, end))

        masked_list = [
            _mask_single_group(
                X_rank_mw,
                group_idx=g,
                group_mode=group_mode,
                mask_value=mask_value,
            )
            for g in groups
        ]

        X_stack = np.stack(masked_list, axis=0)
        metrics = _predict_masked_stack_metrics(
            model,
            X_stack,
            y_rank,
            batch_size=batch_size,
        )

        for g, m in zip(groups, metrics):
            scores[g] = float(m["mae"]) - float(mae_full_rank)

    idx = np.arange(n_groups, dtype=int)
    ranking = np.lexsort((idx, -scores)).astype(int).tolist()
    return ranking, scores


def _group_names(group_mode: GroupMode, n_groups: int) -> List[str]:
    prefix = "s" if group_mode == "sensors" else "w"
    return [f"{prefix}{i + 1}" for i in range(n_groups)]


def _save_group_ablation_ranking_json(
    *,
    out_path: Path,
    dataset: str,
    window_tag: str,
    hidden_tag: str,
    group_mode: GroupMode,
    ranking: Sequence[int],
    scores: np.ndarray,
    mae_full_rank: float,
    mask_value: float,
    ablation_ranking_split: AblationSplit,
    ablation_subsample_meta: Dict[str, Any],
) -> None:
    scores = np.asarray(scores, dtype=float)
    scores_nonneg = np.maximum(scores, 0.0)
    s = float(scores_nonneg.sum())
    scores_norm = scores_nonneg / s if s > 0 else np.zeros_like(scores_nonneg)
    names = _group_names(group_mode, len(scores))

    items: List[Dict[str, Any]] = []
    for rank_pos, g in enumerate([int(x) for x in ranking], start=1):
        items.append(
            {
                "rank": int(rank_pos),
                "group_index_0based": int(g),
                "group_index_1based": int(g + 1),
                "group_name": names[g],
                "score": float(scores[g]),
                "score_norm_nonneg": float(scores_norm[g]),
            }
        )

    payload: Dict[str, Any] = {
        "type": "group_ablation_ranking_fixed_model_masking",
        "group_mode": group_mode,
        "n_groups": int(len(scores)),
        "score_name": "mae_mask_single_group_minus_mae_full_on_ranking_split",
        "definition": {
            "score": "MAE(X with one group masked) - MAE(X full), computed on the ablation ranking split",
            "ranking_rule": "larger score = more important",
            "mask_value": float(mask_value),
            "normalization": "score_norm_nonneg is computed after clipping negative scores to zero; ranking uses raw scores",
            "important_note": "Ranking can be computed on train while retention curves are evaluated on validation to reduce split-level overfitting.",
        },
        "meta": {
            "dataset_name": dataset,
            "window_tag": window_tag,
            "hidden_tag": hidden_tag,
            "ablation_ranking_split": ablation_ranking_split,
            "ablation_subsample": ablation_subsample_meta,
            "mae_full_on_ranking_split": float(mae_full_rank),
        },
        "items": items,
    }

    _json_dump(out_path, payload)


# -------------------------
# Retention curves
# -------------------------

def _retention_from_mae(mae: float, *, mae_full: float, mae_empty: float) -> float:
    denom = float(mae_empty) - float(mae_full)
    if abs(denom) <= 1e-12:
        return float("nan")
    return float((float(mae_empty) - float(mae)) / denom)


def _evaluate_keep_topk_curve(
    *,
    model: Any,
    X_eval_mw: np.ndarray,
    y_eval: np.ndarray,
    group_mode: GroupMode,
    ranking_top_first: Sequence[int],
    method: str,
    random_trial: int,
    dataset: str,
    window_tag: str,
    hidden_tag: str,
    ranking_file: str,
    mae_full: float,
    rmse_full: float,
    mae_empty: float,
    rmse_empty: float,
    mask_value: float,
    batch_size: int,
    ablation_ranking_split: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Builds and evaluates K=0..G masked datasets:
      k=0: keep no groups -> all masked
      k=G: keep all groups -> original input

    Returns:
      curve_rows, auc_row
    """
    _, M, W = X_eval_mw.shape
    n_groups = M if group_mode == "sensors" else W
    ranking = _validate_ranking(ranking_top_first, n_groups, name=f"{method}/{group_mode}")

    X_stack: List[np.ndarray] = []
    fractions: List[float] = []
    ks: List[int] = []

    for k in range(n_groups + 1):
        keep = ranking[:k]
        X_k = _mask_keep_groups(
            X_eval_mw,
            keep_groups=keep,
            group_mode=group_mode,
            mask_value=mask_value,
        )
        X_stack.append(X_k)
        ks.append(int(k))
        fractions.append(float(k) / float(n_groups))

    metrics = _predict_masked_stack_metrics(
        model,
        np.stack(X_stack, axis=0),
        y_eval,
        batch_size=batch_size,
    )

    curve_rows: List[Dict[str, Any]] = []
    retentions: List[float] = []
    maes: List[float] = []
    rmses: List[float] = []

    for k, frac, m in zip(ks, fractions, metrics):
        mae = float(m["mae"])
        rmse = float(m["rmse"])
        retention = _retention_from_mae(mae, mae_full=mae_full, mae_empty=mae_empty)

        retentions.append(retention)
        maes.append(mae)
        rmses.append(rmse)

        curve_rows.append(
            {
                "dataset": dataset,
                "window_tag": window_tag,
                "hidden_tag": hidden_tag,
                "group_mode": group_mode,
                "method": method,
                "random_trial": int(random_trial),
                "k": int(k),
                "n_groups": int(n_groups),
                "fraction_kept": float(frac),
                "mae": mae,
                "rmse": rmse,
                "retention": retention,
                "mae_full": float(mae_full),
                "rmse_full": float(rmse_full),
                "mae_empty": float(mae_empty),
                "rmse_empty": float(rmse_empty),
                "mask_value": float(mask_value),
                "ranking_file": ranking_file,
                "ablation_ranking_split": ablation_ranking_split or "",
            }
        )

    x = np.asarray(fractions, dtype=float)
    r = np.asarray(retentions, dtype=float)
    mae_arr = np.asarray(maes, dtype=float)
    rmse_arr = np.asarray(rmses, dtype=float)

    retention_auc = _trapz(r, x) if np.all(np.isfinite(r)) else float("nan")
    mae_auc = _trapz(mae_arr, x) if np.all(np.isfinite(mae_arr)) else float("nan")
    rmse_auc = _trapz(rmse_arr, x) if np.all(np.isfinite(rmse_arr)) else float("nan")

    r_part = _partial_auc_stats(x, r, lo=PARTIAL_AUC_LO, hi=PARTIAL_AUC_HI)
    mae_part = _partial_auc_stats(x, mae_arr, lo=PARTIAL_AUC_LO, hi=PARTIAL_AUC_HI)
    rmse_part = _partial_auc_stats(x, rmse_arr, lo=PARTIAL_AUC_LO, hi=PARTIAL_AUC_HI)

    auc_row = {
        "dataset": dataset,
        "window_tag": window_tag,
        "hidden_tag": hidden_tag,
        "group_mode": group_mode,
        "method": method,
        "random_trial": int(random_trial),
        "n_groups": int(n_groups),
        "n_points": int(n_groups + 1),

        "retention_auc": retention_auc,
        "retention_auc_20_80": r_part["auc_norm"],
        "retention_auc_20_80_raw": r_part["auc_raw"],
        "retention_at_20": r_part["at_lo"],
        "retention_at_80": r_part["at_hi"],

        "mae_auc": mae_auc,
        "mae_auc_20_80": mae_part["auc_norm"],
        "mae_auc_20_80_raw": mae_part["auc_raw"],
        "mae_at_20": mae_part["at_lo"],
        "mae_at_80": mae_part["at_hi"],

        "rmse_auc": rmse_auc,
        "rmse_auc_20_80": rmse_part["auc_norm"],
        "rmse_auc_20_80_raw": rmse_part["auc_raw"],
        "rmse_at_20": rmse_part["at_lo"],
        "rmse_at_80": rmse_part["at_hi"],

        "partial_auc_lo": float(PARTIAL_AUC_LO),
        "partial_auc_hi": float(PARTIAL_AUC_HI),

        "mae_full": float(mae_full),
        "rmse_full": float(rmse_full),
        "mae_empty": float(mae_empty),
        "rmse_empty": float(rmse_empty),
        "mask_value": float(mask_value),
        "ranking_file": ranking_file,
        "ablation_ranking_split": ablation_ranking_split or "",
    }

    return curve_rows, auc_row


# -------------------------
# One configuration
# -------------------------

def evaluate_one_config(
    *,
    processed_root: Path,
    baseline_dir: Path,
    dataset: str,
    window_tag: str,
    hidden_tag: str,
    group_modes: Sequence[GroupMode],
    deterministic_methods: Sequence[str],
    include_group_ablation: bool,
    include_random: bool,
    n_random: int,
    seed: int,
    mask_value: float,
    batch_size: int,
    save_ablation_rankings: bool,
    ablation_ranking_split: AblationSplit,
    ablation_max_samples: int,
    ablation_group_batch_size: int,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    curve_rows_all: List[Dict[str, Any]] = []
    auc_rows_all: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    arrays = _load_processed_dataset(processed_root, dataset, window_tag)

    X_train_mw = np.asarray(arrays["X_train_mw"], dtype=np.float32)
    y_train = _ensure_1d_y(arrays["y_train"])
    X_val_mw = np.asarray(arrays["X_val_mw"], dtype=np.float32)
    y_val = _ensure_1d_y(arrays["y_val"])

    if X_train_mw.ndim != 3:
        raise ValueError(f"X_train_mw must be 3D [N,M,W], got shape={X_train_mw.shape}")
    if X_val_mw.ndim != 3:
        raise ValueError(f"X_val_mw must be 3D [N,M,W], got shape={X_val_mw.shape}")

    _, M, W = X_val_mw.shape

    model_path = baseline_dir / "model.keras"
    model = _load_keras_model(model_path)

    # Evaluation references are always computed on validation.
    full_metrics = _evaluate_full_metrics(
        model=model,
        X_mw=X_val_mw,
        y=y_val,
        batch_size=batch_size,
    )
    mae_full = float(full_metrics["mae"])
    rmse_full = float(full_metrics["rmse"])

    X_empty = np.full_like(X_val_mw, fill_value=float(mask_value), dtype=np.float32)
    empty_metrics = _evaluate_full_metrics(
        model=model,
        X_mw=X_empty,
        y=y_val,
        batch_size=batch_size,
    )
    mae_empty = float(empty_metrics["mae"])
    rmse_empty = float(empty_metrics["rmse"])

    rankings_dir = baseline_dir / "rankings"

    if verbose:
        print(
            f"\n[{dataset}/{window_tag}/{hidden_tag}] "
            f"X_train={tuple(X_train_mw.shape)} X_val={tuple(X_val_mw.shape)} "
            f"val_mae_full={mae_full:.4f} val_mae_empty={mae_empty:.4f}"
        )

    for group_mode in group_modes:
        n_groups = M if group_mode == "sensors" else W

        if verbose:
            print(f"  group_mode={group_mode} n_groups={n_groups}")

        # 1) Deterministic rankings from JSON.
        for method in deterministic_methods:
            rpath = _ranking_path(rankings_dir, group_mode, method)
            if not rpath.exists():
                errors.append(
                    {
                        "dataset": dataset,
                        "window_tag": window_tag,
                        "hidden_tag": hidden_tag,
                        "group_mode": group_mode,
                        "method": method,
                        "error": f"missing ranking file: {rpath}",
                    }
                )
                if verbose:
                    print(f"    [skip] {method}: missing {rpath.name}")
                continue

            try:
                ranking = _load_ranking_file(rpath, n_groups=n_groups, name=method)

                curve_rows, auc_row = _evaluate_keep_topk_curve(
                    model=model,
                    X_eval_mw=X_val_mw,
                    y_eval=y_val,
                    group_mode=group_mode,
                    ranking_top_first=ranking,
                    method=method,
                    random_trial=-1,
                    dataset=dataset,
                    window_tag=window_tag,
                    hidden_tag=hidden_tag,
                    ranking_file=str(rpath),
                    mae_full=mae_full,
                    rmse_full=rmse_full,
                    mae_empty=mae_empty,
                    rmse_empty=rmse_empty,
                    mask_value=mask_value,
                    batch_size=batch_size,
                )

                curve_rows_all.extend(curve_rows)
                auc_rows_all.append(auc_row)

                if verbose:
                    print(
                        f"    [OK] {method}: "
                        f"AUC={auc_row['retention_auc']:.4f} "
                        f"AUC20-80={auc_row['retention_auc_20_80']:.4f}"
                    )

            except Exception as e:
                errors.append(
                    {
                        "dataset": dataset,
                        "window_tag": window_tag,
                        "hidden_tag": hidden_tag,
                        "group_mode": group_mode,
                        "method": method,
                        "error": str(e),
                    }
                )
                if verbose:
                    print(f"    [ERR] {method}: {e}")

        # 2) Group ablation ranking.
        if include_group_ablation:
            try:
                if ablation_ranking_split == "train":
                    X_rank_base = X_train_mw
                    y_rank_base = y_train
                elif ablation_ranking_split == "val":
                    X_rank_base = X_val_mw
                    y_rank_base = y_val
                else:
                    raise ValueError(f"Invalid ablation_ranking_split={ablation_ranking_split}")

                subsample_seed = _stable_seed(
                    seed,
                    dataset,
                    window_tag,
                    hidden_tag,
                    group_mode,
                    "group_ablation_subsample",
                    ablation_ranking_split,
                )

                X_rank_mw, y_rank, ab_subsample_meta = _subsample_for_ablation(
                    X_mw=X_rank_base,
                    y=y_rank_base,
                    max_samples=int(ablation_max_samples),
                    seed=subsample_seed,
                )

                rank_full_metrics = _evaluate_full_metrics(
                    model=model,
                    X_mw=X_rank_mw,
                    y=y_rank,
                    batch_size=batch_size,
                )
                mae_full_rank = float(rank_full_metrics["mae"])

                ranking_ab, scores_ab = _build_group_ablation_ranking(
                    model=model,
                    X_rank_mw=X_rank_mw,
                    y_rank=y_rank,
                    group_mode=group_mode,
                    mae_full_rank=mae_full_rank,
                    mask_value=mask_value,
                    batch_size=batch_size,
                    ablation_group_batch_size=ablation_group_batch_size,
                )

                ranking_file = ""
                if save_ablation_rankings:
                    out_rank = rankings_dir / f"ranking_{group_mode}_group_ablation_masking.json"
                    _save_group_ablation_ranking_json(
                        out_path=out_rank,
                        dataset=dataset,
                        window_tag=window_tag,
                        hidden_tag=hidden_tag,
                        group_mode=group_mode,
                        ranking=ranking_ab,
                        scores=scores_ab,
                        mae_full_rank=mae_full_rank,
                        mask_value=mask_value,
                        ablation_ranking_split=ablation_ranking_split,
                        ablation_subsample_meta=ab_subsample_meta,
                    )
                    ranking_file = str(out_rank)

                curve_rows, auc_row = _evaluate_keep_topk_curve(
                    model=model,
                    X_eval_mw=X_val_mw,
                    y_eval=y_val,
                    group_mode=group_mode,
                    ranking_top_first=ranking_ab,
                    method="group_ablation",
                    random_trial=-1,
                    dataset=dataset,
                    window_tag=window_tag,
                    hidden_tag=hidden_tag,
                    ranking_file=ranking_file,
                    mae_full=mae_full,
                    rmse_full=rmse_full,
                    mae_empty=mae_empty,
                    rmse_empty=rmse_empty,
                    mask_value=mask_value,
                    batch_size=batch_size,
                    ablation_ranking_split=ablation_ranking_split,
                )

                curve_rows_all.extend(curve_rows)
                auc_rows_all.append(auc_row)

                if verbose:
                    print(
                        f"    [OK] group_ablation "
                        f"(rank_split={ablation_ranking_split}, n_rank={ab_subsample_meta['n_used']}): "
                        f"AUC={auc_row['retention_auc']:.4f} "
                        f"AUC20-80={auc_row['retention_auc_20_80']:.4f}"
                    )

            except Exception as e:
                errors.append(
                    {
                        "dataset": dataset,
                        "window_tag": window_tag,
                        "hidden_tag": hidden_tag,
                        "group_mode": group_mode,
                        "method": "group_ablation",
                        "error": str(e),
                    }
                )
                if verbose:
                    print(f"    [ERR] group_ablation: {e}")

        # 3) Random permutations.
        if include_random and int(n_random) > 0:
            rng = np.random.default_rng(
                _stable_seed(seed, dataset, window_tag, hidden_tag, group_mode)
            )

            random_aucs: List[float] = []
            random_aucs_20_80: List[float] = []

            for trial in range(int(n_random)):
                ranking_r = rng.permutation(n_groups).astype(int).tolist()

                try:
                    curve_rows, auc_row = _evaluate_keep_topk_curve(
                        model=model,
                        X_eval_mw=X_val_mw,
                        y_eval=y_val,
                        group_mode=group_mode,
                        ranking_top_first=ranking_r,
                        method="random_permutation",
                        random_trial=int(trial),
                        dataset=dataset,
                        window_tag=window_tag,
                        hidden_tag=hidden_tag,
                        ranking_file="",
                        mae_full=mae_full,
                        rmse_full=rmse_full,
                        mae_empty=mae_empty,
                        rmse_empty=rmse_empty,
                        mask_value=mask_value,
                        batch_size=batch_size,
                    )

                    curve_rows_all.extend(curve_rows)
                    auc_rows_all.append(auc_row)

                    random_aucs.append(float(auc_row["retention_auc"]))
                    random_aucs_20_80.append(float(auc_row["retention_auc_20_80"]))

                except Exception as e:
                    errors.append(
                        {
                            "dataset": dataset,
                            "window_tag": window_tag,
                            "hidden_tag": hidden_tag,
                            "group_mode": group_mode,
                            "method": "random_permutation",
                            "random_trial": str(trial),
                            "error": str(e),
                        }
                    )
                    if verbose:
                        print(f"    [ERR] random trial {trial}: {e}")

            if verbose and random_aucs:
                print(
                    f"    [OK] random_permutation: n={len(random_aucs)} "
                    f"mean_AUC={np.mean(random_aucs):.4f} std={np.std(random_aucs):.4f} "
                    f"mean_AUC20-80={np.mean(random_aucs_20_80):.4f}"
                )

    return curve_rows_all, auc_rows_all, errors


# -------------------------
# Main batch runner
# -------------------------

def run_batch(
    *,
    processed_root: Path,
    baselines_root: Path,
    out_dir: Path,
    only_dataset: Optional[str],
    only_window_tag: Optional[str],
    only_hidden: Optional[str],
    only_group_mode: Optional[str],
    n_random: int,
    seed: int,
    mask_value: float,
    batch_size: int,
    include_group_ablation: bool,
    include_random: bool,
    save_ablation_rankings: bool,
    ablation_ranking_split: AblationSplit,
    ablation_max_samples: int,
    ablation_group_batch_size: int,
    verbose: bool,
) -> Dict[str, Any]:
    if only_group_mode is not None and only_group_mode not in {"sensors", "windows"}:
        raise ValueError("--only-group-mode must be one of: sensors, windows")

    if ablation_ranking_split not in {"train", "val"}:
        raise ValueError("--ablation-ranking-split must be one of: train, val")

    group_modes: List[GroupMode] = (
        ["sensors", "windows"]
        if only_group_mode is None
        else [only_group_mode]  # type: ignore[list-item]
    )

    deterministic_methods = ["fulltrust_agg_selection", "timetrust_selection", "weights"]

    targets = _find_baseline_folders(
        baselines_root,
        only_dataset=only_dataset,
        only_window_tag=only_window_tag,
        only_hidden=only_hidden,
    )

    all_curve_rows: List[Dict[str, Any]] = []
    all_auc_rows: List[Dict[str, Any]] = []
    all_errors: List[Dict[str, str]] = []

    if verbose:
        print("=== ranking_retention_masking ===")
        print("processed_root:", processed_root)
        print("baselines_root:", baselines_root)
        print("out_dir:", out_dir)
        print("targets:", len(targets))
        print("group_modes:", group_modes)
        print("deterministic_methods:", deterministic_methods)
        print("include_group_ablation:", include_group_ablation)
        print("ablation_ranking_split:", ablation_ranking_split)
        print("ablation_max_samples:", ablation_max_samples)
        print("ablation_group_batch_size:", ablation_group_batch_size)
        print("include_random:", include_random, "n_random:", n_random)
        print("mask_value:", mask_value)
        print("partial_auc:", f"{PARTIAL_AUC_LO:.2f}-{PARTIAL_AUC_HI:.2f}")

    for dataset, window_tag, hidden_tag, baseline_dir in targets:
        try:
            curve_rows, auc_rows, errors = evaluate_one_config(
                processed_root=processed_root,
                baseline_dir=baseline_dir,
                dataset=dataset,
                window_tag=window_tag,
                hidden_tag=hidden_tag,
                group_modes=group_modes,
                deterministic_methods=deterministic_methods,
                include_group_ablation=include_group_ablation,
                include_random=include_random,
                n_random=int(n_random),
                seed=int(seed),
                mask_value=float(mask_value),
                batch_size=int(batch_size),
                save_ablation_rankings=bool(save_ablation_rankings),
                ablation_ranking_split=ablation_ranking_split,
                ablation_max_samples=int(ablation_max_samples),
                ablation_group_batch_size=int(ablation_group_batch_size),
                verbose=verbose,
            )

            all_curve_rows.extend(curve_rows)
            all_auc_rows.extend(auc_rows)
            all_errors.extend(errors)

        except Exception as e:
            all_errors.append(
                {
                    "dataset": dataset,
                    "window_tag": window_tag,
                    "hidden_tag": hidden_tag,
                    "group_mode": "*",
                    "method": "*",
                    "error": str(e),
                }
            )
            if verbose:
                print(f"[ERR] {dataset}/{window_tag}/{hidden_tag}: {e}")

    _safe_mkdir(out_dir)

    curves_path = out_dir / "ranking_retention_masking_curves.csv"
    auc_path = out_dir / "ranking_retention_masking_auc.csv"
    summary_path = out_dir / "ranking_retention_masking_summary.json"

    _write_csv(curves_path, all_curve_rows)
    _write_csv(auc_path, all_auc_rows)

    summary: Dict[str, Any] = {
        "type": "ranking_retention_masking_summary",
        "definition": {
            "protocol": "fixed trained MLP; keep top-k groups according to each ranking; mask all other groups; evaluate without retraining",
            "evaluation_split": "validation",
            "retention": "(mae_empty - mae_k) / (mae_empty - mae_full), computed on validation",
            "auc": "trapezoidal integral over fraction_kept in [0,1]",
            "partial_auc_20_80": "trapezoidal integral over fraction_kept in [0.20,0.80], normalized by interval width",
            "mask_value": float(mask_value),
            "random": "independent random group permutations generated on the fly",
            "group_ablation": (
                "ranking by MAE increase after masking one group at a time; "
                "ranking split is configurable and defaults to train to avoid validation overfitting"
            ),
        },
        "config": {
            "processed_root": str(processed_root),
            "baselines_root": str(baselines_root),
            "out_dir": str(out_dir),
            "only_dataset": only_dataset,
            "only_window_tag": only_window_tag,
            "only_hidden": only_hidden,
            "only_group_mode": only_group_mode,
            "n_random": int(n_random),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "mask_value": float(mask_value),
            "include_group_ablation": bool(include_group_ablation),
            "include_random": bool(include_random),
            "save_ablation_rankings": bool(save_ablation_rankings),
            "ablation_ranking_split": ablation_ranking_split,
            "ablation_max_samples": int(ablation_max_samples),
            "ablation_group_batch_size": int(ablation_group_batch_size),
            "partial_auc_lo": float(PARTIAL_AUC_LO),
            "partial_auc_hi": float(PARTIAL_AUC_HI),
        },
        "counts": {
            "targets_found": int(len(targets)),
            "curve_rows": int(len(all_curve_rows)),
            "auc_rows": int(len(all_auc_rows)),
            "errors": int(len(all_errors)),
        },
        "outputs": {
            "curves_csv": str(curves_path),
            "auc_csv": str(auc_path),
            "summary_json": str(summary_path),
        },
        "errors": all_errors,
    }

    _json_dump(summary_path, summary)

    if verbose:
        print("\n=== SUMMARY ===")
        print("curve rows:", len(all_curve_rows))
        print("auc rows:", len(all_auc_rows))
        print("errors:", len(all_errors))
        print("saved:", curves_path)
        print("saved:", auc_path)
        print("saved:", summary_path)

        if all_errors:
            print("\nFirst errors:")
            for e in all_errors[:10]:
                print(" -", e)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate grouped rankings using fixed-model masking retention. "
            "No MLP retraining is performed."
        )
    )

    parser.add_argument("--processed-root", type=str, default="datasets/processed")
    parser.add_argument("--baselines-root", type=str, default="mlp_baselines")
    parser.add_argument("--out-dir", type=str, default="tables")

    parser.add_argument("--only-dataset", type=str, default=None, help="Optional filter, e.g., FD001")
    parser.add_argument("--only-window-tag", type=str, default=None, help="Optional filter, e.g., W30_step1")
    parser.add_argument("--only-hidden", type=str, default=None, help="Optional filter, e.g., h10")
    parser.add_argument("--only-group-mode", type=str, choices=["sensors", "windows"], default=None)

    parser.add_argument("--n-random", type=int, default=100, help="Number of random permutations per configuration/mode")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask-value", type=float, default=0.0, help="Value used to mask removed groups; 0 is natural for standardized data")
    parser.add_argument("--batch-size", type=int, default=1024, help="Keras predict batch size")

    parser.add_argument("--no-group-ablation", action="store_true", help="Disable group-ablation ranking baseline")
    parser.add_argument("--no-random", action="store_true", help="Disable random-permutation baseline")
    parser.add_argument("--save-ablation-rankings", action="store_true", help="Save group-ablation rankings as JSONs in each rankings/ folder")

    parser.add_argument(
        "--ablation-ranking-split",
        type=str,
        choices=["train", "val"],
        default="train",
        help="Split used to compute the group-ablation ranking. Retention is always evaluated on validation.",
    )
    parser.add_argument(
        "--ablation-max-samples",
        type=int,
        default=0,
        help="If >0, use a deterministic subset of this size from the ablation ranking split.",
    )
    parser.add_argument(
        "--ablation-group-batch-size",
        type=int,
        default=8,
        help="Number of groups to mask/evaluate together when building group-ablation rankings.",
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run_batch(
        processed_root=Path(args.processed_root),
        baselines_root=Path(args.baselines_root),
        out_dir=Path(args.out_dir),
        only_dataset=args.only_dataset,
        only_window_tag=args.only_window_tag,
        only_hidden=args.only_hidden,
        only_group_mode=args.only_group_mode,
        n_random=int(args.n_random),
        seed=int(args.seed),
        mask_value=float(args.mask_value),
        batch_size=int(args.batch_size),
        include_group_ablation=not bool(args.no_group_ablation),
        include_random=not bool(args.no_random),
        save_ablation_rankings=bool(args.save_ablation_rankings),
        ablation_ranking_split=args.ablation_ranking_split,  # type: ignore[arg-type]
        ablation_max_samples=int(args.ablation_max_samples),
        ablation_group_batch_size=int(args.ablation_group_batch_size),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()