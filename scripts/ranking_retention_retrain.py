# scripts/ranking_retention_retrain.py
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from utils.mlp_utils import (
    MLPTrainConfig,
    build_trust_mlp,
    ensure_2d_targets,
    set_global_seed,
)

GroupMode = Literal["sensors", "windows"]

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


def _stable_seed(base_seed: int, *parts: Any) -> int:
    key = "|".join([str(base_seed), *[str(p) for p in parts]])
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# -------------------------
# Naming / traversal helpers
# -------------------------

def _is_hidden_tag(name: str) -> bool:
    return name.startswith("h") and any(ch.isdigit() for ch in name)


def _hidden_tag_to_tuple(h_tag: str) -> Tuple[int, ...]:
    if not h_tag.startswith("h"):
        raise ValueError(f"Invalid hidden tag: {h_tag}")
    parts = h_tag[1:].split("_")
    if any(p.strip() == "" for p in parts):
        raise ValueError(f"Invalid hidden tag: {h_tag}")
    return tuple(int(p) for p in parts)


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
    if method == "group_ablation":
        return rankings_dir / f"ranking_{group_mode}_group_ablation_masking.json"
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
        raise KeyError(f"dataset.npz missing keys={missing}. Available keys={list(arrays.keys())}")
    return arrays


def _load_keras_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.keras at: {model_path}")
    try:
        from tensorflow import keras  # type: ignore
    except Exception as e:
        raise ImportError("Could not import tensorflow.keras.") from e
    return keras.models.load_model(model_path, compile=False)


def _clear_tf_session() -> None:
    try:
        from tensorflow.keras import backend as K  # type: ignore
        K.clear_session()
    except Exception:
        pass
    gc.collect()


def _flatten_mw(X_mw: np.ndarray) -> np.ndarray:
    """
    X_mw: (N, M, W) -> (N, M*W), sensor-major flatten: flat = s*W + w.
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
        out: List[int] = []
        for it in ranking_json["items"]:
            if "group_index_0based" not in it:
                raise KeyError("ranking item missing group_index_0based")
            out.append(int(it["group_index_0based"]))
        return out

    if "ranking" in ranking_json and isinstance(ranking_json["ranking"], list):
        out = []
        for it in ranking_json["ranking"]:
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
# Column mapping
# -------------------------

def _group_to_cols_sensors(sensor_idx_0: int, M: int, W: int) -> np.ndarray:
    s = int(sensor_idx_0)
    return np.arange(s * W, (s + 1) * W, dtype=int)


def _group_to_cols_windows(window_idx_0: int, M: int, W: int) -> np.ndarray:
    w = int(window_idx_0)
    return (np.arange(M, dtype=int) * W + w).astype(int)


def _groups_to_cols(groups: Sequence[int], *, group_mode: GroupMode, M: int, W: int) -> np.ndarray:
    if len(groups) == 0:
        return np.asarray([], dtype=int)
    cols_list: List[np.ndarray] = []
    for g in groups:
        if group_mode == "sensors":
            cols_list.append(_group_to_cols_sensors(int(g), M=M, W=W))
        else:
            cols_list.append(_group_to_cols_windows(int(g), M=M, W=W))
    cols = np.unique(np.concatenate(cols_list).astype(int))
    return np.sort(cols)


def _ks_from_fractions(n_groups: int, fractions: Sequence[float], *, all_k: bool) -> List[int]:
    n_groups = int(n_groups)
    if all_k:
        return list(range(1, n_groups + 1))

    ks: List[int] = []
    for f in fractions:
        ff = float(f)
        if ff <= 0:
            k = 1
        else:
            k = int(np.ceil(ff * n_groups))
        k = max(1, min(n_groups, k))
        ks.append(k)

    ks.append(n_groups)
    return sorted(set(ks))


# -------------------------
# Metrics and training
# -------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = _ensure_1d_y(y_true)
    yp = _ensure_1d_y(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true/y_pred length mismatch: {yt.shape[0]} vs {yp.shape[0]}")
    err = yp - yt
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def _predict_flat(model: Any, X_flat: np.ndarray, *, batch_size: int) -> np.ndarray:
    pred = model.predict(np.asarray(X_flat, dtype=np.float32), batch_size=int(batch_size), verbose=0)
    return np.asarray(pred, dtype=np.float32)


def _mean_predictor_metrics(y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    mean_y = float(np.mean(_ensure_1d_y(y_train)))
    pred = np.full_like(_ensure_1d_y(y_val), fill_value=mean_y, dtype=np.float32)
    out = _compute_metrics(y_val, pred)
    out["mean_y_train"] = mean_y
    return out


def _retention_from_mae(mae: float, *, mae_full_ref: float, mae_mean_ref: float) -> float:
    denom = float(mae_mean_ref) - float(mae_full_ref)
    if abs(denom) <= 1e-12:
        return float("nan")
    return float((float(mae_mean_ref) - float(mae)) / denom)


def _train_eval_selected_cols(
    *,
    X_train_flat: np.ndarray,
    y_train: np.ndarray,
    X_val_flat: np.ndarray,
    y_val: np.ndarray,
    selected_cols: np.ndarray,
    hidden_sizes: Tuple[int, ...],
    learning_rate: float,
    epochs: int,
    train_batch_size: int,
    predict_batch_size: int,
    seed: int,
    keras_verbose: int,
) -> Dict[str, Any]:
    selected_cols = np.asarray(selected_cols, dtype=int)
    if selected_cols.size < 1:
        raise ValueError("selected_cols must contain at least one column")

    _clear_tf_session()
    set_global_seed(int(seed))

    Xtr = np.asarray(X_train_flat[:, selected_cols], dtype=np.float32)
    Xva = np.asarray(X_val_flat[:, selected_cols], dtype=np.float32)
    ytr = ensure_2d_targets(y_train)
    yva = ensure_2d_targets(y_val)

    model = build_trust_mlp(
        input_dim=int(Xtr.shape[1]),
        hidden_sizes=list(hidden_sizes),
        learning_rate=float(learning_rate),
    )

    hist_obj = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=int(epochs),
        batch_size=int(train_batch_size),
        verbose=int(keras_verbose),
    )

    pred_train = _predict_flat(model, Xtr, batch_size=predict_batch_size)
    pred_val = _predict_flat(model, Xva, batch_size=predict_batch_size)

    train_metrics = _compute_metrics(y_train, pred_train)
    val_metrics = _compute_metrics(y_val, pred_val)

    hist = hist_obj.history if hasattr(hist_obj, "history") else {}
    last_train_loss = float(hist.get("loss", [float("nan")])[-1]) if hist else float("nan")
    last_val_loss = float(hist.get("val_loss", [float("nan")])[-1]) if hist else float("nan")

    out = {
        "input_dim": int(Xtr.shape[1]),
        "n_cols": int(selected_cols.size),
        "train_mae": float(train_metrics["mae"]),
        "train_rmse": float(train_metrics["rmse"]),
        "val_mae": float(val_metrics["mae"]),
        "val_rmse": float(val_metrics["rmse"]),
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "epochs": int(epochs),
        "seed": int(seed),
    }

    del model
    _clear_tf_session()
    return out


# -------------------------
# AUC helpers
# -------------------------

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _partial_auc_stats(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lo: float = PARTIAL_AUC_LO,
    hi: float = PARTIAL_AUC_HI,
) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2 or lo < float(np.min(x)) or hi > float(np.max(x)) or hi <= lo:
        return {"at_lo": float("nan"), "at_hi": float("nan"), "auc_raw": float("nan"), "auc_norm": float("nan")}
    if not np.all(np.isfinite(y)):
        return {"at_lo": float("nan"), "at_hi": float("nan"), "auc_raw": float("nan"), "auc_norm": float("nan")}

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y_lo = float(np.interp(lo, x, y))
    y_hi = float(np.interp(hi, x, y))
    mask = (x > lo) & (x < hi)
    x_part = np.concatenate([[lo], x[mask], [hi]]).astype(float)
    y_part = np.concatenate([[y_lo], y[mask], [y_hi]]).astype(float)

    auc_raw = _trapz(y_part, x_part)
    return {
        "at_lo": y_lo,
        "at_hi": y_hi,
        "auc_raw": float(auc_raw),
        "auc_norm": float(auc_raw / (hi - lo)),
    }


def _summarize_curve_auc(curve_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    df_x = np.asarray([float(r["fraction_kept"]) for r in curve_rows], dtype=float)
    retention = np.asarray([float(r["retrain_retention"]) for r in curve_rows], dtype=float)
    mae = np.asarray([float(r["val_mae"]) for r in curve_rows], dtype=float)
    rmse = np.asarray([float(r["val_rmse"]) for r in curve_rows], dtype=float)

    order = np.argsort(df_x)
    x = df_x[order]
    retention = retention[order]
    mae = mae[order]
    rmse = rmse[order]

    r_part = _partial_auc_stats(x, retention)
    mae_part = _partial_auc_stats(x, mae)
    rmse_part = _partial_auc_stats(x, rmse)

    return {
        "retrain_retention_auc": _trapz(retention, x) if np.all(np.isfinite(retention)) else float("nan"),
        "retrain_retention_auc_20_80": r_part["auc_norm"],
        "retrain_retention_auc_20_80_raw": r_part["auc_raw"],
        "retrain_retention_at_20": r_part["at_lo"],
        "retrain_retention_at_80": r_part["at_hi"],
        "val_mae_auc": _trapz(mae, x) if np.all(np.isfinite(mae)) else float("nan"),
        "val_mae_auc_20_80": mae_part["auc_norm"],
        "val_mae_auc_20_80_raw": mae_part["auc_raw"],
        "val_mae_at_20": mae_part["at_lo"],
        "val_mae_at_80": mae_part["at_hi"],
        "val_rmse_auc": _trapz(rmse, x) if np.all(np.isfinite(rmse)) else float("nan"),
        "val_rmse_auc_20_80": rmse_part["auc_norm"],
        "val_rmse_auc_20_80_raw": rmse_part["auc_raw"],
        "val_rmse_at_20": rmse_part["at_lo"],
        "val_rmse_at_80": rmse_part["at_hi"],
    }


# -------------------------
# Evaluation helpers
# -------------------------

def _evaluate_one_ranking_retrain(
    *,
    ranking_top_first: Sequence[int],
    method: str,
    random_trial: int,
    ranking_file: str,
    dataset: str,
    window_tag: str,
    hidden_tag: str,
    group_mode: GroupMode,
    M: int,
    W: int,
    ks: Sequence[int],
    X_train_flat: np.ndarray,
    y_train: np.ndarray,
    X_val_flat: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: Tuple[int, ...],
    learning_rate: float,
    epochs: int,
    train_batch_size: int,
    predict_batch_size: int,
    base_seed: int,
    mae_full_ref: float,
    rmse_full_ref: float,
    mae_mean_ref: float,
    rmse_mean_ref: float,
    mean_y_train: float,
    keras_verbose: int,
    cache: Dict[Tuple[int, ...], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    n_groups = M if group_mode == "sensors" else W
    ranking = _validate_ranking(ranking_top_first, n_groups, name=f"{method}/{group_mode}")

    rows: List[Dict[str, Any]] = []

    # k=0: no model is trained; use mean predictor baseline.
    rows.append(
        {
            "dataset": dataset,
            "window_tag": window_tag,
            "hidden_tag": hidden_tag,
            "group_mode": group_mode,
            "method": method,
            "random_trial": int(random_trial),
            "k": 0,
            "n_groups": int(n_groups),
            "fraction_kept": 0.0,
            "n_features_kept": 0,
            "input_dim": 0,
            "train_mae": float("nan"),
            "train_rmse": float("nan"),
            "val_mae": float(mae_mean_ref),
            "val_rmse": float(rmse_mean_ref),
            "retrain_retention": 0.0,
            "mae_full_ref": float(mae_full_ref),
            "rmse_full_ref": float(rmse_full_ref),
            "mae_mean_ref": float(mae_mean_ref),
            "rmse_mean_ref": float(rmse_mean_ref),
            "mean_y_train": float(mean_y_train),
            "ranking_file": ranking_file,
            "train_seed": -1,
            "cache_hit": False,
        }
    )

    for k in ks:
        k = int(k)
        keep_groups = ranking[:k]
        selected_cols = _groups_to_cols(keep_groups, group_mode=group_mode, M=M, W=W)
        cache_key = tuple(int(c) for c in selected_cols.tolist())

        if cache_key in cache:
            train_out = dict(cache[cache_key])
            cache_hit = True
        else:
            col_hash = hashlib.sha256(",".join(map(str, cache_key)).encode("utf-8")).hexdigest()[:12]
            train_seed = _stable_seed(base_seed, dataset, window_tag, hidden_tag, group_mode, "cols", col_hash)
            train_out = _train_eval_selected_cols(
                X_train_flat=X_train_flat,
                y_train=y_train,
                X_val_flat=X_val_flat,
                y_val=y_val,
                selected_cols=selected_cols,
                hidden_sizes=hidden_sizes,
                learning_rate=learning_rate,
                epochs=epochs,
                train_batch_size=train_batch_size,
                predict_batch_size=predict_batch_size,
                seed=train_seed,
                keras_verbose=keras_verbose,
            )
            cache[cache_key] = dict(train_out)
            cache_hit = False

        val_mae = float(train_out["val_mae"])
        retention = _retention_from_mae(val_mae, mae_full_ref=mae_full_ref, mae_mean_ref=mae_mean_ref)

        rows.append(
            {
                "dataset": dataset,
                "window_tag": window_tag,
                "hidden_tag": hidden_tag,
                "group_mode": group_mode,
                "method": method,
                "random_trial": int(random_trial),
                "k": int(k),
                "n_groups": int(n_groups),
                "fraction_kept": float(k) / float(n_groups),
                "n_features_kept": int(selected_cols.size),
                "input_dim": int(train_out["input_dim"]),
                "train_mae": float(train_out["train_mae"]),
                "train_rmse": float(train_out["train_rmse"]),
                "val_mae": val_mae,
                "val_rmse": float(train_out["val_rmse"]),
                "retrain_retention": float(retention),
                "mae_full_ref": float(mae_full_ref),
                "rmse_full_ref": float(rmse_full_ref),
                "mae_mean_ref": float(mae_mean_ref),
                "rmse_mean_ref": float(rmse_mean_ref),
                "mean_y_train": float(mean_y_train),
                "ranking_file": ranking_file,
                "train_seed": int(train_out["seed"]),
                "epochs": int(train_out["epochs"]),
                "last_train_loss": float(train_out["last_train_loss"]),
                "last_val_loss": float(train_out["last_val_loss"]),
                "cache_hit": bool(cache_hit),
            }
        )

    auc_stats = _summarize_curve_auc(rows)
    auc_row: Dict[str, Any] = {
        "dataset": dataset,
        "window_tag": window_tag,
        "hidden_tag": hidden_tag,
        "group_mode": group_mode,
        "method": method,
        "random_trial": int(random_trial),
        "n_groups": int(n_groups),
        "n_points": int(len(rows)),
        "ks_evaluated": ",".join(str(int(r["k"])) for r in rows),
        "mae_full_ref": float(mae_full_ref),
        "rmse_full_ref": float(rmse_full_ref),
        "mae_mean_ref": float(mae_mean_ref),
        "rmse_mean_ref": float(rmse_mean_ref),
        "mean_y_train": float(mean_y_train),
        "ranking_file": ranking_file,
        "partial_auc_lo": float(PARTIAL_AUC_LO),
        "partial_auc_hi": float(PARTIAL_AUC_HI),
        **auc_stats,
    }
    return rows, auc_row


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
    include_random: bool,
    n_random: int,
    seed: int,
    fractions: Sequence[float],
    all_k: bool,
    learning_rate: float,
    epochs: int,
    train_batch_size: int,
    predict_batch_size: int,
    keras_verbose: int,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    curve_rows_all: List[Dict[str, Any]] = []
    auc_rows_all: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    arrays = _load_processed_dataset(processed_root, dataset, window_tag)
    X_train_mw = np.asarray(arrays["X_train_mw"], dtype=np.float32)
    X_val_mw = np.asarray(arrays["X_val_mw"], dtype=np.float32)
    y_train = _ensure_1d_y(arrays["y_train"])
    y_val = _ensure_1d_y(arrays["y_val"])

    if X_train_mw.ndim != 3 or X_val_mw.ndim != 3:
        raise ValueError(f"X_train_mw/X_val_mw must be 3D. Got {X_train_mw.shape} / {X_val_mw.shape}")
    if X_train_mw.shape[1:] != X_val_mw.shape[1:]:
        raise ValueError(f"Train/val shape mismatch: {X_train_mw.shape} vs {X_val_mw.shape}")

    _, M, W = X_val_mw.shape
    X_train_flat = _flatten_mw(X_train_mw)
    X_val_flat = _flatten_mw(X_val_mw)
    hidden_sizes = _hidden_tag_to_tuple(hidden_tag)

    # Reference metrics: original saved baseline model and mean predictor.
    baseline_model_path = baseline_dir / "model.keras"
    baseline_model = _load_keras_model(baseline_model_path)
    pred_full = _predict_flat(baseline_model, X_val_flat, batch_size=predict_batch_size)
    full_ref = _compute_metrics(y_val, pred_full)
    del baseline_model
    _clear_tf_session()

    mean_ref = _mean_predictor_metrics(y_train, y_val)

    mae_full_ref = float(full_ref["mae"])
    rmse_full_ref = float(full_ref["rmse"])
    mae_mean_ref = float(mean_ref["mae"])
    rmse_mean_ref = float(mean_ref["rmse"])
    mean_y_train = float(mean_ref["mean_y_train"])

    rankings_dir = baseline_dir / "rankings"

    if verbose:
        print(
            f"\n[{dataset}/{window_tag}/{hidden_tag}] "
            f"X_train={tuple(X_train_mw.shape)} X_val={tuple(X_val_mw.shape)} "
            f"ref_mae_full={mae_full_ref:.4f} ref_mae_mean={mae_mean_ref:.4f}"
        )

    # Cache train/eval metrics for identical selected-column subsets within this configuration.
    cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    for group_mode in group_modes:
        n_groups = M if group_mode == "sensors" else W
        ks = _ks_from_fractions(n_groups, fractions, all_k=all_k)

        if verbose:
            print(f"  group_mode={group_mode} n_groups={n_groups} ks={ks}")

        # 1) Deterministic rankings from JSON.
        for method in deterministic_methods:
            rpath = _ranking_path(rankings_dir, group_mode, method)
            if not rpath.exists():
                errors.append({
                    "dataset": dataset,
                    "window_tag": window_tag,
                    "hidden_tag": hidden_tag,
                    "group_mode": group_mode,
                    "method": method,
                    "error": f"missing ranking file: {rpath}",
                })
                if verbose:
                    print(f"    [skip] {method}: missing {rpath.name}")
                continue

            try:
                ranking = _load_ranking_file(rpath, n_groups=n_groups, name=method)
                rows, auc = _evaluate_one_ranking_retrain(
                    ranking_top_first=ranking,
                    method=method,
                    random_trial=-1,
                    ranking_file=str(rpath),
                    dataset=dataset,
                    window_tag=window_tag,
                    hidden_tag=hidden_tag,
                    group_mode=group_mode,
                    M=M,
                    W=W,
                    ks=ks,
                    X_train_flat=X_train_flat,
                    y_train=y_train,
                    X_val_flat=X_val_flat,
                    y_val=y_val,
                    hidden_sizes=hidden_sizes,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    train_batch_size=train_batch_size,
                    predict_batch_size=predict_batch_size,
                    base_seed=seed,
                    mae_full_ref=mae_full_ref,
                    rmse_full_ref=rmse_full_ref,
                    mae_mean_ref=mae_mean_ref,
                    rmse_mean_ref=rmse_mean_ref,
                    mean_y_train=mean_y_train,
                    keras_verbose=keras_verbose,
                    cache=cache,
                )
                curve_rows_all.extend(rows)
                auc_rows_all.append(auc)
                if verbose:
                    print(
                        f"    [OK] {method}: "
                        f"RetAUC={auc['retrain_retention_auc']:.4f} "
                        f"RetAUC20-80={auc['retrain_retention_auc_20_80']:.4f} "
                        f"MAEAUC20-80={auc['val_mae_auc_20_80']:.4f}"
                    )
            except Exception as e:
                errors.append({
                    "dataset": dataset,
                    "window_tag": window_tag,
                    "hidden_tag": hidden_tag,
                    "group_mode": group_mode,
                    "method": method,
                    "error": str(e),
                })
                if verbose:
                    print(f"    [ERR] {method}: {e}")

        # 2) Random permutations.
        if include_random and int(n_random) > 0:
            rng = np.random.default_rng(_stable_seed(seed, dataset, window_tag, hidden_tag, group_mode, "random"))
            random_aucs: List[float] = []
            for trial in range(int(n_random)):
                ranking_r = rng.permutation(n_groups).astype(int).tolist()
                try:
                    rows, auc = _evaluate_one_ranking_retrain(
                        ranking_top_first=ranking_r,
                        method="random_permutation",
                        random_trial=int(trial),
                        ranking_file="",
                        dataset=dataset,
                        window_tag=window_tag,
                        hidden_tag=hidden_tag,
                        group_mode=group_mode,
                        M=M,
                        W=W,
                        ks=ks,
                        X_train_flat=X_train_flat,
                        y_train=y_train,
                        X_val_flat=X_val_flat,
                        y_val=y_val,
                        hidden_sizes=hidden_sizes,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        train_batch_size=train_batch_size,
                        predict_batch_size=predict_batch_size,
                        base_seed=seed,
                        mae_full_ref=mae_full_ref,
                        rmse_full_ref=rmse_full_ref,
                        mae_mean_ref=mae_mean_ref,
                        rmse_mean_ref=rmse_mean_ref,
                        mean_y_train=mean_y_train,
                        keras_verbose=keras_verbose,
                        cache=cache,
                    )
                    curve_rows_all.extend(rows)
                    auc_rows_all.append(auc)
                    random_aucs.append(float(auc["retrain_retention_auc_20_80"]))
                except Exception as e:
                    errors.append({
                        "dataset": dataset,
                        "window_tag": window_tag,
                        "hidden_tag": hidden_tag,
                        "group_mode": group_mode,
                        "method": "random_permutation",
                        "random_trial": str(trial),
                        "error": str(e),
                    })
                    if verbose:
                        print(f"    [ERR] random trial {trial}: {e}")

            if verbose and random_aucs:
                print(
                    f"    [OK] random_permutation: n={len(random_aucs)} "
                    f"mean_RetAUC20-80={np.mean(random_aucs):.4f} std={np.std(random_aucs):.4f}"
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
    fractions: Sequence[float],
    all_k: bool,
    epochs: int,
    train_batch_size: int,
    predict_batch_size: int,
    learning_rate: float,
    keras_verbose: int,
    include_group_ablation: bool,
    include_random: bool,
    verbose: bool,
) -> Dict[str, Any]:
    if only_group_mode is not None and only_group_mode not in {"sensors", "windows"}:
        raise ValueError("--only-group-mode must be one of: sensors, windows")

    group_modes: List[GroupMode] = ["sensors", "windows"] if only_group_mode is None else [only_group_mode]  # type: ignore[list-item]

    deterministic_methods = ["fulltrust_agg_selection", "timetrust_selection", "weights"]
    if include_group_ablation:
        deterministic_methods.append("group_ablation")

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
        print("=== ranking_retention_retrain ===")
        print("processed_root:", processed_root)
        print("baselines_root:", baselines_root)
        print("out_dir:", out_dir)
        print("targets:", len(targets))
        print("group_modes:", group_modes)
        print("deterministic_methods:", deterministic_methods)
        print("include_random:", include_random, "n_random:", n_random)
        print("fractions:", list(fractions), "all_k:", all_k)
        print("epochs:", epochs, "train_batch_size:", train_batch_size, "learning_rate:", learning_rate)
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
                include_random=include_random,
                n_random=int(n_random),
                seed=int(seed),
                fractions=fractions,
                all_k=all_k,
                learning_rate=float(learning_rate),
                epochs=int(epochs),
                train_batch_size=int(train_batch_size),
                predict_batch_size=int(predict_batch_size),
                keras_verbose=int(keras_verbose),
                verbose=verbose,
            )
            all_curve_rows.extend(curve_rows)
            all_auc_rows.extend(auc_rows)
            all_errors.extend(errors)
        except Exception as e:
            all_errors.append({
                "dataset": dataset,
                "window_tag": window_tag,
                "hidden_tag": hidden_tag,
                "group_mode": "*",
                "method": "*",
                "error": str(e),
            })
            if verbose:
                print(f"[ERR] {dataset}/{window_tag}/{hidden_tag}: {e}")

    _safe_mkdir(out_dir)
    curves_path = out_dir / "ranking_retention_retrain_curves.csv"
    auc_path = out_dir / "ranking_retention_retrain_auc.csv"
    summary_path = out_dir / "ranking_retention_retrain_summary.json"

    _write_csv(curves_path, all_curve_rows)
    _write_csv(auc_path, all_auc_rows)

    summary: Dict[str, Any] = {
        "type": "ranking_retention_retrain_summary",
        "definition": {
            "protocol": "For each ranking and top-k group subset, train a new MLP from scratch using only the selected original input columns; evaluate on validation.",
            "k0": "k=0 is represented by a mean-y_train predictor, no MLP is trained.",
            "reference_full": "mae_full_ref is computed using the saved original baseline model.keras on validation.",
            "retrain_retention": "(mae_mean_ref - val_mae_k) / (mae_mean_ref - mae_full_ref). Higher is better.",
            "partial_auc_20_80": "AUC over fraction_kept in [0.20, 0.80], normalized by interval width.",
            "column_order": "Selected columns are sorted in original flattened order, flat=s*W+w.",
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
            "fractions": [float(f) for f in fractions],
            "all_k": bool(all_k),
            "epochs": int(epochs),
            "train_batch_size": int(train_batch_size),
            "predict_batch_size": int(predict_batch_size),
            "learning_rate": float(learning_rate),
            "keras_verbose": int(keras_verbose),
            "include_group_ablation": bool(include_group_ablation),
            "include_random": bool(include_random),
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
            "Evaluate grouped rankings using retrained-subset retention. "
            "For each top-k subset, a new MLP is trained using only selected groups."
        )
    )
    parser.add_argument("--processed-root", type=str, default="datasets/processed")
    parser.add_argument("--baselines-root", type=str, default="mlp_baselines")
    parser.add_argument("--out-dir", type=str, default="tables_retrain")

    parser.add_argument("--only-dataset", type=str, default=None, help="Optional filter, e.g., FD001")
    parser.add_argument("--only-window-tag", type=str, default=None, help="Optional filter, e.g., W30_step1")
    parser.add_argument("--only-hidden", type=str, default=None, help="Optional filter, e.g., h10")
    parser.add_argument("--only-group-mode", type=str, choices=["sensors", "windows"], default=None)

    parser.add_argument("--fractions", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0], help="Fractions of groups to keep. k=ceil(f*n_groups). k=0 mean predictor is always included.")
    parser.add_argument("--all-k", action="store_true", help="Evaluate all k=1..G instead of only --fractions. Much more expensive.")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--keras-verbose", type=int, default=0)

    parser.add_argument("--n-random", type=int, default=0, help="Number of random permutations. Default 0 because retraining is expensive.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-group-ablation", action="store_true", help="Do not read/evaluate saved group-ablation ranking JSONs.")
    parser.add_argument("--no-random", action="store_true", help="Disable random permutations.")
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
        fractions=[float(f) for f in args.fractions],
        all_k=bool(args.all_k),
        epochs=int(args.epochs),
        train_batch_size=int(args.train_batch_size),
        predict_batch_size=int(args.predict_batch_size),
        learning_rate=float(args.learning_rate),
        keras_verbose=int(args.keras_verbose),
        include_group_ablation=not bool(args.no_group_ablation),
        include_random=not bool(args.no_random),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
