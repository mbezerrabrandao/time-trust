#!/usr/bin/env python3
"""
Run Time-TRUST stability experiments without retraining baseline MLPs.

This script keeps the trained baseline MLP fixed and varies only the validation
windows explained by Time-TRUST.

Recommended design:

1) Select n training windows inside each RUL bin.
2) Optionally select n validation windows from the same RUL bin for surrogate validation metrics.
3) Run one Time-TRUST solve on that mini-batch.
4) Repeat with different mini-batches from the same bin.
5) Compare runtime and selected sensor/window groups across draws.

This gives stability across comparable degradation regimes without retraining the
fixed baseline MLP and without launching one expensive MILP per individual
window. The TRUST surrogate is still trained inside each TRUST run, as in the
original pipeline.

Example:
    python scripts/run_trust_stability_experiment.py \
        --dataset FD001 \
        --seq-len 30 \
        --step 1 \
        --hidden 10 10 \
        --modes sensors windows \
        --rul-bins late:0:30 mid:30:80 early:80:999999 \
        --analysis-unit bin \
        --n-per-bin 10 \
        --draws-per-bin 5 \
        --repeat-solve 1 \
        --mlp-mode transfer \
        --milp-time-cap 60 \
        --results-root results/time_trust_stability \
        --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------
# Project path bootstrap
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------
from trust.pipeline import execute_trust_algorithm
from utils.milp_utils import (
    TRUST_MODE_SENSORS,
    TRUST_MODE_WINDOWS,
    MLP_REBUILD,
    MLP_TRANSFER,
    BASELINE_MODE_NONE,
    BASELINE_MODE_CAPPED,
)
from utils.mlp_utils import RANDOM_SEED


# =========================
# Loading helpers
# =========================

def _window_tag(seq_len: int, step: int) -> str:
    return f"W{int(seq_len)}_step{int(step)}"


def _load_processed_dataset(
    processed_root: Path,
    dataset_name: str,
    window_tag: str,
) -> Dict[str, np.ndarray]:
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
    hidden_layers: Sequence[int],
) -> Dict[str, Any]:
    hs_tag = "h" + "_".join(str(int(x)) for x in hidden_layers)
    art_path = baselines_root / dataset_name / window_tag / hs_tag / "baseline_mlp_artifacts.npz"
    if not art_path.exists():
        raise FileNotFoundError(f"Baseline artifact not found: {art_path}")

    data = np.load(art_path, allow_pickle=True)

    if "meta_json" not in data.files:
        raise KeyError(f"baseline_mlp_artifacts.npz missing 'meta_json': {art_path}")

    meta_raw = data["meta_json"]
    meta = json.loads(str(meta_raw.item()))

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

    if "final_metrics" not in meta:
        meta["final_metrics"] = {}

    return {
        "artifact_path": str(art_path),
        "meta": meta,
        "weights": weights,
        "final_metrics": meta.get("final_metrics", {}),
    }


def _baseline_weights_to_trust_format(
    weights: Dict[str, Dict[str, np.ndarray]],
    hidden_layers: Sequence[int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    hidden_Ws: List[np.ndarray] = []
    hidden_bs: List[np.ndarray] = []

    for i in range(1, len(hidden_layers) + 1):
        lk = f"hidden_l{i}"
        if lk not in weights:
            raise KeyError(f"Missing '{lk}' in baseline weights.")
        hidden_Ws.append(np.asarray(weights[lk]["W"], dtype=float))
        hidden_bs.append(np.asarray(weights[lk]["b"], dtype=float).reshape(-1))

    if "rul_output" not in weights:
        raise KeyError("Missing 'rul_output' in baseline weights.")

    W_out = np.asarray(weights["rul_output"]["W"], dtype=float).reshape(-1)
    b_out = np.asarray(weights["rul_output"]["b"], dtype=float).reshape(-1)

    hidden_Ws.append(W_out)
    hidden_bs.append(b_out)

    return hidden_Ws, hidden_bs


def _weights_dict_to_weights_list(
    weights: Dict[str, Dict[str, np.ndarray]],
    hidden_layers: Sequence[int],
) -> List[np.ndarray]:
    weights_list: List[np.ndarray] = []

    for i in range(1, len(hidden_layers) + 1):
        k = f"hidden_l{i}"
        weights_list.append(np.asarray(weights[k]["W"]))
        weights_list.append(np.asarray(weights[k]["b"]))

    weights_list.append(np.asarray(weights["rul_output"]["W"]))
    weights_list.append(np.asarray(weights["rul_output"]["b"]))

    return weights_list


# =========================
# Sampling helpers
# =========================

def _parse_float_token(x: str) -> float:
    x = x.strip().lower()
    if x in {"inf", "+inf", "infty", "infinity"}:
        return float("inf")
    return float(x)


def _parse_rul_bins(specs: Sequence[str]) -> List[Tuple[str, float, float]]:
    """
    Parse bins in the format:
        late:0:30 mid:30:80 early:80:999999

    Intervals use:
        lower <= y < upper

    Use 999999 or inf for the open-ended upper bound.
    """
    bins: List[Tuple[str, float, float]] = []

    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid RUL bin '{spec}'. Expected format name:min:max, e.g. late:0:30"
            )
        name = parts[0].strip()
        lo = _parse_float_token(parts[1])
        hi = _parse_float_token(parts[2])

        if not name:
            raise ValueError(f"Invalid empty bin name in '{spec}'.")
        if not lo < hi:
            raise ValueError(f"Invalid bin '{spec}': min must be < max.")

        bins.append((name, lo, hi))

    return bins


def _sample_indices_for_bins(
    y_val: np.ndarray,
    bins: Sequence[Tuple[str, float, float]],
    n_per_bin: int,
    rng: np.random.Generator,
    replace: bool,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible helper for one-window-per-run analysis.
    Each returned item contains a single idx.
    """
    selected: List[Dict[str, Any]] = []

    for bin_name, lo, hi in bins:
        candidates = np.where((y_val >= lo) & (y_val < hi))[0]
        if len(candidates) == 0:
            print(f"[WARN] No validation instances found for bin {bin_name}: [{lo}, {hi}).")
            continue

        if len(candidates) < n_per_bin and not replace:
            print(
                f"[WARN] Bin {bin_name} has only {len(candidates)} candidates; "
                f"using all of them because --replace-sampling is not set."
            )
            chosen = candidates
        else:
            chosen = rng.choice(candidates, size=n_per_bin, replace=replace)

        for idx in chosen.tolist():
            selected.append(
                {
                    "bin": bin_name,
                    "rul_min": lo,
                    "rul_max": hi,
                    "idx": int(idx),
                    "indices": [int(idx)],
                    "y": float(y_val[int(idx)]),
                    "y_values": [float(y_val[int(idx)])],
                    "draw": 0,
                }
            )

    return selected


def _sample_batches_for_bins(
    y_val: np.ndarray,
    bins: Sequence[Tuple[str, float, float]],
    n_per_bin: int,
    draws_per_bin: int,
    rng: np.random.Generator,
    replace: bool,
) -> List[Dict[str, Any]]:
    """
    Recommended helper for mini-batch-per-run analysis.

    Each returned item contains n_per_bin validation-window indices sampled from
    the same RUL bin. Multiple draws create different mini-batches, enabling
    group-selection stability across comparable degradation states.
    """
    batches: List[Dict[str, Any]] = []

    for bin_name, lo, hi in bins:
        candidates = np.where((y_val >= lo) & (y_val < hi))[0]
        if len(candidates) == 0:
            print(f"[WARN] No validation instances found for bin {bin_name}: [{lo}, {hi}).")
            continue

        for draw in range(int(draws_per_bin)):
            if len(candidates) < n_per_bin and not replace:
                print(
                    f"[WARN] Bin {bin_name} has only {len(candidates)} candidates; "
                    f"using all of them in draw {draw} because --replace-sampling is not set."
                )
                chosen = candidates
            else:
                chosen = rng.choice(candidates, size=n_per_bin, replace=replace)

            chosen = np.asarray(chosen, dtype=int)
            y_values = y_val[chosen]
            batches.append(
                {
                    "bin": bin_name,
                    "rul_min": lo,
                    "rul_max": hi,
                    "draw": int(draw),
                    "indices": [int(i) for i in chosen.tolist()],
                    "idx": int(chosen[0]) if len(chosen) else -1,
                    "y": float(np.mean(y_values)) if len(y_values) else float("nan"),
                    "y_values": [float(v) for v in y_values.tolist()],
                    "batch_size": int(len(chosen)),
                    "y_mean": float(np.mean(y_values)) if len(y_values) else float("nan"),
                    "y_min": float(np.min(y_values)) if len(y_values) else float("nan"),
                    "y_max": float(np.max(y_values)) if len(y_values) else float("nan"),
                }
            )

    return batches


# =========================
# Selection extraction helpers
# =========================

def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _as_group_set(obj: Any) -> Optional[List[str]]:
    """
    Convert a likely selected-group object into a sorted list of group labels.

    Supported forms:
        - boolean mask: [False, True, ...] -> ["1", ...]
        - list of integers/strings: [2, 7] -> ["2", "7"]
        - dict of group -> bool: {"s2": true}
        - dict/list with scores is intentionally not interpreted unless obvious
    """
    if obj is None:
        return None

    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    if isinstance(obj, dict):
        if all(isinstance(v, (bool, np.bool_)) for v in obj.values()):
            return sorted(str(k) for k, v in obj.items() if bool(v))

        for key in ["selected", "selected_groups", "selected_indices", "indices", "groups", "mask"]:
            if key in obj:
                groups = _as_group_set(obj[key])
                if groups is not None:
                    return groups

        return None

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return []

        if all(isinstance(v, (bool, np.bool_)) for v in obj):
            return [str(i) for i, v in enumerate(obj) if bool(v)]

        if all(isinstance(v, (int, np.integer, str)) for v in obj):
            return sorted(str(v) for v in obj)

        return None

    return None


def _iter_nested_items(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            yield path, v
            yield from _iter_nested_items(v, path)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            path = f"{prefix}.{i}" if prefix else str(i)
            yield path, v
            yield from _iter_nested_items(v, path)


def _mode_tokens(mode: str) -> List[str]:
    mode = mode.lower()
    if "sensor" in mode:
        return ["sensor", "sensors"]
    if "window" in mode:
        return ["window", "windows", "temporal"]
    return ["group", "groups"]


def _history_selected_key(mode: str) -> str:
    mode_l = mode.lower()
    if "sensor" in mode_l:
        return "selected_sensors"
    if "window" in mode_l:
        return "selected_windows"
    return "selected_groups"


def _extract_from_history(obj: Any, mode: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Prefer the final selected groups from a TRUST history list.

    Important: earlier versions of this script could accidentally extract
    history.1.selected_sensors, which is an intermediate state. For stability
    analysis we usually want the final non-empty selected_sensors/windows entry.
    """
    if not isinstance(obj, dict):
        return None, None

    history = obj.get("history", None)
    if not isinstance(history, list) or len(history) == 0:
        return None, None

    preferred_key = _history_selected_key(mode)
    fallback_keys = [
        preferred_key,
        "selected_groups",
        "selected_group_indices",
        "selected_indices",
        "selected_mask",
        "active_groups",
    ]

    for i in range(len(history) - 1, -1, -1):
        step = history[i]
        if not isinstance(step, dict):
            continue
        for key in fallback_keys:
            if key in step:
                groups = _as_group_set(step[key])
                if groups is not None:
                    return groups, f"history.{i}.{key}"

    return None, None


def _extract_selected_groups_from_result(result: Dict[str, Any], mode: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Best-effort extraction from execute_trust_algorithm's returned dict.
    """
    # First prefer the final history entry, if available.
    groups, source = _extract_from_history(result, mode)
    if groups is not None:
        return groups, source

    tokens = _mode_tokens(mode)
    positive_words = ["selected", "active", "kept", "chosen", "group", "groups", "mask", "indices"]

    direct_keys = [
        "final_selected_groups",
        "final_selected_sensors",
        "final_selected_windows",
        "selected_groups",
        "selected_group_indices",
        "selected_indices",
        "selected_mask",
        "selected_sensors",
        "selected_sensor_indices",
        "sensor_mask",
        "selected_windows",
        "selected_window_indices",
        "window_mask",
        "groups_selected",
        "active_groups",
    ]
    for key in direct_keys:
        if key in result:
            groups = _as_group_set(result[key])
            if groups is not None:
                return groups, key

    # Avoid returning early intermediate history entries. They are handled above.
    for path, value in _iter_nested_items(result):
        low = path.lower()
        if low.startswith("history."):
            continue
        if any(tok in low for tok in tokens) and any(w in low for w in positive_words):
            groups = _as_group_set(value)
            if groups is not None:
                return groups, path

    return None, None


def _extract_selected_groups_from_out_dir(out_dir: Optional[Path], mode: str) -> Tuple[Optional[List[str]], Optional[str]]:
    if out_dir is None or not out_dir.exists():
        return None, None

    tokens = _mode_tokens(mode)
    positive_words = ["selected", "active", "kept", "chosen", "group", "groups", "mask", "indices"]

    candidate_files: List[Path] = []
    candidate_files.extend(sorted(out_dir.rglob("*.json")))
    candidate_files.extend(sorted(out_dir.rglob("*.npz")))

    candidate_files.sort(
        key=lambda p: (
            not any(w in p.name.lower() for w in positive_words),
            len(p.parts),
            p.name,
        )
    )

    for path in candidate_files[:50]:
        try:
            if path.suffix == ".json":
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)

                groups, source = _extract_from_history(obj, mode)
                if groups is not None:
                    return groups, f"{path}:{source}"

                for nested_path, value in _iter_nested_items(obj):
                    low = nested_path.lower()
                    if low.startswith("history."):
                        continue
                    if any(tok in low for tok in tokens) and any(w in low for w in positive_words):
                        groups = _as_group_set(value)
                        if groups is not None:
                            return groups, f"{path}:{nested_path}"

            elif path.suffix == ".npz":
                data = np.load(path, allow_pickle=True)
                for key in data.files:
                    low = key.lower()
                    if any(tok in low for tok in tokens) and any(w in low for w in positive_words):
                        value = data[key]
                        if value.shape == ():
                            value = value.item()
                        groups = _as_group_set(value)
                        if groups is not None:
                            return groups, f"{path}:{key}"

        except Exception:
            continue

    return None, None


# =========================
# Summary helpers
# =========================

def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(_jsonify(obj), sort_keys=True)


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def _std(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else None
    return float(np.std(vals, ddof=1))


def _median(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.median(vals))


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.percentile(vals, q))


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    A = set(a)
    B = set(b)
    if len(A) == 0 and len(B) == 0:
        return 1.0
    if len(A | B) == 0:
        return 0.0
    return len(A & B) / len(A | B)


def _pairwise_jaccard(group_sets: Sequence[Sequence[str]]) -> Optional[float]:
    sets = [list(s) for s in group_sets if s is not None]
    if len(sets) < 2:
        return None

    scores: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            scores.append(_jaccard(sets[i], sets[j]))

    return float(np.mean(scores)) if scores else None


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _build_summaries(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return:
        stability_rows: runtime + pairwise Jaccard per dataset/mode/bin.
        frequency_rows: selection frequency per dataset/mode/bin/group.
    """
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["mode"]), str(row["bin"]))
        grouped.setdefault(key, []).append(row)

    stability_rows: List[Dict[str, Any]] = []
    frequency_rows: List[Dict[str, Any]] = []

    for (dataset, mode, bin_name), group_rows in sorted(grouped.items()):
        ok_rows = [r for r in group_rows if r.get("status") == "ok"]
        runtimes = [float(r["runtime_seconds"]) for r in ok_rows if r.get("runtime_seconds") not in ("", None)]

        selected_sets: List[List[str]] = []
        for r in ok_rows:
            raw = r.get("selected_groups_json", "")
            if not raw:
                continue
            try:
                groups = json.loads(raw)
            except Exception:
                continue
            if isinstance(groups, list):
                selected_sets.append([str(g) for g in groups])

        runtime_mean = _mean(runtimes)
        runtime_std = _std(runtimes)
        cv = None
        if runtime_mean is not None and runtime_mean > 0 and runtime_std is not None:
            cv = runtime_std / runtime_mean

        stability_rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "bin": bin_name,
                "n_runs": len(group_rows),
                "n_ok": len(ok_rows),
                "n_with_selection": len(selected_sets),
                "runtime_mean_seconds": runtime_mean,
                "runtime_median_seconds": _median(runtimes),
                "runtime_std_seconds": runtime_std,
                "runtime_cv": cv,
                "runtime_p95_seconds": _percentile(runtimes, 95),
                "mean_pairwise_jaccard": _pairwise_jaccard(selected_sets),
            }
        )

        counts: Dict[str, int] = {}
        for groups in selected_sets:
            for g in set(groups):
                counts[g] = counts.get(g, 0) + 1

        denom = len(selected_sets)
        for group_label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            frequency_rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "bin": bin_name,
                    "group": group_label,
                    "count": count,
                    "n_with_selection": denom,
                    "frequency": count / denom if denom else None,
                }
            )

    return stability_rows, frequency_rows


# =========================
# Main experiment
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Time-TRUST runtime and group-selection stability experiments "
            "without retraining the fixed baseline MLP."
        )
    )

    # Data/model selection
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. FD001.")
    parser.add_argument("--processed-root", type=str, default="datasets/processed")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--baselines-root", type=str, default="mlp_baselines")
    parser.add_argument("--hidden", type=int, nargs="+", required=True)

    # Experiment design
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=[TRUST_MODE_SENSORS, TRUST_MODE_WINDOWS],
        choices=[TRUST_MODE_SENSORS, TRUST_MODE_WINDOWS, "sensors", "windows"],
        help="Structured modes to evaluate. Full TRUST is intentionally excluded.",
    )
    parser.add_argument(
        "--rul-bins",
        type=str,
        nargs="+",
        default=["late:0:30", "mid:30:80", "early:80:999999"],
        help="RUL bins as name:min:max. Intervals are lower <= y < upper.",
    )
    parser.add_argument("--n-per-bin", type=int, default=20, help="Number of validation windows sampled per RUL bin or per mini-batch draw.")
    parser.add_argument(
        "--analysis-unit",
        type=str,
        default="bin",
        choices=["bin", "instance"],
        help=(
            "bin: one TRUST run per mini-batch of n windows from the same RUL bin. "
            "instance: one TRUST run per individual window."
        ),
    )
    parser.add_argument(
        "--draw-source",
        type=str,
        default="train",
        choices=["train", "val"],
        help=(
            "Which split is resampled across draws. Use train for selection stability, "
            "because TRUST centroids and surrogate training are based on X_train. "
            "Use val only for runtime/evaluation sensitivity."
        ),
    )
    parser.add_argument(
        "--draws-per-bin",
        type=int,
        default=1,
        help=(
            "Number of different mini-batches sampled per RUL bin. "
            "Used when --analysis-unit bin."
        ),
    )
    parser.add_argument(
        "--repeat-solve",
        type=int,
        default=1,
        help=(
            "Number of repeated solves for the exact same selected instance. "
            "Use >1 only for runtime/solver reproducibility."
        ),
    )
    parser.add_argument(
        "--replace-sampling",
        action="store_true",
        help="Sample with replacement when a RUL bin has fewer candidates than --n-per-bin.",
    )
    parser.add_argument(
        "--vary-seed",
        action="store_true",
        help=(
            "If set, increment the seed across repeated solves. Default keeps seed fixed "
            "to isolate variation due to the explained input."
        ),
    )

    # TRUST configuration
    parser.add_argument("--mlp-mode", type=str, default=MLP_TRANSFER, choices=[MLP_REBUILD, MLP_TRANSFER])
    parser.add_argument("--C", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--milp-time-cap", type=int, default=60)

    # Baseline cap
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default=BASELINE_MODE_CAPPED,
        choices=[BASELINE_MODE_NONE, BASELINE_MODE_CAPPED],
    )
    parser.add_argument("--baseline-slack", type=float, default=0.0)

    # Surrogate hyperparameters. These matter if --mlp-mode rebuild is used.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    # Output / control
    parser.add_argument("--results-root", type=str, default="results/time_trust_stability")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    dataset_name = str(args.dataset)
    window_tag = _window_tag(args.seq_len, args.step)
    hidden_layers = [int(x) for x in args.hidden]
    processed_root = Path(args.processed_root)
    baselines_root = Path(args.baselines_root)
    results_root = Path(args.results_root)

    modes = []
    for mode in args.modes:
        if mode == "sensors":
            modes.append(TRUST_MODE_SENSORS)
        elif mode == "windows":
            modes.append(TRUST_MODE_WINDOWS)
        else:
            modes.append(mode)

    bins = _parse_rul_bins(args.rul_bins)
    rng = np.random.default_rng(int(args.seed))

    arrays = _load_processed_dataset(processed_root, dataset_name, window_tag)
    X_train = np.asarray(arrays["X_train_mw"], dtype=np.float32)
    y_train = np.asarray(arrays["y_train"], dtype=np.float32).reshape(-1)
    X_val = np.asarray(arrays["X_val_mw"], dtype=np.float32)
    y_val = np.asarray(arrays["y_val"], dtype=np.float32).reshape(-1)

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
    baseline_weights_list = _weights_dict_to_weights_list(
        weights=baseline_art["weights"],
        hidden_layers=hidden_layers,
    )
    baseline_metrics = baseline_art.get("final_metrics", None)

    draw_source = str(args.draw_source)
    y_source = y_train if draw_source == "train" else y_val

    if str(args.analysis_unit) == "bin":
        selected_batches = _sample_batches_for_bins(
            y_val=y_source,
            bins=bins,
            n_per_bin=int(args.n_per_bin),
            draws_per_bin=int(args.draws_per_bin),
            rng=rng,
            replace=bool(args.replace_sampling),
        )
    else:
        selected_batches = _sample_indices_for_bins(
            y_val=y_source,
            bins=bins,
            n_per_bin=int(args.n_per_bin),
            rng=rng,
            replace=bool(args.replace_sampling),
        )

    exp_root = results_root / dataset_name / window_tag / ("h" + "_".join(str(h) for h in hidden_layers))
    exp_root.mkdir(parents=True, exist_ok=True)

    if args.verbose or args.dry_run:
        print("\n=== TIME-TRUST STABILITY EXPERIMENT ===")
        print("Dataset:", dataset_name)
        print("Window tag:", window_tag)
        print("Hidden layers:", hidden_layers)
        print("Modes:", modes)
        print("RUL bins:", bins)
        print("X_train:", tuple(X_train.shape), "X_val:", tuple(X_val.shape))
        print("Baseline:", baseline_art["artifact_path"])
        print("Analysis unit:", str(args.analysis_unit))
        print("Draw source:", str(args.draw_source))
        print("Selected batches/runs per mode before repeats:", len(selected_batches))
        print("Windows per bin/draw:", int(args.n_per_bin))
        print("Draws per bin:", int(args.draws_per_bin))
        print("Repeat solve:", int(args.repeat_solve))
        print("Results root:", exp_root)

    plan_path = exp_root / "experiment_plan.json"
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "window_tag": window_tag,
                "hidden_layers": hidden_layers,
                "modes": modes,
                "rul_bins": bins,
                "n_per_bin": int(args.n_per_bin),
                "analysis_unit": str(args.analysis_unit),
                "draw_source": str(args.draw_source),
                "draws_per_bin": int(args.draws_per_bin),
                "repeat_solve": int(args.repeat_solve),
                "mlp_mode": str(args.mlp_mode),
                "C": int(args.C),
                "beta": float(args.beta),
                "milp_time_cap": int(args.milp_time_cap),
                "baseline_mode": str(args.baseline_mode),
                "baseline_slack": float(args.baseline_slack),
                "seed": int(args.seed),
                "vary_seed": bool(args.vary_seed),
                "selected_batches": selected_batches,
            },
            f,
            indent=2,
        )

    if args.dry_run:
        print("\n[DRY RUN] Wrote plan to:", plan_path)
        return

    rows: List[Dict[str, Any]] = []
    run_counter = 0

    for mode in modes:
        for item in selected_batches:
            indices = np.asarray(item["indices"], dtype=int)
            if indices.size == 0:
                continue

            idx = int(item.get("idx", int(indices[0])))
            bin_name = str(item["bin"])
            draw = int(item.get("draw", 0))
            y_anchor = float(item.get("y_mean", item.get("y", float("nan"))))

            if draw_source == "train":
                X_train_subset = X_train[indices]
                y_train_subset = y_train[indices]

                # Validation windows from the same RUL bin are used only for
                # surrogate validation metrics; the TRUST centroids are built
                # from X_train_subset.
                val_candidates = np.where((y_val >= float(item["rul_min"])) & (y_val < float(item["rul_max"])))[0]
                if len(val_candidates) == 0:
                    X_val_subset = X_val
                    y_val_subset = y_val
                else:
                    val_n = min(len(indices), len(val_candidates))
                    val_idx = rng.choice(val_candidates, size=val_n, replace=False)
                    X_val_subset = X_val[val_idx]
                    y_val_subset = y_val[val_idx]
            else:
                # Backward-compatible mode: keep TRUST training fixed and vary
                # only validation/evaluation windows.
                X_train_subset = X_train
                y_train_subset = y_train
                X_val_subset = X_val[indices]
                y_val_subset = y_val[indices]

            for rep in range(int(args.repeat_solve)):
                run_counter += 1
                run_seed = int(args.seed) + run_counter if args.vary_seed else int(args.seed)

                run_dir = (
                    exp_root
                    / str(mode)
                    / bin_name
                    / f"draw_{draw:03d}_n_{len(indices):03d}_rulmean_{y_anchor:.3f}_rep_{rep:03d}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)

                if args.verbose:
                    print(
                        f"\n[RUN {run_counter}] mode={mode} bin={bin_name} "
                        f"draw={draw} n={len(indices)} first_idx={idx} y_mean={y_anchor:.3f} rep={rep} seed={run_seed}"
                    )

                row: Dict[str, Any] = {
                    "dataset": dataset_name,
                    "window_tag": window_tag,
                    "hidden": " ".join(str(h) for h in hidden_layers),
                    "mode": str(mode),
                    "bin": bin_name,
                    "rul_min": item["rul_min"],
                    "rul_max": item["rul_max"],
                    "idx": idx,
                    "y_val": y_anchor,
                    "batch_size": int(len(indices)),
                    "draw_source": draw_source,
                    "draw": draw,
                    "batch_indices_json": _safe_json_dumps([int(i) for i in indices.tolist()]),
                    "batch_y_json": _safe_json_dumps([float(v) for v in y_val_subset.tolist()]),
                    "y_min": float(np.min(y_val_subset)),
                    "y_max": float(np.max(y_val_subset)),
                    "repeat": rep,
                    "seed": run_seed,
                    "status": "ok",
                    "runtime_seconds": "",
                    "out_dir": "",
                    "selected_groups_json": "",
                    "selection_source": "",
                    "error": "",
                }

                start = time.perf_counter()
                try:
                    result = execute_trust_algorithm(
                        dataset_name=dataset_name,
                        window_tag=window_tag,
                        results_root=run_dir,
                        hidden_layers=tuple(hidden_layers),
                        mode=str(mode),
                        C=int(args.C),
                        beta=float(args.beta),
                        mlp_mode=str(args.mlp_mode),
                        milp_time_cap=int(args.milp_time_cap),
                        learning_rate=float(args.lr),
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        baseline_mode=str(args.baseline_mode),
                        baseline_slack=float(args.baseline_slack),
                        X_train=X_train_subset,
                        y_train=y_train_subset,
                        X_val=X_val_subset,
                        y_val=y_val_subset,
                        baseline_hidden_Ws=baseline_hidden_Ws,
                        baseline_hidden_bs=baseline_hidden_bs,
                        baseline_metrics=baseline_metrics,
                        baseline_weights_list=baseline_weights_list,
                        seed=run_seed,
                        verbose=bool(args.verbose),
                        resume=bool(args.resume),
                    )

                    elapsed = time.perf_counter() - start
                    row["runtime_seconds"] = elapsed

                    out_dir_raw = result.get("out_dir", None) if isinstance(result, dict) else None
                    out_dir = Path(out_dir_raw) if out_dir_raw is not None else run_dir
                    row["out_dir"] = str(out_dir)

                    selected_groups, source = (None, None)
                    if isinstance(result, dict):
                        selected_groups, source = _extract_selected_groups_from_result(result, str(mode))

                    if selected_groups is None:
                        selected_groups, source = _extract_selected_groups_from_out_dir(out_dir, str(mode))

                    if selected_groups is not None:
                        row["selected_groups_json"] = _safe_json_dumps(selected_groups)
                        row["selection_source"] = str(source)

                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    row["status"] = "failed"
                    row["runtime_seconds"] = elapsed
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    (run_dir / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    if args.verbose:
                        print("[ERROR]", row["error"])

                rows.append(row)

                _write_csv(
                    exp_root / "runs.csv",
                    rows,
                    fieldnames=[
                        "dataset",
                        "window_tag",
                        "hidden",
                        "mode",
                        "bin",
                        "rul_min",
                        "rul_max",
                        "idx",
                        "y_val",
                        "batch_size",
                        "draw_source",
                        "draw",
                        "batch_indices_json",
                        "batch_y_json",
                        "y_min",
                        "y_max",
                        "repeat",
                        "seed",
                        "status",
                        "runtime_seconds",
                        "out_dir",
                        "selected_groups_json",
                        "selection_source",
                        "error",
                    ],
                )
                with (exp_root / "runs.jsonl").open("w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(_safe_json_dumps(r) + "\n")

    stability_rows, frequency_rows = _build_summaries(rows)

    _write_csv(
        exp_root / "stability_summary.csv",
        stability_rows,
        fieldnames=[
            "dataset",
            "mode",
            "bin",
            "n_runs",
            "n_ok",
            "n_with_selection",
            "runtime_mean_seconds",
            "runtime_median_seconds",
            "runtime_std_seconds",
            "runtime_cv",
            "runtime_p95_seconds",
            "mean_pairwise_jaccard",
        ],
    )

    _write_csv(
        exp_root / "selection_frequency.csv",
        frequency_rows,
        fieldnames=[
            "dataset",
            "mode",
            "bin",
            "group",
            "count",
            "n_with_selection",
            "frequency",
        ],
    )

    print("\n[OK] Stability experiment finished.")
    print("Runs:", exp_root / "runs.csv")
    print("Runtime/Jaccard summary:", exp_root / "stability_summary.csv")
    print("Selection frequencies:", exp_root / "selection_frequency.csv")
    print("Plan:", plan_path)

    if all(not r.get("selected_groups_json") for r in rows if r.get("status") == "ok"):
        print(
            "\n[WARN] No selected groups were automatically extracted. "
            "The runtime summary is still valid, but group-selection stability needs "
            "a project-specific extractor. Search the run output files for the key "
            "that stores selected sensors/windows and add it to direct_keys in "
            "_extract_selected_groups_from_result()."
        )


if __name__ == "__main__":
    main()
