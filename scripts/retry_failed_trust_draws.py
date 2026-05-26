#!/usr/bin/env python3
"""
Retry failed Time-TRUST stability draws from a runs.csv file.

Use case:
    A long stability sweep failed for a few rows because GAMSPy/GAMS could not
    start a network/license session, but successful rows are already saved.
    This script reads the failed rows, reconstructs the exact train mini-batch
    from batch_indices_json, reruns only those draws, and updates runs.csv.

Example:
    python scripts/retry_failed_trust_draws.py \
        --runs-csv results/time_trust_stability_lowrul/FD001/W30_step1/h10_10_10/runs.csv \
        --max-retries 3 \
        --clean-run-dir \
        --update-runs-csv \
        --verbose

Then rerun the post-hoc analyzer:
    python scripts/analyze_trust_selection_stability.py \
        --root results/time_trust_stability_lowrul \
        --out results/time_trust_stability_lowrul/selection_stability_posthoc
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Project path bootstrap
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
# Shared helpers
# =========================

def _window_tag(seq_len: int, step: int) -> str:
    return f"W{int(seq_len)}_step{int(step)}"


def _hidden_tag(hidden_layers: Sequence[int]) -> str:
    return "h" + "_".join(str(int(x)) for x in hidden_layers)


def _parse_hidden(hidden_raw: Any) -> List[int]:
    if isinstance(hidden_raw, (list, tuple)):
        return [int(x) for x in hidden_raw]
    text = str(hidden_raw).replace(",", " ").strip()
    return [int(x) for x in text.split() if x.strip()]


def _parse_json_list(raw: Any) -> List[int]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    return [int(x) for x in json.loads(text)]


def _load_processed_dataset(processed_root: Path, dataset_name: str, window_tag: str) -> Dict[str, np.ndarray]:
    npz_path = processed_root / dataset_name / window_tag / "dataset.npz"
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
    art_path = baselines_root / dataset_name / window_tag / _hidden_tag(hidden_layers) / "baseline_mlp_artifacts.npz"
    if not art_path.exists():
        raise FileNotFoundError(f"Baseline artifact not found: {art_path}")

    data = np.load(art_path, allow_pickle=True)
    if "meta_json" not in data.files:
        raise KeyError(f"baseline_mlp_artifacts.npz missing meta_json: {art_path}")
    meta = json.loads(str(data["meta_json"].item()))

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

    meta.setdefault("final_metrics", {})
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
        layer = f"hidden_l{i}"
        hidden_Ws.append(np.asarray(weights[layer]["W"], dtype=float))
        hidden_bs.append(np.asarray(weights[layer]["b"], dtype=float).reshape(-1))

    hidden_Ws.append(np.asarray(weights["rul_output"]["W"], dtype=float).reshape(-1))
    hidden_bs.append(np.asarray(weights["rul_output"]["b"], dtype=float).reshape(-1))
    return hidden_Ws, hidden_bs


def _weights_dict_to_weights_list(
    weights: Dict[str, Dict[str, np.ndarray]],
    hidden_layers: Sequence[int],
) -> List[np.ndarray]:
    weights_list: List[np.ndarray] = []
    for i in range(1, len(hidden_layers) + 1):
        layer = f"hidden_l{i}"
        weights_list.append(np.asarray(weights[layer]["W"]))
        weights_list.append(np.asarray(weights[layer]["b"]))
    weights_list.append(np.asarray(weights["rul_output"]["W"]))
    weights_list.append(np.asarray(weights["rul_output"]["b"]))
    return weights_list


def _as_group_set(obj: Any) -> Optional[List[str]]:
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, dict):
        for key in ["selected", "selected_groups", "selected_indices", "indices", "groups", "mask"]:
            if key in obj:
                groups = _as_group_set(obj[key])
                if groups is not None:
                    return groups
        if all(isinstance(v, (bool, np.bool_)) for v in obj.values()):
            return sorted(str(k) for k, v in obj.items() if bool(v))
        return None
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return []
        if all(isinstance(v, (bool, np.bool_)) for v in obj):
            return [str(i) for i, v in enumerate(obj) if bool(v)]
        if all(isinstance(v, (int, np.integer, float, np.floating, str)) for v in obj):
            labels = []
            for v in obj:
                if isinstance(v, (int, np.integer)):
                    labels.append(str(int(v)))
                elif isinstance(v, (float, np.floating)) and float(v).is_integer():
                    labels.append(str(int(v)))
                else:
                    labels.append(str(v))
            return sorted(labels, key=lambda x: int(x) if str(x).isdigit() else str(x))
    return None


def _history_selected_key(mode: str) -> str:
    mode_l = mode.lower()
    if "sensor" in mode_l:
        return "selected_sensors"
    if "window" in mode_l:
        return "selected_windows"
    return "selected_groups"


def _extract_selected_groups_from_result(result: Dict[str, Any], mode: str) -> Tuple[Optional[List[str]], Optional[str]]:
    history = result.get("history", None)
    if isinstance(history, list):
        key = _history_selected_key(mode)
        for i in range(len(history) - 1, -1, -1):
            step = history[i]
            if isinstance(step, dict) and key in step:
                groups = _as_group_set(step[key])
                if groups is not None:
                    return groups, f"history.{i}.{key}"
    return None, None


def _safe_json_dumps(obj: Any) -> str:
    def conv(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x
    return json.dumps(conv(obj), sort_keys=True)


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _std(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) == 0:
        return None
    if len(vals) == 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def _median(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.percentile(vals, q)) if vals else None


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
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


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _recompute_summaries(runs_csv: Path) -> None:
    df = pd.read_csv(runs_csv)
    rows = df.to_dict(orient="records")

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("mode", "")), str(row.get("bin", "")))
        grouped.setdefault(key, []).append(row)

    stability_rows: List[Dict[str, Any]] = []
    frequency_rows: List[Dict[str, Any]] = []

    for (dataset, mode, bin_name), group_rows in sorted(grouped.items()):
        ok_rows = [r for r in group_rows if str(r.get("status", "")) == "ok"]
        runtimes = []
        selected_sets: List[List[str]] = []

        for r in ok_rows:
            try:
                runtimes.append(float(r.get("runtime_seconds", "")))
            except Exception:
                pass
            raw = r.get("selected_groups_json", "")
            if isinstance(raw, str) and raw.strip():
                try:
                    selected_sets.append([str(x) for x in json.loads(raw)])
                except Exception:
                    pass

        runtime_mean = _mean(runtimes)
        runtime_std = _std(runtimes)
        cv = runtime_std / runtime_mean if runtime_mean and runtime_mean > 0 and runtime_std is not None else None

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

    summary_fields = [
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
    ]
    freq_fields = ["dataset", "mode", "bin", "group", "count", "n_with_selection", "frequency"]
    _write_csv(runs_csv.parent / "stability_summary.csv", stability_rows, summary_fields)
    _write_csv(runs_csv.parent / "selection_frequency.csv", frequency_rows, freq_fields)


def _infer_combo_dir_from_runs_csv(runs_csv: Path) -> Path:
    # runs.csv is stored at <results_root>/<dataset>/<window_tag>/<hidden_tag>/runs.csv
    return runs_csv.parent


def _reconstruct_run_dir(combo_dir: Path, row: Dict[str, Any]) -> Path:
    mode = str(row["mode"])
    bin_name = str(row["bin"])
    draw = int(row.get("draw", 0))
    n = int(row.get("batch_size", 0))
    y_anchor = float(row.get("y_val", row.get("y_mean", 0.0)))
    rep = int(row.get("repeat", 0))
    return combo_dir / mode / bin_name / f"draw_{draw:03d}_n_{n:03d}_rulmean_{y_anchor:.3f}_rep_{rep:03d}"


# =========================
# Main retry logic
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Retry failed Time-TRUST stability draws from runs.csv.")
    parser.add_argument("--runs-csv", type=str, required=True, help="Path to a specific runs.csv file.")
    parser.add_argument("--processed-root", type=str, default="datasets/processed")
    parser.add_argument("--baselines-root", type=str, default="mlp_baselines")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)

    # TRUST config should match the original sweep.
    parser.add_argument("--mlp-mode", type=str, default=MLP_TRANSFER, choices=[MLP_REBUILD, MLP_TRANSFER])
    parser.add_argument("--C", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--milp-time-cap", type=int, default=60)
    parser.add_argument("--baseline-mode", type=str, default=BASELINE_MODE_CAPPED, choices=[BASELINE_MODE_NONE, BASELINE_MODE_CAPPED])
    parser.add_argument("--baseline-slack", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--only-error-contains", type=str, default="", help="Optional substring filter for failed rows.")
    parser.add_argument("--clean-run-dir", action="store_true", help="Delete the failed run directory before retrying.")
    parser.add_argument("--resume", action="store_true", help="Allow execute_trust_algorithm to resume partial checkpoints.")
    parser.add_argument("--update-runs-csv", action="store_true", help="Update runs.csv in place, with a .bak backup.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    runs_csv = Path(args.runs_csv)
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv}")

    df = pd.read_csv(runs_csv)
    if "status" not in df.columns:
        raise KeyError("runs.csv is missing 'status' column.")

    failed_mask = df["status"].astype(str) != "ok"
    if args.only_error_contains:
        failed_mask &= df.get("error", "").astype(str).str.contains(args.only_error_contains, regex=False, na=False)

    failed_indices = df.index[failed_mask].tolist()
    if not failed_indices:
        print("[OK] No failed rows to retry.")
        return

    combo_dir = _infer_combo_dir_from_runs_csv(runs_csv)

    print(f"[INFO] Found {len(failed_indices)} failed rows in {runs_csv}")
    if args.dry_run:
        print(df.loc[failed_indices, ["dataset", "hidden", "mode", "bin", "draw", "status", "error"]].to_string(index=True))
        return

    # Cache datasets/baselines by (dataset, window_tag, hidden_tag)
    cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    retry_log_rows: List[Dict[str, Any]] = []

    for row_idx in failed_indices:
        row = df.loc[row_idx].to_dict()
        dataset_name = str(row["dataset"])
        window_tag = str(row.get("window_tag", _window_tag(args.seq_len, args.step)))
        hidden_layers = _parse_hidden(row["hidden"])
        hidden_tag = _hidden_tag(hidden_layers)
        mode = str(row["mode"])
        draw = int(row.get("draw", 0))
        seed = int(row.get("seed", RANDOM_SEED))
        batch_indices = _parse_json_list(row.get("batch_indices_json", ""))
        if not batch_indices:
            print(f"[WARN] Row {row_idx}: no batch_indices_json; cannot retry.")
            continue

        cache_key = (dataset_name, window_tag, hidden_tag)
        if cache_key not in cache:
            arrays = _load_processed_dataset(Path(args.processed_root), dataset_name, window_tag)
            X_train = np.asarray(arrays["X_train_mw"], dtype=np.float32)
            y_train = np.asarray(arrays["y_train"], dtype=np.float32).reshape(-1)
            X_val = np.asarray(arrays["X_val_mw"], dtype=np.float32)
            y_val = np.asarray(arrays["y_val"], dtype=np.float32).reshape(-1)

            baseline_art = _load_baseline_artifacts(Path(args.baselines_root), dataset_name, window_tag, hidden_layers)
            baseline_hidden_Ws, baseline_hidden_bs = _baseline_weights_to_trust_format(baseline_art["weights"], hidden_layers)
            baseline_weights_list = _weights_dict_to_weights_list(baseline_art["weights"], hidden_layers)

            cache[cache_key] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "baseline_hidden_Ws": baseline_hidden_Ws,
                "baseline_hidden_bs": baseline_hidden_bs,
                "baseline_weights_list": baseline_weights_list,
                "baseline_metrics": baseline_art.get("final_metrics", None),
            }

        data = cache[cache_key]
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]

        batch_indices_np = np.asarray(batch_indices, dtype=int)
        draw_source = str(row.get("draw_source", "train"))
        if draw_source == "train":
            X_train_subset = X_train[batch_indices_np]
            y_train_subset = y_train[batch_indices_np]

            lo = float(row.get("rul_min", 0.0))
            hi = float(row.get("rul_max", 30.0))
            val_candidates = np.where((y_val >= lo) & (y_val < hi))[0]
            if len(val_candidates) == 0:
                X_val_subset = X_val
                y_val_subset = y_val
            else:
                val_n = min(len(batch_indices_np), len(val_candidates))
                rng = np.random.default_rng(seed + draw + 10007)
                val_idx = rng.choice(val_candidates, size=val_n, replace=False)
                X_val_subset = X_val[val_idx]
                y_val_subset = y_val[val_idx]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
            X_val_subset = X_val[batch_indices_np]
            y_val_subset = y_val[batch_indices_np]

        run_dir = _reconstruct_run_dir(combo_dir, row)

        if args.clean_run_dir and run_dir.exists():
            shutil.rmtree(run_dir)

        print(f"\n[RETRY] row={row_idx} dataset={dataset_name} hidden={hidden_tag} mode={mode} draw={draw}")
        print(f"        run_dir={run_dir}")

        success = False
        last_error = ""
        last_runtime = None
        last_out_dir = ""
        last_selected_groups = ""
        last_selection_source = ""

        for attempt in range(1, int(args.max_retries) + 1):
            start = time.perf_counter()
            try:
                result = execute_trust_algorithm(
                    dataset_name=dataset_name,
                    window_tag=window_tag,
                    results_root=run_dir,
                    hidden_layers=tuple(hidden_layers),
                    mode=mode,
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
                    baseline_hidden_Ws=data["baseline_hidden_Ws"],
                    baseline_hidden_bs=data["baseline_hidden_bs"],
                    baseline_metrics=data["baseline_metrics"],
                    baseline_weights_list=data["baseline_weights_list"],
                    seed=seed,
                    verbose=bool(args.verbose),
                    resume=bool(args.resume),
                )
                elapsed = time.perf_counter() - start
                last_runtime = elapsed
                last_out_dir = str(result.get("out_dir", "")) if isinstance(result, dict) else ""
                groups, source = _extract_selected_groups_from_result(result, mode) if isinstance(result, dict) else (None, None)
                if groups is not None:
                    last_selected_groups = _safe_json_dumps(groups)
                    last_selection_source = str(source)

                success = True
                print(f"[OK] row={row_idx} attempt={attempt} runtime={elapsed:.3f}s")
                break

            except Exception as exc:
                elapsed = time.perf_counter() - start
                last_runtime = elapsed
                last_error = f"{type(exc).__name__}: {exc}"
                err_path = run_dir / f"retry_attempt_{attempt:02d}_traceback.txt"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(traceback.format_exc(), encoding="utf-8")
                print(f"[FAILED] row={row_idx} attempt={attempt} runtime={elapsed:.3f}s error={last_error}")

        retry_log_rows.append(
            {
                "row_idx": row_idx,
                "dataset": dataset_name,
                "hidden": " ".join(str(x) for x in hidden_layers),
                "mode": mode,
                "bin": row.get("bin", ""),
                "draw": draw,
                "success": success,
                "runtime_seconds": last_runtime,
                "out_dir": last_out_dir,
                "selected_groups_json": last_selected_groups,
                "selection_source": last_selection_source,
                "error": "" if success else last_error,
            }
        )

        if success:
            df.at[row_idx, "status"] = "ok"
            df.at[row_idx, "runtime_seconds"] = last_runtime
            df.at[row_idx, "out_dir"] = last_out_dir
            df.at[row_idx, "selected_groups_json"] = last_selected_groups
            df.at[row_idx, "selection_source"] = last_selection_source
            df.at[row_idx, "error"] = ""
        else:
            df.at[row_idx, "runtime_seconds"] = last_runtime
            df.at[row_idx, "error"] = last_error

    retry_log = runs_csv.parent / "retry_failed_draws_log.csv"
    _write_csv(retry_log, retry_log_rows)
    print(f"\n[INFO] Retry log written to {retry_log}")

    if args.update_runs_csv:
        backup = runs_csv.with_suffix(".csv.bak")
        shutil.copy2(runs_csv, backup)
        df.to_csv(runs_csv, index=False)
        _recompute_summaries(runs_csv)
        print(f"[OK] Updated {runs_csv}")
        print(f"[OK] Backup saved as {backup}")
        print(f"[OK] Recomputed {runs_csv.parent / 'stability_summary.csv'}")
        print(f"[OK] Recomputed {runs_csv.parent / 'selection_frequency.csv'}")
    else:
        merged = runs_csv.parent / "runs_after_retry_preview.csv"
        df.to_csv(merged, index=False)
        print(f"[INFO] Preview merged CSV written to {merged}")
        print("[INFO] Use --update-runs-csv to overwrite runs.csv with a .bak backup.")


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 python scripts/retry_failed_trust_draws.py \
  --runs-csv results/time_trust_stability_lowrul/FD001/W30_step1/h10_10_10/runs.csv \
  --max-retries 3 \
  --clean-run-dir \
  --update-runs-csv \
  --verbose

CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 python scripts/retry_failed_trust_draws.py \
  --runs-csv results/time_trust_stability_lowrul/FD004/W30_step1/h10_10/runs.csv \
  --max-retries 3 \
  --clean-run-dir \
  --update-runs-csv \
  --verbose

python scripts/analyze_trust_selection_staexitbility.py \
  --root results/time_trust_stability_lowrul \
  --out results/time_trust_stability_lowrul/selection_stability_posthoc
"""