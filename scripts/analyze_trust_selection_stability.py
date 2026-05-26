#!/usr/bin/env python3
"""
Post-hoc selection-stability analysis for Time-TRUST runs.

This script reads the per-iteration JSON checkpoints already saved by
trust/pipeline.py and computes two complementary stability views:

1) Per-sparsity stability:
   For each n0_value, compare selected sensors/windows across draws using
   pairwise Jaccard. This answers: when 14, 13, 12, ..., groups remain, how
   similar are the selections across draws?

2) Survival stability:
   For each run, compute how long each sensor/window survives during iterative
   elimination. Then compare survival profiles across draws. This answers:
   do the same groups tend to remain important for many elimination steps?

Example:
    python scripts/analyze_trust_selection_stability.py \
        --root results/time_trust_stability_lowrul \
        --out results/time_trust_stability_lowrul/selection_stability_posthoc
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _as_list(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, str):
        # Try JSON first, then Python literal, then a single label.
        try:
            parsed = json.loads(obj)
            return _as_list(parsed)
        except Exception:
            try:
                parsed = ast.literal_eval(obj)
                return _as_list(parsed)
            except Exception:
                return [obj]
    if isinstance(obj, (list, tuple)):
        return sorted(str(int(x)) if isinstance(x, (int, np.integer, float, np.floating)) and float(x).is_integer() else str(x) for x in obj)
    if isinstance(obj, (int, np.integer)):
        return [str(int(obj))]
    return [str(obj)]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return None
    return float(np.dot(a, b) / denom)


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2:
        return None
    if float(np.std(a)) <= 0 or float(np.std(b)) <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Small dependency-free average-rank implementation."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2:
        return None
    return _pearson(_rankdata_average(a), _rankdata_average(b))


def _mean_ignore_none(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return None
    return float(np.mean(xs))


def _pairwise_scores(items: Sequence[Any], fn) -> List[float]:
    scores: List[float] = []
    for a, b in itertools.combinations(items, 2):
        score = fn(a, b)
        if score is not None and math.isfinite(float(score)):
            scores.append(float(score))
    return scores


def _find_runs_csv(root: Path) -> List[Path]:
    if root.is_file() and root.name == "runs.csv":
        return [root]
    return sorted(root.rglob("runs.csv"))


def _iter_json_files(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("iter_*.json"))


def _extract_iteration_records(run_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    mode = str(run_row.get("mode", ""))
    out_dir_raw = str(run_row.get("out_dir", ""))
    out_dir = Path(out_dir_raw)
    if not out_dir.exists():
        return []

    if "sensor" in mode:
        selected_key = "selected_sensors"
    elif "window" in mode:
        selected_key = "selected_windows"
    else:
        selected_key = "selected_original"

    records: List[Dict[str, Any]] = []
    for path in _iter_json_files(out_dir):
        meta = _safe_json_load(path)
        if not meta:
            continue

        if bool(meta.get("is_baseline", False)):
            continue
        if bool(meta.get("is_final_mlp", False)):
            continue

        n0 = meta.get("n0_value", None)
        if n0 is None:
            continue
        try:
            n0_int = int(n0)
        except Exception:
            continue
        if n0_int <= 0:
            continue

        selected = _as_list(meta.get(selected_key, []))
        if not selected:
            continue

        records.append(
            {
                "dataset": run_row.get("dataset", ""),
                "window_tag": run_row.get("window_tag", ""),
                "hidden": run_row.get("hidden", ""),
                "mode": mode,
                "bin": run_row.get("bin", ""),
                "draw_source": run_row.get("draw_source", ""),
                "draw": run_row.get("draw", ""),
                "repeat": run_row.get("repeat", ""),
                "run_status": run_row.get("status", ""),
                "out_dir": out_dir_raw,
                "iter_file": str(path),
                "iter_idx": int(meta.get("iter_idx", -1)),
                "n0_value": n0_int,
                "selected_groups_json": json.dumps(selected),
                "selected_count": len(selected),
                "selected_key": selected_key,
            }
        )

    return records


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _per_n0_stability(iter_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    keys = ["dataset", "window_tag", "hidden", "mode", "bin", "draw_source", "n0_value"]

    for key, g in iter_df.groupby(keys, dropna=False):
        selections = [_as_list(x) for x in g["selected_groups_json"].tolist()]
        scores = _pairwise_scores(selections, _jaccard)
        rows.append(
            {
                "dataset": key[0],
                "window_tag": key[1],
                "hidden": key[2],
                "mode": key[3],
                "bin": key[4],
                "draw_source": key[5],
                "n0_value": key[6],
                "n_runs": int(len(selections)),
                "mean_selected_count": float(np.mean([len(s) for s in selections])) if selections else None,
                "mean_pairwise_jaccard": float(np.mean(scores)) if scores else None,
                "min_pairwise_jaccard": float(np.min(scores)) if scores else None,
                "max_pairwise_jaccard": float(np.max(scores)) if scores else None,
            }
        )

    return pd.DataFrame(rows).sort_values(keys) if rows else pd.DataFrame()


def _survival_tables(iter_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        survival_long: one row per run/group.
        survival_similarity: one row per dataset/mode/bin with pairwise profile similarity.
        survival_frequency: one row per group aggregated across runs.
    """
    run_keys = ["dataset", "window_tag", "hidden", "mode", "bin", "draw_source", "draw", "repeat", "out_dir"]
    group_keys = ["dataset", "window_tag", "hidden", "mode", "bin", "draw_source"]

    survival_rows: List[Dict[str, Any]] = []

    for run_key, g in iter_df.groupby(run_keys, dropna=False):
        selected_by_n0: Dict[int, List[str]] = {}
        universe: set[str] = set()
        n0_values: List[int] = []

        for _, row in g.iterrows():
            n0 = int(row["n0_value"])
            selected = _as_list(row["selected_groups_json"])
            selected_by_n0[n0] = selected
            universe.update(selected)
            n0_values.append(n0)

        if not n0_values:
            continue

        n_steps = len(set(n0_values))
        max_n0 = max(n0_values)

        for group in sorted(universe, key=lambda x: int(x) if str(x).isdigit() else str(x)):
            present_n0 = sorted([n0 for n0, selected in selected_by_n0.items() if group in set(selected)], reverse=True)
            survival_steps = len(present_n0)
            deepest_n0 = min(present_n0) if present_n0 else None
            first_n0 = max(present_n0) if present_n0 else None

            survival_rows.append(
                {
                    "dataset": run_key[0],
                    "window_tag": run_key[1],
                    "hidden": run_key[2],
                    "mode": run_key[3],
                    "bin": run_key[4],
                    "draw_source": run_key[5],
                    "draw": run_key[6],
                    "repeat": run_key[7],
                    "out_dir": run_key[8],
                    "group": group,
                    "survival_steps": int(survival_steps),
                    "survival_fraction": float(survival_steps / n_steps) if n_steps else None,
                    "first_n0_present": first_n0,
                    "deepest_n0_present": deepest_n0,
                    "max_n0_in_run": int(max_n0),
                    "n_steps_in_run": int(n_steps),
                }
            )

    survival_long = pd.DataFrame(survival_rows)
    if survival_long.empty:
        return survival_long, pd.DataFrame(), pd.DataFrame()

    # Frequency / average survival per group across draws.
    freq_rows: List[Dict[str, Any]] = []
    for key, g in survival_long.groupby(group_keys + ["group"], dropna=False):
        freq_rows.append(
            {
                "dataset": key[0],
                "window_tag": key[1],
                "hidden": key[2],
                "mode": key[3],
                "bin": key[4],
                "draw_source": key[5],
                "group": key[6],
                "n_runs": int(len(g)),
                "mean_survival_steps": float(g["survival_steps"].mean()),
                "mean_survival_fraction": float(g["survival_fraction"].mean()),
                "mean_deepest_n0_present": float(g["deepest_n0_present"].mean()),
            }
        )
    survival_frequency = pd.DataFrame(freq_rows)

    # Similarity between whole survival profiles across draws.
    similarity_rows: List[Dict[str, Any]] = []
    for key, g in survival_long.groupby(group_keys, dropna=False):
        universe = sorted(g["group"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        run_profiles: List[np.ndarray] = []
        run_ids: List[str] = []

        for out_dir, rg in g.groupby("out_dir", dropna=False):
            values = {str(row["group"]): float(row["survival_fraction"]) for _, row in rg.iterrows()}
            vec = np.asarray([values.get(group, 0.0) for group in universe], dtype=float)
            run_profiles.append(vec)
            run_ids.append(str(out_dir))

        jac_scores = []
        cos_scores = []
        pear_scores = []
        spear_scores = []

        # Also compute Jaccard on the deepest survivors: groups with deepest_n0_present == 1 if available.
        deepest_sets: List[List[str]] = []
        for out_dir, rg in g.groupby("out_dir", dropna=False):
            if "deepest_n0_present" in rg:
                min_n0 = rg["deepest_n0_present"].min()
                deepest = rg.loc[rg["deepest_n0_present"] == min_n0, "group"].astype(str).tolist()
                deepest_sets.append(deepest)

        jac_scores = _pairwise_scores(deepest_sets, _jaccard)

        for a, b in itertools.combinations(run_profiles, 2):
            c = _cosine(a, b)
            p = _pearson(a, b)
            s = _spearman(a, b)
            if c is not None:
                cos_scores.append(c)
            if p is not None:
                pear_scores.append(p)
            if s is not None:
                spear_scores.append(s)

        similarity_rows.append(
            {
                "dataset": key[0],
                "window_tag": key[1],
                "hidden": key[2],
                "mode": key[3],
                "bin": key[4],
                "draw_source": key[5],
                "n_runs": int(len(run_profiles)),
                "n_groups": int(len(universe)),
                "mean_survival_cosine": float(np.mean(cos_scores)) if cos_scores else None,
                "mean_survival_pearson": float(np.mean(pear_scores)) if pear_scores else None,
                "mean_survival_spearman": float(np.mean(spear_scores)) if spear_scores else None,
                "mean_deepest_set_jaccard": float(np.mean(jac_scores)) if jac_scores else None,
            }
        )

    survival_similarity = pd.DataFrame(similarity_rows)
    return survival_long, survival_similarity, survival_frequency


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc Time-TRUST selection stability analysis.")
    parser.add_argument("--root", type=str, required=True, help="Root containing one or more runs.csv files.")
    parser.add_argument("--out", type=str, default=None, help="Output folder. Defaults to <root>/selection_stability_posthoc.")
    parser.add_argument("--runs-csv", type=str, default=None, help="Optional explicit runs.csv path.")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "selection_stability_posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_csvs = [Path(args.runs_csv)] if args.runs_csv else _find_runs_csv(root)
    if not runs_csvs:
        raise FileNotFoundError(f"No runs.csv files found under {root}")

    all_iter_records: List[Dict[str, Any]] = []
    for runs_csv in runs_csvs:
        runs = pd.read_csv(runs_csv)
        for _, row in runs.iterrows():
            if str(row.get("status", "")) != "ok":
                continue
            records = _extract_iteration_records(row.to_dict())
            all_iter_records.extend(records)

    if not all_iter_records:
        raise RuntimeError("No iteration records extracted. Check that out_dir paths in runs.csv still exist.")

    iter_df = pd.DataFrame(all_iter_records)
    iter_df.to_csv(out_dir / "selection_iterations_long.csv", index=False)

    per_n0 = _per_n0_stability(iter_df)
    per_n0.to_csv(out_dir / "selection_stability_by_n0.csv", index=False)

    survival_long, survival_similarity, survival_frequency = _survival_tables(iter_df)
    survival_long.to_csv(out_dir / "group_survival_long.csv", index=False)
    survival_similarity.to_csv(out_dir / "group_survival_similarity.csv", index=False)
    survival_frequency.to_csv(out_dir / "group_survival_frequency.csv", index=False)

    print("[OK] Post-hoc selection stability analysis finished.")
    print("Output folder:", out_dir)
    print("-", out_dir / "selection_iterations_long.csv")
    print("-", out_dir / "selection_stability_by_n0.csv")
    print("-", out_dir / "group_survival_long.csv")
    print("-", out_dir / "group_survival_similarity.csv")
    print("-", out_dir / "group_survival_frequency.csv")

    print("\nPreview: stability by n0")
    print(per_n0.head(20).to_string(index=False))

    print("\nPreview: survival similarity")
    print(survival_similarity.head(20).to_string(index=False))


if __name__ == "__main__":
    main()