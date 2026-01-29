from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_get(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _read_meta_from_npz(npz_path: Path) -> Dict[str, Any]:
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)

    if "meta_json" not in data:
        raise ValueError(f"Missing meta_json in {npz_path}")

    meta_raw = data["meta_json"]

    # meta_json was stored as np.array(json.dumps(meta), dtype=object)
    # It might come back as 0-d array or 1-element array.
    if hasattr(meta_raw, "shape") and meta_raw.shape == ():
        meta_str = meta_raw.item()
    else:
        meta_str = meta_raw.tolist()
        if isinstance(meta_str, list) and len(meta_str) == 1:
            meta_str = meta_str[0]

    if not isinstance(meta_str, str):
        meta_str = str(meta_str)

    return json.loads(meta_str)


def collect_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for npz_path in root.rglob("baseline_mlp_artifacts.npz"):
        try:
            meta = _read_meta_from_npz(npz_path)
            fm = meta.get("final_metrics", {}) or {}

            row = {
                "dataset": meta.get("dataset_name", npz_path.parts[-4] if len(npz_path.parts) >= 4 else ""),
                "window_tag": meta.get("window_tag", npz_path.parts[-3] if len(npz_path.parts) >= 3 else ""),
                "hidden_tag": meta.get("hidden_sizes_tag", npz_path.parts[-2] if len(npz_path.parts) >= 2 else ""),
                "input_dim": meta.get("input_dim", ""),
                # Main metrics for IEEE CEC table
                "train_mae": _safe_get(fm, "train_mae"),
                "val_mae": _safe_get(fm, "val_mae"),
                "train_rmse": _safe_get(fm, "train_rmse"),
                "val_rmse": _safe_get(fm, "val_rmse"),
                "train_r2": _safe_get(fm, "train_r2"),
                "val_r2": _safe_get(fm, "val_r2"),
                # Optional extras (handy to keep in the CSV)
                "train_mse": _safe_get(fm, "train_mse"),
                "val_mse": _safe_get(fm, "val_mse"),
                "path": str(npz_path.parent),
            }
            rows.append(row)
        except Exception as e:
            # Skip broken runs but keep a hint for debugging
            rows.append(
                {
                    "dataset": "",
                    "window_tag": "",
                    "hidden_tag": "",
                    "input_dim": "",
                    "train_mae": "",
                    "val_mae": "",
                    "train_rmse": "",
                    "val_rmse": "",
                    "train_r2": "",
                    "val_r2": "",
                    "train_mse": "",
                    "val_mse": "",
                    "path": str(npz_path.parent),
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    return rows


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Canonical column order for the paper table
    base_cols = [
        "dataset",
        "window_tag",
        "hidden_tag",
        "train_mae",
        "val_mae",
        "train_rmse",
        "val_rmse",
        "train_r2",
        "val_r2",
        "input_dim",
        "train_mse",
        "val_mse",
        "path",
    ]

    # Include "error" only if present in any row
    has_error = any("error" in r for r in rows)
    cols = base_cols + (["error"] if has_error else [])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> None:
    p = argparse.ArgumentParser(description="Export baseline MLP metrics to CSV from mlp_baselines folder.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/mlp_baselines"),
        help="Root directory that contains dataset/window_tag/hidden_tag folders.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("mlp_baselines_metrics.csv"),
        help="Output CSV path.",
    )
    p.add_argument(
        "--sort",
        action="store_true",
        help="Sort rows by dataset, window_tag, hidden_tag (recommended).",
    )

    args = p.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    rows = collect_rows(root)

    if args.sort:
        rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("window_tag", "")), str(r.get("hidden_tag", ""))))

    write_csv(rows, args.out.resolve())
    print(f"Wrote {len(rows)} rows to: {args.out.resolve()}")


if __name__ == "__main__":
    main()