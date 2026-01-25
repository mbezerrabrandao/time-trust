# scripts/timetrust_rankings.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np


GroupMode = Literal["sensors", "windows"]


# -------------------------
# IO helpers
# -------------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    _safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_int_matrix(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=int)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return arr


# -------------------------
# Results traversal (IGNORE full_)
# -------------------------

def _is_mode_dir(name: str) -> bool:
    # Ignore full_* on purpose (TRUST only)
    return name.startswith("sensors_") or name.startswith("windows_")


def _infer_group_mode(mode_dir_name: str) -> GroupMode:
    if mode_dir_name.startswith("sensors_"):
        return "sensors"
    if mode_dir_name.startswith("windows_"):
        return "windows"
    raise ValueError(f"Unsupported mode_dir (full_* is intentionally ignored): {mode_dir_name}")


def _default_group_names(group_mode: GroupMode, n: int) -> List[str]:
    if group_mode == "sensors":
        return [f"s{i+1}" for i in range(n)]
    return [f"w{i+1}" for i in range(n)]


def _find_experiment_dirs(results_root: Path) -> List[Path]:
    """
    Expected layout:
      results/<dataset>/<window_tag>/<h_tag>/<mode_dir>/vectors_selection_importance.json

    mode_dir must start with sensors_ or windows_ (full_* is ignored).
    """
    exp_dirs: List[Path] = []
    if not results_root.exists():
        raise FileNotFoundError(f"results_root does not exist: {results_root}")

    for ds_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        for win_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()]):
            for h_dir in sorted([p for p in win_dir.iterdir() if p.is_dir()]):
                for mode_dir in sorted([p for p in h_dir.iterdir() if p.is_dir()]):
                    if not _is_mode_dir(mode_dir.name):
                        continue
                    vpath = mode_dir / "vectors_selection_importance.json"
                    if vpath.exists():
                        exp_dirs.append(mode_dir)

    return exp_dirs


# -------------------------
# Ranking logic (selection-based)
# -------------------------

def _first_off_iterpos(sel_mat: np.ndarray) -> np.ndarray:
    """
    sel_mat: (T, G) values 0/1.
    Returns off_iterpos[g] = first t where sel[t,g]==0, else T (survived all iters).
    """
    sel = (sel_mat > 0).astype(int)
    T, G = sel.shape
    off = np.full((G,), T, dtype=int)

    for g in range(G):
        idx = np.where(sel[:, g] == 0)[0]
        if idx.size > 0:
            off[g] = int(idx[0])
    return off


def _ranking_from_survival(off_iterpos: np.ndarray) -> np.ndarray:
    """
    Rank by survival descending (later off is better).
    Tie-breaker: stable by group index.
    """
    score = off_iterpos.astype(int)
    idx = np.arange(score.shape[0], dtype=int)
    return np.lexsort((idx, -score))  # (-score, +idx)


def build_timetrust_ranking_from_selection_json(
    *,
    vselimp_path: Path,
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
    sel_mat = _coerce_int_matrix(data["selection_vector"], "selection_vector")

    T, G = sel_mat.shape
    if iter_idx.shape[0] != T or n0_value.shape[0] != T:
        raise ValueError(
            f"iter_idx/n0_value length mismatch with selection_vector: "
            f"len(iter_idx)={iter_idx.shape[0]}, len(n0_value)={n0_value.shape[0]}, T={T}"
        )

    off_iterpos = _first_off_iterpos(sel_mat)  # (G,)
    survived_all = (off_iterpos == T).astype(int)
    ranking = _ranking_from_survival(off_iterpos)

    if group_names is None:
        group_names = _default_group_names(group_mode, G)
    group_names = list(group_names)

    items: List[Dict[str, Any]] = []
    for rank_pos, g in enumerate(ranking.tolist(), start=1):
        t_off = int(off_iterpos[g])
        turned_off = int(t_off < T)

        items.append(
            {
                "rank": int(rank_pos),
                "group_index_0based": int(g),
                "group_index_1based": int(g + 1),
                "group_name": str(group_names[g]) if g < len(group_names) else str(g + 1),
                "turned_off": int(turned_off),
                "off_iterpos": int(t_off if turned_off else -1),
                "off_iter_idx": int(iter_idx[t_off]) if turned_off else -1,
                "survived_all_iters": int(survived_all[g]),
                "survival_score": int(off_iterpos[g]),  # larger = survived longer
            }
        )

    payload: Dict[str, Any] = {
        "type": "timetrust_group_ranking_selection_based",
        "group_mode": group_mode,
        "n_groups": int(G),
        "n_iters": int(T),
        "definition": {
            "ranking_rule": (
                "Rank groups by when they are first turned off in selection_vector. "
                "First turned off gets last rank, last surviving gets rank 1."
            ),
            "off_iterpos": "first t where selection[t,g] == 0, or T if never turned off",
            "tie_break": "stable by group index",
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
# Batch build: read results/, write into mlp_baselines/.../rankings
# -------------------------

def build_rankings_for_results_tree(
    *,
    results_root: Path,
    baselines_root: Path,
    verbose: bool,
) -> Dict[str, Any]:
    exp_dirs = _find_experiment_dirs(results_root)
    saved: List[str] = []
    errors: List[Dict[str, str]] = []

    for mode_dir in exp_dirs:
        try:
            # results/<ds>/<window>/<h>/<mode_dir>/
            h_dir = mode_dir.parent
            win_dir = h_dir.parent
            ds_dir = win_dir.parent

            dataset_name = ds_dir.name
            window_tag = win_dir.name
            h_tag = h_dir.name
            mode_name = mode_dir.name

            group_mode = _infer_group_mode(mode_name)

            vselimp_path = mode_dir / "vectors_selection_importance.json"

            # Output: mlp_baselines/<ds>/<window>/<h>/rankings/
            out_dir = baselines_root / dataset_name / window_tag / h_tag / "rankings"
            _safe_mkdir(out_dir)

            out_path = out_dir / f"ranking_{group_mode}_timetrust_selection.json"

            extra_meta = {
                "dataset_name": dataset_name,
                "window_tag": window_tag,
                "hidden_tag": h_tag,
                "mode_dir": mode_name,
                "source_json": str(vselimp_path),
                "results_root": str(results_root),
                "baselines_root": str(baselines_root),
            }

            build_timetrust_ranking_from_selection_json(
                vselimp_path=vselimp_path,
                group_mode=group_mode,
                out_json=out_path,
                extra_meta=extra_meta,
            )

            saved.append(str(out_path))

            if verbose:
                print(f"[OK] {dataset_name}/{window_tag}/{h_tag}/{mode_name} -> {out_path}")

        except Exception as e:
            errors.append({"mode_dir": str(mode_dir), "error": str(e)})
            if verbose:
                print(f"[ERR] {mode_dir}: {e}")

    return {"saved": saved, "n_saved": len(saved), "errors": errors, "n_errors": len(errors)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Time-TRUST selection-based rankings (ignore full_*) and save into mlp_baselines/.../rankings."
    )
    parser.add_argument("--results-root", type=str, default="results", help="Root folder with Time-TRUST runs.")
    parser.add_argument("--baselines-root", type=str, default="mlp_baselines", help="Baseline MLP root folder.")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")
    args = parser.parse_args()

    report = build_rankings_for_results_tree(
        results_root=Path(args.results_root),
        baselines_root=Path(args.baselines_root),
        verbose=bool(args.verbose),
    )

    if args.verbose:
        print("\n=== SUMMARY ===")
        print("Saved:", report["n_saved"])
        if report["n_errors"] > 0:
            print("Errors:", report["n_errors"])
            for er in report["errors"][:10]:
                print(" -", er["mode_dir"], "->", er["error"])


if __name__ == "__main__":
    main()