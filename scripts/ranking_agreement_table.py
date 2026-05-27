# scripts/ranking_agreement_table.py
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


GROUP_MODES = ("sensors", "windows")

# Known comparator filenames expected inside:
#   mlp_baselines/<dataset>/<window_tag>/<hidden_tag>/rankings/
KNOWN_COMPARATOR_SUFFIXES: Dict[str, str] = {
    "timetrust_selection": "timetrust_selection",
    "weights": "weights",
    "randomavg": "randomavg",
    # Future-friendly names. These are ignored unless the files exist.
    "group_ablation": "group_ablation",
    "ablation": "ablation",
    "permutation": "permutation",
}


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


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    _safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# -------------------------
# Directory traversal
# -------------------------

def _is_hidden_tag(name: str) -> bool:
    return name.startswith("h") and any(ch.isdigit() for ch in name)


def _find_rankings_dirs(baselines_root: Path) -> List[Path]:
    """
    Finds directories of the form:
      mlp_baselines/<dataset>/<window_tag>/<hidden_tag>/rankings
    """
    if not baselines_root.exists():
        raise FileNotFoundError(f"baselines_root does not exist: {baselines_root}")

    out: List[Path] = []
    for ds_dir in sorted([p for p in baselines_root.iterdir() if p.is_dir()]):
        for win_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()]):
            for h_dir in sorted([p for p in win_dir.iterdir() if p.is_dir() and _is_hidden_tag(p.name)]):
                rdir = h_dir / "rankings"
                if rdir.exists() and rdir.is_dir():
                    out.append(rdir)
    return out


def _parse_rankings_dir(rankings_dir: Path) -> Tuple[str, str, str]:
    """
    rankings_dir = mlp_baselines/<dataset>/<window_tag>/<hidden_tag>/rankings
    Returns: dataset, window_tag, hidden_tag
    """
    h_dir = rankings_dir.parent
    win_dir = h_dir.parent
    ds_dir = win_dir.parent
    return ds_dir.name, win_dir.name, h_dir.name


# -------------------------
# Ranking extraction
# -------------------------

def _extract_ranked_group_indices(ranking_json: Dict[str, Any]) -> List[int]:
    """
    Returns 0-based group indices, best first.

    Supports the formats already used in this project:
      1) New/common format:
         {"items": [{"group_index_0based": 3, ...}, ...]}

      2) Older/simple baseline format:
         {"ranking": [{"index": 4, ...}, ...]}     # index is 1-based
         {"ranking": [{"id": "s4", ...}, ...]}
    """
    if "items" in ranking_json and isinstance(ranking_json["items"], list):
        items = ranking_json["items"]
        out: List[int] = []
        for it in items:
            if "group_index_0based" in it:
                out.append(int(it["group_index_0based"]))
            elif "group_index_1based" in it:
                out.append(int(it["group_index_1based"]) - 1)
            elif "index" in it:
                out.append(int(it["index"]) - 1)
            elif "id" in it:
                out.append(_parse_index_from_id(str(it["id"])))
            else:
                raise KeyError("ranking item missing group_index_0based/group_index_1based/index/id")
        return out

    if "ranking" in ranking_json and isinstance(ranking_json["ranking"], list):
        items = ranking_json["ranking"]
        out = []
        for it in items:
            if "index" in it:
                out.append(int(it["index"]) - 1)
            elif "group_index_0based" in it:
                out.append(int(it["group_index_0based"]))
            elif "group_index_1based" in it:
                out.append(int(it["group_index_1based"]) - 1)
            elif "id" in it:
                out.append(_parse_index_from_id(str(it["id"])))
            else:
                raise KeyError("ranking item missing index/group_index/id")
        return out

    raise KeyError("Unrecognized ranking JSON format. Expected key 'items' or 'ranking'.")


def _parse_index_from_id(s: str) -> int:
    """Parses strings like 's12' or 'w5' into 0-based indices."""
    digits = "".join(c for c in str(s) if c.isdigit())
    if digits == "":
        raise ValueError(f"Cannot parse group index from id={s!r}")
    return int(digits) - 1


def _validate_ranking(best_first: Sequence[int], *, expected_n: Optional[int] = None, label: str = "ranking") -> List[int]:
    r = [int(x) for x in best_first]
    if len(r) == 0:
        raise ValueError(f"{label} is empty")

    if len(set(r)) != len(r):
        raise ValueError(f"{label} contains duplicate group indices: {r}")

    if min(r) < 0:
        raise ValueError(f"{label} contains negative group index: {r}")

    if expected_n is not None:
        expected_n = int(expected_n)
        if len(r) != expected_n:
            raise ValueError(f"{label} length mismatch: got {len(r)}, expected {expected_n}")
        expected = set(range(expected_n))
        got = set(r)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(f"{label} is not a permutation of 0..{expected_n-1}. missing={missing}, extra={extra}")

    return r


def _ranking_positions(best_first: Sequence[int]) -> np.ndarray:
    """
    Converts best-first group list into rank-position vector:
      pos[g] = 0 for most important group, 1 for second, etc.
    """
    r = [int(x) for x in best_first]
    n = len(r)
    pos = np.empty(n, dtype=float)
    for rank_pos, g in enumerate(r):
        pos[int(g)] = float(rank_pos)
    return pos


# -------------------------
# Metrics
# -------------------------

def _spearman_from_rankings(ref_best_first: Sequence[int], cmp_best_first: Sequence[int]) -> float:
    """
    Spearman correlation between rank positions.
    Identical rankings -> 1.0, reversed rankings -> close to -1.0.
    """
    a = _ranking_positions(ref_best_first)
    b = _ranking_positions(cmp_best_first)
    if a.size < 2:
        return float("nan")
    # With complete permutations, variance is non-zero when n >= 2.
    return float(np.corrcoef(a, b)[0, 1])


def _kendall_tau_from_rankings(ref_best_first: Sequence[int], cmp_best_first: Sequence[int]) -> float:
    """
    Kendall tau-a for complete rankings without ties.
    Identical rankings -> 1.0, reversed rankings -> -1.0.
    """
    a = _ranking_positions(ref_best_first)
    b = _ranking_positions(cmp_best_first)
    n = int(a.size)
    if n < 2:
        return float("nan")

    concordant = 0
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            # No ties expected. If present, ignore them.

    denom = n * (n - 1) / 2.0
    if denom <= 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def _overlap_at_k(ref_best_first: Sequence[int], cmp_best_first: Sequence[int], k: int) -> float:
    k = int(k)
    n = min(len(ref_best_first), len(cmp_best_first))
    if k <= 0 or k > n:
        return float("nan")
    a = set(int(x) for x in ref_best_first[:k])
    b = set(int(x) for x in cmp_best_first[:k])
    return float(len(a.intersection(b)) / float(k))


def _rank_mae(ref_best_first: Sequence[int], cmp_best_first: Sequence[int]) -> float:
    a = _ranking_positions(ref_best_first)
    b = _ranking_positions(cmp_best_first)
    return float(np.mean(np.abs(a - b)))


def _rank_rmse(ref_best_first: Sequence[int], cmp_best_first: Sequence[int]) -> float:
    a = _ranking_positions(ref_best_first)
    b = _ranking_positions(cmp_best_first)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _rank_mae_norm(ref_best_first: Sequence[int], cmp_best_first: Sequence[int]) -> float:
    n = len(ref_best_first)
    if n <= 1:
        return float("nan")
    return float(_rank_mae(ref_best_first, cmp_best_first) / float(n - 1))


def compute_ranking_agreement(
    *,
    ref_best_first: Sequence[int],
    cmp_best_first: Sequence[int],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    n = len(ref_best_first)
    ref = _validate_ranking(ref_best_first, expected_n=n, label="reference ranking")
    cmp = _validate_ranking(cmp_best_first, expected_n=n, label="comparator ranking")

    out: Dict[str, Any] = {
        "n_groups": int(n),
        "spearman": _spearman_from_rankings(ref, cmp),
        "kendall": _kendall_tau_from_rankings(ref, cmp),
        "rank_mae": _rank_mae(ref, cmp),
        "rank_mae_norm": _rank_mae_norm(ref, cmp),
        "rank_rmse": _rank_rmse(ref, cmp),
    }

    for k in k_values:
        out[f"overlap_at_{int(k)}"] = _overlap_at_k(ref, cmp, int(k))

    return out


# -------------------------
# Comparator discovery
# -------------------------

def _reference_path(rankings_dir: Path, group_mode: str) -> Path:
    return rankings_dir / f"ranking_{group_mode}_fulltrust_agg_selection.json"


def _known_comparator_paths(rankings_dir: Path, group_mode: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for comparator_name, suffix in KNOWN_COMPARATOR_SUFFIXES.items():
        out[comparator_name] = rankings_dir / f"ranking_{group_mode}_{suffix}.json"
    return out


def _discover_extra_comparators(rankings_dir: Path, group_mode: str) -> Dict[str, Path]:
    """
    Finds additional files matching ranking_<group_mode>_*.json.
    Skips the full TRUST reference and any known comparator already handled.
    """
    known = set(_known_comparator_paths(rankings_dir, group_mode).values())
    ref = _reference_path(rankings_dir, group_mode)
    out: Dict[str, Path] = {}

    prefix = f"ranking_{group_mode}_"
    for p in sorted(rankings_dir.glob(f"{prefix}*.json")):
        if p == ref or p in known:
            continue
        if p.name.endswith("_fulltrust_agg_selection.json"):
            continue
        # ranking_sensors_my_new_method.json -> my_new_method
        stem = p.stem
        comparator = stem.replace(prefix, "", 1)
        comparator = comparator.strip("_")
        if comparator:
            out[comparator] = p
    return out


def _select_comparators(
    *,
    rankings_dir: Path,
    group_mode: str,
    requested: Sequence[str],
    include_extra_rankings: bool,
) -> Dict[str, Path]:
    known = _known_comparator_paths(rankings_dir, group_mode)

    if requested:
        selected = {name: known.get(name, rankings_dir / f"ranking_{group_mode}_{name}.json") for name in requested}
    else:
        selected = dict(known)

    if include_extra_rankings:
        selected.update(_discover_extra_comparators(rankings_dir, group_mode))

    # Only return existing files.
    return {name: p for name, p in selected.items() if p.exists()}


# -------------------------
# Main table builder
# -------------------------

def build_ranking_agreement_table(
    *,
    baselines_root: Path,
    out_csv: Path,
    out_json: Optional[Path],
    k_values: Sequence[int],
    comparators: Sequence[str],
    include_extra_rankings: bool,
    verbose: bool,
) -> Dict[str, Any]:
    rankings_dirs = _find_rankings_dirs(baselines_root)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    missing_reference: List[str] = []

    for rankings_dir in rankings_dirs:
        dataset, window_tag, hidden_tag = _parse_rankings_dir(rankings_dir)

        for group_mode in GROUP_MODES:
            ref_path = _reference_path(rankings_dir, group_mode)
            if not ref_path.exists():
                missing_reference.append(str(ref_path))
                if verbose:
                    print(f"[SKIP] Missing reference: {ref_path}")
                continue

            try:
                ref_json = _load_json(ref_path)
                ref_ranking = _validate_ranking(
                    _extract_ranked_group_indices(ref_json),
                    expected_n=None,
                    label=f"reference {ref_path}",
                )
            except Exception as e:
                errors.append({"path": str(ref_path), "error": str(e)})
                if verbose:
                    print(f"[ERR] Reference failed: {ref_path} -> {e}")
                continue

            cmp_paths = _select_comparators(
                rankings_dir=rankings_dir,
                group_mode=group_mode,
                requested=comparators,
                include_extra_rankings=include_extra_rankings,
            )

            if verbose:
                print(
                    f"[{dataset}/{window_tag}/{hidden_tag}/{group_mode}] "
                    f"reference={ref_path.name} comparators={list(cmp_paths.keys())}"
                )

            for comparator_name, cmp_path in cmp_paths.items():
                try:
                    cmp_json = _load_json(cmp_path)
                    cmp_ranking = _validate_ranking(
                        _extract_ranked_group_indices(cmp_json),
                        expected_n=len(ref_ranking),
                        label=f"comparator {cmp_path}",
                    )

                    metrics = compute_ranking_agreement(
                        ref_best_first=ref_ranking,
                        cmp_best_first=cmp_ranking,
                        k_values=k_values,
                    )

                    row: Dict[str, Any] = {
                        "dataset": dataset,
                        "window_tag": window_tag,
                        "hidden_tag": hidden_tag,
                        "group_mode": group_mode,
                        "reference": "fulltrust_agg_selection",
                        "comparator": comparator_name,
                        "reference_file": str(ref_path),
                        "comparator_file": str(cmp_path),
                    }
                    row.update(metrics)
                    rows.append(row)

                except Exception as e:
                    errors.append({"path": str(cmp_path), "error": str(e)})
                    if verbose:
                        print(f"[ERR] Comparator failed: {cmp_path} -> {e}")

    # Stable row order.
    rows = sorted(rows, key=lambda r: (r["dataset"], r["window_tag"], r["hidden_tag"], r["group_mode"], r["comparator"]))

    overlap_cols = [f"overlap_at_{int(k)}" for k in k_values]
    fieldnames = [
        "dataset",
        "window_tag",
        "hidden_tag",
        "group_mode",
        "reference",
        "comparator",
        "n_groups",
        "spearman",
        "kendall",
        *overlap_cols,
        "rank_mae",
        "rank_mae_norm",
        "rank_rmse",
        "reference_file",
        "comparator_file",
    ]

    _write_csv(out_csv, rows, fieldnames)

    summary = _build_summary(rows)
    report: Dict[str, Any] = {
        "type": "ranking_agreement_table",
        "baselines_root": str(baselines_root),
        "out_csv": str(out_csv),
        "k_values": [int(k) for k in k_values],
        "n_rankings_dirs_found": int(len(rankings_dirs)),
        "n_rows": int(len(rows)),
        "n_errors": int(len(errors)),
        "n_missing_reference": int(len(missing_reference)),
        "summary": summary,
        "errors": errors,
        "missing_reference": missing_reference,
    }

    if out_json is not None:
        _json_dump(out_json, report)

    if verbose:
        print("\n=== SUMMARY ===")
        print("Ranking dirs found:", len(rankings_dirs))
        print("Rows written:", len(rows))
        print("CSV:", out_csv)
        if out_json is not None:
            print("JSON:", out_json)
        print("Missing references:", len(missing_reference))
        print("Errors:", len(errors))

    return report


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _mean(xs: Sequence[float]) -> Optional[float]:
    vals = [_safe_float(x) for x in xs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _build_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compact JSON summary by group_mode and comparator.
    Keeps CSV as the authoritative detailed table.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["group_mode"]), str(row["comparator"]))].append(dict(row))

    out: List[Dict[str, Any]] = []
    for (group_mode, comparator), rs in sorted(groups.items()):
        metric_keys = [k for k in rs[0].keys() if k.startswith("overlap_at_")]
        entry: Dict[str, Any] = {
            "group_mode": group_mode,
            "comparator": comparator,
            "n": int(len(rs)),
            "mean_spearman": _mean([r.get("spearman") for r in rs]),
            "mean_kendall": _mean([r.get("kendall") for r in rs]),
            "mean_rank_mae_norm": _mean([r.get("rank_mae_norm") for r in rs]),
        }
        for key in sorted(metric_keys, key=lambda s: int(s.split("_")[-1])):
            entry[f"mean_{key}"] = _mean([r.get(key) for r in rs])
        out.append(entry)
    return out


# -------------------------
# CLI
# -------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ranking-agreement table using full TRUST aggregated rankings as reference. "
            "Reads mlp_baselines/<dataset>/<window>/<hidden>/rankings/*.json and writes a CSV."
        )
    )
    parser.add_argument("--baselines-root", type=str, default="mlp_baselines", help="Root of baseline MLP folders.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="tables/ranking_agreement_fulltrust_vs_all.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="tables/ranking_agreement_fulltrust_vs_all_summary.json",
        help="Optional JSON report path. Use empty string to disable.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Top-k overlap values to compute.",
    )
    parser.add_argument(
        "--comparators",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional comparator names. Defaults to known existing files: "
            "timetrust_selection weights randomavg group_ablation ablation permutation. "
            "Names map to ranking_<mode>_<name>.json unless they are known aliases."
        ),
    )
    parser.add_argument(
        "--include-extra-rankings",
        action="store_true",
        help="Also compare any extra ranking_<mode>_*.json files found in each rankings directory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    out_json: Optional[Path]
    if str(args.out_json).strip() == "":
        out_json = None
    else:
        out_json = Path(args.out_json)

    build_ranking_agreement_table(
        baselines_root=Path(args.baselines_root),
        out_csv=Path(args.out_csv),
        out_json=out_json,
        k_values=[int(k) for k in args.k_values],
        comparators=[str(c) for c in args.comparators],
        include_extra_rankings=bool(args.include_extra_rankings),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
