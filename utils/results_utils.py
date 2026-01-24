from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def make_run_dir(
    root: Path,
    dataset_name: str,
    w_tag: str,
    h_tag: str,
    trust_mode: str,
    mlp_mode: str,
    baseline_mode: str,
) -> Path:
    run_dir = (
        root
        / dataset_name
        / w_tag
        / h_tag
        / f"trust={trust_mode}__mlp={mlp_mode}__baseline={baseline_mode}"
    )
    (run_dir / "progress").mkdir(parents=True, exist_ok=True)
    (run_dir / "final").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, config: Dict[str, Any]) -> None:
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_iteration_checkpoint(
    run_dir: Path,
    iter_idx: int,
    n0_value: int,
    meta: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    prog_dir = run_dir / "progress"
    json_path = prog_dir / f"iter_{iter_idx:06d}_n0={n0_value}.json"
    npz_path = prog_dir / f"iter_{iter_idx:06d}_n0={n0_value}_state.npz"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(npz_path, **arrays)
    return json_path, npz_path


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    prog_dir = run_dir / "progress"
    if not prog_dir.exists():
        return None
    files = sorted(prog_dir.glob("iter_*_state.npz"))
    return files[-1] if files else None


def load_checkpoint(npz_path: Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    return out