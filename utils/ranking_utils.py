from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from utils.mlp_utils import RANDOM_SEED

GroupMode = Literal["sensors", "windows"]
RandomMode = Literal["per_feature", "per_group"]


@dataclass
class GroupRankingResult:
    group_mode: GroupMode
    n_sensors: int
    n_windows: int
    scores: np.ndarray              # shape: (n_groups,)
    scores_norm: np.ndarray         # shape: (n_groups,)
    ranking: np.ndarray             # group indices sorted desc by score (0-based)
    items: List[Dict[str, Any]]     # JSON-friendly list


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def _flatten_group_mapping(M0: int, W0: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        flat_to_sensor: (M0*W0,) 0-based sensor index
        flat_to_window: (M0*W0,) 0-based window index
    Assumes flatten order: sensor-major (sensor axis first, then window).
    """
    n = int(M0) * int(W0)
    flat = np.arange(n, dtype=int)
    flat_to_sensor = flat // int(W0)
    flat_to_window = flat % int(W0)
    return flat_to_sensor, flat_to_window


def _extract_first_layer_W1_from_baseline(
    *,
    baseline_hidden_Ws: Sequence[np.ndarray],
    expected_input_dim: int,
) -> np.ndarray:
    """
    baseline_hidden_Ws[0] should be W1 with shape (input_dim, hidden_1).
    """
    if baseline_hidden_Ws is None or len(baseline_hidden_Ws) == 0:
        raise ValueError("baseline_hidden_Ws is empty. Cannot extract W1.")

    W1 = np.asarray(baseline_hidden_Ws[0])
    if W1.ndim != 2:
        raise ValueError(f"Expected W1 to be 2D, got shape {W1.shape}")

    if int(W1.shape[0]) != int(expected_input_dim):
        raise ValueError(
            f"W1 input_dim mismatch: W1.shape[0]={W1.shape[0]} "
            f"but expected_input_dim={expected_input_dim}"
        )
    return W1


def build_group_ranking_from_mlp_first_layer(
    *,
    baseline_hidden_Ws: Sequence[np.ndarray],
    M0: int,
    W0: int,
    group_mode: GroupMode,
    group_names: Optional[Sequence[str]] = None,
    out_json: Optional[Path] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> GroupRankingResult:
    """
    Builds a grouped ranking (sensors/windows) from the first-layer weights of a baseline MLP.

    Feature importance:
        I_f = sum_h |W1[f, h]|

    Group importance:
      - sensors: sum_{w} I_{(s,w)}
      - windows: sum_{s} I_{(s,w)}

    Saves JSON if out_json is provided.
    """
    M0 = int(M0)
    W0 = int(W0)
    input_dim = M0 * W0

    W1 = _extract_first_layer_W1_from_baseline(
        baseline_hidden_Ws=baseline_hidden_Ws,
        expected_input_dim=input_dim,
    )

    # Per-feature importance (input_dim,)
    imp_feat = np.sum(np.abs(W1), axis=1).astype(float)

    flat_to_sensor, flat_to_window = _flatten_group_mapping(M0, W0)

    if group_mode == "sensors":
        n_groups = M0
        scores = np.zeros(n_groups, dtype=float)
        for f in range(input_dim):
            scores[int(flat_to_sensor[f])] += float(imp_feat[f])
    elif group_mode == "windows":
        n_groups = W0
        scores = np.zeros(n_groups, dtype=float)
        for f in range(input_dim):
            scores[int(flat_to_window[f])] += float(imp_feat[f])
    else:
        raise ValueError(f"Invalid group_mode={group_mode}")

    scores_norm = _normalize_nonneg(scores)
    ranking = np.argsort(-scores_norm)  # descending

    # Names
    if group_names is None:
        if group_mode == "sensors":
            group_names = [f"s{i+1}" for i in range(M0)]
        else:
            group_names = [f"w{i+1}" for i in range(W0)]
    group_names = list(group_names)

    items: List[Dict[str, Any]] = []
    for rank_pos, g in enumerate(ranking.tolist(), start=1):
        items.append(
            {
                "rank": int(rank_pos),
                "group_index_0based": int(g),
                "group_index_1based": int(g + 1),
                "group_name": str(group_names[g]) if g < len(group_names) else str(g + 1),
                "score": float(scores[g]),
                "score_norm": float(scores_norm[g]),
            }
        )

    result = GroupRankingResult(
        group_mode=group_mode,
        n_sensors=M0,
        n_windows=W0,
        scores=scores,
        scores_norm=scores_norm,
        ranking=ranking,
        items=items,
    )

    if out_json is not None:
        payload: Dict[str, Any] = {
            "type": "group_ranking_mlp_first_layer",
            "group_mode": group_mode,
            "n_sensors": M0,
            "n_windows": W0,
            "input_dim": input_dim,
            "definition": {
                "feature_importance": "I_f = sum_h |W1[f,h]|",
                "group_aggregation": "sum over features in the group",
                "normalization": "scores_norm = scores / sum(scores)",
                "flatten_order": "sensor-major: flat = s*W0 + w",
            },
            "items": items,
        }
        if extra_meta:
            payload["meta"] = dict(extra_meta)
        _json_dump(out_json, payload)

    return result


def build_group_ranking_random_average(
    *,
    M0: int,
    W0: int,
    group_mode: GroupMode,
    n_runs: int = 200,
    seed: int = RANDOM_SEED,
    random_mode: RandomMode = "per_feature",
    group_names: Optional[Sequence[str]] = None,
    out_json: Optional[Path] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> GroupRankingResult:
    """
    Builds a grouped random ranking averaged over multiple random draws.

    random_mode:
      - "per_feature": sample u_f ~ U(0,1) for each feature, aggregate to group, average over runs
      - "per_group": sample u_g ~ U(0,1) for each group, average over runs

    Saves JSON if out_json is provided.
    """
    M0 = int(M0)
    W0 = int(W0)
    input_dim = M0 * W0
    n_runs = int(n_runs)
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0")

    rng = np.random.default_rng(int(seed))
    flat_to_sensor, flat_to_window = _flatten_group_mapping(M0, W0)

    if group_mode == "sensors":
        n_groups = M0
    elif group_mode == "windows":
        n_groups = W0
    else:
        raise ValueError(f"Invalid group_mode={group_mode}")

    scores_acc = np.zeros(n_groups, dtype=float)

    for _ in range(n_runs):
        if random_mode == "per_feature":
            u_feat = rng.random(input_dim)  # (M0*W0,)
            g_scores = np.zeros(n_groups, dtype=float)
            if group_mode == "sensors":
                for f in range(input_dim):
                    g_scores[int(flat_to_sensor[f])] += float(u_feat[f])
            else:
                for f in range(input_dim):
                    g_scores[int(flat_to_window[f])] += float(u_feat[f])
        elif random_mode == "per_group":
            g_scores = rng.random(n_groups).astype(float)
        else:
            raise ValueError(f"Invalid random_mode={random_mode}")

        scores_acc += g_scores

    scores = scores_acc / float(n_runs)
    scores_norm = _normalize_nonneg(scores)
    ranking = np.argsort(-scores_norm)

    if group_names is None:
        if group_mode == "sensors":
            group_names = [f"s{i+1}" for i in range(M0)]
        else:
            group_names = [f"w{i+1}" for i in range(W0)]
    group_names = list(group_names)

    items: List[Dict[str, Any]] = []
    for rank_pos, g in enumerate(ranking.tolist(), start=1):
        items.append(
            {
                "rank": int(rank_pos),
                "group_index_0based": int(g),
                "group_index_1based": int(g + 1),
                "group_name": str(group_names[g]) if g < len(group_names) else str(g + 1),
                "score": float(scores[g]),
                "score_norm": float(scores_norm[g]),
            }
        )

    result = GroupRankingResult(
        group_mode=group_mode,
        n_sensors=M0,
        n_windows=W0,
        scores=scores,
        scores_norm=scores_norm,
        ranking=ranking,
        items=items,
    )

    if out_json is not None:
        payload: Dict[str, Any] = {
            "type": "group_ranking_random_average",
            "group_mode": group_mode,
            "random_mode": random_mode,
            "n_runs": n_runs,
            "seed": int(seed),
            "n_sensors": M0,
            "n_windows": W0,
            "input_dim": input_dim,
            "definition": {
                "sampling": "U(0,1)",
                "aggregation": "sum over features in the group (per_feature) or direct per-group scores (per_group)",
                "averaging": "scores = mean over runs",
                "normalization": "scores_norm = scores / sum(scores)",
                "flatten_order": "sensor-major: flat = s*W0 + w",
            },
            "items": items,
        }
        if extra_meta:
            payload["meta"] = dict(extra_meta)
        _json_dump(out_json, payload)

    return result