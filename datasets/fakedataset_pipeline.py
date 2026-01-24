from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =========================
# Constants
# =========================

RANDOM_SEED = 2026

TARGET_COL = "RUL"

# Reuse your naming convention
SENSOR_NAMES = [f"s{i}" for i in range(1, 22)]  # global pool (we will select 5)
BASE_COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + SENSOR_NAMES


# =========================
# Config
# =========================

@dataclass
class FakeDatasetConfig:
    dataset_id: str = "FAKE_DATASET"

    # Windowing
    seq_len: int = 10
    step: int = 1

    # Split by unit
    val_ratio: float = 0.1

    # Synthetic generation
    n_units: int = 100
    min_cycles: int = 25
    max_cycles: int = 60

    # Sensors
    n_sensors_keep: int = 5

    # Signal/noise
    noise_std: float = 0.05
    seed: int = RANDOM_SEED


# =========================
# Feature mapping helpers
# =========================

def flatten_windows_mw(X_mw: np.ndarray) -> np.ndarray:
    """
    Converts [N, M, W] to [N, M*W], keeping a stable flattening order.
    """
    if X_mw.ndim != 3:
        raise ValueError(f"Expected X_mw to be 3D [N,M,W], got {X_mw.shape}")
    N, M, W = X_mw.shape
    return X_mw.reshape(N, M * W)


def build_flat_feature_mapping(
    kept_sensors: List[str],
    seq_len: int,
) -> Dict[str, object]:
    """
    Builds mapping arrays for flattened features m=1..(M*W).

    Flattening order must match: X_mw.reshape(N, M*W)
    where M axis comes first, then W.

    For m_idx in [0..M*W-1]:
        sensor_idx = m_idx // W
        window_idx = m_idx % W
    """
    M = len(kept_sensors)
    W = int(seq_len)
    n_features_flat = M * W

    flat_to_sensor = np.empty((n_features_flat,), dtype=np.int32)
    flat_to_window = np.empty((n_features_flat,), dtype=np.int32)
    flat_sensor_key = np.empty((n_features_flat,), dtype=object)

    for m_idx in range(n_features_flat):
        s_idx = m_idx // W          # 0..M-1
        w_idx = m_idx % W           # 0..W-1
        flat_to_sensor[m_idx] = s_idx + 1   # 1-based
        flat_to_window[m_idx] = w_idx + 1   # 1-based
        flat_sensor_key[m_idx] = kept_sensors[s_idx]

    return {
        "n_sensors": M,
        "window_length": W,
        "n_features_flat": n_features_flat,
        "flat_to_sensor": flat_to_sensor,
        "flat_to_window": flat_to_window,
        "flat_sensor_key": flat_sensor_key,
    }


# =========================
# Saving and loading (same structure as yours)
# =========================

def processed_output_dir(dataset_id: str, cfg: FakeDatasetConfig, root: Optional[Path] = None) -> Path:
    """
    Root defaults to current working directory.
    """
    if root is None:
        root = Path.cwd()
    return root / "datasets" / "processed" / dataset_id / f"W{cfg.seq_len}_step{cfg.step}"


def save_processed_dataset(
    out_dir: Path,
    X_train_mw: np.ndarray,
    y_train: np.ndarray,
    X_val_mw: np.ndarray,
    y_val: np.ndarray,
    metadata: Dict,
    mapping: Optional[Dict[str, object]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(
        X_train_mw=X_train_mw,
        y_train=y_train,
        X_val_mw=X_val_mw,
        y_val=y_val,
        X_train_flat=flatten_windows_mw(X_train_mw),
        X_val_flat=flatten_windows_mw(X_val_mw),
    )

    if mapping is not None:
        payload.update(
            n_sensors=np.array(mapping["n_sensors"], dtype=np.int32),
            window_length=np.array(mapping["window_length"], dtype=np.int32),
            n_features_flat=np.array(mapping["n_features_flat"], dtype=np.int32),
            flat_to_sensor=mapping["flat_to_sensor"],
            flat_to_window=mapping["flat_to_window"],
            flat_sensor_key=mapping["flat_sensor_key"],
        )

    np.savez_compressed(out_dir / "dataset.npz", **payload)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# =========================
# Synthetic data generation
# =========================

def _set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _generate_unit_timeseries(
    rng: np.random.Generator,
    T: int,
    sensors: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a unit time-series:
      feats_tm: [T, M]
      rul:      [T]  (RUL decreases to 0 at end-of-life)
    """
    M = len(sensors)

    # Base trends + oscillations + noise
    t = np.arange(T, dtype=np.float32)

    # Create correlated sensor signals
    base = rng.normal(0.0, 1.0, size=(M,)).astype(np.float32)
    slope = rng.normal(0.0, 0.02, size=(M,)).astype(np.float32)
    amp = rng.uniform(0.2, 1.0, size=(M,)).astype(np.float32)
    freq = rng.uniform(0.03, 0.2, size=(M,)).astype(np.float32)

    feats_tm = np.zeros((T, M), dtype=np.float32)
    for j in range(M):
        feats_tm[:, j] = (
            base[j]
            + slope[j] * t
            + amp[j] * np.sin(freq[j] * t)
            + rng.normal(0.0, 0.1, size=T).astype(np.float32)
        )

    # RUL definition: max_cycle - cycle (cycle starts at 1)
    cycle = np.arange(1, T + 1, dtype=np.int32)
    rul = (T - cycle).astype(np.float32)

    return feats_tm, rul


def split_by_unit_ids(
    unit_ids: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Splits unit ids into train/val lists.
    """
    rng = np.random.default_rng(seed)
    unit_ids = unit_ids.copy()
    rng.shuffle(unit_ids)

    n_val = int(len(unit_ids) * val_ratio)
    val_units = unit_ids[:n_val].tolist()
    train_units = unit_ids[n_val:].tolist()
    return train_units, val_units


def generate_windows_mw_from_units(
    units_data: Dict[int, Dict[str, np.ndarray]],
    feature_cols: List[str],
    seq_len: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sliding windows per unit from prebuilt unit arrays.

      X: [N, M, W] where M=len(feature_cols) and W=seq_len
      y: [N]
    Label is the RUL at the last timestep in the window.
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for unit_id, pack in units_data.items():
        feats_tm = pack["feats_tm"]  # [T, M]
        rul_t = pack["rul_t"]        # [T]
        T = feats_tm.shape[0]

        for start in range(0, T - seq_len + 1, step):
            end = start + seq_len
            window_tm = feats_tm[start:end]      # [W, M]
            window_mw = window_tm.T              # [M, W]
            X_list.append(window_mw)
            y_list.append(float(rul_t[end - 1]))

    if len(X_list) == 0:
        return (
            np.empty((0, len(feature_cols), seq_len), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def standardize_by_train(
    X_train_mw: np.ndarray,
    X_val_mw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Standardize features using train statistics (mean/std per feature index in MxW space).
    """
    mu = X_train_mw.mean(axis=0, keepdims=True)
    sigma = X_train_mw.std(axis=0, keepdims=True)
    sigma = np.maximum(sigma, 1e-8)

    X_train_n = (X_train_mw - mu) / sigma
    X_val_n = (X_val_mw - mu) / sigma

    scaler_meta = {
        "type": "zscore_numpy",
        "mean_": mu.reshape(-1).tolist(),
        "scale_": sigma.reshape(-1).tolist(),
        "n_features_in_": int(mu.size),
        "notes": "mean/std computed over X_train_mw across samples, per (sensor, window) entry",
    }
    return X_train_n.astype(np.float32), X_val_n.astype(np.float32), scaler_meta


# =========================
# Pipeline for FAKE_DATASET
# =========================

def process_fake_dataset(cfg: FakeDatasetConfig) -> Tuple[Path, Dict]:
    """
    Full synthetic pipeline:
      - generates unit time-series
      - split by unit ids
      - windowing to X_mw
      - normalize using train only
      - save to datasets/processed/FAKE_DATASET/W10_step1/
    """
    rng = _set_seed(cfg.seed)

    # Select 5 sensors from the global pool, keep stable naming
    kept = SENSOR_NAMES[: cfg.n_sensors_keep]

    # Generate per-unit data
    unit_ids = np.arange(1, cfg.n_units + 1, dtype=int)
    units_data: Dict[int, Dict[str, np.ndarray]] = {}

    for unit_id in unit_ids:
        T = int(rng.integers(cfg.min_cycles, cfg.max_cycles + 1))
        feats_tm, rul_t = _generate_unit_timeseries(rng=rng, T=T, sensors=kept)
        units_data[int(unit_id)] = {
            "feats_tm": feats_tm,
            "rul_t": rul_t,
            "T": np.array(T, dtype=np.int32),
        }

    # Split by unit
    train_units, val_units = split_by_unit_ids(
        unit_ids=unit_ids,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    units_train = {u: units_data[u] for u in train_units}
    units_val = {u: units_data[u] for u in val_units}

    # Windowing (mw: sensor-major)
    X_train_mw, y_train = generate_windows_mw_from_units(
        units_data=units_train,
        feature_cols=kept,
        seq_len=cfg.seq_len,
        step=cfg.step,
    )
    X_val_mw, y_val = generate_windows_mw_from_units(
        units_data=units_val,
        feature_cols=kept,
        seq_len=cfg.seq_len,
        step=cfg.step,
    )

    # Normalize (fit on train only)
    X_train_mw, X_val_mw, scaler_meta = standardize_by_train(X_train_mw, X_val_mw)

    # Mapping
    mapping = build_flat_feature_mapping(kept_sensors=kept, seq_len=cfg.seq_len)

    # Metadata
    meta = {
        "dataset_id": cfg.dataset_id,
        "source": "synthetic",
        "split": {
            "val_ratio": cfg.val_ratio,
            "seed": cfg.seed,
            "train_units": train_units,
            "val_units": val_units,
            "n_units_total": int(cfg.n_units),
            "n_units_train": int(len(train_units)),
            "n_units_val": int(len(val_units)),
        },
        "windowing": {
            "seq_len": cfg.seq_len,
            "step": cfg.step,
            "X_train_mw_shape": list(X_train_mw.shape),
            "X_val_mw_shape": list(X_val_mw.shape),
        },
        "features": {
            "kept_sensors": kept,
            "dropped_sensors": [s for s in SENSOR_NAMES if s not in kept],
            "notes": "kept_sensors are a subset of s1..s21",
        },
        "scaler": scaler_meta,
        "target": TARGET_COL,
        "notes": {
            "rul_definition": "RUL = max_cycle(unit) - cycle (synthetic, per unit)",
            "normalization_fit": "train split only",
            "constant_sensor_rule": "not applied (synthetic)",
            "test_set_used": False,
        },
        "structure": {
            "n_sensors": mapping["n_sensors"],
            "window_length": mapping["window_length"],
            "n_features_flat": mapping["n_features_flat"],
            "flatten_order": "mw (sensor-major): m_idx = sensor_idx*W + window_idx",
        },
        "feature_mapping": {
            "flat_to_sensor_1based": mapping["flat_to_sensor"].tolist(),
            "flat_to_window_1based": mapping["flat_to_window"].tolist(),
            "flat_sensor_key": mapping["flat_sensor_key"].tolist(),
        },
        "synthetic_generation": {
            "n_units": int(cfg.n_units),
            "min_cycles": int(cfg.min_cycles),
            "max_cycles": int(cfg.max_cycles),
            "noise_std": float(cfg.noise_std),
            "seed": int(cfg.seed),
        },
    }

    out_dir = processed_output_dir(cfg.dataset_id, cfg, root=Path.cwd())
    save_processed_dataset(out_dir, X_train_mw, y_train, X_val_mw, y_val, meta, mapping=mapping)

    return out_dir, meta


def main() -> None:
    cfg = FakeDatasetConfig(
        dataset_id="FAKE_DATASET",
        seq_len=10,
        step=1,
        val_ratio=0.1,
        n_units=100,
        min_cycles=25,
        max_cycles=60,
        n_sensors_keep=5,
        noise_std=0.05,
        seed=RANDOM_SEED,
    )
    out_dir, _ = process_fake_dataset(cfg)
    print(f"[OK] {cfg.dataset_id} saved to: {out_dir}")


if __name__ == "__main__":
    main()