from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import kagglehub


# =========================
# Constants
# =========================

RANDOM_SEED = 2026

SENSORS_LIST = [
    "(Fan inlet temperature) (°R)",
    "(LPC outlet temperature) (°R)",
    "(HPC outlet temperature) (°R)",
    "(LPT outlet temperature) (°R)",
    "(Fan inlet Pressure) (psia)",
    "(Bypass-duct pressure) (psia)",
    "(HPC outlet pressure) (psia)",
    "(Physical fan speed) (rpm)",
    "(Physical core speed) (rpm)",
    "(Engine pressure ratio (P50/P2))",
    "(HPC outlet Static pressure) (psia)",
    "(Ratio of fuel flow to Ps30) (pps/psia)",
    "(Corrected fan speed) (rpm)",
    "(Corrected core speed) (rpm)",
    "(Bypass Ratio)",
    "(Burner fuel-air ratio)",
    "(Bleed Enthalpy)",
    "(Required fan speed)",
    "(Required fan conversion speed)",
    "(High-pressure turbines Cool air flow)",
    "(Low-pressure turbines Cool air flow)",
]

SENSOR_NAMES = [f"s{i}" for i in range(1, 22)]
SENSORS_DICTIONARY = {f"s{i+1}": SENSORS_LIST[i] for i in range(len(SENSORS_LIST))}

BASE_COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + SENSOR_NAMES
TARGET_COL = "RUL"


# =========================
# Config
# =========================

@dataclass
class CmapssConfig:
    seq_len: int = 30
    step: int = 1
    val_ratio: float = 0.1
    eps_constant_sensor: float = 1e-3
    seed: int = RANDOM_SEED
    dataset_ids: Tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")


# =========================
# Download and load
# =========================

def download_cmapss() -> Path:
    """
    Downloads the dataset via kagglehub and returns the folder path.
    """
    path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
    return Path(path)


def find_cmapss_txt_folder(root_path: Path) -> Path:
    """
    Locates the folder that contains train_FD001.txt etc.
    """
    for root, _, files in os.walk(root_path):
        if "train_FD001.txt" in files:
            return Path(root)
    raise FileNotFoundError("Could not locate train_FD001.txt under the downloaded path.")


def load_train_split_only(data_path: Path, dataset_id: str) -> pd.DataFrame:
    """
    Loads train_FD00x.txt and assigns column names.
    Computes RUL per unit using max cycle - cycle.
    """
    train_file = data_path / f"train_{dataset_id}.txt"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing file: {train_file}")

    df_train = pd.read_csv(train_file, sep=r"\s+", header=None)
    df_train.columns = BASE_COLUMNS

    # Compute max cycle per unit, then RUL
    max_cycle = df_train.groupby("unit")["cycle"].max().reset_index()
    max_cycle.rename(columns={"cycle": "max_cycle"}, inplace=True)
    df_train = df_train.merge(max_cycle, on="unit", how="left")
    df_train[TARGET_COL] = df_train["max_cycle"] - df_train["cycle"]
    df_train.drop(columns=["max_cycle"], inplace=True)

    return df_train


# =========================
# Preprocessing helpers
# =========================

def split_by_unit(
    df: pd.DataFrame,
    unit_col: str = "unit",
    val_ratio: float = 0.1,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    Splits units into train/val and returns dataframes plus unit lists.
    """
    rng = np.random.default_rng(seed)
    units = df[unit_col].unique()
    rng.shuffle(units)

    n_val = int(len(units) * val_ratio)
    val_units = units[:n_val].tolist()
    train_units = units[n_val:].tolist()

    df_train = df[df[unit_col].isin(train_units)].copy()
    df_val = df[df[unit_col].isin(val_units)].copy()

    return df_train, df_val, train_units, val_units


def remove_constant_sensors(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
    eps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Drops sensors with (train) std <= eps (or non-finite std).
    Applies the same drop list to validation.
    """
    stds = df_train[feature_cols].std(axis=0, ddof=0)
    dropped = [c for c, s in stds.items() if (not np.isfinite(s)) or (s <= eps)]
    kept = [c for c in feature_cols if c not in dropped]

    df_train_clean = df_train.drop(columns=dropped, errors="ignore").copy()
    df_val_clean = df_val.drop(columns=dropped, errors="ignore").copy()

    return df_train_clean, df_val_clean, dropped, kept


def normalize_sensors(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Fits StandardScaler on df_train and transforms df_train and df_val.
    """
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    df_train_n = df_train.copy()
    df_val_n = df_val.copy()

    df_train_n[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_val_n[feature_cols] = scaler.transform(df_val[feature_cols].values)

    return scaler, df_train_n, df_val_n


def generate_windows_mw(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = TARGET_COL,
    unit_col: str = "unit",
    cycle_col: str = "cycle",
    seq_len: int = 30,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sliding windows per unit:
      X: [N, M, W] where M=len(feature_cols) and W=seq_len
      y: [N]
    Label is the RUL at the last timestep in the window.
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for _, g in df.groupby(unit_col):
        g = g.sort_values(cycle_col).reset_index(drop=True)
        feats = g[feature_cols].values  # [T, M]
        labels = g[label_col].values   # [T]

        T = feats.shape[0]
        for start in range(0, T - seq_len + 1, step):
            end = start + seq_len
            window_tm = feats[start:end]      # [W, M]
            window_mw = window_tm.T           # [M, W]
            X_list.append(window_mw)
            y_list.append(labels[end - 1])

    if len(X_list) == 0:
        return np.empty((0, len(feature_cols), seq_len), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


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
# Saving and loading
# =========================

def processed_output_dir(dataset_id: str, cfg: CmapssConfig, root: Optional[Path] = None) -> Path:
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
    mapping: Optional[Dict[str, object]] = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays (mw and flat variants)
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


def load_processed_dataset(out_dir: Path) -> Dict[str, object]:
    """
    Loads dataset.npz and metadata.json.
    """
    data = np.load(out_dir / "dataset.npz", allow_pickle=True)
    with open(out_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "arrays": {k: data[k] for k in data.files},
        "metadata": meta,
    }


# =========================
# Pipeline for a single FD
# =========================

def process_one_fd(dataset_id: str, cfg: CmapssConfig) -> Tuple[Path, Dict]:
    """
    Full preprocessing pipeline for one dataset_id (FD001..FD004).
    Saves outputs to datasets/processed/<dataset_id>/W{seq_len}_step{step}/
    """
    root = download_cmapss()
    data_path = find_cmapss_txt_folder(root)

    df = load_train_split_only(data_path, dataset_id)

    # Split by unit (train/val)
    df_train, df_val, train_units, val_units = split_by_unit(
        df,
        unit_col="unit",
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    # Constant sensor removal (train only stats)
    feature_cols = SENSOR_NAMES.copy()
    df_train, df_val, dropped, kept = remove_constant_sensors(
        df_train=df_train,
        df_val=df_val,
        feature_cols=feature_cols,
        eps=cfg.eps_constant_sensor,
    )

    # Normalize (fit on train split only)
    scaler, df_train_n, df_val_n = normalize_sensors(df_train, df_val, kept)

    # Windows
    X_train_mw, y_train = generate_windows_mw(
        df=df_train_n,
        feature_cols=kept,
        seq_len=cfg.seq_len,
        step=cfg.step,
    )
    X_val_mw, y_val = generate_windows_mw(
        df=df_val_n,
        feature_cols=kept,
        seq_len=cfg.seq_len,
        step=cfg.step,
    )

    # Metadata
    kept_sensor_descriptions = {s: SENSORS_DICTIONARY[s] for s in kept}
    dropped_sensor_descriptions = {s: SENSORS_DICTIONARY[s] for s in dropped}
    mapping = build_flat_feature_mapping(kept_sensors=kept, seq_len=cfg.seq_len)

    meta = {
        "dataset_id": dataset_id,
        "source": "behrad3d/nasa-cmaps (kagglehub)",
        "split": {
            "val_ratio": cfg.val_ratio,
            "seed": cfg.seed,
            "train_units": train_units,
            "val_units": val_units,
            "n_units_total": int(df["unit"].nunique()),
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
            "all_sensors": SENSOR_NAMES,
            "kept_sensors": kept,
            "dropped_sensors": dropped,
            "kept_sensor_descriptions": kept_sensor_descriptions,
            "dropped_sensor_descriptions": dropped_sensor_descriptions,
        },
        "scaler": {
            "type": "StandardScaler",
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "var_": scaler.var_.tolist(),
            "n_features_in_": int(scaler.n_features_in_),
        },
        "target": TARGET_COL,
        "notes": {
            "rul_definition": "RUL = max_cycle(unit) - cycle",
            "normalization_fit": "train split only",
            "constant_sensor_rule": f"std(train) <= {cfg.eps_constant_sensor}",
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
    }

    out_dir = processed_output_dir(dataset_id, cfg, root=Path.cwd())
    save_processed_dataset(out_dir, X_train_mw, y_train, X_val_mw, y_val, meta, mapping=mapping)

    return out_dir, meta


# =========================
# Pipeline for FD001-FD004
# =========================

def process_all_fds(cfg: Optional[CmapssConfig] = None) -> Dict[str, Dict]:
    """
    Runs the preprocessing pipeline for FD001-FD004 and saves all outputs.
    Returns metadata per dataset_id.
    """
    if cfg is None:
        cfg = CmapssConfig()

    results: Dict[str, Dict] = {}
    for ds in cfg.dataset_ids:
        out_dir, meta = process_one_fd(ds, cfg)
        results[ds] = {
            "out_dir": str(out_dir),
            "metadata": meta,
        }
        print(f"[OK] {ds} saved to: {out_dir}")

    return results


def main() -> None:
    cfg = CmapssConfig(
        seq_len=30,
        step=1,
        val_ratio=0.1,
        eps_constant_sensor=1e-3,
        seed=RANDOM_SEED,
    )
    process_all_fds(cfg)


if __name__ == "__main__":
    main()