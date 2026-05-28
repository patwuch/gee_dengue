"""
scripts/OpenDengue/XGBoost/utils.py
--------------------------------------
Shared helpers for the OpenDengue XGBoost (and Random Forest) experiments.

Public surface mirrors scripts/Indonesia/XGBoost/utils.py so both datasets
are driven by the same tune / train / eval pattern with only the config YAML
differing:

    load_data(cfg, run_cfg=None)        → pd.DataFrame
    build_feature_columns(df, cfg)      → list[str]
    split_train_test(df, cols, cfg, region=None)
                                        → (df_train_val, df_test)
    calculate_sample_weights(y)         → np.ndarray
    resolve_paths(cfg)                  → dict[str, Path]
    validation_strategy(cfg)            → 'walk' | 'kfold'
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _find_project_root(start: Path, marker: str = ".git") -> Path:
    for parent in start.resolve().parents:
        if (parent / marker).exists():
            return parent
    return start.resolve()


def resolve_paths(cfg: dict) -> dict[str, Path]:
    """
    Return canonical project paths.  cfg may optionally contain a 'paths'
    block to override; otherwise paths are derived from the git root.
    """
    root    = _find_project_root(Path(__file__))
    reports = root / "reports"
    defaults = {
        "project_root":  root,
        "processed_dir": root / "data" / "processed",
        "models_dir":    root / "models",
        "figures_dir":   reports / "figures",
        "tables_dir":    reports / "tables",
    }
    overrides = cfg.get("paths", {})
    return {k: Path(overrides.get(k, v)) for k, v in defaults.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: dict, run_cfg: dict | None = None) -> pd.DataFrame:
    """
    Load the OpenDengue CSV and prepare lag features.

    Supports both standalone mode (cfg contains data_path) and
    Snakemake mode (run_cfg may override the path).
    """
    run_cfg  = run_cfg or {}
    data_path = run_cfg.get("data") or cfg.get("data_path")
    if data_path is None:
        raise ValueError("data_path not set in config or run_cfg")

    df = pd.read_csv(data_path)

    time_col = cfg.get("time_column", "Date")
    unit_col = cfg.get("unit_column", "name")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([unit_col, time_col]).reset_index(drop=True)

    # Lag-1 target (epidemic feature) — optional
    target_col = cfg.get("target_column") or cfg.get("target", {}).get("column", "IR")
    if cfg.get("add_lag_IR", False):
        df[f"{target_col}_lag1"] = df.groupby(unit_col)[target_col].shift(1)

    # Lagged env features
    lag_steps = cfg.get("lag_steps", [1, 2, 3])
    for var in cfg.get("env_vars", []):
        if var in df.columns:
            for lag in lag_steps:
                df[f"{var}_lag{lag}"] = df.groupby(unit_col)[var].shift(lag)

    # Encode classifier target if label map is present
    label_map = cfg.get("target", {}).get("labels")
    if label_map:
        df[target_col] = df[target_col].replace(label_map).infer_objects(copy=False)

    return df


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

def build_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """
    Build the ordered feature column list from config.
    Mirrors Indonesia utils.build_feature_columns.
    """
    use_landuse = cfg.get("use_landuse", False)
    lag_steps   = cfg.get("lag_steps", [1, 2, 3])
    target_col  = cfg.get("target_column") or cfg.get("target", {}).get("column", "IR")
    time_col    = cfg.get("time_column", "Date")
    unit_col    = cfg.get("unit_column", "name")
    admin_col   = cfg.get("admin_column", "admin")

    excluded = {target_col, time_col, unit_col, admin_col}

    cols: list[str] = []

    for var in cfg.get("env_vars", []):
        if var in df.columns:
            cols.append(var)

    if use_landuse:
        for var in cfg.get("lulc_vars", []):
            if var in df.columns:
                cols.append(var)

    # Lagged env vars
    for var in cfg.get("env_vars", []):
        for lag in lag_steps:
            lagged = f"{var}_lag{lag}"
            if lagged in df.columns:
                cols.append(lagged)

    # Lag-1 IR if added
    if cfg.get("add_lag_IR", False):
        lag_ir = f"{target_col}_lag1"
        if lag_ir in df.columns:
            cols.append(lag_ir)

    # Deduplicate, drop excluded
    seen: set[str] = set()
    result: list[str] = []
    for c in cols:
        if c not in seen and c not in excluded:
            seen.add(c)
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_train_test(
    df: pd.DataFrame,
    variable_columns: list[str],
    cfg: dict,
    region: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (df_train_val, df_test).

    Temporal split: last `test_months` months → test;
    remainder → train/val.  Region filtering optional.
    """
    time_col   = cfg.get("time_column", "Date")
    unit_col   = cfg.get("unit_column", "name")
    target_col = cfg.get("target_column") or cfg.get("target", {}).get("column", "IR")

    if region is not None:
        df = df[df[unit_col] == region].copy()

    df = df.dropna(subset=variable_columns + [target_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    test_months = cfg.get("split", {}).get("test_months", 12)
    unique_dates = sorted(df[time_col].unique())
    cutoff       = unique_dates[-test_months] if len(unique_dates) > test_months else unique_dates[0]

    df_train_val = df[df[time_col] < cutoff].copy()
    df_test      = df[df[time_col] >= cutoff].copy()
    return df_train_val, df_test


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def calculate_sample_weights(y: np.ndarray) -> np.ndarray:
    unique_classes, counts = np.unique(y, return_counts=True)
    total  = len(y)
    n_cls  = len(unique_classes)
    wmap   = {cls: total / (n_cls * cnt) for cls, cnt in zip(unique_classes, counts)}
    return np.array([wmap[cls] for cls in y])


# ---------------------------------------------------------------------------
# Validation strategy
# ---------------------------------------------------------------------------

def validation_strategy(run_cfg: dict) -> str:
    """Return 'walk' or 'kfold' from run_cfg (Snakemake) or cfg (standalone)."""
    strat = run_cfg.get("validation", run_cfg.get("strategy", "walk"))
    if isinstance(strat, dict):
        strat = strat.get("strategy", "walk")
    return strat
