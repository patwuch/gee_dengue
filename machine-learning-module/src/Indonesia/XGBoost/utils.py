"""
scripts/Indonesia/XGBoost/utils.py
-----------------------------------
Shared helpers for the Indonesia XGBoost experiments.

The public surface intentionally mirrors scripts/OpenDengue/XGBoost/utils.py
so both datasets can be driven by the same tune / train / eval entry points
with only the config YAML changing:

    load_data(cfg)                  → pd.DataFrame
    build_feature_columns(df, cfg)  → list[str]
    split_train_test(df, variable_columns, cfg, region=None)
                                    → (df_train_val, df_test)
    calculate_sample_weights(y)     → np.ndarray
    resolve_paths(cfg)              → dict[str, Path]
    validation_strategy(cfg)        → 'walk' | 'kfold'
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
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Paths  (mirrors resolve_paths in OpenDengue utils)
# ---------------------------------------------------------------------------

def find_project_root(start: Path, marker: str = ".git") -> Path:
    """Walk up from *start* until a directory containing *marker* is found."""
    for parent in start.resolve().parents:
        if (parent / marker).exists():
            return parent
    return start.resolve()


def resolve_paths(cfg: dict) -> dict[str, Path]:
    """
    Return a dict of canonical project paths derived from the git root.

    Keys mirror the OpenDengue version so calling code stays consistent:
        processed_dir, models_dir, figures_dir, tables_dir, sqlite_dir
    """
    root = find_project_root(Path(__file__))
    reports = root / "reports"
    return {
        "project_root":  root,
        "raw_dir":       root / "data" / "raw",
        "interim_dir":   root / "data" / "interim",
        "processed_dir": root / "data" / "processed",
        "external_dir":  root / "data" / "external",
        "models_dir":    root / "models" / "xgboost",
        "figures_dir":   reports / "figures",
        "tables_dir":    reports / "tables",
        "sqlite_dir":    root / "sqlite",
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: dict) -> pd.DataFrame:
    """
    Load and prepare the Indonesia dengue CSV.

    Applies:
      - Risk_Category label encoding (classifier only)
      - datetime conversion for the time column
      - Region_Group reassignment if needed (regressor dataset)
      - lag-1 Incidence_Rate feature (when epidemic_vars is non-empty)
      - lag features for env_vars and climate_vars
    """
    paths = resolve_paths(cfg)
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]

    csv_path = paths["processed_dir"] / data_cfg["processed_file"]
    df = pd.read_csv(csv_path)

    time_col   = data_cfg["time_column"]
    unit_col   = data_cfg["unit_column"]
    region_col = data_cfg.get("region_column", "Region_Group")

    # Region reassignment (regressor dataset uses a different CSV that
    # has 'Region' instead of 'Region_Group')
    if region_col not in df.columns and "Region" in df.columns:
        df[region_col] = df["Region"].replace(
            {"Maluku Islands": "Maluku-Papua", "Papua": "Maluku-Papua"}
        )

    df[time_col] = pd.to_datetime(df[time_col])

    # Encode classifier target
    target_cfg = cfg.get("target", {})
    if cfg.get("task") == "classifier" and "labels" in target_cfg:
        df[target_cfg["column"]] = (
            df[target_cfg["column"]]
            .replace(target_cfg["labels"])
            .infer_objects(copy=False)
            .astype("int32")
        )

    # Lag-1 Incidence_Rate (epidemic feature)
    if "Incidence_Rate_lag1" in feat_cfg.get("epidemic_vars", []):
        df["Incidence_Rate_lag1"] = df.groupby(unit_col)["Incidence_Rate"].shift(1)

    # Sort before creating lags
    df = df.sort_values([time_col, unit_col])

    # Lagged env + climate features
    lag_steps = feat_cfg.get("lag_steps", [1, 2, 3])
    for var in feat_cfg.get("env_vars", []) + feat_cfg.get("climate_vars", []):
        if var in df.columns:
            for lag in lag_steps:
                df[f"{var}_lag{lag}"] = df.groupby(unit_col)[var].shift(lag)

    return df


# ---------------------------------------------------------------------------
# Feature columns  (mirrors build_feature_columns in OpenDengue utils)
# ---------------------------------------------------------------------------

def build_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """
    Return the ordered list of feature columns that should be passed to the model.

    Follows the same inclusion logic as the original scripts:
      env_vars + climate_vars + epidemic_vars
      [+ land_use_vars if use_landuse]
      + all lag variants of env_vars and climate_vars
    """
    feat_cfg = cfg["features"]
    target   = cfg["target"]["column"]
    meta     = [
        cfg["data"]["time_column"],
        cfg["data"]["unit_column"],
        cfg["data"].get("region_column", "Region_Group"),
        "Incidence_Rate",
    ]

    cols: list[str] = []

    base_groups = (
        feat_cfg.get("env_vars", [])
        + feat_cfg.get("climate_vars", [])
        + feat_cfg.get("epidemic_vars", [])
    )
    for var in base_groups:
        if var in df.columns:
            cols.append(var)

    if feat_cfg.get("use_landuse", False):
        for var in feat_cfg.get("land_use_vars", []):
            if var in df.columns:
                cols.append(var)

    lag_steps = feat_cfg.get("lag_steps", [1, 2, 3])
    for var in feat_cfg.get("env_vars", []) + feat_cfg.get("climate_vars", []):
        for lag in lag_steps:
            lagged = f"{var}_lag{lag}"
            if lagged in df.columns:
                cols.append(lagged)

    # Deduplicate while preserving order; drop metadata / target columns
    seen: set[str] = set()
    result: list[str] = []
    excluded = set(meta) | {target}
    for c in cols:
        if c not in seen and c not in excluded:
            seen.add(c)
            result.append(c)

    return result


# ---------------------------------------------------------------------------
# Train / test split  (mirrors split_train_test in OpenDengue utils)
# ---------------------------------------------------------------------------

def split_train_test(
    df: pd.DataFrame,
    variable_columns: list[str],
    cfg: dict,
    region: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (df_train_val, df_test) filtered to *region* if supplied.

    Split strategy is driven by cfg["data"]["split_strategy"]:
      'year'   → rows with time_col.year == test_year go to df_test (classifier default)
      'random' → sklearn train_test_split (regressor default)
    """
    target       = cfg["target"]["column"]
    time_col     = cfg["data"]["time_column"]
    region_col   = cfg["data"].get("region_column", "Region_Group")
    split_strat  = cfg["data"].get("split_strategy", "year")

    if region is not None:
        df = df[df[region_col] == region].copy()

    df = df.dropna(subset=variable_columns + [target])

    if split_strat == "year":
        test_year = cfg["data"]["test_year"]
        df_train_val = df[df[time_col].dt.year < test_year].copy()
        df_test      = df[df[time_col].dt.year == test_year].copy()
    else:
        from sklearn.model_selection import train_test_split
        test_size    = cfg["data"].get("test_size", 0.2)
        rand_state   = cfg["training"].get("random_state", 64)
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=rand_state
        )

    return df_train_val, df_test


# ---------------------------------------------------------------------------
# Sample weights  (identical to OpenDengue version)
# ---------------------------------------------------------------------------

def calculate_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Inverse-frequency sample weights to handle class imbalance.
    Identical signature and behaviour to the OpenDengue utils version.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_cls = len(unique_classes)
    weight_map = {cls: total / (n_cls * cnt) for cls, cnt in zip(unique_classes, counts)}
    return np.array([weight_map[cls] for cls in y])


# ---------------------------------------------------------------------------
# Validation strategy helper  (mirrors OpenDengue utils)
# ---------------------------------------------------------------------------

def validation_strategy(cfg: dict) -> str:
    """Return 'walk' or 'kfold' from the config."""
    return cfg.get("validation", {}).get("strategy", "walk")


# ---------------------------------------------------------------------------
# CV splits
# ---------------------------------------------------------------------------

def build_cv_splits(
    df_sorted: pd.DataFrame,
    cfg: dict,
) -> list[tuple]:
    """
    Build cross-validation index pairs [(train_idx, val_idx), ...].

    Mirrors the split-building logic that was duplicated across the original scripts.
    Uses cfg['validation'] to choose walk-forward or k-fold.
    """
    val_cfg   = cfg.get("validation", {})
    strat     = val_cfg.get("strategy", "walk")
    time_col  = cfg["data"]["time_column"]
    rand_state = cfg["training"].get("random_state", 64)

    if strat == "walk":
        test_window = val_cfg.get("walk_test_window", 12)
        init_window = val_cfg.get("walk_init_window", 36)
        unique_months = sorted(df_sorted[time_col].unique())
        splits = []
        for i in range(init_window, len(unique_months) - test_window + 1, test_window):
            train_months = set(unique_months[:i])
            val_months   = set(unique_months[i : i + test_window])
            tr_idx  = df_sorted.index[df_sorted[time_col].isin(train_months)].tolist()
            val_idx = df_sorted.index[df_sorted[time_col].isin(val_months)].tolist()
            if tr_idx and val_idx:
                splits.append((tr_idx, val_idx))
        return splits

    else:  # kfold
        from sklearn.model_selection import KFold
        n_splits = val_cfg.get("kfold_n_splits", 5)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rand_state)
        return list(kf.split(df_sorted))
