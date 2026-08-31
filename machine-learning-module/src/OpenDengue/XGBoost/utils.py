"""
scripts/OpenDengue/XGBoost/utils.py
--------------------------------------
Shared helpers for the OpenDengue XGBoost (and Random Forest) experiments.

Supports both the legacy flat config schema and the current nested schema:
  - data_path (flat)  →  data.path (nested)
  - env_vars (flat)   →  features.env_vars (nested)
  - lulc_vars (flat)  →  features.land_use_vars (nested)
  - use_landuse       →  features.use_landuse
  - lag_steps         →  preprocessing.lag_steps
  - add_lag_IR        →  preprocessing.add_lag_target
  - time_column       →  data.time_column
  - unit_column       →  data.unit_column
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Schema helpers (support flat legacy keys and new nested keys)
# ---------------------------------------------------------------------------

def _data_path(cfg: dict, run_cfg: dict | None = None) -> str:
    run_cfg = run_cfg or {}
    candidate = run_cfg.get("data") or cfg.get("data")
    if isinstance(candidate, dict):
        return candidate.get("path") or cfg.get("data_path", "")
    return candidate or cfg.get("data_path", "")


def _time_col(cfg: dict) -> str:
    return cfg.get("time_column") or cfg.get("data", {}).get("time_column", "Date")


def _unit_col(cfg: dict) -> str:
    return cfg.get("unit_column") or cfg.get("data", {}).get("unit_column", "name")


def _admin_col(cfg: dict) -> str:
    return cfg.get("admin_column") or cfg.get("data", {}).get("admin_column", "admin")


def _region_col(cfg: dict) -> str:
    """Column the {region} wildcard filters on (e.g. country for multi-country scope).

    Falls back to unit_column when no explicit region_column is configured, matching
    the single-level scope used by earlier (single-country) experiments.
    """
    return (cfg.get("region_column")
            or cfg.get("data", {}).get("region_column")
            or _unit_col(cfg))


def _target_col(cfg: dict) -> str:
    return cfg.get("target_column") or cfg.get("target", {}).get("column", "IR")


def _env_vars(cfg: dict) -> list[str]:
    return cfg.get("env_vars") or cfg.get("features", {}).get("env_vars", [])


def _land_use_vars(cfg: dict) -> list[str]:
    return cfg.get("lulc_vars") or cfg.get("features", {}).get("land_use_vars", [])


def _use_landuse(cfg: dict) -> bool:
    return cfg.get("use_landuse", cfg.get("features", {}).get("use_landuse", False))


def _lag_steps(cfg: dict) -> list[int]:
    return cfg.get("lag_steps") or cfg.get("preprocessing", {}).get("lag_steps", [1, 2, 3])


def _add_lag_target(cfg: dict) -> bool:
    return (cfg.get("add_lag_IR", False)
            or cfg.get("preprocessing", {}).get("add_lag_target", False))


def _experiment_name(cfg: dict) -> str:
    return (cfg.get("name")
            or cfg.get("experiment", {}).get("name", "opendengue"))


def _split_cfg(cfg: dict) -> dict:
    return cfg.get("split") or cfg.get("data", {}).get("split", {})


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _find_project_root(start: Path, marker: str = ".git") -> Path:
    for parent in start.resolve().parents:
        if (parent / marker).exists():
            return parent
    return start.resolve()


def _find_module_root(start: Path, marker: str = "Snakefile") -> Path:
    for parent in start.resolve().parents:
        if (parent / marker).exists():
            return parent
    return start.resolve()


def resolve_paths(cfg: dict) -> dict[str, Path]:
    root    = _find_module_root(Path(__file__))
    reports = root / "reports"
    defaults = {
        "project_root":  root,
        "processed_dir": root / "data" / "processed",
        "models_dir":    root / "models",
        "figures_dir":   reports / "figures",
        "tables_dir":    reports / "tables",
        "results_dir":   root / "results" / "XGBoost",
    }
    overrides = cfg.get("paths", {})
    return {k: Path(overrides.get(k, v)) for k, v in defaults.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: dict, run_cfg: dict | None = None) -> pd.DataFrame:
    run_cfg   = run_cfg or {}
    data_path = _data_path(cfg, run_cfg)
    if not data_path:
        raise ValueError("data path not set in config (data.path or data_path)")

    data_path = Path(data_path)
    if not data_path.is_absolute() and not data_path.exists():
        data_path = _find_project_root(Path(__file__)) / data_path

    df = pd.read_csv(data_path)

    time_col = _time_col(cfg)
    unit_col = _unit_col(cfg)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([unit_col, time_col]).reset_index(drop=True)

    target_col = _target_col(cfg)
    if _add_lag_target(cfg):
        df[f"{target_col}_lag1"] = df.groupby(unit_col)[target_col].shift(1)

    lag_steps = _lag_steps(cfg)
    for var in _env_vars(cfg):
        if var in df.columns:
            for lag in lag_steps:
                df[f"{var}_lag{lag}"] = df.groupby(unit_col)[var].shift(lag)

    label_map = cfg.get("target", {}).get("labels")
    if label_map:
        df[target_col] = df[target_col].replace(label_map).infer_objects(copy=False)

    return df


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

def build_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    use_landuse = _use_landuse(cfg)
    lag_steps   = _lag_steps(cfg)
    target_col  = _target_col(cfg)
    time_col    = _time_col(cfg)
    unit_col    = _unit_col(cfg)
    admin_col   = _admin_col(cfg)
    excluded    = {target_col, time_col, unit_col, admin_col}

    cols: list[str] = []

    for var in _env_vars(cfg):
        if var in df.columns:
            cols.append(var)

    if use_landuse:
        for var in _land_use_vars(cfg):
            if var in df.columns:
                cols.append(var)

    for var in _env_vars(cfg):
        for lag in lag_steps:
            lagged = f"{var}_lag{lag}"
            if lagged in df.columns:
                cols.append(lagged)

    if _add_lag_target(cfg):
        lag_col = f"{target_col}_lag1"
        if lag_col in df.columns:
            cols.append(lag_col)

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
    time_col   = _time_col(cfg)
    region_col = _region_col(cfg)
    target_col = _target_col(cfg)

    if region is not None:
        df = df[df[region_col] == region].copy()

    df = df.dropna(subset=variable_columns + [target_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    test_months  = _split_cfg(cfg).get("test_months", 12)
    unique_dates = sorted(df[time_col].unique())
    cutoff       = unique_dates[-test_months] if len(unique_dates) > test_months else unique_dates[0]

    return df[df[time_col] < cutoff].copy(), df[df[time_col] >= cutoff].copy()


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def fit_target_preprocessing(df_train: pd.DataFrame, cfg: dict) -> dict:
    """Fit log/deseasonalise/scale transforms for a regression target on train data only."""
    prep       = cfg.get("preprocessing", {})
    target_col = _target_col(cfg)
    time_col   = _time_col(cfg)

    artifacts = {
        "log_transform": bool(prep.get("log_transform", False)),
        "deseasonalise": bool(prep.get("deseasonalise", False)),
        "scale_target":  bool(prep.get("scale_target", False)),
    }

    y = df_train[target_col].astype(float)
    if artifacts["log_transform"]:
        y = np.log1p(y)

    if artifacts["deseasonalise"]:
        months = pd.to_datetime(df_train[time_col]).dt.month
        artifacts["seasonal_means"] = y.groupby(months).mean().to_dict()
        artifacts["seasonal_mean_fallback"] = float(y.mean())
        y = y - months.map(artifacts["seasonal_means"]).fillna(artifacts["seasonal_mean_fallback"])

    if artifacts["scale_target"]:
        scaler = StandardScaler()
        scaler.fit(y.values.reshape(-1, 1))
        artifacts["scaler"] = scaler

    return artifacts


def apply_target_preprocessing(df: pd.DataFrame, cfg: dict, artifacts: dict) -> pd.DataFrame:
    """Transform the target column of `df` using previously fitted artifacts."""
    df = df.copy()
    target_col = _target_col(cfg)
    time_col   = _time_col(cfg)

    y = df[target_col].astype(float)
    if artifacts["log_transform"]:
        y = np.log1p(y)

    if artifacts["deseasonalise"]:
        months = pd.to_datetime(df[time_col]).dt.month
        y = y - months.map(artifacts["seasonal_means"]).fillna(artifacts["seasonal_mean_fallback"])

    if artifacts["scale_target"]:
        y = pd.Series(artifacts["scaler"].transform(y.values.reshape(-1, 1)).ravel(), index=y.index)

    df[target_col] = y
    return df


def invert_target(y: np.ndarray, months: pd.Series, artifacts: dict) -> np.ndarray:
    """Invert a prediction/target array back to the raw target scale."""
    y = np.asarray(y, dtype=float)

    if artifacts["scale_target"]:
        y = artifacts["scaler"].inverse_transform(y.reshape(-1, 1)).ravel()

    if artifacts["deseasonalise"]:
        y = y + months.map(artifacts["seasonal_means"]).fillna(artifacts["seasonal_mean_fallback"]).values

    if artifacts["log_transform"]:
        y = np.expm1(y)

    return y


def calculate_sample_weights(y: np.ndarray) -> np.ndarray:
    unique_classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_cls = len(unique_classes)
    wmap  = {cls: total / (n_cls * cnt) for cls, cnt in zip(unique_classes, counts)}
    return np.array([wmap[cls] for cls in y])


# ---------------------------------------------------------------------------
# Validation strategy
# ---------------------------------------------------------------------------

def validation_strategy(cfg: dict) -> str:
    strat = cfg.get("validation", cfg.get("strategy", "walk"))
    if isinstance(strat, dict):
        strat = strat.get("strategy", "walk")
    return strat
