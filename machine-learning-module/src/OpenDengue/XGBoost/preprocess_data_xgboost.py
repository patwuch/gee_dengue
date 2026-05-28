from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess CSV → Pytorch tensor files for XGBoost")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_transform = cfg["log_transform"]
    test_months   = cfg["split"]["test_months"]
    val_fraction  = cfg["split"]["val_fraction"]

    keep_cols = ["admin", "name", "Date", cfg.get("target_column")] + cfg.get("env_vars") + cfg.get("lulc_vars")

    df = pd.read_csv(cfg.get("data_path"))
    print(df.columns)
    df = df.filter(items=keep_cols)
    print("after filter:", df.columns.tolist())
    df = df.sort_values(["name", "Date"]).reset_index(drop=True)
    # Add lags for env_vars if specified
    if cfg.get("add_lag_IR", False):
        df = (df.groupby("name", group_keys=False)
              .apply(lambda g: _add_lags(g, columns=[cfg.get("target_column")], lags=[1]))
            )
    if cfg.get("log_transform", False):
        df[cfg.get("target_column")] = np.log1p(df[cfg.get("target_column")])

    print(df.columns)
    print(df.sample(10))
    

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _add_lags(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Add lagged columns for each variable×lag pair within one node's time series."""
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df



if __name__ == "__main__":
    main()