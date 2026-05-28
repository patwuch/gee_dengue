from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def log_transform(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply log1p to the target column in-place."""
    col = cfg.get("target_column")
    df[col] = np.log1p(df[col])
    return df


def add_lags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add lagged columns for the target column, grouped by node."""
    col  = cfg.get("target_column")
    lags = cfg.get("lags", [1])
    df = (
        df.groupby("name", group_keys=False)
          .apply(lambda g: _add_lags(g, columns=[col], lags=lags))
    )
    return df


def _add_lags(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Add lagged columns for each variable×lag pair within one node's time series."""
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def fit_scaler(x_train: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    return scaler, x_train_scaled

def apply_scaler(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(x)

def deseasonalise_target(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Subtract monthly means from the target column."""
    col = cfg.get("target_column")
    df[col] = df[col] - df.groupby(df["Date"].dt.month)[col].transform("mean")
    return df