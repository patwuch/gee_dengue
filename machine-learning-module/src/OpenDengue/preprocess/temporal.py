from __future__ import annotations

import numpy as np
import torch


def build_snapshot(
    t: int,
    window_size: int,
    incidence_norm: np.ndarray,
    monthly_norm: np.ndarray,
    yearly_expanded_norm: np.ndarray,
) -> torch.Tensor:
    """
    Stack three feature sources into a single node feature matrix [num_regions, num_features]
    at timestep t.
    """
    # Rolling incidence history: [num_regions, window_size]
    incidence_window = incidence_norm[t - window_size:t].T

    # Monthly env at time t: [num_regions, num_monthly_feats]
    monthly_feats = monthly_norm[t]

    # Yearly env at time t (already expanded): [num_regions, num_yearly_feats]
    yearly_feats = yearly_expanded_norm[t]

    x = np.hstack([incidence_window, monthly_feats, yearly_feats])
    return torch.tensor(x, dtype=torch.float)


def temporal_split(df, cfg: dict):
    target      = cfg["target_column"]
    test_months = cfg["split"]["test_months"]
    val_frac    = cfg["split"]["val_fraction"]

    # find the cutoff dates by counting back from the last date
    all_dates   = sorted(df["Date"].unique())
    test_cutoff = all_dates[-test_months]      # e.g. last 6 months → cutoff at month -6
    
    # find val cutoff within the remaining train dates
    train_dates = [d for d in all_dates if d < test_cutoff]
    val_cutoff  = train_dates[-int(len(train_dates) * val_frac)]

    # split by date
    train_df    = df[df["Date"] < val_cutoff]
    val_df      = df[(df["Date"] >= val_cutoff) & (df["Date"] < test_cutoff)]
    test_df     = df[df["Date"] >= test_cutoff]

    # separate x and y
    drop_cols   = ["name", "Date", "node_idx", "admin", target]

    x_train     = train_df.drop(columns=drop_cols).values
    y_train     = train_df[target].values

    x_val       = val_df.drop(columns=drop_cols).values
    y_val       = val_df[target].values

    x_test      = test_df.drop(columns=drop_cols).values
    y_test      = test_df[target].values

    return x_train, y_train, x_val, y_val, x_test, y_test

def create_windows(x_train, y_train, x_val, y_val, x_test, y_test, window_size):
    """
    Create rolling window snapshots for train/val/test sets.
    Each snapshot is a list of (x_window, y_window) pairs for each time step.
    """
    snapshots = {"train": [], "val": [], "test": []}
    for t in range(window_size, len(x_train)):
        snapshots["train"].append((x_train[t - window_size:t], y_train[t]))
    for t in range(window_size, len(x_val)):
        snapshots["val"].append((x_val[t - window_size:t], y_val[t]))
    for t in range(window_size, len(x_test)):
        snapshots["test"].append((x_test[t - window_size:t], y_test[t]))
    return snapshots