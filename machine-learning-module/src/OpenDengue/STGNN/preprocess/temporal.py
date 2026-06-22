from __future__ import annotations

import numpy as np
import pandas as pd
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


def temporal_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df chronologically into train, val, and test by date."""
    test_months = cfg["data"]["split"]["test_months"]
    val_frac    = cfg["data"]["split"]["val_fraction"]
    date_col    = cfg["data"]["time_column"]

    all_dates   = sorted(df[date_col].unique())
    test_cutoff = all_dates[-test_months]

    train_dates = [d for d in all_dates if d < test_cutoff]
    val_cutoff  = train_dates[-int(len(train_dates) * val_frac)]

    train_df = df[df[date_col] < val_cutoff].copy()
    val_df   = df[(df[date_col] >= val_cutoff) & (df[date_col] < test_cutoff)].copy()
    test_df  = df[df[date_col] >= test_cutoff].copy()

    return train_df, val_df, test_df

def _build_feature_tensor(data: dict) -> torch.Tensor:
    """Concatenate inc (NaN→0), env, lulc, quality into (T, N, F)."""
    inc = data["inc"].clone()
    inc[torch.isnan(inc)] = 0.0
    parts = [inc, data["env"], data["lulc"]]
    if data.get("quality") is not None:
        parts.append(data["quality"])
    return torch.cat(parts, dim=-1)


def create_windows(tensors: dict, window_size: int, node_index: dict, cfg: dict) -> dict:
    snapshots = {}

    for split, data in tensors.items():
        x    = _build_feature_tensor(data)
        inc  = data["inc"].clone()
        inc[torch.isnan(inc)] = 0.0
        mask = data["mask"]

        if window_size >= len(x):
            raise ValueError(
                f"window_size {window_size} >= number of timesteps {len(x)} "
                f"in {split} split."
            )

        snapshots[split] = [
            (x[t-window_size:t], inc[t], mask[t])
            for t in range(window_size, len(x))
        ]

    return snapshots


def create_inference_windows(tensors: dict, window_size: int) -> list:
    """Build windows that cover all test months.

    The first window's context is drawn from the tail of the train+val data,
    so every test month gets a prediction (not just the last len(test)-window_size).
    """
    pre_x = torch.cat(
        [_build_feature_tensor(tensors["train"]),
         _build_feature_tensor(tensors["val"])],
        dim=0,
    )[-window_size:]                              # (window_size, N, F)

    test_x    = _build_feature_tensor(tensors["test"])   # (T_test, N, F)
    test_inc  = tensors["test"]["inc"].clone()
    test_inc[torch.isnan(test_inc)] = 0.0
    test_mask = tensors["test"]["mask"]

    full_x = torch.cat([pre_x, test_x], dim=0)   # (window_size + T_test, N, F)
    T_test = len(test_x)

    return [
        (full_x[t - window_size:t], test_inc[t - window_size], test_mask[t - window_size])
        for t in range(window_size, window_size + T_test)
    ]