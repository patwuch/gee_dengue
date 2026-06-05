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

def create_windows(tensors: dict, window_size: int, node_index: dict, cfg: dict) -> dict:
    snapshots = {}

    for split, data in tensors.items():
        inc  = data["inc"].clone()
        inc[torch.isnan(inc)] = 0.0

        parts = [inc, data["env"], data["lulc"]]
        if data.get("quality") is not None:
            parts.append(data["quality"])

        x    = torch.cat(parts, dim=-1)
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