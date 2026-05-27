from __future__ import annotations

import pandas as pd


def load_data(cfg: dict) -> pd.DataFrame:
    """Read CSV, filter to relevant columns, and sort by node and time."""
    keep_cols = (
        ["admin", "name", "Date", cfg.get("target_column")]
        + cfg.get("env_vars", [])
        + cfg.get("lulc_vars", [])
    )

    df = pd.read_csv(str(cfg.get("data_path")))
    df = df.filter(items=keep_cols)
    df = df.sort_values(["name", "Date"]).reset_index(drop=True)
    return df


def build_node_index(df: pd.DataFrame) -> dict:
    """Map node names to integer indices."""
    node_names = df["name"].unique().tolist()
    node_name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    df["node_idx"] = df["name"].map(node_name_to_idx)
    return node_name_to_idx