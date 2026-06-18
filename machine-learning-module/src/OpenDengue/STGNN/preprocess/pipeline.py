import argparse
import pandas as pd
import torch
from utils import load_config, save_tensors, save_scaler, save_edge_index, get_window_sizes
from dataset import load_data, build_node_index
from features import log_transform, separate_sources, build_masks, fill_missing_inc, reshape_all, deseasonalise_target
from features import fill_node_from_donors, impute_env_naive, assert_no_nans, encode_quality_flags
from features import diagnose_feature_composition, scale_sources
from graph import build_edge_index
from temporal import create_windows, temporal_split


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess CSV → Pytorch tensor files for Graph Learning"
    )
    p.add_argument("--config", required=True, help="Path to config.yaml")
    return p.parse_args()


def main(config_path: str):
    cfg  = load_config(config_path)
    prep = cfg.get("preprocessing", {})

    df = load_data(cfg)
    print("Data loaded.")
    print(df["data_quality"].value_counts(dropna=False))

    # ── Sanity checks ────────────────────────────────────────────────────────
    node_col = cfg["data"]["unit_column"]
    time_col = cfg["data"]["time_column"]
    target   = cfg["target_column"]

    dupes = df.groupby([time_col, node_col]).size()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        print(f"WARNING: {len(dupes)} duplicate (time, node) pairs found:")
        print(dupes.to_string())
    else:
        print("No duplicates found.")

    # ── Diagnose raw NaNs (excluding known Labuan issue) ─────────────────────
    env_cols  = [c for c in df.columns if c not in [
        node_col, time_col, target,
        "dengue_total", "population_sum", "IR", "IR_quality",
        "data_quality", "adm_0_name",
    ]]
    labuan_df = df[df[node_col] != "W.P. Labuan"]
    for col in env_cols:
        n_nan = labuan_df[col].isna().sum()
        if n_nan > 0:
            bad = labuan_df[labuan_df[col].isna()][[node_col, time_col, col]]
            print(f"\n{col}: {n_nan} NaNs")
            print(f"  Nodes affected: {bad[node_col].unique().tolist()}")
            print(f"  Time range: {bad[time_col].min()} → {bad[time_col].max()}")
            print(f"  Consecutive? {bad[time_col].nunique() == n_nan // bad[node_col].nunique()}")

    # ── Standard preprocessing ───────────────────────────────────────────────
    # NOTE: log_transform and deseasonalise run before the split intentionally:
    #   - NaNs are still present here, so deseasonalise correctly excludes them
    #     from monthly mean calculation rather than treating them as zeros.
    #   - fillna(0) on incidence happens later in fill_missing_inc, after
    #     build_masks has already captured NaN positions.
    df = log_transform(df, cfg)        if prep.get("log_transform")  else df
    print("Log transform applied.")
    df = deseasonalise_target(df, cfg) if prep.get("deseasonalise")  else df
    print("Deseasonalisation applied.")

    # ── Encode categorical quality flags ─────────────────────────────────────
    quality_vars = cfg.get("features", {}).get("quality_vars", [])
    if quality_vars:
        df, quality_dummies = encode_quality_flags(df, quality_vars)
        cfg["features"]["quality_dummy_vars"] = quality_dummies
        print(f"Quality flags encoded: {quality_dummies}")

    node_index = build_node_index(df, cfg)
    print("Node index built.")
    edge_index = build_edge_index(node_index)
    print("Edge index built.")
    save_edge_index(edge_index, cfg)

    train_df, val_df, test_df = temporal_split(df, cfg)
    sources = separate_sources(train_df, val_df, test_df, cfg)

    # ── Incidence NaN handling ───────────────────────────────────────────────
    # Order matters: masks must be built before filling so that NaN positions
    # are captured while still present, then zeros are filled explicitly before
    # scaling so scale_sources can assume clean input.
    masks   = build_masks(sources)
    sources = fill_missing_inc(sources)
    scaled  = scale_sources(sources)

    # Write scaled values back into the DataFrames so reshape_all sees scaled data.
    # scale_sources operates on flat numpy arrays; we put them back in place here.
    env_vars = cfg.get("features", {}).get("env_vars", [])
    for df_obj, split_key in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df_obj[target]   = scaled["inc"][split_key].reshape(-1)
        df_obj[env_vars] = scaled["env"][split_key]

    tensors = reshape_all(masks, train_df, val_df, test_df, node_index, cfg)

    # ── Naive imputation strategy ────────────────────────────────────────────
    # Stage 1: Labuan filled — residual NaNs expected (LST_Night etc.)
    tensors = fill_node_from_donors(
        tensors,
        node_index,
        receiver = "W.P. Labuan",
        donors   = ["Sabah", "Brunei And Muara", "Belait", "Temburong", "Tutong"],
    )
    assert_no_nans(tensors, stage="post-labuan-fill", keys=["env", "lulc"],
                raise_on_fail=False)   # ← warn only

    # Stage 2: ffill → bfill → global mean for remaining sparse NaNs
    tensors = impute_env_naive(tensors, node_index, cfg)
    assert_no_nans(tensors, stage="post-imputation",  keys=["env", "lulc"],
                raise_on_fail=True)    # ← hard stop

    # ── Window creation ──────────────────────────────────────────────────────
    for window_size in get_window_sizes(cfg):
        snapshots = create_windows(tensors, window_size, node_index, cfg)

        if window_size == get_window_sizes(cfg)[0]:   # only check first window size
            diagnose_feature_composition(snapshots, cfg)

        save_tensors(snapshots, cfg, window_size)

    save_scaler(scaled["scalers"], cfg)

if "snakemake" in dir():
    main(config_path=snakemake.params.cfg)
elif __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)