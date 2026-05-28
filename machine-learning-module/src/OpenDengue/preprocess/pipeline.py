from __future__ import annotations

import argparse

from utils import load_config, save_tensors
from dataset import load_data, build_node_index
from features import log_transform, add_lags, fit_scaler, apply_scaler, deseasonalise_target
from graph import build_edge_index, add_self_loops
from temporal import build_snapshot, create_windows, temporal_split


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess CSV → Pytorch tensor files for Graph Learning"
    )
    p.add_argument("--config", required=True, help="Path to config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    df             = load_data(cfg)
    print("Data loaded.")
    df             = add_lags(df, cfg) if cfg.get("add_lag_IR") else df
    print("Lags added.")
    df             = log_transform(df, cfg) if cfg.get("log_transform") else df
    print("Log transform applied.")
    df             = deseasonalise_target(df, cfg) if cfg.get("deseasonalise") else df
    print("Deseasonalisation applied.")
    node_index     = build_node_index(df)
    edge_index     = add_self_loops(build_edge_index(df, cfg))


    x_train, y_train, x_val, y_val, x_test, y_test = temporal_split(df, cfg)
    scaler, x_train                  = fit_scaler(x_train)
    x_val                            = apply_scaler(x_val, scaler)
    x_test                           = apply_scaler(x_test, scaler)

    for window_size in cfg["sweep"]["window_sizes"]:
        snapshots = create_windows(
            x_train, y_train,
            x_val,   y_val,
            x_test,  y_test,
            window_size
        )
        save_tensors(snapshots, cfg, window_size)
    save_scaler(scaler, cfg)



if __name__ == "__main__":
    main()