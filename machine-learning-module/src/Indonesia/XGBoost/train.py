"""
scripts/Indonesia/XGBoost/train.py
------------------------------------
Step 2: Train XGBoost on the full train+val split using tuned hyperparameters.

Reads  : config YAML  (--config)
         reports/tables/{experiment.name}/{region}_params.csv  (written by tune.py)
Writes : models/xgboost/{experiment.name}/{region}.json

Usage
-----
# National scope:
python scripts/Indonesia/XGBoost/train.py \\
    --config config/indonesia/xgboost_national_classifier.yaml

# Regional scope — single region:
python scripts/Indonesia/XGBoost/train.py \\
    --config config/indonesia/xgboost_regional_classifier.yaml \\
    --region Sumatera

# Regional scope — all regions:
python scripts/Indonesia/XGBoost/train.py \\
    --config config/indonesia/xgboost_regional_classifier.yaml

Mirrors the interface of scripts/OpenDengue/XGBoost/train_xgboost.py.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import xgboost

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_feature_columns,
    calculate_sample_weights,
    load_config,
    load_data,
    resolve_paths,
    split_train_test,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost — Indonesia")
    p.add_argument("--config", required=True, help="Path to experiment YAML config")
    p.add_argument("--region", default=None,
                   help="Region name (regional configs only). Omit to iterate all.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load hyperparameters from the params CSV written by tune.py
# ---------------------------------------------------------------------------

def _load_hyperparams(paths: dict, cfg: dict, region_label: str) -> tuple[dict, int]:
    """
    Read the params CSV for *region_label* and return (hyperparams_dict, n_estimators).
    Raises FileNotFoundError with a helpful message if tune.py has not been run yet.
    """
    study_name = cfg["experiment"]["name"]
    params_csv = paths["tables_dir"] / study_name / f"{region_label}_params.csv"

    if not params_csv.exists():
        raise FileNotFoundError(
            f"Params CSV not found: {params_csv}\n"
            f"Run tune.py first:\n"
            f"  python scripts/Indonesia/XGBoost/tune.py --config {cfg.get('_config_path', '<config>')}"
        )

    row = pd.read_csv(params_csv).iloc[0]

    hp: dict = {
        "gamma":            float(row["gamma"]),
        "max_depth":        int(row["max_depth"]),
        "reg_alpha":        float(row["reg_alpha"]),
        "subsample":        float(row["subsample"]),
        "reg_lambda":       float(row["reg_lambda"]),
        "learning_rate":    float(row["learning_rate"]),
        "colsample_bytree": float(row["colsample_bytree"]),
        "min_child_weight": int(row["min_child_weight"]),
    }
    n_estimators = int(row["n_estimators"])
    return hp, n_estimators


# ---------------------------------------------------------------------------
# Single-region train
# ---------------------------------------------------------------------------

def train_region(region: str | None, cfg: dict, paths: dict) -> None:
    region_label = region if region is not None else "National"
    study_name   = cfg["experiment"]["name"]
    task         = cfg.get("task", "classifier")
    device       = cfg["training"].get("device", "cuda")

    print(f"\n[train:{region_label}] Loading hyperparams …")
    hp, n_estimators = _load_hyperparams(paths, cfg, region_label)

    df               = load_data(cfg)
    variable_columns = build_feature_columns(df, cfg)
    target           = cfg["target"]["column"]

    df_train_val, _  = split_train_test(df, variable_columns, cfg, region=region)

    if df_train_val.empty:
        print(f"[train:{region_label}] No training data — skipping.")
        return

    X_train = df_train_val[variable_columns]
    y_train = df_train_val[target].values.flatten()

    print(f"[train:{region_label}] {len(X_train)} samples | {len(variable_columns)} features")

    # Build XGBoost params
    num_cls = int(pd.Series(y_train).nunique()) if task == "classifier" else None
    xgb_params: dict = {
        **hp,
        "tree_method": "hist",
        "device":      device,
    }
    if task == "classifier":
        if num_cls > 2:
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = num_cls
        else:
            xgb_params["objective"] = "binary:logistic"
    else:
        xgb_params["objective"] = "reg:squarederror"

    # Train
    if task == "classifier":
        sw     = calculate_sample_weights(y_train)
        dtrain = xgboost.DMatrix(X_train, label=y_train, weight=sw)
    else:
        dtrain = xgboost.DMatrix(X_train, label=y_train)

    model = xgboost.train(xgb_params, dtrain, num_boost_round=n_estimators)

    # Save model
    out_dir = paths["models_dir"] / study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{region_label}.json"
    model.save_model(str(model_path))
    print(f"[train:{region_label}] Saved model → {model_path}")

    del dtrain, model
    gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args  = parse_args()
    cfg   = load_config(args.config)
    cfg["_config_path"] = args.config   # used in error messages
    paths = resolve_paths(cfg)
    scope = cfg.get("scope", "national")

    if scope == "national":
        train_region(None, cfg, paths)
    elif args.region:
        train_region(args.region, cfg, paths)
    else:
        df      = load_data(cfg)
        regions = df[cfg["data"]["region_column"]].unique()
        for region in regions:
            train_region(region, cfg, paths)


if __name__ == "__main__":
    main()
