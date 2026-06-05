"""
src/OpenDengue/XGBoost/train_xgboost.py
----------------------------------------
Snakemake Step 2 (per region): Train XGBoost with tuned hyperparams.

Wildcards : {study}, {region}
Reads     : reports/tables/{study}/{region}_params.csv  (via run_cfg["hyperparams"])
Writes    : models/xgboost/{study}/{region}.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import xgboost

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    _experiment_name,
    build_feature_columns,
    calculate_sample_weights,
    load_config,
    load_data,
    resolve_paths,
    split_train_test,
)

# ---------------------------------------------------------------------------
# Context: Snakemake (workflow) OR standalone CLI
# ---------------------------------------------------------------------------
try:
    cfg: dict      = snakemake.config             # noqa: F821
    run_cfg: dict  = cfg
    region: str    = snakemake.wildcards.region   # noqa: F821
    study: str     = _experiment_name(cfg)

    model_out = Path(snakemake.output.model)      # noqa: F821
    model_out.parent.mkdir(parents=True, exist_ok=True)

except NameError:
    _p = argparse.ArgumentParser(description="Train XGBoost — OpenDengue (standalone)")
    _p.add_argument("--config",  required=True, help="Path to experiment YAML config")
    _p.add_argument("--region",  required=True, help="Region / unit name (e.g. 'Thailand')")
    _p.add_argument("--study",   default=None,  help="Study name")
    _p.add_argument("--params",  required=True, help="Path to params CSV from tune step")
    _args = _p.parse_args()

    cfg     = load_config(_args.config)
    run_cfg = cfg
    region  = _args.region
    study   = _args.study or _experiment_name(cfg)

    cfg["use_landuse"] = cfg.get("use_landuse", False)

    _paths    = resolve_paths(cfg)
    model_out = _paths["models_dir"] / "xgboost" / study / f"{region}.json"
    model_out.parent.mkdir(parents=True, exist_ok=True)

    _row    = pd.read_csv(_args.params).iloc[0]
    run_cfg = {**cfg, "hyperparams": {k: v for k, v in _row.items() if k != "Region"}}

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
hp           = dict(run_cfg.get("hyperparams", {}))
rand_state   = cfg.get("training", {}).get("random_state", 64)
n_estimators = int(hp.pop("n_estimators", 200))

# int-cast params that Optuna emits as float via CSV round-trip
for _int_key in ("max_depth", "min_child_weight"):
    if _int_key in hp:
        hp[_int_key] = int(hp[_int_key])

# ---------------------------------------------------------------------------
# Load data, filter to region
# ---------------------------------------------------------------------------
df = load_data(cfg, run_cfg)
variable_columns = build_feature_columns(df, cfg)
target      = cfg.get("target_column") or cfg.get("target", {}).get("column", "IR")
num_classes = int(df[target].nunique())

df_train_val, _ = split_train_test(df, variable_columns, cfg, region=region)

X_train = df_train_val[variable_columns]
y_train = df_train_val[target].values.flatten()

print(f"[train_xgb:{region}] {len(X_train)} samples | {len(variable_columns)} features | n_estimators={n_estimators}")
print(f"[train_xgb:{region}] Hyperparams: {hp}")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
params = {
    "objective":   "multi:softprob" if num_classes > 2 else "binary:logistic",
    "tree_method": "hist",
    "device":      "cuda",
    "seed":        rand_state,
    **hp,
}
if num_classes > 2:
    params["num_class"] = num_classes

sw     = calculate_sample_weights(y_train)
dtrain = xgboost.DMatrix(X_train, label=y_train, weight=sw)

model = xgboost.train(params, dtrain, num_boost_round=n_estimators)
model.save_model(str(model_out))
print(f"[train_xgb:{region}] Saved → {model_out}")
