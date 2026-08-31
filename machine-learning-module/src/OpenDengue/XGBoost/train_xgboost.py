"""
src/OpenDengue/XGBoost/train_xgboost.py
-----------------------------------------
Snakemake Step 2 (per region): Train an XGBoost regressor with tuned hyperparams.

Wildcards : {region}
Reads     : results/XGBoost/{name}/{region}/params.csv
Writes    : results/XGBoost/{name}/{region}/model.json
            results/XGBoost/{name}/{region}/preprocessing.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import xgboost

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    _experiment_name,
    build_feature_columns,
    fit_target_preprocessing,
    apply_target_preprocessing,
    load_config,
    load_data,
    resolve_paths,
    split_train_test,
)

# ---------------------------------------------------------------------------
# Context: Snakemake (workflow) OR standalone CLI
# ---------------------------------------------------------------------------
try:
    cfg: dict   = snakemake.config             # noqa: F821
    region: str = snakemake.wildcards.region   # noqa: F821
    study: str  = _experiment_name(cfg)

    model_out  = Path(snakemake.output.model)  # noqa: F821
    params_csv = Path(snakemake.input.params_csv)  # noqa: F821
    model_out.parent.mkdir(parents=True, exist_ok=True)

except NameError:
    _p = argparse.ArgumentParser(description="Train XGBoost regressor — OpenDengue (standalone)")
    _p.add_argument("--config",  required=True, help="Path to experiment YAML config")
    _p.add_argument("--region",  required=True, help="Region / unit name (e.g. 'Thailand')")
    _p.add_argument("--params",  required=True, help="Path to params CSV from tune step")
    _args = _p.parse_args()

    cfg    = load_config(_args.config)
    region = _args.region
    study  = _experiment_name(cfg)

    _paths     = resolve_paths(cfg)
    model_out  = _paths["results_dir"] / study / region / "model.json"
    params_csv = Path(_args.params)
    model_out.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
_row         = pd.read_csv(params_csv).iloc[0]
hp           = {k: v for k, v in _row.items() if k != "Region"}
device       = cfg.get("tune", {}).get("device", "cuda")
rand_state   = cfg.get("training", {}).get("random_state", 64)
n_estimators = int(hp.pop("n_estimators", 200))

for _int_key in ("max_depth", "min_child_weight"):
    if _int_key in hp:
        hp[_int_key] = int(hp[_int_key])

# ---------------------------------------------------------------------------
# Load data, filter to region, fit + apply target preprocessing on train data
# ---------------------------------------------------------------------------
df = load_data(cfg)
variable_columns = build_feature_columns(df, cfg)
target = cfg.get("target_column") or cfg.get("target", {}).get("column", "IR")

df_train_val, _ = split_train_test(df, variable_columns, cfg, region=region)

artifacts = fit_target_preprocessing(df_train_val, cfg)
df_train_val = apply_target_preprocessing(df_train_val, cfg, artifacts)

X_train = df_train_val[variable_columns]
y_train = df_train_val[target].values.flatten()

print(f"[train_xgb:{region}] {len(X_train)} samples | {len(variable_columns)} features | n_estimators={n_estimators}")
print(f"[train_xgb:{region}] Hyperparams: {hp}")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
params = {
    "objective":   "reg:squarederror",
    "tree_method": "hist",
    "device":      device,
    "seed":        rand_state,
    **hp,
}

dtrain = xgboost.DMatrix(X_train, label=y_train)
model = xgboost.train(params, dtrain, num_boost_round=n_estimators)
model.save_model(str(model_out))
print(f"[train_xgb:{region}] Saved model → {model_out}")

# ---------------------------------------------------------------------------
# Persist preprocessing artifacts alongside the model (needed by eval step)
# ---------------------------------------------------------------------------
prep_path = model_out.with_name("preprocessing.pkl")
with open(prep_path, "wb") as f:
    pickle.dump(artifacts, f)
print(f"[train_xgb:{region}] Saved preprocessing artifacts → {prep_path}")
