"""
src/OpenDengue/XGBoost/eval_xgboost.py
----------------------------------------
Snakemake Step 3 (per region): Regression metrics for a trained XGBoost model.

Wildcards : {region}
Reads     : results/XGBoost/{name}/{region}/model.json
            results/XGBoost/{name}/{region}/preprocessing.pkl  (written by train_xgboost.py)
Writes    : results/XGBoost/{name}/{region}/results.csv
            results/XGBoost/{name}/{region}/pred_vs_actual_{split}.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    _experiment_name,
    _time_col,
    build_feature_columns,
    invert_target,
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

    model_path  = Path(snakemake.input.model)         # noqa: F821
    results_csv = Path(snakemake.output.results_csv)  # noqa: F821
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    region_dir = results_csv.parent
    region_dir.mkdir(parents=True, exist_ok=True)

    prep_path = model_path.with_name("preprocessing.pkl")

except NameError:
    # --- Standalone CLI mode ---
    _p = argparse.ArgumentParser(description="Evaluate XGBoost regressor — OpenDengue (standalone)")
    _p.add_argument("--config",  required=True, help="Path to experiment YAML config")
    _p.add_argument("--region",  required=True, help="Region / unit name (e.g. 'Thailand')")
    _p.add_argument("--model",   default=None,  help="Override model .json path")
    _p.add_argument("--preprocessing", default=None, help="Override preprocessing .pkl path")
    _args = _p.parse_args()

    cfg    = load_config(_args.config)
    region = _args.region
    study  = _experiment_name(cfg)

    paths      = resolve_paths(cfg)
    region_dir = paths["results_dir"] / study / region
    region_dir.mkdir(parents=True, exist_ok=True)

    model_path  = Path(_args.model) if _args.model else region_dir / "model.json"
    prep_path   = Path(_args.preprocessing) if _args.preprocessing else model_path.with_name("preprocessing.pkl")
    results_csv = region_dir / "results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data, filter to region
# ---------------------------------------------------------------------------
df = load_data(cfg)
variable_columns = build_feature_columns(df, cfg)
target   = cfg.get("target_column") or cfg.get("target", {}).get("column", "IR")
time_col = _time_col(cfg)

df_train_val, df_test = split_train_test(df, variable_columns, cfg, region=region)

with open(prep_path, "rb") as f:
    artifacts = pickle.load(f)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = xgboost.Booster()
model.load_model(str(model_path))
print(f"[evaluate:{region}] Loaded model from {model_path}")

# ---------------------------------------------------------------------------
# Evaluate each split (metrics reported on the raw, inverse-transformed target)
# ---------------------------------------------------------------------------
datasets = {
    "Train_Val": df_train_val,
    "Test":      df_test,
}

all_results = []

for set_name, df_split in datasets.items():
    print(f"\n[evaluate:{region}] --- {set_name} ---")

    X_set = df_split[variable_columns]
    y_true_raw = df_split[target].values.flatten()
    months = pd.to_datetime(df_split[time_col]).dt.month

    dset       = xgboost.DMatrix(X_set)
    y_pred_raw = invert_target(model.predict(dset), months, artifacts)

    rmse = mean_squared_error(y_true_raw, y_pred_raw) ** 0.5
    mae  = mean_absolute_error(y_true_raw, y_pred_raw)
    r2   = r2_score(y_true_raw, y_pred_raw)

    print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | n={len(y_true_raw)}")

    # Predicted vs. actual scatter plot
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true_raw, y_pred_raw, alpha=0.5, s=12)
    lims = [min(y_true_raw.min(), y_pred_raw.min()), max(y_true_raw.max(), y_pred_raw.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual — {region} ({set_name})")
    fig_path = region_dir / f"pred_vs_actual_{set_name}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {fig_path}")

    all_results.append({
        "Region":  region,
        "Dataset": set_name,
        "N":       len(y_true_raw),
        "RMSE":    rmse,
        "MAE":     mae,
        "R2":      r2,
    })

# ---------------------------------------------------------------------------
# Save per-region CSV
# ---------------------------------------------------------------------------
pd.DataFrame(all_results).to_csv(results_csv, index=False, float_format="%.4f")
print(f"\n[evaluate:{region}] Saved results → {results_csv}")
