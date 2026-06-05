"""
scripts/tune_RF.py
------------------
Snakemake Step 1 (per region): RandomizedSearchCV hyperparameter search for RF.

Wildcards : {study}, {region}
Reads     : reports/tables/{study}_regions.txt
Writes    : reports/tables/{study}/{region}_rf_params.csv

If run_cfg contains a 'hyperparams' key, search is skipped and those
fixed values are written directly to the params CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

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
    cfg: dict       = snakemake.config             # noqa: F821
    run_cfg: dict   = cfg
    region: str     = snakemake.wildcards.region   # noqa: F821
    study: str      = _experiment_name(cfg)

    params_csv = Path(snakemake.output.params_csv)  # noqa: F821
    params_csv.parent.mkdir(parents=True, exist_ok=True)

except NameError:
    # --- Standalone CLI mode ---
    _p = argparse.ArgumentParser(description="RandomizedSearch tune — OpenDengue RF (standalone)")
    _p.add_argument("--config",   required=True, help="Path to experiment YAML config")
    _p.add_argument("--region",   required=True, help="Region / unit name (e.g. 'Thailand')")
    _p.add_argument("--study",    default=None,  help="Study name (defaults to config name)")
    _p.add_argument("--n-trials", type=int, default=None)
    _args = _p.parse_args()

    cfg     = load_config(_args.config)
    run_cfg = cfg
    region  = _args.region
    study   = _args.study or _experiment_name(cfg)

    cfg["use_landuse"] = cfg.get("use_landuse", False)

    _paths     = resolve_paths(cfg)
    params_csv = _paths["tables_dir"] / study / f"{region}_rf_params.csv"
    params_csv.parent.mkdir(parents=True, exist_ok=True)

    if _args.n_trials:
        cfg.setdefault("optuna", {})["n_trials"] = _args.n_trials

n_iter     = cfg.get("optuna", {}).get("n_trials", 40)
rand_state = cfg.get("training", {}).get("random_state", 64)

# ---------------------------------------------------------------------------
# Load data, filter to region
# ---------------------------------------------------------------------------
df = load_data(cfg, run_cfg)
variable_columns = build_feature_columns(df, cfg)
target = cfg.get("target_column") or cfg.get("target", {}).get("column", "IR")

df_train_val, _ = split_train_test(df, variable_columns, cfg, region=region)

X_train = df_train_val[variable_columns].values
y_train = df_train_val[target].values.flatten()
sample_weights = calculate_sample_weights(y_train)

print(f"[tune_rf:{region}] {len(X_train)} samples | n_iter={n_iter}")

# ---------------------------------------------------------------------------
# If hyperparams are fixed, skip search and write directly
# ---------------------------------------------------------------------------
fixed = run_cfg.get("hyperparams")
if fixed:
    hp = dict(fixed)
    print(f"[tune_rf:{region}] Fixed hyperparams from run_cfg — skipping search")
    pd.DataFrame([{"Region": region, **hp}]).to_csv(params_csv, index=False)
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
param_dist = {
    "n_estimators":      [100, 200, 300, 500, 750, 1000],
    "max_depth":         [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
    "max_features":      ["sqrt", "log2", 0.3, 0.5],
}

base = RandomForestClassifier(n_jobs=-1, random_state=rand_state)
search = RandomizedSearchCV(
    base,
    param_distributions=param_dist,
    n_iter=n_iter,
    cv=3,
    scoring="f1_macro",
    refit=False,
    random_state=rand_state,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train, sample_weight=sample_weights)

best = search.best_params_
best_score = search.best_score_
print(f"[tune_rf:{region}] Best f1_macro={best_score:.4f} | params={best}")

# ---------------------------------------------------------------------------
# Write params CSV
# ---------------------------------------------------------------------------
pd.DataFrame([{"Region": region, **best}]).to_csv(params_csv, index=False)
print(f"[tune_rf:{region}] Params → {params_csv}")
