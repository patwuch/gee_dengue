"""
scripts/train_RF.py
-------------------
Snakemake Step 1 (per region): Train a Random Forest classifier.

Hyperparameters are read from run_cfg['hyperparams'] in config.yaml.
If absent, sklearn defaults are used.

Wildcards : {study}, {region}
Reads     : reports/tables/{study}_regions.txt
Writes    : models/rf/{study}/{region}.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
# Context: Snakemake (workflow) OR standalone CLI
# ---------------------------------------------------------------------------
try:
    cfg: dict       = snakemake.config             # noqa: F821
    run_cfg: dict   = snakemake.params.run_cfg     # noqa: F821
    region: str     = snakemake.wildcards.region   # noqa: F821
    study: str      = snakemake.wildcards.study    # noqa: F821

    cfg["use_landuse"] = run_cfg.get("landuse") == "t"

    model_out = Path(snakemake.output.model)  # noqa: F821
    model_out.parent.mkdir(parents=True, exist_ok=True)

    _n_jobs = snakemake.threads  # noqa: F821

except NameError:
    # --- Standalone CLI mode ---
    _p = argparse.ArgumentParser(description="Train Random Forest — OpenDengue (standalone)")
    _p.add_argument("--config",  required=True, help="Path to experiment YAML config")
    _p.add_argument("--region",  required=True, help="Region / unit name (e.g. 'Thailand')")
    _p.add_argument("--study",   default=None,  help="Study name")
    _p.add_argument("--params",  default=None,
                    help="Path to params CSV from tune_RF.py. "
                         "If omitted, uses hyperparams from config or sklearn defaults.")
    _p.add_argument("--n-jobs",  type=int, default=-1)
    _args = _p.parse_args()

    cfg     = load_config(_args.config)
    run_cfg = cfg
    region  = _args.region
    study   = _args.study or cfg.get("experiment", {}).get("name", "opendengue")
    _n_jobs = _args.n_jobs

    cfg["use_landuse"] = cfg.get("use_landuse", False)

    _paths    = resolve_paths(cfg)
    model_out = _paths["models_dir"] / "rf" / study / f"{region}.joblib"
    model_out.parent.mkdir(parents=True, exist_ok=True)

    # Load hyperparams from params CSV if provided
    if _args.params:
        _row = pd.read_csv(_args.params).iloc[0]
        _hp_override = {k: v for k, v in _row.items() if k != "Region"}
        run_cfg = {**cfg, "hyperparams": _hp_override}

# ---------------------------------------------------------------------------
# Hyperparameters — from run_cfg or sklearn defaults
# ---------------------------------------------------------------------------
hp = dict(run_cfg.get("hyperparams", {}))
print(f"[train_rf:{region}] Hyperparams: {hp if hp else 'sklearn defaults'}")

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

print(f"[train_rf:{region}] {len(X_train)} samples, {len(variable_columns)} features")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
model = RandomForestClassifier(
    **hp,
    n_jobs=_n_jobs,
    random_state=cfg.get("training", {}).get("random_state", 64),
)
model.fit(X_train, y_train, sample_weight=sample_weights)

joblib.dump(model, model_out)
print(f"[train_rf:{region}] Saved model → {model_out}")
