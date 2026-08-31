"""
src/OpenDengue/XGBoost/tune_xgboost.py
---------------------------------------
Snakemake Step 1 (per region): Optuna hyperparameter search for XGBoost regression.

Wildcards : {region}
Writes    : results/XGBoost/{name}/{region}/params.csv

If cfg contains a 'hyperparams' key, Optuna is skipped and those fixed
values are written directly to the params CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    _experiment_name,
    _split_cfg,
    _time_col,
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

    params_csv = Path(snakemake.output.params_csv)  # noqa: F821
    params_csv.parent.mkdir(parents=True, exist_ok=True)

except NameError:
    # --- Standalone CLI mode ---
    _p = argparse.ArgumentParser(description="Optuna tune — OpenDengue XGBoost regressor (standalone)")
    _p.add_argument("--config",   required=True, help="Path to experiment YAML config")
    _p.add_argument("--region",   required=True, help="Region / unit name (e.g. 'Thailand')")
    _p.add_argument("--n-trials", type=int, default=None)
    _args = _p.parse_args()

    cfg    = load_config(_args.config)
    region = _args.region
    study  = _experiment_name(cfg)

    _paths     = resolve_paths(cfg)
    params_csv = _paths["results_dir"] / study / region / "params.csv"
    params_csv.parent.mkdir(parents=True, exist_ok=True)

    if _args.n_trials:
        cfg.setdefault("tune", {})["n_trials"] = _args.n_trials

tune_cfg     = cfg.get("tune", {})
n_trials     = tune_cfg.get("n_trials", 40)
device       = tune_cfg.get("device", "cuda")
search_space = tune_cfg.get("search_space", {}).get("xgboost", {})
rand_state   = cfg.get("training", {}).get("random_state", 64)
time_col     = _time_col(cfg)

# Defaults used for any hyperparameter not given a spec in the config's
# tune.search_space.xgboost block.
_DEFAULT_SEARCH_SPACE = {
    "learning_rate":    {"type": "log_uniform", "low": 1e-4, "high": 0.3},
    "max_depth":        {"type": "int",         "low": 2,    "high": 10},
    "n_estimators":     {"type": "int",          "low": 50,   "high": 500},
    "subsample":        {"type": "uniform",      "low": 0.5,  "high": 1.0},
    "colsample_bytree": {"type": "uniform",      "low": 0.5,  "high": 1.0},
    "min_child_weight": {"type": "int",          "low": 1,    "high": 20},
    "gamma":            {"type": "log_uniform",  "low": 1e-2, "high": 10.0},
    "reg_alpha":        {"type": "log_uniform",  "low": 1e-3, "high": 10.0},
    "reg_lambda":       {"type": "log_uniform",  "low": 0.1,  "high": 100.0},
}


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    kind = spec["type"]
    if kind == "log_uniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if kind == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search space type: {kind!r} for {name!r}")

# ---------------------------------------------------------------------------
# Load data, filter to region, split, fit target preprocessing (train only)
# ---------------------------------------------------------------------------
df = load_data(cfg)
variable_columns = build_feature_columns(df, cfg)
target = cfg.get("target_column") or cfg.get("target", {}).get("column", "IR")

df_train_val, _ = split_train_test(df, variable_columns, cfg, region=region)

print(f"[tune_xgb:{region}] {len(df_train_val)} samples | trials={n_trials}")

# ---------------------------------------------------------------------------
# If hyperparams are fixed, skip Optuna and write directly
# ---------------------------------------------------------------------------
fixed = cfg.get("hyperparams")
if fixed:
    hp = dict(fixed)
    print(f"[tune_xgb:{region}] Fixed hyperparams from config — skipping Optuna")
    pd.DataFrame([{"Region": region, **hp}]).to_csv(params_csv, index=False)
    raise SystemExit(0)

artifacts = fit_target_preprocessing(df_train_val, cfg)
df_train_val = apply_target_preprocessing(df_train_val, cfg, artifacts)

# ---------------------------------------------------------------------------
# Build walk-forward CV splits
# ---------------------------------------------------------------------------
def _walk_forward_splits(df_sorted: pd.DataFrame, test_window: int = 6, init_window: int = 24):
    unique_times = sorted(df_sorted[time_col].unique())
    splits = []
    for i in range(init_window, len(unique_times) - test_window + 1, test_window):
        train_times = set(unique_times[:i])
        val_times   = set(unique_times[i: i + test_window])
        tr_idx  = df_sorted.index[df_sorted[time_col].isin(train_times)].tolist()
        val_idx = df_sorted.index[df_sorted[time_col].isin(val_times)].tolist()
        if tr_idx and val_idx:
            splits.append((tr_idx, val_idx))
    return splits


df_sorted = df_train_val.sort_values(time_col).reset_index(drop=True)
X_all = df_sorted[variable_columns]
y_all = df_sorted[target]

test_window = cfg.get("training", {}).get("walk_forward_test_window", 6)
init_window = cfg.get("training", {}).get("walk_forward_init_window", 24)
cv_splits = _walk_forward_splits(df_sorted, test_window=test_window, init_window=init_window)

if not cv_splits:
    # Too few timesteps for the requested windows — fall back to a single 80/20 split.
    cutoff = int(len(df_sorted) * 0.8)
    cv_splits = [(list(range(cutoff)), list(range(cutoff, len(df_sorted))))]

print(f"[tune_xgb:{region}] {len(cv_splits)} CV folds")


# ---------------------------------------------------------------------------
# Optuna objective (minimize RMSE on the transformed target)
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    hp = {
        name: _suggest(trial, name, search_space.get(name, default_spec))
        for name, default_spec in _DEFAULT_SEARCH_SPACE.items()
    }
    n_estimators = hp.pop("n_estimators")
    params = {
        "objective":   "reg:squarederror",
        "tree_method": "hist",
        "device":      device,
        "seed":        rand_state,
        **hp,
    }

    fold_rmses = []
    for tr_idx, val_idx in cv_splits:
        X_tr, y_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
        X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]

        dtrain = xgboost.DMatrix(X_tr, label=y_tr)
        dval   = xgboost.DMatrix(X_val, label=y_val)

        booster = xgboost.train(
            params, dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            verbose_eval=False,
        )
        preds = booster.predict(dval)
        fold_rmses.append(mean_squared_error(y_val, preds) ** 0.5)

    return float(np.mean(fold_rmses))


study_obj = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=rand_state))
study_obj.optimize(objective, n_trials=n_trials, show_progress_bar=False)

best = study_obj.best_params
best_score = study_obj.best_value
print(f"[tune_xgb:{region}] Best RMSE (transformed target)={best_score:.4f} | params={best}")

# ---------------------------------------------------------------------------
# Write params CSV
# ---------------------------------------------------------------------------
pd.DataFrame([{"Region": region, **best}]).to_csv(params_csv, index=False)
print(f"[tune_xgb:{region}] Params → {params_csv}")
