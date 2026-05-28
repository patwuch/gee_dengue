"""
scripts/Indonesia/XGBoost/tune.py
----------------------------------
Step 1: Optuna hyperparameter search for XGBoost (Indonesia datasets).

Reads  : config YAML  (--config)
Writes : reports/tables/{experiment.name}/{region}_params.csv
         (one row per region; 'National' for national-scope configs)

Usage
-----
# National scope (single run):
python scripts/Indonesia/XGBoost/tune.py \\
    --config config/indonesia/xgboost_national_classifier.yaml

# Regional scope (one call per region, or omit --region to loop all):
python scripts/Indonesia/XGBoost/tune.py \\
    --config config/indonesia/xgboost_regional_classifier.yaml \\
    --region Sumatera

Mirrors the interface of scripts/OpenDengue/XGBoost/tune_xgboost.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import pandas as pd
import xgboost
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_cv_splits,
    build_feature_columns,
    calculate_sample_weights,
    load_config,
    load_data,
    resolve_paths,
    split_train_test,
    validation_strategy,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter search — Indonesia XGBoost")
    p.add_argument("--config",  required=True, help="Path to experiment YAML config")
    p.add_argument("--region",  default=None,
                   help="Region name to tune (regional configs only). "
                        "Omit to iterate all regions.")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override n_trials from config")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(X_all, y_all, cv_splits, cfg: dict):
    """Return a closure over the data and splits for Optuna to call."""
    task      = cfg.get("task", "classifier")
    num_cls   = int(pd.Series(y_all).nunique()) if task == "classifier" else None
    device    = cfg["training"].get("device", "cuda")

    def objective(trial: optuna.Trial) -> float:
        if task == "classifier":
            objective_fn = "multi:softprob" if num_cls > 2 else "binary:logistic"
        else:
            objective_fn = "reg:squarederror"

        params: dict = {
            "objective":        objective_fn,
            "tree_method":      "hist",
            "device":           device,
            "seed":             cfg["training"].get("random_state", 64),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 2, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "gamma":            trial.suggest_float("gamma", 1e-2, 10.0, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        }
        if task == "classifier" and num_cls > 2:
            params["num_class"] = num_cls

        n_estimators = trial.suggest_int("n_estimators", 50, 1000)

        all_preds, all_true = [], []

        for tr_idx, val_idx in cv_splits:
            X_tr  = X_all.iloc[tr_idx]
            y_tr  = y_all.iloc[tr_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all.iloc[val_idx]

            if task == "classifier":
                sw = calculate_sample_weights(y_tr.values)
                dtrain = xgboost.DMatrix(X_tr, label=y_tr, weight=sw)
            else:
                dtrain = xgboost.DMatrix(X_tr, label=y_tr)
            dval = xgboost.DMatrix(X_val, label=y_val)

            booster = xgboost.train(
                params, dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, "val")],
                verbose_eval=False,
            )

            preds_prob = booster.predict(dval)
            if task == "classifier":
                preds = (
                    preds_prob.argmax(axis=1) if num_cls > 2
                    else (preds_prob > 0.5).astype(int)
                )
            else:
                preds = preds_prob

            all_preds.extend(preds.tolist())
            all_true.extend(y_val.values.tolist())

        if not all_true:
            return 0.0 if task == "classifier" else float("inf")

        if task == "classifier":
            return f1_score(all_true, all_preds, average="macro", zero_division=0)
        else:
            return float(mean_squared_error(all_true, all_preds) ** 0.5)

    return objective


# ---------------------------------------------------------------------------
# Single-region tune
# ---------------------------------------------------------------------------

def tune_region(region: str | None, cfg: dict, n_trials: int, paths: dict) -> None:
    """
    Run Optuna for *region* (pass None for national-scope configs).
    Writes one row to reports/tables/{experiment.name}/{region}_params.csv.
    """
    region_label = region if region is not None else "National"
    study_name   = cfg["experiment"]["name"]
    val_strat    = validation_strategy(cfg)
    rand_state   = cfg["training"].get("random_state", 64)
    task         = cfg.get("task", "classifier")

    print(f"\n[tune:{region_label}] strategy={val_strat} | task={task} | trials={n_trials}")

    df               = load_data(cfg)
    variable_columns = build_feature_columns(df, cfg)
    target           = cfg["target"]["column"]

    df_train_val, _  = split_train_test(df, variable_columns, cfg, region=region)

    if df_train_val.empty:
        print(f"[tune:{region_label}] No training data — skipping.")
        return

    df_sorted = df_train_val.sort_values(cfg["data"]["time_column"]).reset_index(drop=True)
    X_all     = df_sorted[variable_columns]
    y_all     = df_sorted[target]

    cv_splits = build_cv_splits(df_sorted, cfg)
    print(f"[tune:{region_label}] {len(df_train_val)} samples | {len(cv_splits)} CV folds")

    direction  = "maximize" if task == "classifier" else "minimize"
    study      = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=rand_state),
    )
    objective  = _make_objective(X_all, y_all, cv_splits, cfg)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best       = study.best_params
    best_score = study.best_value
    metric     = "f1_macro" if task == "classifier" else "RMSE"
    print(f"[tune:{region_label}] Best {metric}={best_score:.4f} | params={best}")

    # Write params CSV
    out_dir = paths["tables_dir"] / study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{region_label}_params.csv"
    pd.DataFrame([{"Region": region_label, **best}]).to_csv(out_csv, index=False)
    print(f"[tune:{region_label}] Params → {out_csv}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)
    paths  = resolve_paths(cfg)
    n_trials = args.n_trials or cfg["training"].get("n_trials", 40)
    scope  = cfg.get("scope", "national")

    if scope == "national":
        tune_region(None, cfg, n_trials, paths)
    elif args.region:
        tune_region(args.region, cfg, n_trials, paths)
    else:
        # Loop every region in the data
        df = load_data(cfg)
        regions = df[cfg["data"]["region_column"]].unique()
        for region in regions:
            tune_region(region, cfg, n_trials, paths)


if __name__ == "__main__":
    main()
