"""
scripts/Indonesia/XGBoost/eval.py
-----------------------------------
Step 3: Evaluate a trained model and (optionally) produce SHAP plots.

Reads  : config YAML  (--config)
         models/xgboost/{experiment.name}/{region}.json  (written by train.py)
Writes : reports/tables/{experiment.name}/{region}_results.csv
         reports/figures/{experiment.name}/{region}/confusion_matrix_{split}.png
         reports/figures/{experiment.name}/{region}/beeswarm/  (if --shap)
         reports/figures/{experiment.name}/{region}/dependence/ (if --shap)

Usage
-----
# National scope (with SHAP):
python scripts/Indonesia/XGBoost/eval.py \\
    --config config/indonesia/xgboost_national_classifier.yaml --shap

# Regional scope — all regions, no SHAP:
python scripts/Indonesia/XGBoost/eval.py \\
    --config config/indonesia/xgboost_regional_classifier.yaml

Mirrors the interface of scripts/OpenDengue/XGBoost/eval_xgboost.py.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_feature_columns,
    load_config,
    load_data,
    resolve_paths,
    split_train_test,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate XGBoost — Indonesia")
    p.add_argument("--config", required=True, help="Path to experiment YAML config")
    p.add_argument("--region", default=None,
                   help="Region name (regional configs only). Omit to iterate all.")
    p.add_argument("--shap", action="store_true",
                   help="Compute and save SHAP beeswarm + dependence plots")
    p.add_argument("--shap-top-n", type=int, default=5,
                   help="Number of top features to show in dependence plots (default: 5)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def _run_shap(
    model: xgboost.Booster,
    X_set: pd.DataFrame,
    y_set: np.ndarray,
    num_classes: int,
    excluded_cols: list[str],
    region_label: str,
    set_name: str,
    fig_dir: Path,
    top_n: int,
) -> None:
    """
    Compute SHAP values and save beeswarm + dependence plots.
    GPU Tree explainer used if available; falls back to CPU Tree explainer.
    """
    try:
        import shap
    except ImportError:
        print(f"[eval:{region_label}] shap not installed — skipping SHAP.")
        return

    beeswarm_dir  = fig_dir / "beeswarm"
    dependence_dir = fig_dir / "dependence"
    beeswarm_dir.mkdir(parents=True, exist_ok=True)
    dependence_dir.mkdir(parents=True, exist_ok=True)

    try:
        explainer = shap.explainers.GPUTree(
            model, feature_perturbation="tree_path_dependent"
        )
    except Exception:
        explainer = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent"
        )

    shap_values = explainer.shap_values(X_set, check_additivity=False)

    included_cols = [c for c in X_set.columns if c not in excluded_cols]

    # --- Beeswarm plots (one per class) ---
    for cls_idx in range(num_classes):
        sv = shap_values[cls_idx] if num_classes > 2 else shap_values
        sv_plot = sv[:, [i for i, c in enumerate(X_set.columns) if c in included_cols]]
        X_plot  = X_set[included_cols]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv_plot, X_plot, show=False)
        plt.title(f"SHAP Beeswarm — Class {cls_idx} ({set_name})")
        out = beeswarm_dir / f"class{cls_idx}_beeswarm_{set_name}.png"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  [SHAP] Saved beeswarm → {out}")

    # --- Dependence plots (top N features per class, excluding epidemic vars) ---
    for cls_idx in range(num_classes):
        sv = shap_values[cls_idx] if num_classes > 2 else shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        sorted_feats = [X_set.columns[i] for i in mean_abs.argsort()[::-1]]
        top_feats = [f for f in sorted_feats if f not in excluded_cols][:top_n]

        for feat in top_feats:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feat, sv, X_set, interaction_index="auto", show=False)
            plt.title(f"SHAP Dependence — {feat}, Class {cls_idx} ({set_name})")
            safe_feat = feat.replace("+", "PLUS").replace(".", "_")
            out = dependence_dir / f"class{cls_idx}_{safe_feat}_dep_{set_name}.png"
            plt.savefig(out, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"  [SHAP] Saved dependence → {out}")


# ---------------------------------------------------------------------------
# Single-region eval
# ---------------------------------------------------------------------------

def eval_region(
    region: str | None,
    cfg: dict,
    paths: dict,
    run_shap: bool,
    shap_top_n: int,
) -> list[dict]:
    region_label = region if region is not None else "National"
    study_name   = cfg["experiment"]["name"]
    task         = cfg.get("task", "classifier")

    model_path = paths["models_dir"] / study_name / f"{region_label}.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run train.py first:\n"
            f"  python scripts/Indonesia/XGBoost/train.py --config {cfg.get('_config_path', '<config>')}"
        )

    model = xgboost.Booster()
    model.load_model(str(model_path))
    print(f"\n[eval:{region_label}] Loaded model from {model_path}")

    df               = load_data(cfg)
    variable_columns = build_feature_columns(df, cfg)
    target           = cfg["target"]["column"]

    df_train_val, df_test = split_train_test(df, variable_columns, cfg, region=region)
    num_classes = int(df[target].nunique()) if task == "classifier" else None
    class_labels = list(range(num_classes)) if num_classes else None

    fig_dir = paths["figures_dir"] / study_name / region_label
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Columns to exclude from SHAP visualisation (but not from model input)
    excluded_from_shap = cfg["features"].get("epidemic_vars", [])

    datasets = {
        "Train_Val": (df_train_val[variable_columns], df_train_val[target].values.flatten()),
        "Test":      (df_test[variable_columns],      df_test[target].values.flatten()),
    }

    all_results = []

    for set_name, (X_set, y_set) in datasets.items():
        print(f"[eval:{region_label}] --- {set_name} ---")

        dset        = xgboost.DMatrix(X_set)
        y_pred_prob = model.predict(dset)

        if task == "classifier":
            y_pred = (
                y_pred_prob.argmax(axis=1) if num_classes > 2
                else (y_pred_prob > 0.5).astype(int)
            )

            accuracy      = accuracy_score(y_set, y_pred)
            report        = classification_report(y_set, y_pred, output_dict=True)
            target_counts = pd.Series(y_set).value_counts().to_dict()
            cm            = confusion_matrix(y_set, y_pred, labels=class_labels)

            # Confusion matrix plot
            fig_size = max(4, num_classes * 2)
            plt.figure(figsize=(fig_size, fig_size))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix — {region_label} ({set_name})")
            cm_path = fig_dir / f"confusion_matrix_{set_name}.png"
            plt.savefig(cm_path, bbox_inches="tight")
            plt.close()
            print(f"  Accuracy: {accuracy:.4f} | Counts: {target_counts}")
            print(f"  Saved confusion matrix → {cm_path}")

            row: dict = {"Region": region_label, "Dataset": set_name, "Accuracy": accuracy}
            for i in class_labels:
                row[f"Class_{i}_count"] = target_counts.get(i, 0)
                if str(i) in report:
                    row[f"Precision_{i}"] = report[str(i)]["precision"]
                    row[f"Recall_{i}"]    = report[str(i)]["recall"]
                    row[f"F1_{i}"]        = report[str(i)]["f1-score"]
            row["Macro_Precision"]    = report["macro avg"]["precision"]
            row["Macro_Recall"]       = report["macro avg"]["recall"]
            row["Macro_F1"]           = report["macro avg"]["f1-score"]
            row["Weighted_Precision"] = report["weighted avg"]["precision"]
            row["Weighted_Recall"]    = report["weighted avg"]["recall"]
            row["Weighted_F1"]        = report["weighted avg"]["f1-score"]

        else:  # regressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            y_pred = y_pred_prob  # already 1-D for regression
            rmse   = float(mean_squared_error(y_set, y_pred) ** 0.5)
            mae    = float(mean_absolute_error(y_set, y_pred))
            r2     = float(r2_score(y_set, y_pred))
            print(f"  RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")
            row = {"Region": region_label, "Dataset": set_name,
                   "RMSE": rmse, "MAE": mae, "R2": r2}

        all_results.append(row)

        # SHAP (optional, classifier only for now)
        if run_shap and task == "classifier":
            _run_shap(
                model, X_set, y_set, num_classes,
                excluded_from_shap, region_label, set_name, fig_dir, shap_top_n,
            )

    del model
    gc.collect()
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args  = parse_args()
    cfg   = load_config(args.config)
    cfg["_config_path"] = args.config
    paths = resolve_paths(cfg)
    scope = cfg.get("scope", "national")

    all_results = []

    if scope == "national":
        all_results.extend(eval_region(None, cfg, paths, args.shap, args.shap_top_n))
    elif args.region:
        all_results.extend(eval_region(args.region, cfg, paths, args.shap, args.shap_top_n))
    else:
        df      = load_data(cfg)
        regions = df[cfg["data"]["region_column"]].unique()
        for region in regions:
            try:
                all_results.extend(eval_region(region, cfg, paths, args.shap, args.shap_top_n))
            except FileNotFoundError as e:
                print(f"[eval:{region}] Skipping — {e}")

    # Save combined results table
    if all_results:
        study_name = cfg["experiment"]["name"]
        out_dir    = paths["tables_dir"] / study_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv    = out_dir / "results.csv"
        pd.DataFrame(all_results).to_csv(out_csv, index=False, float_format="%.4f")
        print(f"\nResults saved → {out_csv}")


if __name__ == "__main__":
    main()
