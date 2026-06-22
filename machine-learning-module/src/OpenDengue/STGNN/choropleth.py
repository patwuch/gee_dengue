"""
choropleth.py

Produce two figures saved to results/STGNN/<name>/:

  1. actual_ir_14months.png
       One choropleth per test month (14 panels), actual IR only.

  2. pred_vs_actual.png
       Side-by-side actual vs predicted IR for every predicted month.
       Covers all test months when the inference split is used
       (requires re-running preprocessing after the inference-window change);
       otherwise covers only the last len(test) - window_size months.

Invoked via Snakemake (choropleth_stgnn rule) or directly:
    python src/OpenDengue/STGNN/choropleth.py \
        --config config/OpenDengue/stgnn_logIR_regressor.yaml
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")


# ── Path helpers ──────────────────────────────────────────────────────────────

def _git_root() -> Path:
    for p in (Path(__file__).resolve(),) + tuple(Path(__file__).resolve().parents):
        if (p / ".git").exists():
            return p
    return Path(__file__).resolve().parent


_ROOT       = _git_root()
_GEOM_PATH  = _ROOT / "data/processed/dengue-infection/geoparquet/gaul_2024_sea_filtered.parquet"
_CSV_PATH   = _ROOT / "data/interim/machine-learning/SEA_dengue_env_monthly_2011-2018.csv"
_RESULTS    = _ROOT / "machine-learning-module/results/STGNN"


def _results_dir(cfg: dict) -> Path:
    return _RESULTS / cfg["name"]


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _node_names(cfg: dict) -> list[str]:
    """adm_1_name list in node-index order (alphabetical, matches preprocessing)."""
    df = pd.read_csv(str(_CSV_PATH), usecols=["adm_1_name"])
    return df["adm_1_name"].sort_values().unique().tolist()


def _load_actuals(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(str(_CSV_PATH), usecols=["adm_1_name", "year_month", "IR"])
    df["year_month"] = pd.to_datetime(df["year_month"])
    n_test     = cfg["data"]["split"]["test_months"]
    test_dates = sorted(df["year_month"].unique())[-n_test:]
    return df[df["year_month"].isin(test_dates)].copy()


def _load_predictions(cfg: dict) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Return (W, N) IR predictions and matching timestamps."""
    npz   = np.load(_results_dir(cfg) / "test_predictions.npz")
    preds = npz["predictions"]
    if preds.ndim == 3:
        preds = preds.squeeze(-1)

    with open(_results_dir(cfg) / "best_params.json") as f:
        window_size = json.load(f)["window_size"]

    df         = pd.read_csv(str(_CSV_PATH), usecols=["year_month"])
    all_dates  = sorted(pd.to_datetime(df["year_month"].unique()))
    n_test     = cfg["data"]["split"]["test_months"]
    test_dates = all_dates[-n_test:]

    W = preds.shape[0]
    if W == n_test:
        pred_dates = test_dates           # inference split: all 14 months
    else:
        pred_dates = test_dates[window_size:]   # legacy: last n_test - window_size months

    assert len(pred_dates) == W, (
        f"Mismatch: {W} prediction windows but {len(pred_dates)} dates."
    )
    return preds, pred_dates


def _load_geometry() -> gpd.GeoDataFrame:
    return gpd.read_parquet(_GEOM_PATH)[["name", "geometry"]].copy()


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _norm(values: np.ndarray) -> matplotlib.colors.Normalize:
    valid = values[~np.isnan(values)]
    lo = float(np.percentile(valid, 2))  if len(valid) else 0
    hi = float(np.percentile(valid, 98)) if len(valid) else 1
    return matplotlib.colors.Normalize(vmin=lo, vmax=hi)


def _draw_map(ax, gdf, col, norm, cmap, title):
    gdf.plot(
        column=col, ax=ax, cmap=cmap, norm=norm,
        missing_kwds={"color": "lightgrey"},
        linewidth=0.3, edgecolor="white",
    )
    ax.set_title(title, fontsize=9, pad=4)
    ax.axis("off")


def _colorbar(fig, axes, norm, cmap, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, shrink=0.85)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


# ── Figure 1: actual IR for all test months ───────────────────────────────────

def plot_actuals(cfg: dict, out_path: Path) -> None:
    gdf     = _load_geometry()
    actuals = _load_actuals(cfg)
    dates   = sorted(actuals["year_month"].unique())
    n       = len(dates)
    ncols   = 4
    nrows   = int(np.ceil(n / ncols))
    cmap    = "YlOrRd"

    norm = _norm(actuals["IR"].dropna().values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    flat = axes.flatten()

    for i, date in enumerate(dates):
        month_df = actuals[actuals["year_month"] == date][["adm_1_name", "IR"]]
        merged   = gdf.merge(month_df, left_on="name", right_on="adm_1_name", how="left")
        _draw_map(flat[i], merged, "IR", norm, cmap, date.strftime("%b %Y"))

    for j in range(i + 1, len(flat)):
        flat[j].set_visible(False)

    _colorbar(fig, flat[:n], norm, cmap, "Incidence Rate (IR)")
    date_range = f"{dates[0].strftime('%b %Y')} – {dates[-1].strftime('%b %Y')}"
    fig.suptitle(f"Actual Dengue IR — {n} Test Months ({date_range})", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: actual vs predicted ─────────────────────────────────────────────

def plot_pred_vs_actual(cfg: dict, out_path: Path) -> None:
    gdf        = _load_geometry()
    actuals    = _load_actuals(cfg)
    preds, pred_dates = _load_predictions(cfg)
    nodes      = _node_names(cfg)
    n_test     = cfg["data"]["split"]["test_months"]
    n          = len(pred_dates)
    cmap       = "YlOrRd"

    all_actual = actuals[actuals["year_month"].isin(pred_dates)]["IR"].dropna().values
    all_vals   = np.concatenate([all_actual, preds[~np.isnan(preds)].flatten()])
    norm       = _norm(all_vals)

    ncols = 2
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 4))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row, (date, pred_row) in enumerate(zip(pred_dates, preds)):
        month_df = actuals[actuals["year_month"] == date][["adm_1_name", "IR"]]
        merged_a = gdf.merge(month_df, left_on="name", right_on="adm_1_name", how="left")
        _draw_map(axes[row, 0], merged_a, "IR", norm, cmap, f"Actual — {date.strftime('%b %Y')}")

        pred_df  = pd.DataFrame({"adm_1_name": nodes, "IR_pred": pred_row})
        merged_p = gdf.merge(pred_df, left_on="name", right_on="adm_1_name", how="left")
        _draw_map(axes[row, 1], merged_p, "IR_pred", norm, cmap, f"Predicted — {date.strftime('%b %Y')}")

    _colorbar(fig, axes, norm, cmap, "Incidence Rate (IR)")
    coverage = "all test months" if n == n_test else f"{n} of {n_test} test months"
    fig.suptitle(f"Actual vs Predicted Dengue IR ({coverage})", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Entry points ──────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    cfg     = _load_cfg(config_path)
    out_dir = _results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "actual_ir_14months.png"
    p2 = out_dir / "pred_vs_actual.png"

    print("Generating actual IR choropleths…")
    plot_actuals(cfg, p1)
    print(f"  → {p1}")

    print("Generating predicted vs actual choropleths…")
    plot_pred_vs_actual(cfg, p2)
    print(f"  → {p2}")


if "snakemake" in dir():
    main(config_path=snakemake.params.cfg)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(config_path=parser.parse_args().config)
