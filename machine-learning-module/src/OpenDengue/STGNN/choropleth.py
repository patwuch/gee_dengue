"""
choropleth.py

Produce a figure saved to results/STGNN/<name>/:

  pred_vs_actual.png
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


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _node_names(csv_path: str) -> list[str]:
    """adm_1_name list in node-index order (alphabetical, matches preprocessing)."""
    df = pd.read_csv(csv_path, usecols=["adm_1_name"])
    return df["adm_1_name"].sort_values().unique().tolist()


def _load_actuals(csv_path: str, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["adm_1_name", "year_month", "IR"])
    df["year_month"] = pd.to_datetime(df["year_month"])
    n_test     = cfg["data"]["split"]["test_months"]
    test_dates = sorted(df["year_month"].unique())[-n_test:]
    return df[df["year_month"].isin(test_dates)].copy()


def _load_predictions(csv_path: str, out_dir: Path) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Return (W, N) IR predictions and matching timestamps."""
    npz   = np.load(out_dir / "test_predictions.npz")
    preds = npz["predictions"]
    if preds.ndim == 3:
        preds = preds.squeeze(-1)

    with open(out_dir / "best_params.json") as f:
        window_size = json.load(f)["window_size"]

    df         = pd.read_csv(csv_path, usecols=["year_month"])
    all_dates  = sorted(pd.to_datetime(df["year_month"].unique()))
    n_test     = 14  # cfg is not available here; this is overridden by _load_actuals
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


def _load_geometry(geom_path: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(geom_path)[["name", "geometry"]].copy()


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


# ── Figure: actual vs predicted ───────────────────────────────────────────────

def plot_pred_vs_actual(cfg: dict, csv_path: str, geom_path: str,
                        out_dir: Path, out_path: Path) -> None:
    gdf        = _load_geometry(geom_path)
    actuals    = _load_actuals(csv_path, cfg)
    preds, pred_dates = _load_predictions(csv_path, out_dir)
    nodes      = _node_names(csv_path)
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

def main(config_path: str, out_dir: Path, csv_path: str, geom_path: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(config_path)

    p2 = out_dir / "pred_vs_actual.png"

    print("Generating predicted vs actual choropleths…")
    plot_pred_vs_actual(cfg, csv_path, geom_path, out_dir, p2)
    print(f"  → {p2}")


if "snakemake" in dir():
    main(config_path=snakemake.params.cfg,
         out_dir=Path(snakemake.params.results_dir),
         csv_path=snakemake.params.csv_path,
         geom_path=snakemake.params.geom_path)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv-path", default=None,
                        help="Path to the merged CSV (default: computed from project root).")
    parser.add_argument("--geom-path", default=None,
                        help="Path to the geometry Parquet (default: computed from project root).")
    args = parser.parse_args()
    cfg   = _load_cfg(args.config)

    def _git_root() -> Path:
        for p in (Path(__file__).resolve(),) + tuple(Path(__file__).resolve().parents):
            if (p / ".git").exists():
                return p
        return Path(__file__).resolve().parent

    _root     = _git_root()
    csv_path  = args.csv_path or str(_root / "data/interim/machine-learning/SEA_dengue_env_monthly_2011-2018.csv")
    geom_path = args.geom_path or str(_root / "data/processed/dengue-infection/geoparquet/gaul_2024_sea_filtered.parquet")

    main(config_path=args.config,
         out_dir=Path("results/STGNN") / cfg["name"],
         csv_path=csv_path,
         geom_path=geom_path)