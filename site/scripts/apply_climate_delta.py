"""
Apply climate delta bias correction to raw GEE zonal statistics.

The STGNN was trained on 2011-2018 data. Feeding 2026 observations directly
produces covariate shift — features like temperature are systematically outside
the training distribution. This script corrects for that shift by subtracting
the per-variable, per-calendar-month delta between the current climatology
(estimated from the training CSV plus any recent GEE history) and the
2011-2018 training period baseline.

Inputs:
    site/data/latest_zonal_stats.json          raw GEE observations for current month
    machine-learning-module/data/interim/...   2011-2018 training CSV (baseline)

Outputs:
    site/data/climate_deltas.json              per-variable monthly deltas
    site/data/delta_corrected_stats.json       corrected observations fed to inference
"""

import json
import pathlib
import datetime
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent.parent
TRAINING_CSV = ROOT / "machine-learning-module/data/interim/SEA_dengue_env_monthly_2011-2018.csv"
STATS_PATH   = pathlib.Path(__file__).parent.parent / "data/latest_zonal_stats.json"
DELTAS_PATH  = pathlib.Path(__file__).parent.parent / "data/climate_deltas.json"
OUTPUT_PATH  = pathlib.Path(__file__).parent.parent / "data/delta_corrected_stats.json"

ENV_FEATURES = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max_mean",
    "temperature_2m_min_mean",
    "potential_evaporation_sum_mean",
    "total_evaporation_sum_mean",
    "evaporation_from_bare_soil_sum_mean",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_mean",
    "evaporation_from_the_top_of_canopy_sum_mean",
    "evaporation_from_vegetation_transpiration_sum_mean",
    "LST_Day_1km_mean",
    "LST_Night_1km_mean",
    "NDVI_mean",
    "EVI_mean",
]


def load_training_climatology(features: list[str]) -> dict[str, dict[str, float]]:
    """
    Compute per-variable, per-calendar-month means from the training CSV.
    Returns {feature: {month_str: mean}} e.g. {"temperature_2m_mean": {"01": 28.3, ...}}
    """
    df = pd.read_csv(TRAINING_CSV, parse_dates=["year_month"])
    df["calendar_month"] = df["year_month"].dt.month.apply(lambda m: f"{m:02d}")

    climatology: dict[str, dict[str, float]] = {}
    for feat in features:
        if feat not in df.columns:
            continue
        monthly = df.groupby("calendar_month")[feat].mean()
        climatology[feat] = monthly.to_dict()

    return climatology


def compute_deltas(
    stats: dict,
    climatology: dict[str, dict[str, float]],
    calendar_month: str,
) -> dict[str, float]:
    """
    For each feature, compute mean(current_obs) - training_climatology[month].
    This is the climate shift that needs to be subtracted before inference.
    """
    regions = stats.get("regions", {})
    deltas: dict[str, float] = {}

    for feat in ENV_FEATURES:
        if feat not in climatology:
            continue
        baseline = climatology[feat].get(calendar_month)
        if baseline is None:
            continue

        current_vals = [
            r[feat] for r in regions.values()
            if isinstance(r.get(feat), (int, float)) and not np.isnan(r[feat])
        ]
        if not current_vals:
            deltas[feat] = 0.0
            continue

        current_mean = float(np.mean(current_vals))
        deltas[feat] = round(current_mean - baseline, 6)

    return deltas


def apply_correction(
    stats: dict,
    deltas: dict[str, float],
) -> dict:
    """Subtract the monthly delta from each region's raw observations."""
    corrected = {k: v for k, v in stats.items() if k != "regions"}
    corrected["bias_corrected"] = True
    corrected["regions"] = {}

    for region, vals in stats.get("regions", {}).items():
        region_corrected = dict(vals)
        for feat, delta in deltas.items():
            if feat in region_corrected and isinstance(region_corrected[feat], (int, float)):
                region_corrected[feat] = round(region_corrected[feat] - delta, 6)
        corrected["regions"][region] = region_corrected

    return corrected


def load_precomputed_deltas(calendar_month: str) -> dict[str, float] | None:
    """
    Read the stable batch-derived delta for this calendar month from
    climate_deltas.json if it was produced by compute_climate_delta.py.

    Returns None if the file is missing or doesn't contain this month,
    triggering the single-month fallback in main().
    """
    if not DELTAS_PATH.exists():
        return None
    all_deltas = json.loads(DELTAS_PATH.read_text())
    month_deltas = all_deltas.get(calendar_month)
    # Distinguish a pre-computed file (has all 12 months as keys) from a
    # file written by a previous single-month fallback run (has fewer keys
    # and each value is a dict rather than a float per feature).
    if not month_deltas:
        return None
    if not isinstance(next(iter(month_deltas.values())), (int, float)):
        return None  # old format written by this script's fallback path
    return {k: float(v) for k, v in month_deltas.items()}


def main() -> None:
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Zonal stats not found: {STATS_PATH}\nRun fetch_zonal_stats.py first.")

    stats = json.loads(STATS_PATH.read_text())
    target_month = stats.get("target_month", "")

    try:
        dt = datetime.datetime.strptime(target_month, "%Y-%m")
        calendar_month = f"{dt.month:02d}"
    except ValueError:
        calendar_month = datetime.datetime.now().strftime("%m")

    # ── Prefer pre-computed stable deltas from compute_climate_delta.py ──────
    deltas = load_precomputed_deltas(calendar_month)

    if deltas is not None:
        print(f"Using pre-computed batch delta for month {calendar_month} "
              f"(from {DELTAS_PATH.name}) — stable 2019-2026 vs 2011-2018 baseline.")
    else:
        # Fallback: estimate delta from this month's single GEE observation.
        # Noisier than the batch delta but works before the batch job is run.
        print(f"No pre-computed delta found — estimating from single-month GEE obs "
              f"(run zonal-statistics-module/compute_climate_delta.py for a stable delta).")
        print(f"Computing climatology baseline from {TRAINING_CSV} ...")
        climatology = load_training_climatology(ENV_FEATURES)
        print(f"Computing deltas for calendar month {calendar_month} ...")
        deltas = compute_deltas(stats, climatology, calendar_month)

        # Write fallback deltas back so the site can display them, but don't
        # overwrite pre-computed month entries that may already be there.
        all_deltas = json.loads(DELTAS_PATH.read_text()) if DELTAS_PATH.exists() else {}
        if calendar_month not in all_deltas:
            all_deltas[calendar_month] = deltas
            DELTAS_PATH.write_text(json.dumps(all_deltas, indent=2))

    corrected = apply_correction(stats, deltas)
    OUTPUT_PATH.write_text(json.dumps(corrected, indent=2))
    print(f"Delta-corrected stats saved → {OUTPUT_PATH}")

    temp_delta = deltas.get("temperature_2m_mean", 0)
    prec_delta = deltas.get("precipitation_sum", 0)
    print(f"\nDelta summary (positive = current warmer/wetter than 2011-2018 baseline):")
    print(f"  temperature_2m_mean : {temp_delta:+.3f} °C")
    print(f"  precipitation_sum   : {prec_delta:+.6f}")


if __name__ == "__main__":
    main()
