"""
Fetch monthly zonal statistics for SEA admin-1 regions from Google Earth Engine.

Reuses gee_ops.py and products.py from the zonal-statistics-module so QA masks,
band transforms, collection IDs, and reducers stay in sync with the offline app.

No Snakemake needed here: for a single month and a fixed product set the overhead
of chunking, GEE concurrency management, and resume/retry does not apply.

Outputs:
    site/data/latest_zonal_stats.json

Required env var (set as GitHub Actions secret):
    GEE_SERVICE_ACCOUNT  — full contents of a GEE service account key JSON

Local usage (with gcloud ADC):
    earthengine authenticate
    python fetch_zonal_stats.py --month 2026-05
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import tempfile

import ee
import geopandas as gpd

ROOT        = pathlib.Path(__file__).parent.parent.parent
ZSM_ROOT    = ROOT / "zonal-statistics-module"
sys.path.insert(0, str(ZSM_ROOT))

from gee_ops import apply_qa_mask, build_compound_reducer   # noqa: E402
from products import PRODUCT_REGISTRY                        # noqa: E402

REGIONS_PATH = ROOT / "dengue-infection-module/data/processed/geoparquet/gaul_2024_sea_filtered.parquet"
OUTPUT_PATH  = pathlib.Path(__file__).parent.parent / "data/latest_zonal_stats.json"

ADM1_COL = "ADM1_NAME"
ADM0_COL = "ADM0_NAME"

# ── Feature map: STGNN config name → (product_key, band, stat) ──────────────
# stat is applied temporally first; spatial reduction always uses mean.
FEATURE_MAP = {
    "precipitation_sum":                                          ("ERA5_LAND", "total_precipitation_sum",    "SUM"),
    "temperature_2m_mean":                                        ("ERA5_LAND", "temperature_2m",             "MEAN"),
    "temperature_2m_max_mean":                                    ("ERA5_LAND", "temperature_2m_max",         "MEAN"),
    "temperature_2m_min_mean":                                    ("ERA5_LAND", "temperature_2m_min",         "MEAN"),
    "potential_evaporation_sum_mean":                             ("ERA5_LAND", "potential_evaporation_sum",  "MEAN"),
    "total_evaporation_sum_mean":                                 ("ERA5_LAND", "total_evaporation_sum",      "MEAN"),
    "evaporation_from_bare_soil_sum_mean":                        ("ERA5_LAND", "evaporation_from_bare_soil_sum", "MEAN"),
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_mean": ("ERA5_LAND", "evaporation_from_open_water_surfaces_excluding_oceans_sum", "MEAN"),
    "evaporation_from_the_top_of_canopy_sum_mean":                ("ERA5_LAND", "evaporation_from_the_top_of_canopy_sum", "MEAN"),
    "evaporation_from_vegetation_transpiration_sum_mean":         ("ERA5_LAND", "evaporation_from_vegetation_transpiration_sum", "MEAN"),
    "LST_Day_1km_mean":   ("MODIS_LST",      "LST_Day_1km", "MEAN"),
    "LST_Night_1km_mean": ("MODIS_LST",      "LST_Night_1km", "MEAN"),
    "NDVI_mean":          ("MODIS_NDVI_EVI", "NDVI", "MEAN"),
    "EVI_mean":           ("MODIS_NDVI_EVI", "EVI",  "MEAN"),
}

# MODIS LULC is annual — fetched separately and appended as LC_Type1_pct_class{1-17}
LULC_PRODUCT   = "MODIS_LULC"
LULC_BAND      = "LC_Type1"
LULC_N_CLASSES = 17
LULC_YEAR      = 2023   # most recent year in the registry


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate() -> None:
    sa_json = os.environ.get("GEE_SERVICE_ACCOUNT")
    if sa_json:
        key = json.loads(sa_json)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(key, f)
            key_path = f.name
        credentials = ee.ServiceAccountCredentials(key["client_email"], key_path)
        ee.Initialize(credentials)
    else:
        ee.Authenticate()
        ee.Initialize()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gdf_to_ee(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    return ee.FeatureCollection(json.loads(gdf.to_json()))


def apply_band_transform(image: ee.Image, band: str, transform: dict | None) -> ee.Image:
    """Apply scale/offset from products.py band_transform to a single-band image."""
    if transform is None:
        return image
    return (
        image
        .select([band])
        .multiply(transform["scale"])
        .add(transform["offset"])
        .rename([band])
    )


def target_month_dates(target_month: str) -> tuple[str, str]:
    dt = datetime.datetime.strptime(target_month, "%Y-%m")
    start = dt.strftime("%Y-%m-01")
    if dt.month == 12:
        end = f"{dt.year + 1}-01-01"
    else:
        end = f"{dt.year}-{dt.month + 1:02d}-01"
    return start, end


def default_target_month() -> str:
    """Two months ago — most recent month with complete ERA5-Land data."""
    today = datetime.date.today()
    m1 = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
    m2 = (m1 - datetime.timedelta(days=1)).replace(day=1)
    return m2.strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Per-product fetchers
# ---------------------------------------------------------------------------

def fetch_era5_band(
    band: str,
    stat: str,
    start: str,
    end: str,
    fc: ee.FeatureCollection,
) -> dict[str, float]:
    """
    Temporal: reduce daily ERA5-Land images to a single monthly stat image.
    Spatial:  mean over each admin-1 region.
    Returns {region_id: value}.
    """
    info      = PRODUCT_REGISTRY["ERA5_LAND"]
    transform = info["content"][band].get("band_transform")
    scale     = info["scale"]

    temporal_reducer = build_compound_reducer([stat])
    col = (
        ee.ImageCollection(info["ee_collection"])
        .filterDate(start, end)
        .select([band])
    )

    # Reduce temporally to one image, then apply offset (K→°C where needed)
    reduced = col.reduce(temporal_reducer).rename([band])
    if transform:
        reduced = apply_band_transform(reduced, band, transform)

    result = reduced.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
    )
    return {
        f["properties"].get(ADM1_COL, ""): f["properties"].get("mean")
        for f in result.getInfo()["features"]
    }


def fetch_modis_band(
    product_key: str,
    band: str,
    start: str,
    end: str,
    fc: ee.FeatureCollection,
) -> dict[str, float]:
    """
    Fetch a MODIS composite band for the target month.
    Applies QA mask and band transform from products.py, then reduces spatially.
    """
    info      = PRODUCT_REGISTRY[product_key]
    band_cfg  = info["content"][band]
    transform = band_cfg.get("band_transform")
    qa_cfg    = band_cfg.get("qa_mask")
    scale     = info["scale"]

    col = ee.ImageCollection(info["ee_collection"]).filterDate(start, end)

    def process(img: ee.Image) -> ee.Image:
        img = apply_qa_mask(img, qa_cfg) if qa_cfg else img
        img = apply_band_transform(img.select([band]), band, transform) if transform else img.select([band])
        return img

    mean_img = col.map(process).mean().rename([band])

    result = mean_img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
    )
    return {
        f["properties"].get(ADM1_COL, ""): f["properties"].get("mean")
        for f in result.getInfo()["features"]
    }


def fetch_lulc(
    fc: ee.FeatureCollection,
    year: int = LULC_YEAR,
) -> dict[str, dict[str, float]]:
    """
    Fetch MODIS LULC histogram for a given year and return per-class fractions.
    Returns {region: {"LC_Type1_pct_class1": 0.xx, ..., "LC_Type1_pct_class17": 0.xx}}.
    """
    info  = PRODUCT_REGISTRY[LULC_PRODUCT]
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    col   = ee.ImageCollection(info["ee_collection"]).filterDate(start, end)
    image = col.first().select([LULC_BAND])

    result = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=info["scale"],
    )

    region_lulc: dict[str, dict[str, float]] = {}
    for feat in result.getInfo()["features"]:
        props  = feat["properties"]
        region = props.get(ADM1_COL, "")
        hist   = props.get("histogram", {})
        total  = sum(hist.values()) if hist else 1

        fracs: dict[str, float] = {}
        for cls in range(1, LULC_N_CLASSES + 1):
            count = hist.get(str(cls), hist.get(cls, 0))
            fracs[f"LC_Type1_pct_class{cls}"] = round(count / total, 6) if total else 0.0
        region_lulc[region] = fracs

    return region_lulc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(target_month: str | None = None) -> None:
    if target_month is None:
        target_month = default_target_month()

    start, end = target_month_dates(target_month)
    print(f"Target: {target_month}  ({start} → {end})")

    authenticate()

    print("Loading region boundaries ...")
    gdf = gpd.read_parquet(REGIONS_PATH).to_crs("EPSG:4326")
    fc  = gdf_to_ee(gdf)

    # Build per-region skeleton from the GDF so every region appears even if
    # a GEE call returns no data for that feature.
    region_dict: dict[str, dict] = {}
    for _, row in gdf.iterrows():
        region  = row.get(ADM1_COL, "")
        country = row.get(ADM0_COL, "")
        if region:
            region_dict[region] = {"country": country}

    # ── ERA5-Land bands ──────────────────────────────────────────────────────
    for feat_name, (prod_key, band, stat) in FEATURE_MAP.items():
        if prod_key != "ERA5_LAND":
            continue
        print(f"  ERA5_LAND / {band} ({stat}) ...")
        values = fetch_era5_band(band, stat, start, end, fc)
        for region, val in values.items():
            if region in region_dict:
                region_dict[region][feat_name] = val

    # ── MODIS LST ────────────────────────────────────────────────────────────
    for feat_name, (prod_key, band, _) in FEATURE_MAP.items():
        if prod_key != "MODIS_LST":
            continue
        print(f"  MODIS_LST / {band} ...")
        values = fetch_modis_band("MODIS_LST", band, start, end, fc)
        for region, val in values.items():
            if region in region_dict:
                region_dict[region][feat_name] = val

    # ── MODIS NDVI / EVI ─────────────────────────────────────────────────────
    for feat_name, (prod_key, band, _) in FEATURE_MAP.items():
        if prod_key != "MODIS_NDVI_EVI":
            continue
        print(f"  MODIS_NDVI_EVI / {band} ...")
        values = fetch_modis_band("MODIS_NDVI_EVI", band, start, end, fc)
        for region, val in values.items():
            if region in region_dict:
                region_dict[region][feat_name] = val

    # ── MODIS LULC (annual, not per-month) ───────────────────────────────────
    print(f"  MODIS_LULC / LC_Type1 (year {LULC_YEAR}) ...")
    lulc_data = fetch_lulc(fc, year=LULC_YEAR)
    for region, fracs in lulc_data.items():
        if region in region_dict:
            region_dict[region].update(fracs)

    # Normalise NaN → None for JSON serialisation
    for region in region_dict:
        for k, v in region_dict[region].items():
            if isinstance(v, float) and v != v:  # NaN check
                region_dict[region][k] = None

    output = {
        "target_month":   target_month,
        "fetched_at":     datetime.datetime.utcnow().isoformat() + "Z",
        "n_regions":      len(region_dict),
        "bias_corrected": False,
        "regions":        region_dict,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {OUTPUT_PATH}  ({len(region_dict)} regions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--month", default=None,
        help="YYYY-MM to fetch. Defaults to two months ago (ERA5-Land availability lag).",
    )
    args = parser.parse_args()
    main(args.month)
