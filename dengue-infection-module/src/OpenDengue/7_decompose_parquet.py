from pathlib import Path

import geopandas as gpd

_PIPELINE_ROOT = Path(__file__).resolve().parents[2]

GEO_PATH = (
    _PIPELINE_ROOT
    / "data" / "external" / "geoparquet"
    / "gaul_2024_sea_filtered.parquet"
)
OUT_DIR = (
    _PIPELINE_ROOT
    / "data" / "external" / "geoparquet"
    / "decomposed_sea_parquet"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

filtered_sea = gpd.read_parquet(GEO_PATH)

null_rows = filtered_sea[filtered_sea["admin"].isna()]
if not null_rows.empty:
    print(null_rows)

print(filtered_sea["admin"].apply(type).unique())

gdf_reset = filtered_sea.reset_index(drop=True)

for admin_value in gdf_reset["admin"].unique():
    subset_gdf = gdf_reset.loc[gdf_reset["admin"] == admin_value]
    if not subset_gdf.empty:
        filename = OUT_DIR / f"{str(admin_value).replace(' ', '_')}.parquet"
        subset_gdf.to_parquet(filename)
        print(f"Saved {filename.name}  ({len(subset_gdf)} rows)")

print(f"\nDone. {gdf_reset['admin'].nunique()} country files written to {OUT_DIR}")
