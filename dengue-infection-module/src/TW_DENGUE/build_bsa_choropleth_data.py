"""
Prepare simplified BSA polygons + annual case counts for a self-contained SVG
choropleth (Tainan + Kaohsiung combined, 2015 and 2023 separately).

Outputs one compact JSON per year to
dengue-infection-module/results/TW_DENGUE/choropleth/bsa_choropleth_<year>.json
with each BSA's SVG path (in a shared planar meter grid, EPSG:3826, y-flipped
for screen coordinates) and its annual case count / population / incidence.
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "dengue-infection-module" / "results" / "TW_DENGUE" / "choropleth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TN_GEOJSON = PROJECT_ROOT / "data" / "external" / "machine-learning" / "geojson" / "bsa_gjs.geojson"
KH_SHP = PROJECT_ROOT / "data" / "external" / "machine-learning" / "KH_SHP" / "G97_64000_U0200_2015.shp"
BSA_CSV = PROJECT_ROOT / "data" / "processed" / "dengue-infection" / "TW_DENGUE" / "bsa_weekly_incidence_2015_2023.csv"

SIMPLIFY_TOLERANCE_M = 15  # metres; BSAs are ~100-300 residents, small polygons

YEARS = [2015, 2023]


def load_geometries() -> gpd.GeoDataFrame:
    tn = gpd.read_file(TN_GEOJSON)[["CODEBASE", "geometry"]].to_crs(3826)
    tn["city"] = "Tainan"
    kh = gpd.read_file(KH_SHP)[["CODEBASE", "geometry"]]
    kh = kh.set_crs(3826, allow_override=True) if kh.crs is None else kh.to_crs(3826)
    kh["city"] = "Kaohsiung"
    geo = pd.concat([tn, kh], ignore_index=True)
    geo = geo.drop_duplicates(subset=["city", "CODEBASE"])
    geo["geometry"] = geo["geometry"].simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
    geo = geo[~geo["geometry"].is_empty & geo["geometry"].notna()]
    return geo


def ring_to_path(coords, tx, ty, sy):
    pts = [f"{(x - tx):.0f},{sy - (y - ty):.0f}" for x, y, *_ in coords]
    return "M" + "L".join(pts) + "Z"


def geometry_to_path(geom, tx, ty, sy):
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    else:
        return None
    parts = []
    for poly in polys:
        parts.append(ring_to_path(poly.exterior.coords, tx, ty, sy))
        for interior in poly.interiors:
            parts.append(ring_to_path(interior.coords, tx, ty, sy))
    return "".join(parts)


def main():
    geo = load_geometries()
    minx, miny, maxx, maxy = geo.total_bounds
    tx, ty = minx, miny
    sy = maxy - miny  # flip: svg y grows downward, projected y grows northward
    width, height = maxx - minx, maxy - miny
    print(f"combined bounds: {width:.0f}m x {height:.0f}m, {len(geo)} BSA polygons")

    bsa = pd.read_csv(BSA_CSV)
    annual = bsa.groupby(["city", "year", "CODEBASE"]).agg(
        cases=("cases", "sum"), population=("population", "first")
    ).reset_index()

    for year in YEARS:
        yr_cases = annual[annual["year"] == year]
        # left join: a BSA absent from this year's population snapshot (created/
        # redistricted between 2015 and 2023) must stay on the map as "no data",
        # not silently disappear -- an inner join here makes newly-redistricted
        # BSAs (e.g. along river/floodplain boundaries) look like blank gaps in
        # the earlier year instead of what they are: not yet a distinct BSA.
        merged = geo.merge(yr_cases, on=["city", "CODEBASE"], how="left")
        no_data = merged["cases"].isna().sum()
        print(f"{year}: {len(merged)} BSAs total, {no_data} with no population snapshot this year")

        features = []
        for row in merged.itertuples():
            path = geometry_to_path(row.geometry, tx, ty, sy)
            if path is None:
                continue
            has_data = pd.notna(row.cases)
            inc = (row.cases / row.population * 100_000) if has_data and row.population else 0
            features.append({
                "c": row.CODEBASE,
                "city": row.city,
                "cases": int(row.cases) if has_data else None,
                "pop": int(row.population) if has_data else None,
                "inc": round(inc, 1) if has_data else None,
                "d": path,
            })

        out = {
            "year": year,
            "width": round(width, 1),
            "height": round(height, 1),
            "features": features,
        }
        out_path = OUT_DIR / f"bsa_choropleth_{year}.json"
        out_path.write_text(json.dumps(out, separators=(",", ":")))
        print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(features)} features)")


if __name__ == "__main__":
    main()
