import geopandas as gpd
import pandas as pd
from multiprocessing import Pool
import functools

CSV_PATH = snakemake.input[0]
SHP_PATH = snakemake.input[1]
OUT_PATH = snakemake.output[0]
n_workers = snakemake.threads

# CSV adm_0_name (uppercase) → GAUL gaul0_name
# Only entries that differ from plain .title() are listed.
COUNTRY_MAP = {
    "BRUNEI DARUSSALAM": "Brunei Darussalam",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "Lao People's Democratic Republic",
    "VIET NAM": "Viet Nam",
}

# Manual adm_1 name mappings where CSV name does not match GAUL gaul1_name
# (or gaul2_name for Myanmar) case-insensitively.
# Key: (gaul0_name, csv adm_1_name uppercase) → dissolved-L1 'name' value
MANUAL_ADM1_MAP = {
    # Brunei — CSV appends " DISTRICT"
    ("Brunei Darussalam", "BELAIT DISTRICT"): "Belait",
    ("Brunei Darussalam", "BRUNEI MUARA DISTRICT"): "Brunei And Muara",
    ("Brunei Darussalam", "TEMBURONG DISTRICT"): "Temburong",
    ("Brunei Darussalam", "TUTONG DISTRICT"): "Tutong",
    # Indonesia — abbreviated names
    ("Indonesia", "BABEL"): "Kepulauan Bangka Belitung",
    # ("Indonesia", "BANGKA BELITUNG"): "Kepulauan Bangka Belitung",
    # Malaysia — federal territories have "W.P." prefix in GAUL
    ("Malaysia", "KUALA LUMPUR"): "W.P. Kuala Lumpur",
    ("Malaysia", "LABUAN"): "W.P. Labuan",
    # Thailand — spelling variants
    ("Thailand", "BUNGKAN"): "Bueng Kan",
    ("Thailand", "BURIRAM"): "Buri Ram",
    ("Thailand", "CHAINAT"): "Chai Nat",
    ("Thailand", "CHONBURI"): "Chon Buri",
    ("Thailand", "KAMPAENG PHET"): "Kamphaeng Phet",
    ("Thailand", "LOPBURI"): "Lop Buri",
    ("Thailand", "NONG BUA LAMPHU"): "Nong Bua Lam Phu",
    ("Thailand", "PHACHINBURI"): "Prachin Buri",
    ("Thailand", "PHRA NAKHON SI AYUDHYA"): "Phra Nakhon Si Ayutthaya",
    ("Thailand", "PRACHUAP KHILIKHAN"): "Prachuap Khiri Khan",
    ("Thailand", "SAMUT PRAKARN"): "Samut Prakan",
    ("Thailand", "SAMUT SONGKHAM"): "Samut Songkhram",
    ("Thailand", "SI SAKET"): "Si Sa Ket",
    ("Thailand", "SINGBURI"): "Sing Buri",
    ("Thailand", "SUPHANBURI"): "Suphan Buri",
    ("Thailand", "TRAD"): "Trat",
}

# CSV rows whose adm_1_name refers to the whole country (national-level data).
# All L2 polygons for that country are dissolved into a single polygon.
NATIONAL_LEVEL = {
    ("Cambodia", "CAMBODIA"),
    ("Lao People's Democratic Republic", "LAO PEOPLE'S DEMOCRATIC REPUBLIC"),
    ("Singapore", "SINGAPORE"),
    ("Viet Nam", "VIET NAM"),
}


def dissolve_country(args):
    """Dissolve a single country's L2 polygons to L1. Run in worker."""
    country, group, use_gaul2 = args
    if use_gaul2:
        result = (
            group.dissolve(by=["gaul0_name", "gaul2_name"])
            .reset_index()[["gaul0_name", "gaul2_name", "geometry"]]
            .rename(columns={"gaul0_name": "admin", "gaul2_name": "name"})
        )
    else:
        result = (
            group.dissolve(by=["gaul0_name", "gaul1_name"])
            .reset_index()[["gaul0_name", "gaul1_name", "geometry"]]
            .rename(columns={"gaul0_name": "admin", "gaul1_name": "name"})
        )
    return result


def dissolve_to_l1(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve GAUL L2 polygons to admin-1 level, using all available cores."""
    tasks = []
    for country, group in gdf.groupby("gaul0_name"):
        use_gaul2 = (country == "Myanmar")
        tasks.append((country, group.copy(), use_gaul2))

    with Pool(processes=n_workers) as pool:
        frames = pool.map(dissolve_country, tasks)

    l1 = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(l1, geometry="geometry", crs=gdf.crs)


def resolve_csv_pair(args):
    """Resolve a single (admin, adm_1_name) CSV pair. Run in worker."""
    admin, adm_1_name, adm1_up, l1_records = args

    if (mapped := MANUAL_ADM1_MAP.get((admin, adm1_up))):
        return ("keep", mapped)

    if (admin, adm1_up) in NATIONAL_LEVEL:
        return ("national", admin)

    # Case-insensitive match against pre-filtered country L1 records
    for name, name_upper in l1_records:
        if name_upper == adm1_up:
            return ("keep", name)

    return ("warn", f"{admin!r} / {adm_1_name!r}")


def data():
    df = pd.read_csv(CSV_PATH)    
    gdf = gpd.read_file(SHP_PATH)
    df = df[~((df["adm_0_name"].str.upper() == "INDONESIA") & (df["adm_1_name"].str.upper() == "BANGKA BELITUNG"))]

    df["admin_key"] = df["adm_0_name"].apply(
        lambda x: COUNTRY_MAP.get(x, x.title())
    )
    df["adm1_upper"] = df["adm_1_name"].str.upper()
    

    print("Dissolving L2 → L1 (parallel)...")
    l1_gdf = dissolve_to_l1(gdf)
    l1_gdf["name_upper"] = l1_gdf["name"].str.upper()

    csv_pairs = df[["admin_key", "adm_1_name", "adm1_upper"]].drop_duplicates()

    # Pre-group L1 records by country so workers don't receive the full GDF
    l1_by_country = {
        admin: list(zip(grp["name"], grp["name_upper"]))
        for admin, grp in l1_gdf.groupby("admin")
    }

    tasks = [
        (row["admin_key"], row["adm_1_name"], row["adm1_upper"],
         l1_by_country.get(row["admin_key"], []))
        for _, row in csv_pairs.iterrows()
    ]

    print("Resolving CSV pairs (parallel)...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(resolve_csv_pair, tasks)

    keep_names: set[str] = set()
    national_countries: set[str] = set()
    national_rows: list[gpd.GeoDataFrame] = []

    for tag, value in results:
        if tag == "keep":
            keep_names.add(value)
        elif tag == "national":
            national_countries.add(value)
        else:  # warn
            print(f"[WARN] No shapefile match for: {value}")
    
    from collections import Counter

    resolved_names = [value for tag, value in results if tag == "keep"]
    dupes = {name: count for name, count in Counter(resolved_names).items() if count > 1}
    print("Names resolved by multiple CSV rows:", dupes)

    for country in national_countries:
        country_polys = gdf[gdf["gaul0_name"] == country]
        if country_polys.empty:
            print(f"[WARN] No L2 polygons found for country: {country!r}")
            continue
        dissolved_geom = country_polys.geometry.union_all()
        national_rows.append(
            gpd.GeoDataFrame(
                [{"name": country, "admin": country, "geometry": dissolved_geom}],
                crs=gdf.crs,
            )
        )

    filtered = l1_gdf[l1_gdf["name"].isin(keep_names)].drop(columns=["name_upper"])

    if national_rows:
        filtered = pd.concat([filtered] + national_rows, ignore_index=True)
        filtered = gpd.GeoDataFrame(filtered, geometry="geometry", crs=gdf.crs)

    print(f"Original GAUL L2 rows  : {len(gdf)}")
    print(f"Dissolved L1 rows      : {len(l1_gdf)}")
    print(f"Filtered output rows   : {len(filtered)}")
    print(f"  — matched L1 regions : {len(filtered) - len(national_rows)}")
    print(f"  — national dissolves : {len(national_rows)}")
    print(f"Saving to {OUT_PATH}")
    filtered.to_parquet(OUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    data()