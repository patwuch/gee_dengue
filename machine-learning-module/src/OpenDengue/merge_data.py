import pandas as pd
from pathlib import Path
import json

# ── Paths ─────────────────────────────────────────────────────────────────────


ERA5_PATH     = Path(snakemake.input.era5)
CHIRPS_PATH   = Path(snakemake.input.chirps)
LST_PATH      = Path(snakemake.input.modis_lst)
NDVI_EVI_PATH = Path(snakemake.input.modis_veg)
LULC_PATH     = Path(snakemake.input.modis_lulc)
POP_PATH      = Path(snakemake.input.worldpop)
DENGUE_PATH   = Path(snakemake.input.dengue)

MERGED_PATH   = Path(snakemake.output.merged)

JOIN_KEYS = ["admin", "name", "year_month"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp()
    return df


def expand_annual_to_monthly(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Repeat annual rows for each month of the year."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["Date"]).dt.year
    df = df.merge(pd.DataFrame({"month": range(1, 13)}), how="cross")
    df["year_month"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df[["admin", "name", "year_month"] + value_cols]


def histogram_to_pct(hist) -> dict:
    """Convert a pixel-count histogram dict/string to fractional percentages."""
    if isinstance(hist, str):
        hist = json.loads(hist)
    if not isinstance(hist, dict) or not hist or sum(hist.values()) == 0:
        return {}
    total = sum(hist.values())
    return {int(k): v / total for k, v in hist.items()}


# ── Name mappings ─────────────────────────────────────────────────────────────
COUNTRY_MAP = {
    "BRUNEI DARUSSALAM":                  "Brunei Darussalam",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC":   "Lao People's Democratic Republic",
    "VIET NAM":                           "Viet Nam",
}

NATIONAL_LEVEL = {
    "Cambodia",
    "Lao People's Democratic Republic",
    "Singapore",
    "Viet Nam",
}

MANUAL_ADM1_MAP = {
    # Brunei
    ("Brunei Darussalam", "BELAIT DISTRICT"):       "Belait",
    ("Brunei Darussalam", "BRUNEI MUARA DISTRICT"): "Brunei And Muara",
    ("Brunei Darussalam", "TEMBURONG DISTRICT"):    "Temburong",
    ("Brunei Darussalam", "TUTONG DISTRICT"):       "Tutong",
    # Indonesia
    ("Indonesia", "BABEL"):                "Kepulauan Bangka Belitung",
    ("Indonesia", "BANGKA BELITUNG"):      "Kepulauan Bangka Belitung",
    ("Indonesia", "D.I YOGYA"):            "Daerah Istimewa Yogyakarta",
    ("Indonesia", "KALIMANTAN SELATA"):    "Kalimantan Selatan",
    ("Indonesia", "KEPULAUAN-RIAU"):       "Kepulauan Riau",
    ("Indonesia", "NUSATENGGARA BARAT"):   "Nusa Tenggara Barat",
    ("Indonesia", "NUSATENGGARA TIMUR"):   "Nusa Tenggara Timur",
    ("Indonesia", "SULAWESI SELATA"):      "Sulawesi Selatan",
    ("Indonesia", "SUMATERA SELATA"):      "Sumatera Selatan",
    # Malaysia — explicit mapping for all states to match shapefile names
    ("Malaysia", "JOHOR"):                 "Johor",
    ("Malaysia", "KEDAH"):                 "Kedah",
    ("Malaysia", "KELANTAN"):              "Kelantan",
    ("Malaysia", "MELAKA"):                "Melaka",
    ("Malaysia", "NEGERI SEMBILAN"):       "Negeri Sembilan",
    ("Malaysia", "PAHANG"):                "Pahang",
    ("Malaysia", "PERAK"):                 "Perak",
    ("Malaysia", "PERLIS"):                "Perlis",
    ("Malaysia", "PULAU PINANG"):          "Pulau Pinang",
    ("Malaysia", "SABAH"):                 "Sabah",
    ("Malaysia", "SARAWAK"):               "Sarawak",
    ("Malaysia", "SELANGOR"):              "Selangor",
    ("Malaysia", "TERENGGANU"):            "Terengganu",
    ("Malaysia", "KUALA LUMPUR"):          "W.P. Kuala Lumpur",
    ("Malaysia", "LABUAN"):                "W.P. Labuan",
    # Myanmar
    ("Myanmar", "AYAYARWADDY"):            "Ayeyarwady",
    ("Myanmar", "BAGO (E)"):               "Bago (East)",
    ("Myanmar", "BAGO (W)"):               "Bago (West)",
    ("Myanmar", "NAYPYITAW"):              "Nay Pyi Taw",
    ("Myanmar", "SHAN (N)"):               "Shan (North)",
    ("Myanmar", "SHAN (S)"):               "Shan (South)",
    # Philippines
    ("Philippines", "REGION 2"):           "Region Ii (Cagayan Valley)",
    ("Philippines", "REGION 4"):           "Region Iv-A (Calabarzon)",
    # Thailand
    ("Thailand", "BUNGKAN"):               "Bueng Kan",
    ("Thailand", "BURIRAM"):               "Buri Ram",
    ("Thailand", "CHAINAT"):               "Chai Nat",
    ("Thailand", "CHONBURI"):              "Chon Buri",
    ("Thailand", "KAMPAENG PHET"):         "Kamphaeng Phet",
    ("Thailand", "LOPBURI"):               "Lop Buri",
    ("Thailand", "NONG BUA LAMPHU"):       "Nong Bua Lam Phu",
    ("Thailand", "PHACHINBURI"):           "Prachin Buri",
    ("Thailand", "PHRA NAKHON SI AYUDHYA"): "Phra Nakhon Si Ayutthaya",
    ("Thailand", "PRACHUAP KHILIKHAN"):    "Prachuap Khiri Khan",
    ("Thailand", "SAMUT PRAKARN"):         "Samut Prakan",
    ("Thailand", "SAMUT SONGKHAM"):        "Samut Songkhram",
    ("Thailand", "SI SAKET"):              "Si Sa Ket",
    ("Thailand", "SINGBURI"):              "Sing Buri",
    ("Thailand", "SUPHANBURI"):            "Suphan Buri",
    ("Thailand", "TRAD"):                  "Trat",
}


def resolve_shp_name(admin_key: str, adm1_upper: str) -> str:
    """Map a raw ADM1 name to the canonical shapefile name."""
    if admin_key in NATIONAL_LEVEL:
        return admin_key
    return MANUAL_ADM1_MAP.get((admin_key, adm1_upper), adm1_upper.title())


# ── Build env dataset ─────────────────────────────────────────────────────────
print("Processing ERA5_LAND...")
era5      = to_monthly(pd.read_parquet(ERA5_PATH))
era5_cols = [c for c in era5.columns if c.endswith(("_sum", "_mean"))]
era_monthly = era5.groupby(JOIN_KEYS)[era5_cols].mean().reset_index()
print(f"  ERA5 shape: {era_monthly.shape}")

print("Processing CHIRPS...")
chirps         = to_monthly(pd.read_parquet(CHIRPS_PATH))
chirps_monthly = chirps.groupby(JOIN_KEYS)["precipitation_sum"].mean().reset_index()
print(f"  CHIRPS shape: {chirps_monthly.shape}")

print("Processing MODIS_LST...")
lst         = to_monthly(pd.read_parquet(LST_PATH))
lst_monthly = lst.groupby(JOIN_KEYS)[["LST_Day_1km_mean", "LST_Night_1km_mean"]].mean().reset_index()
print(f"  MODIS_LST shape: {lst_monthly.shape}")

print("Processing MODIS_NDVI_EVI...")
ndvi         = to_monthly(pd.read_parquet(NDVI_EVI_PATH))
ndvi_monthly = ndvi.groupby(JOIN_KEYS)[["NDVI_mean", "EVI_mean"]].mean().reset_index()
print(f"  MODIS_NDVI_EVI shape: {ndvi_monthly.shape}")

print("Processing MODIS_LULC...")
lulc         = pd.read_parquet(LULC_PATH)
hist_expanded = (
    lulc["LC_Type1_histogram"]
    .apply(histogram_to_pct)
    .apply(pd.Series)
)
hist_expanded        = hist_expanded[sorted(hist_expanded.columns)]
hist_expanded.columns = [f"LC_Type1_pct_class{c}" for c in hist_expanded.columns]
hist_expanded        = hist_expanded.fillna(0)
lulc_hist    = pd.concat([lulc[["admin", "name", "Date"]], hist_expanded], axis=1)
lulc_monthly = expand_annual_to_monthly(lulc_hist, list(hist_expanded.columns))
print(f"  MODIS_LULC shape: {lulc_monthly.shape}")

print("Processing WorldPop...")
pop         = pd.read_parquet(POP_PATH)
pop_monthly = expand_annual_to_monthly(pop, ["population_sum"])
print(f"  WorldPop shape: {pop_monthly.shape}")

print("\nMerging env datasets...")
combined_env = chirps_monthly.copy()
for ds_name, ds_df in [
    ("ERA5_LAND",      era_monthly),
    ("MODIS_LST",      lst_monthly),
    ("MODIS_NDVI_EVI", ndvi_monthly),
    ("MODIS_LULC",     lulc_monthly),
    ("WorldPop",       pop_monthly),
]:
    combined_env = combined_env.merge(ds_df, on=JOIN_KEYS, how="outer")
    print(f"  after {ds_name}: {combined_env.shape}")

combined_env.rename(columns={"year_month": "Date"}, inplace=True)
id_cols   = ["admin", "name", "Date"]
data_cols = [c for c in combined_env.columns if c not in id_cols]
combined_env  = combined_env[id_cols + data_cols]
print(f"\nCombined env columns: {combined_env.columns.tolist()}")
# ── Load and normalise dengue ─────────────────────────────────────────────────
env = combined_env.copy()
print("\nLoading dengue...")
dengue = pd.read_csv(
    DENGUE_PATH,
    parse_dates=["calendar_start_date"],
    usecols=[
        "adm_0_name", "adm_1_name", "calendar_start_date",
        "dengue_total", "S_res", "T_res", "data_quality",
    ],
)

dengue = dengue[~(
    (dengue["adm_0_name"].str.upper() == "INDONESIA") &
    (dengue["adm_1_name"].str.upper() == "BANGKA BELITUNG")
)]

dengue["year_month"] = dengue["calendar_start_date"].dt.to_period("M").dt.to_timestamp().astype("datetime64[us]")
env["Date"]          = pd.to_datetime(env["Date"]).astype("datetime64[us]")
dengue["year_month"] = pd.to_datetime(dengue["year_month"]).astype("datetime64[us]")

print(f"Env date range   : {env['Date'].min()} → {env['Date'].max()}")
print(f"Dengue date range: {dengue['year_month'].min()} → {dengue['year_month'].max()}")

# ── Normalise dengue keys to match env (shapefile) names ─────────────────────
dengue["adm_0_name"] = dengue["adm_0_name"].str.strip().map(COUNTRY_MAP).fillna(
    dengue["adm_0_name"].str.strip().str.title()
)
dengue["adm_1_name"] = dengue.apply(
    lambda r: resolve_shp_name(r["adm_0_name"], r["adm_1_name"].strip()),
    axis=1,
)

# ── Key diagnostics ───────────────────────────────────────────────────────────
env_keys    = set(zip(env["admin"],          env["name"]))
dengue_keys = set(zip(dengue["adm_0_name"],  dengue["adm_1_name"]))
still_missing = dengue_keys - env_keys
print(f"\nStill unmatched dengue keys ({len(still_missing)}):")
for k in sorted(still_missing):
    print(" ", k)

# ── Merge env onto dengue (dengue is authoritative) ──────────────────────────
merged = dengue.merge(
    env,
    left_on=["adm_0_name", "adm_1_name", "year_month"],
    right_on=["admin", "name", "Date"],
    how="left",
).drop(columns=["admin", "name", "Date"])



merged["data_quality"] = merged["data_quality"].fillna("UNMATCHED_ENV")
# ── Incidence rate ────────────────────────────────────────────────────────────
merged["IR"] = merged["dengue_total"] / merged["population_sum"] * 100_000

# IR values derived from imputed, missing, or unmatched dengue counts are flagged
# so downstream models can mask or down-weight them without re-deriving from nullity.
_UNRELIABLE = {
    "IMPUTED_BORROW",
    "IMPUTED_NAN",
    "PLACEHOLDER_HAS_ADM1_YEAR",
    "PLACEHOLDER_HAS_ADM0_YEAR",
    "PLACEHOLDER_NO_SOURCE",
    "UNMATCHED_ENV",
}

merged["IR_quality"] = merged["data_quality"].where(
    merged["data_quality"].isin(_UNRELIABLE), other="RELIABLE"
)

# ── Diagnostics ───────────────────────────────────────────────────────────────
total   = len(merged)
matched = merged["dengue_total"].notna().sum()

print(f"\nMerged shape          : {merged.shape}")
print(f"Rows with dengue_total: {matched} / {total} ({100 * matched / total:.1f}%)")
print(f"\nData quality breakdown:\n{merged['data_quality'].value_counts(dropna=False)}")
print(f"\nIR quality breakdown:\n{merged['IR_quality'].value_counts(dropna=False)}")
print(f"\nNA counts:\n{merged[['dengue_total', 'population_sum', 'IR']].isna().sum()}")
print(f"\nIR summary:\n{merged['IR'].describe()}")

# ── Save ──────────────────────────────────────────────────────────────────────
merged.to_csv(MERGED_PATH, index=False)
print(f"\nSaved: {MERGED_PATH}")