import random
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import Queen
from matplotlib.colors import Normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
_PIPELINE_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    _PIPELINE_ROOT
    / "data" / "interim" / "OpenDengue"
    / "filtered_sea_2011_2018_SLVC_imputed.csv"
)
POP_PATH = (
    _PIPELINE_ROOT
    / "data" / "raw" / "OpenDengue"
    / "WorldPop_2011-01-01_to_2018-12-31.parquet"
)
GEO_PATH = (
    _PIPELINE_ROOT
    / "data" / "external" / "geoparquet"
    / "gaul_2024_sea_filtered.parquet"
)
RESULTS_DIR = _PIPELINE_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
N_SLICES = 10

# ── Name mappings (mirrors script 5) ─────────────────────────────────────────
COUNTRY_MAP = {
    "BRUNEI DARUSSALAM": "Brunei Darussalam",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "Lao People's Democratic Republic",
    "VIET NAM": "Viet Nam",
}

MANUAL_ADM1_MAP = {
    ("Brunei Darussalam", "BELAIT DISTRICT"): "Belait",
    ("Brunei Darussalam", "BRUNEI MUARA DISTRICT"): "Brunei And Muara",
    ("Brunei Darussalam", "TEMBURONG DISTRICT"): "Temburong",
    ("Brunei Darussalam", "TUTONG DISTRICT"): "Tutong",
    ("Indonesia", "BABEL"): "Kepulauan Bangka Belitung",
    ("Indonesia", "BANGKA BELITUNG"): "Kepulauan Bangka Belitung",
    ("Malaysia", "KUALA LUMPUR"): "W.P. Kuala Lumpur",
    ("Malaysia", "LABUAN"): "W.P. Labuan",
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

NATIONAL_LEVEL = {
    ("Cambodia", "CAMBODIA"),
    ("Lao People's Democratic Republic", "LAO PEOPLE'S DEMOCRATIC REPUBLIC"),
    ("Singapore", "SINGAPORE"),
    ("Viet Nam", "VIET NAM"),
}


def resolve_shp_name(admin_key: str, adm1_upper: str) -> str | None:
    if (admin_key, adm1_upper) in NATIONAL_LEVEL:
        return admin_key
    manual = MANUAL_ADM1_MAP.get((admin_key, adm1_upper))
    if manual:
        return manual
    return adm1_upper.title()


# ── Load population data ──────────────────────────────────────────────────────
print("Loading WorldPop population data...")
pop_df = pd.read_parquet(POP_PATH)
pop_df["year"] = pd.to_datetime(pop_df["Date"].astype(str)).dt.year
pop_lkp = (
    pop_df[["admin", "name", "year", "population_sum"]]
    .rename(columns={"name": "shp_name_geo"})
    .copy()
)

# ── Load dengue and geometry data ─────────────────────────────────────────────
print("Loading dengue and geometry data...")
df = pd.read_csv(CSV_PATH, parse_dates=["calendar_start_date"])
gdf_base = gpd.read_parquet(GEO_PATH)

# Keep only Admin1 monthly rows
df = df[(df["S_res"] == "Admin1") & (df["T_res"] == "Month")].copy()

# Build shapefile name lookup
df["admin_key"] = df["adm_0_name"].apply(lambda x: COUNTRY_MAP.get(x, x.title()))
df["adm1_upper"] = df["adm_1_name"].str.upper()
df["shp_name"] = df.apply(
    lambda r: resolve_shp_name(r["admin_key"], r["adm1_upper"]), axis=1
)
df = df[df["shp_name"].notna()].copy()

df["year_month"] = df["calendar_start_date"].dt.to_period("M").astype(str)
df["year"] = df["calendar_start_date"].dt.year

# Aggregate dengue_total; retain year for WorldPop join
df_agg = (
    df.groupby(["shp_name", "year_month", "year"], as_index=False)["dengue_total"]
    .sum(min_count=1)
)

# ── Build joined GeoDataFrame ─────────────────────────────────────────────────
# Include 'admin' (country) from gdf_base for the WorldPop join key
joined = gdf_base[["admin", "name", "geometry"]].merge(
    df_agg, left_on="name", right_on="shp_name", how="inner"
)
joined = joined.rename(columns={"name": "shp_name_geo"})

# ── Join WorldPop and compute incidence rate per 100 000 population ───────────
joined = joined.merge(pop_lkp, on=["admin", "shp_name_geo", "year"], how="left")
joined["ir_per_100k"] = joined["dengue_total"] / joined["population_sum"] * 100_000

n_no_pop = joined["population_sum"].isna().sum()
if n_no_pop:
    print(f"[WARN] {n_no_pop} rows have no WorldPop match — IR will be NaN for these.")

# ── Randomly select 10 time slices ───────────────────────────────────────────
all_slices = sorted(joined["year_month"].unique())
random.seed(RANDOM_SEED)
selected_slices = sorted(random.sample(all_slices, min(N_SLICES, len(all_slices))))

print(f"\nRandomly selected {len(selected_slices)} time slices (seed={RANDOM_SEED}):")
for s in selected_slices:
    print(f"  {s}")

# ── Moran's I per time slice (on IR per 100k) ─────────────────────────────────
results = []

for ym in selected_slices:
    slice_gdf = joined[joined["year_month"] == ym].copy()
    slice_gdf = slice_gdf.dropna(subset=["ir_per_100k"])
    slice_gdf = slice_gdf.reset_index(drop=True)

    if len(slice_gdf) < 4:
        print(f"[SKIP] {ym}: only {len(slice_gdf)} valid regions after NaN removal.")
        results.append({"year_month": ym, "n": len(slice_gdf),
                        "moran_i": np.nan, "p_sim": np.nan, "z_sim": np.nan})
        continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w = Queen.from_dataframe(slice_gdf, silence_warnings=True)
        except Exception:
            from libpysal.weights import KNN
            w = KNN.from_dataframe(slice_gdf, k=4)

    w.transform = "r"

    y = slice_gdf["ir_per_100k"].values.astype(float)
    mi = Moran(y, w, permutations=999)

    results.append({
        "year_month": ym,
        "n": len(slice_gdf),
        "moran_i": mi.I,
        "expected_i": mi.EI,
        "z_sim": mi.z_sim,
        "p_sim": mi.p_sim,
        "significant": mi.p_sim < 0.05,
    })
    sig = "**" if mi.p_sim < 0.01 else ("*" if mi.p_sim < 0.05 else "")
    print(
        f"  {ym}  I={mi.I:+.4f}  E[I]={mi.EI:.4f}  "
        f"z={mi.z_sim:+.3f}  p={mi.p_sim:.4f} {sig}  n={len(slice_gdf)}"
    )

results_df = pd.DataFrame(results)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n── Moran's I Results (IR per 100k) ──────────────────────────────────")
print(results_df.to_string(index=False, float_format="{:.4f}".format))

# ── Choropleth maps ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, N_SLICES // 2, figsize=(20, 8))
axes = axes.flatten()

norm_global = Normalize(
    vmin=joined["ir_per_100k"].quantile(0.05),
    vmax=joined["ir_per_100k"].quantile(0.95),
)

for ax, ym in zip(axes, selected_slices):
    slice_gdf = joined[joined["year_month"] == ym].copy().reset_index(drop=True)
    slice_gdf["plot_val"] = np.log1p(slice_gdf["ir_per_100k"].fillna(0))
    slice_gdf.plot(
        column="plot_val",
        ax=ax,
        cmap="YlOrRd",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        linewidth=0.3,
        edgecolor="white",
    )

    row = results_df[results_df["year_month"] == ym].iloc[0]
    sig_label = (
        "p<0.01" if row["p_sim"] < 0.01
        else "p<0.05" if row["p_sim"] < 0.05
        else "n.s."
    )
    ax.set_title(
        f"{ym}\nI={row['moran_i']:.3f}  {sig_label}",
        fontsize=8,
    )
    ax.axis("off")

fig.suptitle(
    f"Dengue IR/100k — Moran's I, {N_SLICES} random monthly slices (SEA, 2011–2018)",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
slices_path = RESULTS_DIR / "morans_i_slices.png"
plt.savefig(slices_path, dpi=150, bbox_inches="tight")
print(f"\nMap saved to {slices_path}")
plt.show()

# ── Moran scatter plot for the most significant slice ────────────────────────
best = results_df.dropna(subset=["moran_i"]).sort_values("p_sim").iloc[0]
ym_best = best["year_month"]

slice_gdf = (
    joined[joined["year_month"] == ym_best]
    .dropna(subset=["ir_per_100k"])
    .reset_index(drop=True)
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    w_best = Queen.from_dataframe(slice_gdf, silence_warnings=True)
w_best.transform = "r"

y_best = slice_gdf["ir_per_100k"].values.astype(float)
mi_best = Moran(y_best, w_best, permutations=999)

fig2, ax2 = plt.subplots(figsize=(6, 5))
y_std = (y_best - y_best.mean()) / y_best.std()

from libpysal.weights import lag_spatial
y_lag = lag_spatial(w_best, y_best)
y_lag_std = (y_lag - y_lag.mean()) / y_lag.std()

ax2.scatter(y_std, y_lag_std, alpha=0.6, edgecolors="k", linewidths=0.4)
ax2.axhline(0, color="grey", lw=0.8, linestyle="--")
ax2.axvline(0, color="grey", lw=0.8, linestyle="--")

m, b = np.polyfit(y_std, y_lag_std, 1)
x_line = np.linspace(y_std.min(), y_std.max(), 100)
ax2.plot(x_line, m * x_line + b, color="tomato", lw=1.5,
         label=f"slope ≈ I = {mi_best.I:.4f}")

ax2.set_xlabel("Standardised IR per 100k")
ax2.set_ylabel("Spatial lag (standardised)")
ax2.set_title(f"Moran scatter — {ym_best}\nI={mi_best.I:.4f}  p={mi_best.p_sim:.4f}")
ax2.legend(fontsize=9)
plt.tight_layout()
scatter_path = RESULTS_DIR / "morans_i_scatter.png"
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
print(f"Moran scatter saved to {scatter_path}")
plt.show()
