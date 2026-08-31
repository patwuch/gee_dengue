"""
Weekly BSA-level dengue incidence for the 2015 and 2023 Tainan/Kaohsiung outbreaks.

Case location: residence BSA (居住..., column '最小統計區' / CODEBASE), restricted to
locally-acquired cases (是否境外移入 == '否'), following the convention already used in
dengue-infection-module/notebooks/tn_dengue_cleaning.ipynb. Population is the BSA
population snapshot for the matching outbreak year (Tainan: Dec of each year;
Kaohsiung: Jun of each year -- the only snapshots available for each city).

Outputs (data/processed/dengue-infection/TW_DENGUE/):
  - tainan_kaohsiung_2015_2023_weekly_cases.csv   city-level weekly case counts (both outbreak years)
  - bsa_weekly_incidence_2015_2023.csv            BSA x week case counts + incidence per 100k
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "dengue-infection" / "TW_DENGUE"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "dengue-infection" / "TW_DENGUE"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY_MAP = {"台南市": "Tainan", "高雄市": "Kaohsiung"}
OUTBREAK_YEARS = [2015, 2023]


def year_weeks(year: int) -> pd.DatetimeIndex:
    """Monday-anchored weeks covering `year`, aligned with `week = date - weekday`."""
    jan1 = pd.Timestamp(f"{year}-01-01")
    start = jan1 - pd.to_timedelta(jan1.weekday(), unit="D")
    return pd.date_range(start, f"{year}-12-31", freq="7D")

POP_FILES = {
    ("Tainan", 2015): RAW_DIR / "BSA_POP_TN_2015.csv",
    ("Tainan", 2023): RAW_DIR / "BSA_POP_TN_2023.csv",
    ("Kaohsiung", 2015): RAW_DIR / "KH_BSA_population" / "2015年6月高雄市統計區人口統計_最小統計區" / "104年6月高雄市統計區人口統計_最小統計區.csv",
    ("Kaohsiung", 2023): RAW_DIR / "KH_BSA_population" / "2023年6月高雄市統計區人口統計_最小統計區" / "112年6月高雄市統計區人口統計_最小統計區.csv",
}


def load_pop(path: Path) -> pd.DataFrame:
    # The 2015 snapshots are Big5, the 2023 snapshots are UTF-8 -- detect per file.
    # CPython's Big5 incremental decoder also mis-splits multibyte sequences across
    # TextIOWrapper's read buffer, so decode the whole byte string at once rather
    # than via `open(encoding=...)` or pandas' own encoding handling.
    import io
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("big5")
    df = pd.read_csv(io.StringIO(text), skiprows=[1])
    df["P_CNT"] = pd.to_numeric(df["P_CNT"], errors="coerce")
    return df[["CODEBASE", "P_CNT"]].dropna(subset=["CODEBASE"]).drop_duplicates("CODEBASE")


def main():
    daily = pd.read_csv(RAW_DIR / "Dengue_Daily.csv")
    daily["發病日"] = pd.to_datetime(daily["發病日"], errors="coerce")
    daily = daily.dropna(subset=["發病日"])
    daily["year"] = daily["發病日"].dt.year
    daily["week"] = daily["發病日"] - pd.to_timedelta(daily["發病日"].dt.weekday, unit="D")

    sub = daily[daily["感染縣市"].isin(CITY_MAP) & daily["year"].isin(OUTBREAK_YEARS)].copy()
    sub["city"] = sub["感染縣市"].map(CITY_MAP)

    # --- 1. city-level weekly case counts for the two outbreak years ---
    city_weekly = (
        sub.groupby(["city", "year", "week"])["確定病例數"].sum().reset_index()
        .rename(columns={"確定病例數": "cases"})
    )
    full_rows = []
    for (city, year), grp in city_weekly.groupby(["city", "year"]):
        full_idx = year_weeks(year)
        s = grp.set_index("week")["cases"].reindex(full_idx, fill_value=0)
        full_rows.append(pd.DataFrame({"city": city, "year": year, "week": full_idx, "cases": s.values}))
    city_weekly_full = pd.concat(full_rows, ignore_index=True).sort_values(["city", "year", "week"])
    city_out = OUT_DIR / "tainan_kaohsiung_2015_2023_weekly_cases.csv"
    city_weekly_full.to_csv(city_out, index=False)
    print(f"wrote {city_out} ({len(city_weekly_full)} rows)")

    # --- 2. BSA-level weekly incidence, local cases only ---
    local = sub[sub["是否境外移入"] == "否"].copy()
    dropped_imported = len(sub) - len(local)
    local = local.dropna(subset=["最小統計區"])
    dropped_no_bsa = sub[sub["是否境外移入"] == "否"]["最小統計區"].isna().sum()
    print(f"dropped {dropped_imported} imported cases, {dropped_no_bsa} local cases missing residence BSA")

    bsa_frames = []
    for city, year in [("Tainan", 2015), ("Tainan", 2023), ("Kaohsiung", 2015), ("Kaohsiung", 2023)]:
        pop = load_pop(POP_FILES[(city, year)])
        cases_yr = local[(local["city"] == city) & (local["year"] == year)]

        bsa_case_week = (
            cases_yr.groupby(["最小統計區", "week"])["確定病例數"].sum()
            .reset_index().rename(columns={"最小統計區": "CODEBASE", "確定病例數": "cases"})
        )

        full_weeks = year_weeks(year)
        grid = pd.MultiIndex.from_product([pop["CODEBASE"], full_weeks], names=["CODEBASE", "week"]).to_frame(index=False)
        grid = grid.merge(bsa_case_week, on=["CODEBASE", "week"], how="left")
        grid["cases"] = grid["cases"].fillna(0).astype(int)
        grid = grid.merge(pop, on="CODEBASE", how="left")
        grid["incidence_per_100k"] = grid["cases"] / grid["P_CNT"] * 100_000
        grid["city"] = city
        grid["year"] = year

        unmatched = bsa_case_week.loc[~bsa_case_week["CODEBASE"].isin(pop["CODEBASE"]), "cases"].sum()
        if unmatched:
            print(f"{city} {year}: {unmatched} cases in BSA codes absent from the population snapshot (boundary drift)")

        bsa_frames.append(grid[["city", "year", "week", "CODEBASE", "P_CNT", "cases", "incidence_per_100k"]])

    bsa_out_df = pd.concat(bsa_frames, ignore_index=True).rename(columns={"P_CNT": "population"})
    bsa_out = OUT_DIR / "bsa_weekly_incidence_2015_2023.csv"
    bsa_out_df.to_csv(bsa_out, index=False)
    print(f"wrote {bsa_out} ({len(bsa_out_df)} rows, {bsa_out_df['CODEBASE'].nunique()} unique BSAs)")

    pop_totals = {
        (city, year): load_pop(POP_FILES[(city, year)])["P_CNT"].sum()
        for city, year in POP_FILES
    }
    case_totals = local.groupby(["city", "year"])["確定病例數"].sum()
    for (city, year) in POP_FILES:
        n_bsa = bsa_out_df[(bsa_out_df.city == city) & (bsa_out_df.year == year)]["CODEBASE"].nunique()
        cases = case_totals.get((city, year), 0)
        pop = pop_totals[(city, year)]
        print(f"{city} {year}: {n_bsa} BSAs, {pop:,.0f} population, {cases:,.0f} local cases -> {cases/pop*100_000:,.1f} per 100k (annual)")


if __name__ == "__main__":
    main()
