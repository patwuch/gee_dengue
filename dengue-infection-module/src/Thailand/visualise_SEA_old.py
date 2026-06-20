import sys
import warnings
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── CONFIG ────────────────────────────────────────────────────────────────────
def _find_git_root(start):
    from pathlib import Path
    for p in (start,) + tuple(start.parents):
        if (p / ".git").exists():
            return p
    return start

from pathlib import Path
INPUT_PATH = str(_find_git_root(Path(__file__).resolve()) / "data" / "raw" / "dengue-infection" / "Temporal_extract_V1_3.csv")
OUTPUT_PATH = "coverage_sea_long_temporal.html"
DATE_START = "1991-01-01"
DATE_END = "2025-12-31"

NATIONAL_ROW_LABEL = "― National (Admin0) ―"

COUNTRIES = [
    "MALAYSIA", "INDONESIA", "VIET NAM", "THAILAND", "PHILIPPINES",
    "SINGAPORE", "LAO PEOPLE'S DEMOCRATIC REPUBLIC", "CAMBODIA",
    "MYANMAR", "TIMOR-LESTE", "BRUNEI DARUSSALAM",
]

DISPLAY_NAMES = {
    "CAMBODIA": "Cambodia", "INDONESIA": "Indonesia",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "Laos", "MALAYSIA": "Malaysia",
    "PHILIPPINES": "Philippines", "SINGAPORE": "Singapore",
    "THAILAND": "Thailand", "VIET NAM": "Viet Nam",
    "TIMOR-LESTE": "Timor-Leste", "BRUNEI DARUSSALAM": "Brunei",
    "MYANMAR": "Myanmar",
}

RESOLUTION_ALIASES = {
    "Week": "Weekly", "week": "Weekly", "Weekly": "Weekly",
    "WEEKLY": "Weekly", "wk": "Weekly", "W": "Weekly",
    "Month": "Month", "month": "Month", "Monthly": "Month", "MONTHLY": "Month",
    "Year": "Year", "year": "Year", "Annual": "Year", "ANNUAL": "Year",
    "Yearly": "Year", "YEARLY": "Year",
}
RESOLUTION_PRIORITY = {"Year": 1, "Month": 2, "Weekly": 3}
RESOLUTION_ORDER = ["Missing", "Year", "Month", "Weekly"]
RESOLUTION_COLORS = {
    "Missing": "#FFFFFF", "Year": "#AED6F1",
    "Month": "#2980B9", "Weekly": "#1A2D6B",
}
MAX_RESOLUTION_INDEX = len(RESOLUTION_ORDER) - 1
RESOLUTION_COLORSCALE = [
    [i / MAX_RESOLUTION_INDEX, RESOLUTION_COLORS[name]]
    for i, name in enumerate(RESOLUTION_ORDER)
]
RESOLUTION_CODES = {name: idx for idx, name in enumerate(RESOLUTION_ORDER)}

PROVINCE_ALIASES = {
    "MYANMAR": {
        "AYAYARWADDY": "AYEYARWADY", "BAGO (E)": "BAGO (EAST)",
        "BAGO (W": "BAGO (WEST)", "NAYPYITAW": "NAY PYI TAW",
        "SHAN (N)": "SHAN (NORTH)", "SHAN (S)": "SHAN (SOUTH)",
        "MONGAR": None,
    }
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalize_resolution(value):
    """Map raw T_res strings to canonical {Year, Month, Weekly}."""
    if pd.isna(value):
        return value
    label = str(value).strip()
    if not label:
        return value
    normalized = RESOLUTION_ALIASES.get(label, label.title())
    if normalized not in RESOLUTION_ORDER and normalized != "Missing":
        warnings.warn(
            f"[T_res] Unrecognized resolution label after normalization: "
            f"'{label}' → '{normalized}'. Add it to RESOLUTION_ALIASES.",
            stacklevel=2,
        )
    return normalized


def floor_to_month_start(series: pd.Series) -> pd.Series:
    """Snap any date to the first day of its month (fixes week-start dates)."""
    return series.dt.to_period("M").dt.to_timestamp()


def best_resolution(series: pd.Series) -> str:
    """Return the highest-priority T_res label in a group."""
    ranked = series.map(RESOLUTION_PRIORITY)
    if ranked.isna().all():
        return series.iloc[0]
    return series.iloc[ranked.fillna(0).values.argmax()]


def build_resolution_rows(sub: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Given a subset of df with a label column already set, normalize T_res,
    assign rank, expand yearly rows to monthly, then dedup keeping highest rank.
    """
    rows = sub[["adm_1_name", "calendar_start_date", "T_res"]].copy()
    rows = rows.rename(columns={"adm_1_name": label_col})
    rows["T_res"] = rows["T_res"].apply(normalize_resolution)
    rows["rank"] = rows["T_res"].map(RESOLUTION_PRIORITY).fillna(0)

    yearly_mask = rows["T_res"] == "Year"
    if yearly_mask.any():
        expanded = []
        for _, row in rows[yearly_mask].iterrows():
            year = row["calendar_start_date"].year
            for month in range(1, 13):
                new_row = row.copy()
                new_row["calendar_start_date"] = pd.Timestamp(year=year, month=month, day=1)
                expanded.append(new_row)
        rows = pd.concat(
            [rows[~yearly_mask], pd.DataFrame(expanded)],
            ignore_index=True,
        )

    rows = (
        rows
        .sort_values([label_col, "calendar_start_date", "rank"], ascending=[True, True, False])
        .drop_duplicates([label_col, "calendar_start_date"], keep="first")
    )
    return rows


# ── LOAD & VALIDATE ───────────────────────────────────────────────────────────

print(f"Loading: {INPUT_PATH}")
try:
    df = pd.read_csv(INPUT_PATH, low_memory=False)
except FileNotFoundError:
    sys.exit(f"[ERROR] Input file not found: {INPUT_PATH}")

REQUIRED_COLS = {"adm_0_name", "adm_1_name", "calendar_start_date", "T_res", "dengue_total"}
missing_cols = REQUIRED_COLS - set(df.columns)
if missing_cols:
    sys.exit(f"[ERROR] Missing required columns: {missing_cols}")

df["calendar_start_date"] = pd.to_datetime(df["calendar_start_date"], errors="coerce")
n_bad_dates = df["calendar_start_date"].isna().sum()
if n_bad_dates:
    warnings.warn(f"[DATE] {n_bad_dates} rows had unparseable dates and will be dropped.")
    df = df[df["calendar_start_date"].notna()]

raw_tres = df["T_res"].dropna().unique()
unmapped = [v for v in raw_tres if str(v).strip() not in RESOLUTION_ALIASES]
if unmapped:
    warnings.warn(
        f"[T_res] The following raw values are not in RESOLUTION_ALIASES and will "
        f"fall back to .title(): {sorted(unmapped)}\n"
        f"  → Add them explicitly to RESOLUTION_ALIASES to silence this warning."
    )

countries_in_data = set(df["adm_0_name"].unique())
missing_countries = [c for c in COUNTRIES if c not in countries_in_data]
if missing_countries:
    warnings.warn(f"[COUNTRIES] These countries are in COUNTRIES list but absent from data: {missing_countries}")

n_non_first = (df["calendar_start_date"].dt.day != 1).sum()
if n_non_first:
    print(f"[DATE] Flooring {n_non_first} non-month-start dates to month-start "
          f"(e.g. week-start dates from weekly resolution rows).")
df["calendar_start_date"] = floor_to_month_start(df["calendar_start_date"])

all_months = pd.date_range(DATE_START, DATE_END, freq="MS")

# ── PHILIPPINES: collapse adm2 → adm1 ────────────────────────────────────────
ph_mask = df["adm_0_name"] == "PHILIPPINES"
n_ph_before = ph_mask.sum()
ph_collapsed = (
    df[ph_mask]
    .groupby(["adm_0_name", "adm_1_name", "calendar_start_date"], as_index=False)
    .agg(dengue_total=("dengue_total", "sum"), T_res=("T_res", best_resolution))
)
print(f"[PHL] Collapsed {n_ph_before} adm2 rows → {len(ph_collapsed)} adm1 rows.")
df = pd.concat([df[~ph_mask], ph_collapsed], ignore_index=True)

# ── BUILD PER-COUNTRY PIVOT ───────────────────────────────────────────────────
country_data = {}
for country in COUNTRIES:
    sub = df[df["adm_0_name"] == country].copy()

    if sub.empty:
        warnings.warn(f"[{country}] No rows found — skipping.")
        empty_grid = pd.DataFrame(index=["(no data)"], columns=all_months)
        country_data[country] = (empty_grid.notna(), all_months, empty_grid, {})
        continue

    # Province name aliases
    if country in PROVINCE_ALIASES:
        aliases = PROVINCE_ALIASES[country]
        drop_names = {k for k, v in aliases.items() if v is None}
        rename_map = {k: v for k, v in aliases.items() if v is not None}
        n_dropped = sub["adm_1_name"].isin(drop_names).sum()
        if n_dropped:
            print(f"[{country}] Dropping {n_dropped} rows with bad province names: {drop_names}")
        sub = sub[~sub["adm_1_name"].isin(drop_names)]
        sub["adm_1_name"] = sub["adm_1_name"].replace(rename_map)

    # ── Split national vs province rows ──────────────────────────────────────
    national_sub = sub[sub["adm_1_name"].isna()].copy()
    province_sub = sub[sub["adm_1_name"].notna()].copy()

    has_provinces = not province_sub.empty

    # ── Build national row grid (always, regardless of province presence) ─────
    national_grid_col = None  # will be a single-row DataFrame or None
    if not national_sub.empty:
        national_sub["adm_1_name"] = NATIONAL_ROW_LABEL
        nat_rows = build_resolution_rows(national_sub, label_col="adm_1_name")

        n_nat_weekly = (nat_rows["T_res"] == "Weekly").sum()
        if n_nat_weekly:
            print(f"[{country}] {n_nat_weekly} weekly rows in national (Admin0) data.")

        nat_grid = (
            nat_rows.pivot(index="adm_1_name", columns="calendar_start_date", values="T_res")
            .reindex(columns=all_months)
        )
        national_grid_col = nat_grid

        # Check for out-of-range national weekly dates
        nat_weekly_dates = nat_rows.loc[nat_rows["T_res"] == "Weekly", "calendar_start_date"].unique()
        out_of_range = [d for d in nat_weekly_dates if d not in all_months]
        if out_of_range:
            warnings.warn(
                f"[{country}] {len(out_of_range)} national weekly dates fall outside grid range "
                f"({DATE_START}–{DATE_END}): "
                f"{sorted(out_of_range)[:5]}{'...' if len(out_of_range) > 5 else ''}"
            )
    else:
        print(f"[{country}] No national-level (Admin0) rows found.")

    # ── Build province grid ───────────────────────────────────────────────────
    if has_provinces:
        n_prov_national_dropped = sub["adm_1_name"].isna().sum()
        if n_prov_national_dropped:
            print(f"[{country}] National rows handled separately; "
                  f"building province grid from {len(province_sub)} province rows.")

        prov_rows = build_resolution_rows(province_sub, label_col="adm_1_name")

        n_weekly = (prov_rows["T_res"] == "Weekly").sum()
        n_weekly_after = n_weekly  # already deduped inside build_resolution_rows
        if n_weekly == 0:
            print(f"[{country}] No weekly rows in province data (may exist at national level).")
        else:
            print(f"[{country}] {n_weekly_after} weekly province rows retained.")

        provinces = sorted(province_sub["adm_1_name"].unique())
        prov_grid = (
            prov_rows.pivot(index="adm_1_name", columns="calendar_start_date", values="T_res")
            .reindex(columns=all_months)
            .reindex(provinces)
        )
    else:
        # Country has only national-level data; province_sub is empty
        warnings.warn(f"[{country}] No province-level rows found; only national row will appear.")
        prov_grid = pd.DataFrame(columns=all_months)

    # ── Combine: national row pinned at top, provinces below ─────────────────
    if national_grid_col is not None:
        type_grid = pd.concat([national_grid_col, prov_grid])
    else:
        type_grid = prov_grid

    if type_grid.empty:
        warnings.warn(f"[{country}] Combined grid is empty — no data to display.")
        type_grid = pd.DataFrame(index=["(no data)"], columns=all_months)

    # Per-row metadata: flag which rows are national so hover can note it
    is_national_row = {row: (row == NATIONAL_ROW_LABEL) for row in type_grid.index}

    presence = type_grid.notna()
    country_data[country] = (presence, all_months, type_grid, is_national_row)


# ── SUMMARY BAR ───────────────────────────────────────────────────────────────
# Exclude the national row from coverage % (it's not a province-month)
summary_rows = []
for country, (presence, months, type_grid, is_national_row) in country_data.items():
    province_rows = [r for r in type_grid.index if not is_national_row.get(r, False)]
    prov_presence = presence.loc[province_rows] if province_rows else presence.iloc[0:0]
    n_provinces = len(prov_presence)
    expected = n_provinces * len(months)
    observed = int(prov_presence.values.sum()) if expected > 0 else 0
    summary_rows.append({
        "country": DISPLAY_NAMES[country],
        "provinces": n_provinces,
        "coverage_pct": 100 * observed / expected if expected > 0 else 0,
    })
summary_df = pd.DataFrame(summary_rows).sort_values("coverage_pct", ascending=True)

# ── LAYOUT ────────────────────────────────────────────────────────────────────
n_countries = len(COUNTRIES)
row_heights = [0.12] + [max(0.04, len(country_data[c][0]) / 350) for c in COUNTRIES]
total = sum(row_heights)
row_heights = [h / total for h in row_heights]

fig = make_subplots(
    rows=n_countries + 1, cols=1,
    subplot_titles=["Overall Coverage (% of expected province-months)"]
    + [DISPLAY_NAMES[c] for c in COUNTRIES],
    row_heights=row_heights,
    vertical_spacing=0.02,
)

fig.add_trace(
    go.Bar(
        x=summary_df["coverage_pct"], y=summary_df["country"],
        orientation="h", marker_color="#2196F3",
        text=[f"{v:.1f}%" for v in summary_df["coverage_pct"]],
        textposition="outside", showlegend=False,
    ),
    row=1, col=1,
)
fig.update_xaxes(range=[0, 110], row=1, col=1, title_text="% covered")

for i, country in enumerate(COUNTRIES, start=2):
    _, months, type_grid, is_national_row = country_data[country]
    provinces = list(type_grid.index)
    display_grid = type_grid.fillna("Missing")
    display_grid = display_grid.where(display_grid.isin(RESOLUTION_ORDER), "Missing")
    z = display_grid.replace(RESOLUTION_CODES).values
    resolution_labels = display_grid.values

    # Build customdata: for national row, append the clarifying note
    customdata = []
    for row_label, res_row in zip(provinces, resolution_labels):
        if is_national_row.get(row_label, False):
            customdata.append([
                f"{v} (national aggregate only — not a province)" if v != "Missing" else v
                for v in res_row
            ])
        else:
            customdata.append(list(res_row))

    year_ticks = [m for m in months if m.month == 1]
    year_tick_indices = [list(months).index(m) for m in year_ticks]
    year_tick_labels = [str(m.year) for m in year_ticks]
    text_dates = [[m.strftime("%b %Y") for m in months] for _ in provinces]

    heatmap_kwargs = dict(
        z=z, x=list(range(len(months))), y=provinces,
        colorscale=RESOLUTION_COLORSCALE,
        zmin=0, zmax=len(RESOLUTION_ORDER) - 1,
        showscale=(i == 2),
        hovertemplate="<b>%{y}</b><br>%{text}<br>Resolution: %{customdata}<extra></extra>",
        text=text_dates,
        customdata=customdata,
    )
    if i == 2:
        heatmap_kwargs["colorbar"] = dict(
            tickmode="array",
            tickvals=list(range(len(RESOLUTION_ORDER))),
            ticktext=RESOLUTION_ORDER,
            lenmode="fraction", len=0.75,
        )
    fig.add_trace(go.Heatmap(**heatmap_kwargs), row=i, col=1)
    fig.update_xaxes(
        tickvals=year_tick_indices, ticktext=year_tick_labels,
        tickangle=45, row=i, col=1,
    )
    fig.update_yaxes(tickfont=dict(size=9), row=i, col=1)

total_height = 300 + sum(max(120, len(country_data[c][0]) * 14) for c in COUNTRIES)
fig.update_layout(
    title=dict(
        text="OpenDengue SEA — Province Coverage & Temporal Resolution (Longest Span)",
        font=dict(size=16),
    ),
    height=total_height, width=1400,
    paper_bgcolor="white", plot_bgcolor="white",
    margin=dict(l=200, r=60, t=80, b=40),
)

fig.write_html(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")