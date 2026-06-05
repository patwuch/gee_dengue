import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH = "/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/data/raw/OpenDengue/Spatial_extract_V1_3.csv"
OUTPUT_PATH = "coverage_thailand_dual.html"
COUNTRY = "THAILAND"
DATE_START = "1960-01-01"
DATE_END = "2025-12-31"

# ── Resolution config ─────────────────────────────────────────────────────────

RESOLUTION_ALIASES = {
    "Week": "Weekly",
    "week": "Weekly",
    "Weekly": "Weekly",
}
RESOLUTION_PRIORITY = {"Year": 1, "Month": 2, "Weekly": 3}
RESOLUTION_ORDER = ["Missing", "Year", "Month", "Weekly"]
RESOLUTION_COLORS = {
    "Missing": "#FFFFFF",
    "Year": "#AED6F1",
    "Month": "#2980B9",
    "Weekly": "#1A2D6B",
}
RESOLUTION_CODES = {name: idx for idx, name in enumerate(RESOLUTION_ORDER)}
MAX_RESOLUTION_INDEX = len(RESOLUTION_ORDER) - 1
RESOLUTION_COLORSCALE = [
    [i / MAX_RESOLUTION_INDEX, RESOLUTION_COLORS[name]]
    for i, name in enumerate(RESOLUTION_ORDER)
]


def normalize_resolution(value):
    if pd.isna(value):
        return value
    label = str(value).strip()
    if not label:
        return value
    return RESOLUTION_ALIASES.get(label, label.title())


# ── Load & preprocess data ────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH, low_memory=False)
df["calendar_start_date"] = pd.to_datetime(df["calendar_start_date"])
df = df[df["adm_0_name"] == COUNTRY].copy()
df["T_res"] = df["T_res"].apply(normalize_resolution)

# FIX: snap weekly dates to month-start so they align with the MS grid,
# but only after recording the original resolution.
# This preserves the fact that weekly data exists for that month.
df["month_start"] = df["calendar_start_date"].values.astype("datetime64[M]")

all_months = pd.date_range(DATE_START, DATE_END, freq="MS")

year_tick_indices = [i for i, m in enumerate(all_months) if m.month == 1]
year_tick_labels = [str(m.year) for m in all_months if m.month == 1]


# ── Generic grid builder ──────────────────────────────────────────────────────

def build_resolution_grid(sub: pd.DataFrame, level_col: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a (entities × months) resolution grid for a given admin level column.
    Entities with no data at this level are excluded.
    National-level rows (level_col is NaN) are included as a separate 'National' row.
    """
    sub = sub.copy()

    # Label NaN as 'National' so they aren't dropped
    sub[level_col] = sub[level_col].fillna("National (adm_0)")

    entities = sorted(sub[level_col].unique())
    res = sub[[level_col, "month_start", "T_res"]].copy()
    res["rank"] = res["T_res"].map(RESOLUTION_PRIORITY).fillna(0)

    # Expand yearly rows to cover all 12 months of that year
    yearly = res[res["T_res"] == "Year"].copy()
    if not yearly.empty:
        expanded_rows = []
        for _, row in yearly.iterrows():
            year = row["month_start"].year
            for month in range(1, 13):
                new_row = row.copy()
                new_row["month_start"] = pd.Timestamp(year=year, month=month, day=1)
                expanded_rows.append(new_row)
        res = pd.concat(
            [pd.DataFrame(expanded_rows), res[res["T_res"] != "Year"]],
            ignore_index=True,
        )

    # Keep highest-resolution entry per (entity, month)
    res = (
        res.sort_values([level_col, "month_start", "rank"], ascending=[True, True, False])
        .drop_duplicates([level_col, "month_start"], keep="first")
    )

    type_grid = (
        res.pivot(index=level_col, columns="month_start", values="T_res")
        .reindex(columns=all_months)
        .reindex(entities)
    )

    display_grid = type_grid.fillna("Missing")
    display_grid = display_grid.where(display_grid.isin(RESOLUTION_ORDER), "Missing")

    return display_grid, entities


def make_heatmap_trace(display_grid: pd.DataFrame, entities: list[str], showscale: bool = True):
    z = display_grid.replace(RESOLUTION_CODES).values
    resolution_labels = display_grid.values
    text_dates = [[m.strftime("%b %Y") for m in all_months] for _ in entities]

    return go.Heatmap(
        z=z,
        x=list(range(len(all_months))),
        y=entities,
        colorscale=RESOLUTION_COLORSCALE,
        zmin=0,
        zmax=MAX_RESOLUTION_INDEX,
        colorbar=dict(
            title="Resolution",
            tickmode="array",
            tickvals=list(range(len(RESOLUTION_ORDER))),
            ticktext=RESOLUTION_ORDER,
        ),
        showscale=showscale,
        hovertemplate="<b>%{y}</b><br>%{text}<br>Resolution: %{customdata}<extra></extra>",
        text=text_dates,
        customdata=resolution_labels.tolist(),
    )


# ── Build grids ───────────────────────────────────────────────────────────────

# adm_1: all rows (national rows become "National (adm_0)")
grid_adm1, entities_adm1 = build_resolution_grid(df, "adm_1_name")

# adm_2: only rows where adm_2 exists; national/province-only rows are included
# as their own aggregate label so nothing is silently dropped
df_adm2 = df.copy()
df_adm2["adm_2_name"] = df_adm2["adm_2_name"].fillna(
    df_adm2["adm_1_name"].fillna("National (adm_0)") + " (no adm_2)"
)
grid_adm2, entities_adm2 = build_resolution_grid(df_adm2, "adm_2_name")

# ── Build figure ──────────────────────────────────────────────────────────────

row_height_px = 14
h1 = max(300, len(entities_adm1) * row_height_px + 120)
h2 = max(300, len(entities_adm2) * row_height_px + 120)
total_height = h1 + h2 + 120  # gap between panels

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        f"Province level (adm_1) — {len(entities_adm1)} entities",
        f"District level (adm_2) — {len(entities_adm2)} entities",
    ],
    vertical_spacing=80 / total_height,
    row_heights=[h1 / total_height, h2 / total_height],
)

fig.add_trace(make_heatmap_trace(grid_adm1, entities_adm1, showscale=True), row=1, col=1)
fig.add_trace(make_heatmap_trace(grid_adm2, entities_adm2, showscale=False), row=2, col=1)

for row in (1, 2):
    fig.update_xaxes(
        tickvals=year_tick_indices,
        ticktext=year_tick_labels,
        tickangle=45,
        title_text="Year",
        row=row,
        col=1,
    )
    fig.update_yaxes(tickfont=dict(size=8), title_text="Admin unit", row=row, col=1)

fig.update_layout(
    title=dict(
        text=(
            f"OpenDengue Thailand — Coverage & Temporal Resolution "
            f"({DATE_START[:4]}–{DATE_END[:4]})"
        ),
        font=dict(size=16),
    ),
    height=total_height,
    width=1400,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=220, r=60, t=80, b=80),
)

# ── Save ──────────────────────────────────────────────────────────────────────

fig.write_html(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")
print(f"  adm_1 entities : {len(entities_adm1)}")
print(f"  adm_2 entities : {len(entities_adm2)}")