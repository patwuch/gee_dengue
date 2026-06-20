import pandas as pd
from pathlib import Path


def _find_git_root(start: Path) -> Path:
    for p in (start,) + tuple(start.parents):
        if (p / ".git").exists():
            return p
    return start


PROJECT_ROOT = _find_git_root(Path(__file__).resolve())
DATA_ROOT = PROJECT_ROOT / "data"

SEA_COUNTRIES = [
    "MALAYSIA",
    "INDONESIA",
    "VIET NAM",
    "THAILAND",
    "PHILIPPINES",
    "SINGAPORE",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
    "CAMBODIA",
    'MYANMAR',
    'BRUNEI DARUSSALAM'
]

df = pd.read_csv(DATA_ROOT / "interim" / "dengue-infection" / "spatial_SEARO_WPRO_EMRO_2000_2025.csv")
filtered = df[df["adm_0_name"].isin(SEA_COUNTRIES)]
filtered["calendar_start_date"] = pd.to_datetime(filtered["calendar_start_date"])

filtered = filtered[
    (filtered['calendar_start_date'] >= pd.to_datetime("2011-01-01")) &
    (filtered['calendar_start_date'] <= pd.to_datetime("2018-12-31"))
]
print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(filtered)}")
print(f"Countries found: {sorted(filtered['adm_0_name'].unique())}")

filtered.to_csv(DATA_ROOT / "interim" / "dengue-infection" / "filtered_sea_2011_2018.csv", index=False)
