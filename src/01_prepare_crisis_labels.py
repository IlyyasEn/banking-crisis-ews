"""
Step 1 - Prepare banking crisis onset labels.

Input : data/raw/SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx
        (IMF Laeven & Valencia Systemic Banking Crises Database, 2026 update)
        Download from the IMF working paper page (search "Systemic Banking
        Crises Database 1970-2025" on imf.org) and place it in data/raw/.

Output: data/processed/crisis_panel.csv
        One row per country-year (1970-2025) x 24 countries, with a binary
        `crisis_t` flag marking the FIRST year of each systemic banking
        crisis episode (not every year the crisis lasted).
"""

import pandas as pd
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils import clean_country_name, clean_year

RAW_PATH = "data/raw/SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx"
OUT_PATH = "data/processed/crisis_panel.csv"

PANEL_24 = [
    "Argentina", "Brazil", "Mexico", "Chile", "Colombia",
    "Turkey", "Russia", "Poland", "Hungary",
    "Egypt", "Nigeria", "Morocco",
    "India", "Indonesia", "Thailand", "Philippines",
    "Costa Rica", "Ecuador", "Ghana", "Kazakhstan",
    "Kenya", "Ukraine", "Uruguay", "Viet Nam",
]

# The IMF source spells these differently than our panel / World Bank convention
NAME_MAP = {"Türkiye": "Turkey", "Vietnam": "Viet Nam"}

# Starts at 1971, not 1970: every feature is lagged by 1 year (uses year t-1
# data), and our indicator sources start in 1970, so 1970 itself can never
# have a valid feature row (there's no 1969 data to lag from). Including it
# would just add fully-empty rows that inflate missingness stats downstream.
YEAR_RANGE = range(1971, 2026)


def main():
    df = pd.read_excel(RAW_PATH, sheet_name="Crisis Resolution and Outcomes")

    df["country_clean"] = df["Country"].apply(clean_country_name).replace(NAME_MAP)
    df["start_year"] = df["Start"].apply(clean_year)

    onset_set = set(
        zip(
            df[df["country_clean"].isin(PANEL_24)]["country_clean"],
            df[df["country_clean"].isin(PANEL_24)]["start_year"].dropna().astype(int),
        )
    )

    grid = pd.MultiIndex.from_product(
        [PANEL_24, YEAR_RANGE], names=["country", "prediction_year"]
    ).to_frame(index=False)
    # Note: prediction_year here is the RAW year; the +1 lag alignment to
    # "prediction_year" (t) vs indicator year (t-1) happens in script 02.
    grid["crisis_t"] = grid.apply(
        lambda r: 1 if (r["country"], r["prediction_year"]) in onset_set else 0, axis=1
    )

    os.makedirs("data/processed", exist_ok=True)
    grid.to_csv(OUT_PATH, index=False)

    print(f"Saved {OUT_PATH}: {grid.shape[0]} rows, {int(grid['crisis_t'].sum())} crisis onsets")
    print(grid.groupby("country")["crisis_t"].sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
