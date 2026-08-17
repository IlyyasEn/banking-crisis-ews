"""
Step 2 - Merge macro/financial indicators onto the crisis panel.

Inputs:
  data/raw/P_Data_Extract_From_World_Development_Indicators.xlsx
      World Bank Databank export (Data tab) with 6 indicators for the
      24-country panel: domestic credit to private sector (% GDP),
      GDP growth, inflation, reserves in months of imports, current
      account balance (% GDP). Interest rate spread was collected but
      DROPPED (49-51% missing at crisis onsets - too sparse to trust).

  data/raw/exchange_rate_depreciation.csv
      Country-year nominal exchange rate, annual % change (depreciation
      positive). Source: World Bank / IMF exchange rate series.
      Columns required: country, prediction_year, exchange_rate_depreciation

  data/processed/crisis_panel.csv
      Output of 01_prepare_crisis_labels.py

Output:
  data/processed/banking_crisis_model_ready.csv
      Final modeling table: country, prediction_year, 6 lagged features,
      crisis_t. All features are lagged by 1 year (t-1) relative to the
      crisis label at year t, to avoid look-ahead bias.
"""

import pandas as pd
import numpy as np
import os

WB_RAW_PATH = "data/raw/P_Data_Extract_From_World_Development_Indicators.xlsx"
FX_RAW_PATH = "data/raw/exchange_rate_depreciation.csv"
CRISIS_PATH = "data/processed/crisis_panel.csv"
OUT_PATH = "data/processed/banking_crisis_model_ready.csv"

RENAME_MAP = {
    "FS.AST.PRVT.GD.ZS": "credit_to_gdp",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "FI.RES.TOTL.MO": "reserves_months_imports",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
}
NAME_FIX = {"Egypt, Arab Rep.": "Egypt", "Russian Federation": "Russia", "Turkiye": "Turkey"}


def load_wb_indicators():
    df = pd.read_excel(WB_RAW_PATH, sheet_name="Data")
    df = df.dropna(subset=["Series Code"]).copy()
    year_cols = [c for c in df.columns if "[YR" in c]

    long = df.melt(
        id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],
        value_vars=year_cols,
        var_name="year_raw",
        value_name="value",
    )
    long["year"] = long["year_raw"].str.extract(r"(\d{4})").astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["country"] = long["Country Name"].replace(NAME_FIX)

    wide = long.pivot_table(index=["country", "year"], columns="Series Code", values="value").reset_index()
    wide = wide.rename(columns=RENAME_MAP)
    # Keep only the indicators we actually use (interest rate spread / bank
    # capital ratio / NPL ratio are intentionally excluded - see README)
    keep = ["country", "year"] + list(RENAME_MAP.values())
    return wide[[c for c in keep if c in wide.columns]]


def main():
    indicators = load_wb_indicators()
    fx = pd.read_csv(FX_RAW_PATH)

    feature_cols_wb = list(RENAME_MAP.values())

    # Apply 1-year lag: year t's crisis label pairs with year (t-1)'s indicators
    lagged = indicators.copy()
    lagged["prediction_year"] = lagged["year"] + 1
    lagged = lagged.drop(columns=["year"])
    lagged = lagged.rename(columns={c: c + "_t_minus_1" for c in feature_cols_wb})

    crisis = pd.read_csv(CRISIS_PATH)
    final = crisis.merge(lagged, on=["country", "prediction_year"], how="left")
    final = final.merge(fx, on=["country", "prediction_year"], how="left")

    os.makedirs("data/processed", exist_ok=True)
    final.to_csv(OUT_PATH, index=False)

    feature_cols = [c + "_t_minus_1" for c in feature_cols_wb] + ["exchange_rate_depreciation"]
    print(f"Saved {OUT_PATH}: {final.shape}")
    print("\nMissingness per feature:")
    print((final[feature_cols].isna().mean() * 100).round(1))


if __name__ == "__main__":
    main()
