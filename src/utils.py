"""
Shared utility functions for the Banking Crisis EWS pipeline.

Key principle used throughout: any statistic used to fill a missing value
(median, fallback, etc.) must be computed ONLY from the training partition
currently in scope - never from validation/test countries, and never using
a country's own future years to fill its own past gaps.
"""

import pandas as pd
import numpy as np


def clean_country_name(name):
    """Strip IMF footnote markers (e.g. 'Sri Lanka 5/') and whitespace."""
    if pd.isna(name):
        return name
    import re
    name = re.sub(r"\s*\d+/\s*$", "", str(name))
    return name.strip()


def clean_year(val):
    """Extract a 4-digit year from a possibly footnoted/string value."""
    if pd.isna(val):
        return None
    import re
    m = re.search(r"(\d{4})", str(val))
    return int(m.group(1)) if m else None


def fold_safe_impute(train_df, apply_df, cols):
    """
    Past-only imputation strategy for tabular models (Logistic Regression,
    Random Forest).

    - A missing value first uses an earlier observed value from the same
      country. This is safe for held-out countries because it uses only that
      country's history available at prediction time.
    - Remaining gaps fall back to the same-year median across TRAIN countries
      only (never touching validation/test countries).
    - The final fallback is a train-only expanding median, so future years
      cannot fill an earlier missing value.
    - If no historical data exists at all (the earliest panel years), use 0.0
      as an explicit no-information placeholder rather than borrowing from a
      future year. Models that need dense inputs can therefore run without
      introducing look-ahead bias.

    train_df : DataFrame used to compute all fill statistics (must exclude
               the current outer test fold's countries).
    apply_df : DataFrame the fill is applied to (can be train or test).
    cols     : list of feature columns to impute.
    """
    out = apply_df.sort_values(["country", "prediction_year"]).copy()
    train = train_df.sort_values(["country", "prediction_year"]).copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        train[col] = pd.to_numeric(train[col], errors="coerce")
        year_medians = train.loc[train[col].notna()].groupby("prediction_year")[col].median()
        global_by_year = year_medians.expanding().median()
        # Add training history before the rows being filled. If a country is
        # present in both frames, sorting keeps only earlier/same-year values
        # available to the current row; if it is held out, only its own past
        # observations are available.
        history = pd.concat(
            [
                train[["country", "prediction_year", col]].assign(_apply_row=-1, _source=0),
                out[["country", "prediction_year", col]].assign(_apply_row=np.arange(len(out)), _source=1),
            ],
            ignore_index=True,
        ).sort_values(["country", "prediction_year", "_source"])
        history["_filled"] = history.groupby("country")[col].ffill()
        filled = history.loc[history["_source"] == 1].sort_values("_apply_row")["_filled"].to_numpy()
        out[col] = pd.Series(filled, index=out.index, dtype=float)
        still_na = out[col].isna()
        if still_na.any():
            out.loc[still_na, col] = out.loc[still_na, "prediction_year"].map(year_medians)
        still_na = out[col].isna()
        if still_na.any():
            out.loc[still_na, col] = out.loc[still_na, "prediction_year"].map(global_by_year)
        out[col] = out[col].fillna(0.0)
    return out


def fold_safe_fill(train_df, apply_df, cols):
    """
    Forward-fill (past-only) strategy for sequence models (LSTM).

    - Within each country, missing values are forward-filled from that
      country's own earlier years only (never backward-filled from the
      future).
    - Leading gaps (no prior value exists yet) fall back to the same-year
      median across TRAIN countries only, then an expanding train-only median.
      If neither exists in the earliest years, the final no-information
      placeholder is 0.0 rather than a future-derived statistic.
    """
    out = apply_df.sort_values(["country", "prediction_year"]).copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        year_medians = train_df.loc[train_df[col].notna()].groupby("prediction_year")[col].median()
        global_by_year = year_medians.expanding().median()
        out[col] = out.groupby("country")[col].transform(lambda x: x.ffill())
        still_na = out[col].isna()
        if still_na.any():
            out.loc[still_na, col] = out.loc[still_na, "prediction_year"].map(year_medians)
        still_na = out[col].isna()
        if still_na.any():
            out.loc[still_na, col] = out.loc[still_na, "prediction_year"].map(global_by_year)
        out[col] = out[col].fillna(0.0)
    return out


FEATURE_COLS = [
    "gdp_growth_t_minus_1",
    "credit_to_gdp_t_minus_1",
    "inflation_t_minus_1",
    "reserves_months_imports_t_minus_1",
    "current_account_gdp_t_minus_1",
    "exchange_rate_depreciation",
]

TARGET = "crisis_t"
