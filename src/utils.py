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
    Expanding, past-only imputation strategy for tabular models
    (Logistic Regression, Random Forest).

    - Country-level median is computed ONLY from `train_df`.
    - Any still-missing value falls back to the same-year median computed
      across all TRAIN countries (never touching test-fold countries).
    - Final fallback is the global train-only median.

    train_df : DataFrame used to compute all fill statistics (must exclude
               the current outer test fold's countries).
    apply_df : DataFrame the fill is applied to (can be train or test).
    cols     : list of feature columns to impute.
    """
    out = apply_df.copy()
    year_medians = train_df.groupby("prediction_year")[cols].median()
    global_medians = train_df[cols].median()

    for col in cols:
        country_med = train_df.groupby("country")[col].median()
        out[col] = out.apply(
            lambda r: r[col] if pd.notna(r[col]) else country_med.get(r["country"], np.nan),
            axis=1,
        )
        still_na = out[col].isna()
        if still_na.any():
            out.loc[still_na, col] = out.loc[still_na, "prediction_year"].map(year_medians[col])
        still_na = out[col].isna()
        if still_na.any():
            out.loc[still_na, col] = global_medians[col]
    return out


def fold_safe_fill(train_df, apply_df, cols):
    """
    Forward-fill (past-only) strategy for sequence models (LSTM).

    - Within each country, missing values are forward-filled from that
      country's own earlier years only (never backward-filled from the
      future).
    - Leading gaps (no prior value exists yet) fall back to the same-year
      median across TRAIN countries only.
    """
    out = apply_df.sort_values(["country", "prediction_year"]).copy()
    year_medians = train_df.groupby("prediction_year")[cols].median()
    global_medians = train_df[cols].median()

    for col in cols:
        out[col] = out.groupby("country")[col].transform(lambda x: x.ffill())
        still_na = out[col].isna()
        if still_na.any():
            out.loc[still_na, col] = out.loc[still_na, "prediction_year"].map(year_medians[col])
        still_na = out[col].isna()
        if still_na.any():
            out.loc[still_na, col] = global_medians[col]
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
