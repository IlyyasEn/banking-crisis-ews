"""Country-grouped fold helpers."""

from __future__ import annotations

import pandas as pd


def build_balanced_country_folds(
    data: pd.DataFrame,
    country_col: str = "country",
    target_col: str = "crisis_t",
    n_folds: int = 6,
) -> pd.DataFrame:
    """Assign countries to folds while balancing crisis onsets greedily."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    crisis_per_country = (
        data.groupby(country_col)[target_col]
        .sum()
        .sort_values(ascending=False)
    )
    if crisis_per_country.empty:
        raise ValueError("data must contain at least one country")

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    fold_totals = [0.0] * n_folds

    for country, n_crisis in crisis_per_country.items():
        min_fold = fold_totals.index(min(fold_totals))
        folds[min_fold].append(country)
        fold_totals[min_fold] += float(n_crisis)

    rows = [
        {country_col: country, "fold": fold_id}
        for fold_id, countries in enumerate(folds)
        for country in countries
    ]
    return pd.DataFrame(rows)
