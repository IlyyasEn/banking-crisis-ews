"""
Step 3 - Build country-grouped cross-validation folds.

Why grouped-by-country: a random row-level split would leak information
(e.g. training on Argentina 2001 and testing on Argentina 2002 of the same
crisis episode). Splitting by whole country guarantees a country's data
never appears in both train and test within a fold.

Folds are built with a greedy balancing algorithm so each fold ends up
with a similar number of crisis onsets (not just a similar number of
countries) - otherwise a fold could randomly end up with 0-1 crises,
making its score meaningless.

Output: data/processed/country_fold_assignments.csv
"""

import pandas as pd
import os

IN_PATH = "data/processed/banking_crisis_model_ready.csv"
OUT_PATH = "data/processed/country_fold_assignments.csv"
N_FOLDS = 6


def main():
    df = pd.read_csv(IN_PATH)
    crisis_per_country = df.groupby("country")["crisis_t"].sum().sort_values(ascending=False)

    folds = [[] for _ in range(N_FOLDS)]
    fold_totals = [0] * N_FOLDS

    # Greedy: assign each country (largest crisis count first) to whichever
    # fold currently has the lowest crisis total.
    for country, n_crisis in crisis_per_country.items():
        min_fold = fold_totals.index(min(fold_totals))
        folds[min_fold].append(country)
        fold_totals[min_fold] += n_crisis

    fold_map = {c: i for i, countries in enumerate(folds) for c in countries}
    fold_df = pd.DataFrame(list(fold_map.items()), columns=["country", "fold"])

    os.makedirs("data/processed", exist_ok=True)
    fold_df.to_csv(OUT_PATH, index=False)

    print(f"Saved {OUT_PATH}")
    for i, countries in enumerate(folds):
        total = sum(crisis_per_country[c] for c in countries)
        print(f"  Fold {i}: {countries} -> {total} crisis onsets")


if __name__ == "__main__":
    main()
