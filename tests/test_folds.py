import pandas as pd

from banking_ews.validation import build_balanced_country_folds


def test_balanced_country_folds_keep_countries_unique():
    data = pd.DataFrame(
        {
            "country": ["A", "A", "B", "C", "D", "E"],
            "crisis_t": [1, 1, 1, 0, 0, 0],
        }
    )

    folds = build_balanced_country_folds(data, n_folds=3)

    assert folds["country"].is_unique
    assert set(folds["country"]) == {"A", "B", "C", "D", "E"}
    assert set(folds["fold"]) == {0, 1, 2}
