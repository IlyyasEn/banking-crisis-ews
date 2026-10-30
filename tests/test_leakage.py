import pandas as pd

from utils import fold_safe_fill


def test_fold_safe_fill_does_not_backfill_future_values():
    train = pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "prediction_year": [2000, 2001, 2000, 2001],
            "feature": [1.0, 2.0, 10.0, 20.0],
        }
    )
    apply = pd.DataFrame(
        {
            "country": ["C", "C"],
            "prediction_year": [2000, 2001],
            "feature": [None, 99.0],
        }
    )

    out = fold_safe_fill(train, apply, ["feature"]).sort_values("prediction_year")

    assert out.iloc[0]["feature"] == 5.5
    assert out.iloc[1]["feature"] == 99.0


def test_fold_safe_fill_uses_train_only_cross_country_fallback():
    train = pd.DataFrame(
        {
            "country": ["A", "B"],
            "prediction_year": [2000, 2000],
            "feature": [2.0, 4.0],
        }
    )
    validation = pd.DataFrame(
        {
            "country": ["C"],
            "prediction_year": [2000],
            "feature": [None],
        }
    )

    out = fold_safe_fill(train, validation, ["feature"])

    assert out.iloc[0]["feature"] == 3.0
