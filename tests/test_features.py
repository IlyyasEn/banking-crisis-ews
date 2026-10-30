import pandas as pd

from utils import fold_safe_impute


def test_fold_safe_impute_uses_past_only_train_fallbacks():
    train = pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "prediction_year": [2000, 2001, 2000, 2001],
            "feature": [1.0, 3.0, 10.0, 30.0],
        }
    )
    apply = pd.DataFrame(
        {
            "country": ["C", "A"],
            "prediction_year": [2001, 2002],
            "feature": [None, None],
        }
    )

    out = fold_safe_impute(train, apply, ["feature"])

    assert out.loc[0, "feature"] == 16.5
    assert out.loc[1, "feature"] == 3.0


def test_fold_safe_impute_never_uses_a_future_country_value():
    train = pd.DataFrame(
        {
            "country": ["A", "A", "A", "B", "B"],
            "prediction_year": [2000, 2001, 2002, 2000, 2001],
            "feature": [1.0, None, 100.0, 10.0, 20.0],
        }
    )

    out = fold_safe_impute(train, train, ["feature"]).sort_values("prediction_year")

    assert out.loc[(out["country"] == "A") & (out["prediction_year"] == 2001), "feature"].item() == 1.0


def test_fold_safe_impute_uses_zero_when_no_past_train_data_exists():
    train = pd.DataFrame(
        {
            "country": ["A", "B"],
            "prediction_year": [2000, 2000],
            "feature": [None, None],
        }
    )

    out = fold_safe_impute(train, train, ["feature"])

    assert out["feature"].eq(0.0).all()
