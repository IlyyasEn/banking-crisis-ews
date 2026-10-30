def test_project_imports():
    import banking_ews

    assert banking_ews is not None


def test_processed_panel_shape_and_target():
    import pandas as pd

    df = pd.read_csv("data/processed/banking_crisis_model_ready.csv")
    lagged_features = [
        "credit_to_gdp_t_minus_1",
        "gdp_growth_t_minus_1",
        "inflation_t_minus_1",
        "reserves_months_imports_t_minus_1",
        "current_account_gdp_t_minus_1",
    ]

    assert df.shape[0] == 24 * 55
    assert df["country"].nunique() == 24
    assert df["prediction_year"].min() == 1971
    assert df["prediction_year"].max() == 2025
    assert int(df["crisis_t"].sum()) == 46
    assert set(lagged_features).issubset(df.columns)
    assert all(c.endswith("_t_minus_1") for c in lagged_features)
