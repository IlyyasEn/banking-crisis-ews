import pandas as pd


def test_candidate_panel_has_lagged_external_features_and_coverage_report():
    panel = pd.read_csv("data/processed/banking_crisis_extended_candidates.csv")
    coverage = pd.read_csv("results/candidate_feature_coverage.csv")

    external_features = {
        "external_debt_service_exports_t_minus_1",
        "reserves_to_short_term_debt_t_minus_1",
    }
    assert external_features.issubset(panel.columns)
    assert panel.shape[0] == 24 * 55
    assert set(coverage["feature"]).issuperset(external_features)
    assert coverage.loc[
        coverage["feature"] == "external_debt_service_exports_t_minus_1",
        "crisis_onset_coverage_pct",
    ].item() >= 70
