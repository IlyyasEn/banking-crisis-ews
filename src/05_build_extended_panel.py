"""Create a candidate-feature panel and its crisis-onset coverage report."""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_PATH = Path("data/processed/banking_crisis_model_ready.csv")
CANDIDATES_PATH = Path("data/raw/world_bank_candidate_features.csv")
OUT_PATH = Path("data/processed/banking_crisis_extended_candidates.csv")
REPORT_PATH = Path("results/candidate_feature_coverage.csv")


def main():
    base = pd.read_csv(BASE_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)
    candidates["value"] = pd.to_numeric(candidates["value"], errors="coerce")

    wide = candidates.pivot_table(
        index=["country", "year"], columns="feature", values="value", aggfunc="first"
    ).reset_index()
    wide["reserves_to_short_term_debt"] = np.where(
        wide["short_term_external_debt_usd"] > 0,
        wide["reserves_usd"] / wide["short_term_external_debt_usd"],
        np.nan,
    )
    candidate_cols = [
        "bank_capital_to_assets",
        "nonperforming_loans_ratio",
        "external_debt_service_exports",
        "net_barter_terms_of_trade",
        "reserves_to_short_term_debt",
    ]
    wide["prediction_year"] = wide["year"] + 1
    wide = wide.drop(columns=["year", "reserves_usd", "short_term_external_debt_usd"])
    wide = wide.rename(columns={column: f"{column}_t_minus_1" for column in candidate_cols})

    enriched = base.merge(wide, on=["country", "prediction_year"], how="left")
    candidate_cols = [f"{column}_t_minus_1" for column in candidate_cols]
    onset = enriched[enriched["crisis_t"] == 1]
    report = pd.DataFrame(
        {
            "feature": candidate_cols,
            "panel_coverage_pct": [enriched[column].notna().mean() * 100 for column in candidate_cols],
            "crisis_onset_coverage_pct": [onset[column].notna().mean() * 100 for column in candidate_cols],
            "countries_with_data": [enriched.loc[enriched[column].notna(), "country"].nunique() for column in candidate_cols],
        }
    ).sort_values("crisis_onset_coverage_pct", ascending=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUT_PATH, index=False)
    report.to_csv(REPORT_PATH, index=False)
    print(f"Saved {OUT_PATH}")
    print(f"Saved {REPORT_PATH}")
    print(report.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
