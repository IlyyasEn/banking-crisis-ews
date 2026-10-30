"""Download candidate banking and external-vulnerability features from WDI.

The file is deliberately kept separate from the curated baseline panel: data
availability must be audited before any new variable is promoted into a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen

import pandas as pd


OUT_PATH = Path("data/raw/world_bank_candidate_features.csv")
YEARS = range(1970, 2026)

COUNTRY_CODES = {
    "Argentina": "ARG", "Brazil": "BRA", "Mexico": "MEX", "Chile": "CHL",
    "Colombia": "COL", "Turkey": "TUR", "Russia": "RUS", "Poland": "POL",
    "Hungary": "HUN", "Egypt": "EGY", "Nigeria": "NGA", "Morocco": "MAR",
    "India": "IND", "Indonesia": "IDN", "Thailand": "THA", "Philippines": "PHL",
    "Costa Rica": "CRI", "Ecuador": "ECU", "Ghana": "GHA", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Ukraine": "UKR", "Uruguay": "URY", "Viet Nam": "VNM",
}

INDICATORS = {
    "FB.BNK.CAPA.ZS": "bank_capital_to_assets",
    "FB.AST.NPER.ZS": "nonperforming_loans_ratio",
    "DT.TDS.DECT.EX.ZS": "external_debt_service_exports",
    "TT.PRI.MRCH.XD.WD": "net_barter_terms_of_trade",
    "FI.RES.TOTL.CD": "reserves_usd",
    "DT.DOD.DSTC.CD": "short_term_external_debt_usd",
}


def fetch_indicator(country_code: str, indicator_code: str) -> list[dict]:
    url = (
        "https://api.worldbank.org/v2/country/"
        f"{country_code}/indicator/{indicator_code}?format=json&per_page=100"
    )
    with urlopen(url, timeout=15) as response:
        payload = json.load(response)
    return payload[1] or []


def main():
    rows, failures = [], []
    requests = [
        (country, country_code, indicator_code, feature)
        for country, country_code in COUNTRY_CODES.items()
        for indicator_code, feature in INDICATORS.items()
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_indicator, country_code, indicator_code):
            (country, indicator_code, feature)
            for country, country_code, indicator_code, feature in requests
        }
        for future in as_completed(futures):
            country, indicator_code, feature = futures[future]
            try:
                records = future.result()
            except Exception as error:
                failures.append(f"{country}/{indicator_code}: {error}")
                continue
            for record in records:
                year = int(record["date"])
                if year in YEARS:
                    rows.append(
                        {
                            "country": country,
                            "year": year,
                            "indicator_code": indicator_code,
                            "feature": feature,
                            "value": record["value"],
                        }
                    )

    result = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    coverage = result.dropna(subset=["value"]).groupby("feature").agg(
        observations=("value", "size"), countries=("country", "nunique")
    )
    print(f"Saved {OUT_PATH}: {len(result)} country-year-indicator records")
    print(coverage.to_string())
    if failures:
        print("Failed requests:")
        print("\n".join(failures))


if __name__ == "__main__":
    main()
