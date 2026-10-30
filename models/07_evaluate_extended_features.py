"""Compare baseline and external-vulnerability feature sets with nested CV."""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import FEATURE_COLS, TARGET, fold_safe_impute

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/banking_crisis_extended_candidates.csv"
FOLDS_PATH = "data/processed/country_fold_assignments.csv"
OUT_PATH = "results/extended_feature_cv_results.csv"

EXTERNAL_FEATURES = [
    "external_debt_service_exports_t_minus_1",
    "reserves_to_short_term_debt_t_minus_1",
]
FEATURE_SETS = {
    "baseline": FEATURE_COLS,
    "external_vulnerability": FEATURE_COLS + EXTERNAL_FEATURES,
}
GRIDS = {
    "LogReg": [{"C": value} for value in (0.01, 0.1, 1, 10)],
    "LightGBM": [
        {"max_depth": depth, "learning_rate": rate, "n_estimators": estimators}
        for depth in (2, 3)
        for rate in (0.05, 0.1)
        for estimators in (100, 200)
    ],
}


def scale_pos_weight(target):
    return (len(target) - target.sum()) / max(target.sum(), 1)


def fit_predict(model_name, params, train_features, train_target, apply_features):
    if model_name == "LogReg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l1", solver="liblinear", class_weight="balanced", max_iter=1000, **params
            ),
        )
    else:
        model = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight(train_target), n_jobs=1, random_state=42,
            verbosity=-1, **params
        )
    model.fit(train_features, train_target)
    return model.predict_proba(apply_features)[:, 1]


def main():
    data = pd.read_csv(DATA_PATH).merge(pd.read_csv(FOLDS_PATH), on="country")
    results = {}

    for feature_set_name, features in FEATURE_SETS.items():
        for model_name, grid in GRIDS.items():
            key = f"{model_name}_{feature_set_name}"
            aucs, praucs = [], []
            for test_fold in sorted(data["fold"].unique()):
                test = data[data["fold"] == test_fold].copy()
                train_all = data[data["fold"] != test_fold].copy()
                countries = train_all["country"].unique()
                shuffled = np.random.RandomState(int(test_fold)).permutation(countries)
                validation_countries = shuffled[:max(1, len(shuffled) // 5)]
                inner_train = train_all[~train_all["country"].isin(validation_countries)]
                inner_validation = train_all[train_all["country"].isin(validation_countries)]

                best_score, best_params = -1.0, grid[0]
                if model_name == "LogReg":
                    inner_train_ready = fold_safe_impute(inner_train, inner_train, features)
                    inner_validation_ready = fold_safe_impute(inner_train, inner_validation, features)
                else:
                    inner_train_ready, inner_validation_ready = inner_train, inner_validation
                for params in grid:
                    if inner_train[TARGET].sum() < 2 or inner_validation[TARGET].sum() < 1:
                        continue
                    scores = fit_predict(
                        model_name, params, inner_train_ready[features], inner_train[TARGET],
                        inner_validation_ready[features],
                    )
                    score = average_precision_score(inner_validation[TARGET], scores)
                    if score > best_score:
                        best_score, best_params = score, params

                if model_name == "LogReg":
                    full_train = fold_safe_impute(train_all, train_all, features)
                    full_test = fold_safe_impute(train_all, test, features)
                else:
                    full_train, full_test = train_all, test
                scores = fit_predict(
                    model_name, best_params, full_train[features], train_all[TARGET], full_test[features]
                )
                aucs.append(roc_auc_score(test[TARGET], scores))
                praucs.append(average_precision_score(test[TARGET], scores))

            results[key] = {
                "AUC_mean": np.mean(aucs), "AUC_std": np.std(aucs),
                "PRAUC_mean": np.mean(praucs), "PRAUC_std": np.std(praucs),
            }

    summary = pd.DataFrame(results).T.sort_values("PRAUC_mean", ascending=False)
    os.makedirs("results", exist_ok=True)
    summary.to_csv(OUT_PATH)
    print(summary.round(3))


if __name__ == "__main__":
    main()
