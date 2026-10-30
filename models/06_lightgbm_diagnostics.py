"""
Step 6 - LightGBM threshold selection and SHAP explainability.

For each outer country fold:
  1. Tune the alert threshold on an inner validation-country split only.
  2. Retrain LightGBM on all outer-train countries.
  3. Apply the chosen threshold once to the untouched outer-test countries.
  4. Collect fold metrics and SHAP feature importance.

Outputs:
  results/lightgbm_threshold_diagnostics.csv
  results/lightgbm_shap_importance.csv
"""

import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from banking_ews.evaluation.thresholds import select_threshold, threshold_metrics
from utils import FEATURE_COLS, TARGET

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/banking_crisis_model_ready.csv"
FOLDS_PATH = "data/processed/country_fold_assignments.csv"
THRESHOLD_OUT = "results/lightgbm_threshold_diagnostics.csv"
SHAP_OUT = "results/lightgbm_shap_importance.csv"

PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "n_jobs": 1,
    "random_state": 42,
    "verbosity": -1,
}


def scale_pos_weight(y):
    n_pos = int(np.sum(y))
    return (len(y) - n_pos) / max(n_pos, 1)


def fit_lightgbm(train_df):
    y = train_df[TARGET]
    model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight(y), **PARAMS)
    model.fit(train_df[FEATURE_COLS], y)
    return model


def shap_values_for_positive_class(model, X):
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        return values[1]
    if getattr(values, "ndim", 0) == 3:
        return values[:, :, 1]
    return values


def main():
    df = pd.read_csv(DATA_PATH).merge(pd.read_csv(FOLDS_PATH), on="country")
    threshold_rows = []
    shap_rows = []

    for test_fold in sorted(df["fold"].unique()):
        test_countries = df.loc[df["fold"] == test_fold, "country"].unique()
        train_countries = df.loc[df["fold"] != test_fold, "country"].unique()
        train_all = df[df["country"].isin(train_countries)].copy()
        test = df[df["country"].isin(test_countries)].copy()

        rng = np.random.RandomState(int(test_fold))
        shuffled = rng.permutation(train_countries)
        n_inner_val = max(1, len(shuffled) // 5)
        inner_val_c = shuffled[:n_inner_val]
        inner_train_c = shuffled[n_inner_val:]

        inner_train = train_all[train_all["country"].isin(inner_train_c)]
        inner_val = train_all[train_all["country"].isin(inner_val_c)]

        threshold = 0.5
        validation_selection = {
            "threshold": threshold,
            "precision": 0.0,
            "recall": 0.0,
            "f_beta": 0.0,
            "alert_rate": 0.0,
        }
        if inner_train[TARGET].sum() >= 2 and inner_val[TARGET].sum() >= 1:
            inner_model = fit_lightgbm(inner_train)
            val_scores = inner_model.predict_proba(inner_val[FEATURE_COLS])[:, 1]
            validation_selection = select_threshold(inner_val[TARGET], val_scores, beta=2.0)
            threshold = validation_selection["threshold"]

        final_model = fit_lightgbm(train_all)
        test_scores = final_model.predict_proba(test[FEATURE_COLS])[:, 1]
        test_metrics = threshold_metrics(test[TARGET], test_scores, threshold, beta=2.0)

        row = {
            "fold": int(test_fold),
            "test_countries": "; ".join(sorted(test_countries)),
            "test_onsets": int(test[TARGET].sum()),
            "roc_auc": roc_auc_score(test[TARGET], test_scores) if test[TARGET].sum() > 0 else np.nan,
            "pr_auc": average_precision_score(test[TARGET], test_scores) if test[TARGET].sum() > 0 else np.nan,
            "selected_threshold": threshold,
            "validation_f2": validation_selection["f_beta"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f2": test_metrics["f_beta"],
            "test_alert_rate": test_metrics["alert_rate"],
        }
        threshold_rows.append(row)

        fold_shap = np.abs(shap_values_for_positive_class(final_model, test[FEATURE_COLS])).mean(axis=0)
        for feature, mean_abs_shap in zip(FEATURE_COLS, fold_shap):
            shap_rows.append(
                {
                    "fold": int(test_fold),
                    "feature": feature,
                    "mean_abs_shap": float(mean_abs_shap),
                }
            )

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(threshold_rows).to_csv(THRESHOLD_OUT, index=False)

    shap_summary = (
        pd.DataFrame(shap_rows)
        .groupby("feature")["mean_abs_shap"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_abs_shap", "std": "std_abs_shap"})
        .sort_values("mean_abs_shap", ascending=False)
    )
    shap_summary.to_csv(SHAP_OUT, index=False)

    print(f"Saved {THRESHOLD_OUT}")
    print(pd.DataFrame(threshold_rows).round(3).to_string(index=False))
    print(f"\nSaved {SHAP_OUT}")
    print(shap_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
