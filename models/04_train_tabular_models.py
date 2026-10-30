"""
Step 4 - Train and evaluate 4 tabular models with nested cross-validation.

Design (see README for the full diagram):

  For each of the 6 OUTER country folds (used once, as the test set):
    1. The remaining ~20 "train" countries are split AGAIN into an inner
       train / inner validation set.
    2. A small hyperparameter grid is tried for each model, scored on the
       inner validation set only (PR-AUC). The best config is kept.
    3. The winning config is retrained on the FULL outer-train set.
    4. It is scored ONCE on the untouched outer test fold.

This guarantees the outer test countries never influence any modeling
decision (hyperparameters, imputation statistics, or feature choices) -
only the final score.

Models:
  - Logistic Regression / Random Forest -> use fold_safe_impute (median-based)
  - XGBoost / LightGBM -> NaN left as-is (both handle missing values natively)

Output: results/nested_cv_results.csv
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import fold_safe_impute, FEATURE_COLS, TARGET

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/banking_crisis_model_ready.csv"
FOLDS_PATH = "data/processed/country_fold_assignments.csv"
OUT_PATH = "results/nested_cv_results.csv"

GRIDS = {
    "LogReg": [{"C": c} for c in [0.01, 0.1, 1, 10]],
    "RandomForest": [{"n_estimators": n, "max_depth": d} for n in [100, 300] for d in [3, 5, None]],
    "XGBoost": [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n}
        for d in [2, 3] for lr in [0.05, 0.1] for n in [100, 200]
    ],
    "LightGBM": [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n}
        for d in [2, 3] for lr in [0.05, 0.1] for n in [100, 200]
    ],
}
NATIVE_NAN_MODELS = {"XGBoost", "LightGBM"}
IMPUTE_MODELS = {"LogReg", "RandomForest"}


def scale_pos_weight(y):
    n_pos = y.sum()
    return (len(y) - n_pos) / max(n_pos, 1)


def fit_predict(model_name, params, Xtr, ytr, Xval):
    spw = scale_pos_weight(ytr)
    if model_name == "LogReg":
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l1", solver="liblinear", class_weight="balanced", max_iter=1000, **params
            ),
        )
    elif model_name == "RandomForest":
        m = RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=42, **params)
    elif model_name == "XGBoost":
        m = xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="aucpr", random_state=42,
                               n_jobs=1, verbosity=0, **params)
    elif model_name == "LightGBM":
        m = lgb.LGBMClassifier(scale_pos_weight=spw, n_jobs=1, random_state=42, verbosity=-1, **params)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xval)[:, 1], m


def main():
    df = pd.read_csv(DATA_PATH)
    folds = pd.read_csv(FOLDS_PATH)
    df = df.merge(folds, on="country")

    results = {m: {"auc": [], "prauc": []} for m in GRIDS}
    results["Basel_benchmark"] = {"auc": [], "prauc": []}

    for test_fold in sorted(df["fold"].unique()):
        test_countries = df[df["fold"] == test_fold]["country"].unique()
        train_countries_all = df[df["fold"] != test_fold]["country"].unique()

        train_all_df = df[df["country"].isin(train_countries_all)].copy()
        test_df = df[df["country"].isin(test_countries)].copy()

        rng = np.random.RandomState(test_fold)
        shuffled = rng.permutation(train_countries_all)
        n_inner_val = max(1, len(shuffled) // 5)
        inner_val_c, inner_train_c = shuffled[:n_inner_val], shuffled[n_inner_val:]
        inner_train_df = train_all_df[train_all_df["country"].isin(inner_train_c)]
        inner_val_df = train_all_df[train_all_df["country"].isin(inner_val_c)]

        for model_name, grid in GRIDS.items():
            best_score, best_params = -1, grid[0]
            for params in grid:
                if model_name in IMPUTE_MODELS:
                    itr = fold_safe_impute(inner_train_df, inner_train_df, FEATURE_COLS)
                    ival = fold_safe_impute(inner_train_df, inner_val_df, FEATURE_COLS)
                else:
                    itr, ival = inner_train_df, inner_val_df
                Xtr, ytr = itr[FEATURE_COLS], itr[TARGET]
                Xval, yval = ival[FEATURE_COLS], ival[TARGET]
                if ytr.sum() < 2 or yval.sum() < 1:
                    continue
                try:
                    preds, _ = fit_predict(model_name, params, Xtr, ytr, Xval)
                    score = average_precision_score(yval, preds)
                    if score > best_score:
                        best_score, best_params = score, params
                except Exception:
                    continue

            if model_name in IMPUTE_MODELS:
                full_tr = fold_safe_impute(train_all_df, train_all_df, FEATURE_COLS)
                full_te = fold_safe_impute(train_all_df, test_df, FEATURE_COLS)
            else:
                full_tr, full_te = train_all_df, test_df

            Xtr, ytr = full_tr[FEATURE_COLS], full_tr[TARGET]
            Xte, yte = full_te[FEATURE_COLS], full_te[TARGET]
            preds, _ = fit_predict(model_name, best_params, Xtr, ytr, Xte)

            if yte.sum() > 0:
                results[model_name]["auc"].append(roc_auc_score(yte, preds))
                results[model_name]["prauc"].append(average_precision_score(yte, preds))

        # Basel III-style benchmark: raw credit-to-GDP level as the score
        te_scores = test_df["credit_to_gdp_t_minus_1"].fillna(test_df["credit_to_gdp_t_minus_1"].median())
        yte = test_df[TARGET]
        if yte.sum() > 0:
            results["Basel_benchmark"]["auc"].append(roc_auc_score(yte, te_scores))
            results["Basel_benchmark"]["prauc"].append(average_precision_score(yte, te_scores))

    summary = pd.DataFrame({
        m: {
            "AUC_mean": np.mean(r["auc"]), "AUC_std": np.std(r["auc"]),
            "PRAUC_mean": np.mean(r["prauc"]), "PRAUC_std": np.std(r["prauc"]),
        }
        for m, r in results.items()
    }).T.sort_values("PRAUC_mean", ascending=False)

    os.makedirs("results", exist_ok=True)
    summary.to_csv(OUT_PATH)
    print(summary.round(3))


if __name__ == "__main__":
    main()
