"""
Step 5 - Train and evaluate the LSTM model with the same nested CV design
used for the tabular models (see 04_train_tabular_models.py for the full
explanation of the outer/inner fold structure).

Sequence construction: each training example is a 5-year rolling window of
the 6 features for one country, used to predict whether a crisis starts in
the year immediately following the window.

Missing values: forward-fill only (never backward-fill, which would leak
future values into a past gap) via fold_safe_fill, with a train-only
cross-sectional fallback for leading gaps.

Output: results/lstm_cv_results.csv
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import fold_safe_fill, FEATURE_COLS, TARGET

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.random.set_seed(42)

DATA_PATH = "data/processed/banking_crisis_model_ready.csv"
FOLDS_PATH = "data/processed/country_fold_assignments.csv"
OUT_PATH = "results/lstm_cv_results.csv"
WINDOW = 5
ARCH_GRID = [{"units": 8}, {"units": 16}]


def build_sequences(data, countries, cols, window=WINDOW):
    X, y = [], []
    for c in countries:
        sub = data[data["country"] == c].sort_values("prediction_year").reset_index(drop=True)
        vals, targets = sub[cols].values, sub[TARGET].values
        for i in range(window, len(sub)):
            X.append(vals[i - window:i])
            y.append(targets[i])
    return np.array(X), np.array(y)


def build_lstm(units, window, n_features):
    m = models.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(units, dropout=0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m


def main():
    df = pd.read_csv(DATA_PATH)
    folds = pd.read_csv(FOLDS_PATH)
    df = df.merge(folds, on="country")

    results = {"auc": [], "prauc": []}

    for test_fold in sorted(df["fold"].unique()):
        test_countries = df[df["fold"] == test_fold]["country"].unique()
        train_countries_all = df[df["fold"] != test_fold]["country"].unique()
        train_all_df = df[df["country"].isin(train_countries_all)]

        rng = np.random.RandomState(test_fold)
        shuffled = rng.permutation(train_countries_all)
        n_inner_val = max(1, len(shuffled) // 5)
        inner_val_c, inner_train_c = shuffled[:n_inner_val], shuffled[n_inner_val:]

        inner_train_df = fold_safe_fill(train_all_df, train_all_df[train_all_df["country"].isin(inner_train_c)], FEATURE_COLS)
        inner_val_df = fold_safe_fill(train_all_df, train_all_df[train_all_df["country"].isin(inner_val_c)], FEATURE_COLS)

        scaler = StandardScaler().fit(inner_train_df[FEATURE_COLS])
        itr_s = inner_train_df.copy(); itr_s[FEATURE_COLS] = scaler.transform(inner_train_df[FEATURE_COLS])
        ival_s = inner_val_df.copy(); ival_s[FEATURE_COLS] = scaler.transform(inner_val_df[FEATURE_COLS])

        Xtr, ytr = build_sequences(itr_s, inner_train_c, FEATURE_COLS)
        Xval, yval = build_sequences(ival_s, inner_val_c, FEATURE_COLS)

        best_score, best_units = -1, 8
        if ytr.sum() >= 2 and yval.sum() >= 1:
            for arch in ARCH_GRID:
                m = build_lstm(arch["units"], WINDOW, len(FEATURE_COLS))
                cw = {0: 1.0, 1: (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)}
                m.fit(Xtr, ytr, epochs=30, batch_size=16, class_weight=cw, verbose=0,
                      validation_data=(Xval, yval),
                      callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
                preds = m.predict(Xval, verbose=0).ravel()
                score = average_precision_score(yval, preds)
                if score > best_score:
                    best_score, best_units = score, arch["units"]

        full_tr = fold_safe_fill(train_all_df, train_all_df, FEATURE_COLS)
        full_te = fold_safe_fill(train_all_df, df[df["country"].isin(test_countries)], FEATURE_COLS)

        scaler_full = StandardScaler().fit(full_tr[FEATURE_COLS])
        full_tr_s = full_tr.copy(); full_tr_s[FEATURE_COLS] = scaler_full.transform(full_tr[FEATURE_COLS])
        full_te_s = full_te.copy(); full_te_s[FEATURE_COLS] = scaler_full.transform(full_te[FEATURE_COLS])

        Xtr_f, ytr_f = build_sequences(full_tr_s, train_countries_all, FEATURE_COLS)
        Xte_f, yte_f = build_sequences(full_te_s, test_countries, FEATURE_COLS)

        if ytr_f.sum() >= 2 and len(Xte_f) > 0:
            m = build_lstm(best_units, WINDOW, len(FEATURE_COLS))
            cw = {0: 1.0, 1: (len(ytr_f) - ytr_f.sum()) / max(ytr_f.sum(), 1)}
            m.fit(Xtr_f, ytr_f, epochs=30, batch_size=16, class_weight=cw, verbose=0)
            preds = m.predict(Xte_f, verbose=0).ravel()
            if yte_f.sum() > 0:
                results["auc"].append(roc_auc_score(yte_f, preds))
                results["prauc"].append(average_precision_score(yte_f, preds))
                print(f"Fold {test_fold}: units={best_units}, onsets={int(yte_f.sum())}/{len(yte_f)}, "
                      f"AUC={results['auc'][-1]:.3f}")

    auc_arr, pr_arr = np.array(results["auc"]), np.array(results["prauc"])
    summary = pd.DataFrame({"LSTM": {
        "AUC_mean": auc_arr.mean(), "AUC_std": auc_arr.std(),
        "PRAUC_mean": pr_arr.mean(), "PRAUC_std": pr_arr.std(),
    }}).T

    os.makedirs("results", exist_ok=True)
    summary.to_csv(OUT_PATH)
    print(f"\nLSTM final: AUC = {auc_arr.mean():.3f} +/- {auc_arr.std():.3f}, "
          f"PR-AUC = {pr_arr.mean():.3f} +/- {pr_arr.std():.3f}")


if __name__ == "__main__":
    main()
