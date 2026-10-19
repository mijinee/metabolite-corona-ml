"""
Metabolite class prediction for unannotated features (KNN, Random Forest, XGBoost).

Input:  Master_Processed_Conservative_Data.xlsx, Other_Unknown_Manual_Curation.xlsx
Output: metabolite_class_predictions.xlsx
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, f1_score,
                             precision_recall_fscore_support, roc_curve)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from config import CURATION_PATH, OUT_CLASSIFY, PRED_PATH, PROCESSED_PATH
from utils.metabolite_classes import apply_merges, build_feature_table, load_curation

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — KNN and RandomForest will still run.")

OUT_CLASSIFY.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.40
CV_FOLDS     = 3
TEST_SIZE    = 0.20
KNN_K        = 5
FEATURE_COLS = ["MZ", "MassDefect", "Polarity"]


def make_xgb():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         use_label_encoder=False, eval_metric="mlogloss",
                         random_state=42, n_jobs=-1)


def build_models() -> dict:
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=KNN_K, weights="distance", metric="euclidean"),
        "RandomForest": RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                               min_samples_leaf=2, random_state=42, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = make_xgb()
    return models


def evaluate_models(X_tr, X_te, y_tr, y_te, models, le, mode, save_path):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    classes = list(range(len(le.classes_)))
    y_te_bin = label_binarize(y_te, classes=classes)

    cv_rows, test_rows, perclass_rows, roc_rows = [], [], [], []

    for name, model in models.items():
        Xtr, Xte = X_tr[name], X_te[name]

        cv_res = cross_validate(model, Xtr, y_tr, cv=cv,
                                scoring={"f1_macro": "f1_macro", "f1_weighted": "f1_weighted"})
        cv_rows.append({
            "Model": name,
            "Mean_F1_Macro_CV":    round(cv_res["test_f1_macro"].mean(), 4),
            "Std_F1_Macro_CV":     round(cv_res["test_f1_macro"].std(), 4),
            "Mean_F1_Weighted_CV": round(cv_res["test_f1_weighted"].mean(), 4),
            "Std_F1_Weighted_CV":  round(cv_res["test_f1_weighted"].std(), 4),
        })

        if name == "XGBoost":
            model = make_xgb()
        model.fit(Xtr, y_tr)
        y_pred  = model.predict(Xte)
        y_score = model.predict_proba(Xte)

        test_rows.append({
            "Model": name,
            "Macro_F1_Test":    round(f1_score(y_te, y_pred, average="macro",    zero_division=0), 4),
            "Weighted_F1_Test": round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        })
        print(classification_report(le.inverse_transform(y_te),
                                    le.inverse_transform(y_pred), zero_division=0))

        ps, rs, fs, ss = precision_recall_fscore_support(y_te, y_pred, labels=classes, zero_division=0)
        for i, cls in enumerate(le.classes_):
            perclass_rows.append({"Model": name, "Class": cls,
                                  "Precision": round(float(ps[i]), 4),
                                  "Recall":    round(float(rs[i]), 4),
                                  "F1":        round(float(fs[i]), 4),
                                  "Support_test": int(ss[i])})

        for i, cls in enumerate(le.classes_):
            if y_te_bin[:, i].sum() > 0:
                fpr, tpr, thr = roc_curve(y_te_bin[:, i], y_score[:, i])
                for f, t, th in zip(fpr, tpr, thr):
                    roc_rows.append({"Mode": mode, "Model": name, "Class": cls,
                                     "FPR": round(float(f), 5), "TPR": round(float(t), 5),
                                     "Threshold": round(float(th), 5)})
    return (pd.DataFrame(cv_rows), pd.DataFrame(test_rows),
            pd.DataFrame(perclass_rows), pd.DataFrame(roc_rows))


def predict_mode(raw: dict, mode: str, curation_labels: dict):
    print(f"\n{'=' * 60}\n  {mode}\n{'=' * 60}")
    df = build_feature_table(raw[f"{mode}_gap"], mode, curation_labels)

    labeled_mask = df["IsAnnotated"] & ~df["RuleClass"].isin(["Other/Unknown", "Cannot Classify"])
    df_labeled   = df[labeled_mask].copy()
    df_predict   = df[~labeled_mask].copy()

    if labeled_mask.sum() < 10:
        print("Not enough labeled data to train.")
        return None

    df_labeled["Class"] = apply_merges(df_labeled["RuleClass"], mode)
    counts = df_labeled["Class"].value_counts()
    df_labeled = df_labeled[df_labeled["Class"].isin(counts[counts >= 3].index)].copy()
    df_labeled = df_labeled[df_labeled[FEATURE_COLS].notna().all(axis=1)].copy()

    le = LabelEncoder()
    y  = le.fit_transform(df_labeled["Class"].values)
    X  = df_labeled[FEATURE_COLS].values

    spl = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    tr_idx, te_idx = next(spl.split(X, y))

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X[tr_idx])
    X_te_sc  = scaler.transform(X[te_idx])
    X_pr_raw = np.nan_to_num(df_predict[FEATURE_COLS].values.copy())

    X_tr = {"KNN": X_tr_sc, "RandomForest": X[tr_idx], "XGBoost": X[tr_idx]}
    X_te = {"KNN": X_te_sc, "RandomForest": X[te_idx], "XGBoost": X[te_idx]}

    models = build_models()
    cv_sum, test_sum, perclass, roc_df = evaluate_models(
        X_tr, X_te, y[tr_idx], y[te_idx], models, le, mode, OUT_CLASSIFY)
    perclass.insert(0, "Mode", mode)

    # Retrain on the full labeled set for final predictions
    scaler_full = StandardScaler()
    X_full      = {"KNN": scaler_full.fit_transform(X), "RandomForest": X, "XGBoost": X}
    X_pr_full   = {"KNN": scaler_full.transform(X_pr_raw),
                   "RandomForest": X_pr_raw, "XGBoost": X_pr_raw}

    df_pred_out = df_predict.copy().set_index("FeatureID")
    df_pred_out["Split"] = "Predict"
    for name, model in models.items():
        if name == "XGBoost":
            model = make_xgb()
        model.fit(X_full[name], y)
        conf   = np.max(model.predict_proba(X_pr_full[name]), axis=1)
        labels = le.inverse_transform(model.predict(X_pr_full[name]))
        df_pred_out[f"Predicted_{name}"]  = np.where(conf >= CONFIDENCE_THRESHOLD, labels, "Low Confidence")
        df_pred_out[f"Confidence_{name}"] = conf.round(4)

    df_lab_out = df_labeled.copy().set_index("FeatureID")
    split_lbl  = pd.Series("Train", index=range(len(df_labeled)))
    split_lbl.iloc[te_idx] = "Test"
    df_lab_out["Split"] = split_lbl.values
    for name in models:
        df_lab_out[f"Predicted_{name}"]  = df_labeled["Class"].values
        df_lab_out[f"Confidence_{name}"] = 1.0

    df_final = pd.concat([df_lab_out, df_pred_out]).sort_index().reset_index()
    return df_final, cv_sum, test_sum, perclass, roc_df


if __name__ == "__main__":
    raw             = pd.read_excel(PROCESSED_PATH, sheet_name=None)
    curation_labels = load_curation(CURATION_PATH)

    all_cv, all_test, all_perclass, all_roc = [], [], [], []

    with pd.ExcelWriter(PRED_PATH, engine="openpyxl") as writer:
        for mode in ["HILIC", "RP"]:
            result = predict_mode(raw, mode, curation_labels)
            if result is None:
                continue
            df_out, cv_sum, test_sum, perclass, roc_df = result
            cv_sum.insert(0, "Mode", mode)
            test_sum.insert(0, "Mode", mode)
            all_cv.append(cv_sum); all_test.append(test_sum)
            all_perclass.append(perclass); all_roc.append(roc_df)
            df_out.to_excel(writer, sheet_name=f"{mode}_Predictions", index=False)

        if all_cv:
            cv_all   = pd.concat(all_cv,   ignore_index=True)
            test_all = pd.concat(all_test, ignore_index=True)
            pd.concat(all_perclass, ignore_index=True).to_excel(writer, sheet_name="PerClass_TestSet_Scores", index=False)
            pd.concat(all_roc,      ignore_index=True).to_excel(writer, sheet_name="ROC_Raw_Data",           index=False)
            cv_all.to_excel(writer,   sheet_name="CV_Summary",      index=False)
            test_all.to_excel(writer, sheet_name="TestSet_Summary", index=False)
            cv_all.merge(test_all, on=["Mode", "Model"]).to_excel(
                writer, sheet_name="Performance_Overview", index=False)

    print(f"Done. Results saved to: {PRED_PATH}")
