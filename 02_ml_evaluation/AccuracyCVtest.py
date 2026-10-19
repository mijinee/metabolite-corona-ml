"""
Classifier performance: cross-validation, test set, confusion matrices.

Input:  Master_Processed_Conservative_Data.xlsx, Other_Unknown_Manual_Curation.xlsx
Output: metabolite_class_predictions_accuracy.xlsx, confusion_matrices/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_curve)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from config import CURATION_PATH, OUT_ML_EVAL, PROCESSED_PATH
from utils.metabolite_classes import apply_merges, build_feature_table, load_curation

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. KNN and Random Forest will still run.")

SAVE_PATH = OUT_ML_EVAL
SAVE_PATH.mkdir(parents=True, exist_ok=True)

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


def perclass_cv_accuracy(X, y, model, le, scale=False) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    fold_accs = {cls: [] for cls in le.classes_}
    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_val = sc.transform(X_val)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        for cls_idx, cls_name in enumerate(le.classes_):
            mask = y_val == cls_idx
            if mask.sum() > 0:
                fold_accs[cls_name].append(float((y_pred[mask] == cls_idx).mean()))
    return {cls: {"accs": v,
                  "mean": float(np.mean(v)) if v else np.nan,
                  "std": float(np.std(v)) if len(v) > 1 else 0.0,
                  "n_folds_present": len(v)}
            for cls, v in fold_accs.items()}


def evaluate_models(X_tr_d, X_te_d, y_train, y_test, models, le, mode_name, save_path):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_rows, test_rows, perclass_rows, roc_rows, perclass_cv_rows = [], [], [], [], []
    test_preds = {}
    classes    = list(range(len(le.classes_)))
    y_test_bin = label_binarize(y_test, classes=classes)

    for name, model in models.items():
        X_tr, X_te = X_tr_d[name], X_te_d[name]

        print(f"\n{name}: {CV_FOLDS}-fold CV on train set (n={len(y_train)})")
        scoring = {"f1macro": "f1_macro", "f1weighted": "f1_weighted", "accuracy": "accuracy"}
        cv_res = cross_validate(model, X_tr, y_train, cv=cv, scoring=scoring)
        print(f"Macro F1: {cv_res['test_f1macro'].mean():.4f} ± {cv_res['test_f1macro'].std():.4f}")
        print(f"Accuracy: {cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}")
        cv_rows.append({"Model": name,
                        "MeanF1MacroCV":    round(cv_res["test_f1macro"].mean(), 4),
                        "StdF1MacroCV":     round(cv_res["test_f1macro"].std(), 4),
                        "MeanF1WeightedCV": round(cv_res["test_f1weighted"].mean(), 4),
                        "StdF1WeightedCV":  round(cv_res["test_f1weighted"].std(), 4),
                        "MeanAccuracyCV":   round(cv_res["test_accuracy"].mean(), 4),
                        "StdAccuracyCV":    round(cv_res["test_accuracy"].std(), 4)})

        pcv = perclass_cv_accuracy(X_tr, y_train, model, le, scale=(name == "KNN"))
        for cls_name, st in pcv.items():
            perclass_cv_rows.append({"Mode": mode_name, "Model": name, "Class": cls_name,
                                     "MeanAccuracyCV": round(st["mean"], 4),
                                     "StdAccuracyCV": round(st["std"], 4),
                                     "NFoldsPresent": st["n_folds_present"],
                                     "FoldAccuracies": str([round(v, 4) for v in st["accs"]])})

        print(f"{name}: test set evaluation (n={len(y_test)})")
        if name == "XGBoost":
            model = make_xgb()
        model.fit(X_tr, y_train)
        y_pred  = model.predict(X_te)
        y_score = model.predict_proba(X_te)
        test_preds[name] = y_pred

        print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred), zero_division=0))
        test_rows.append({"Model": name,
                          "MacroF1Test":    round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
                          "WeightedF1Test": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                          "AccuracyTest":   round(accuracy_score(y_test, y_pred), 4)})

        ps, rs, fs, ss = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
        for i, cls_name in enumerate(le.classes_):
            perclass_rows.append({"Mode": mode_name, "Model": name, "Class": cls_name,
                                  "Precision": round(float(ps[i]), 4), "Recall": round(float(rs[i]), 4),
                                  "F1": round(float(fs[i]), 4), "AccuracyTest": round(float(rs[i]), 4),
                                  "Support_test": int(ss[i])})

        for i, cls in enumerate(le.classes_):
            if y_test_bin[:, i].sum() > 0:
                fpr, tpr, thr = roc_curve(y_test_bin[:, i], y_score[:, i])
                for f, t, th in zip(fpr, tpr, thr):
                    roc_rows.append({"Mode": mode_name, "Model": name, "Class": cls,
                                     "FPR": round(float(f), 5), "TPR": round(float(t), 5),
                                     "Threshold": round(float(th), 5)})
    return (pd.DataFrame(cv_rows), pd.DataFrame(test_rows), pd.DataFrame(perclass_rows),
            pd.DataFrame(roc_rows), pd.DataFrame(perclass_cv_rows), test_preds)


def save_confusion_matrices(y_te, test_preds, classes_list, mode_name):
    cm_dir = SAVE_PATH / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for model_name, y_pred in test_preds.items():
        cm = confusion_matrix(y_te, y_pred, labels=list(range(len(classes_list))))
        pd.DataFrame(cm, index=classes_list, columns=classes_list).to_excel(
            cm_dir / f"{mode_name}_CM_{model_name}.xlsx")


def predict_metabolite_classes(raw, mode_name, curation_labels=None):
    print("\n" + "=" * 60 + f"\nProcessing: {mode_name}\n" + "=" * 60)
    df = build_feature_table(raw[f"{mode_name}_gap"], mode_name, curation_labels)

    labeled_mask = df["IsAnnotated"] & ~df["RuleClass"].isin(["Other/Unknown", "Cannot Classify"])
    df_labeled   = df[labeled_mask].copy()
    df_predict   = df[~labeled_mask].copy()
    if labeled_mask.sum() < 10:
        print("Not enough labeled data to train. Skipping.")
        return None

    df_labeled["Class"] = apply_merges(df_labeled["RuleClass"], mode_name)
    counts = df_labeled["Class"].value_counts()
    df_labeled = df_labeled[df_labeled["Class"].isin(counts[counts >= 3].index)].copy()
    df_labeled = df_labeled[df_labeled[FEATURE_COLS].notna().all(axis=1)].copy()

    le    = LabelEncoder()
    y_all = le.fit_transform(df_labeled["Class"].values)
    X_all = df_labeled[FEATURE_COLS].values

    splitter       = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    tr_idx, te_idx = next(splitter.split(X_all, y_all))
    y_tr, y_te     = y_all[tr_idx], y_all[te_idx]

    scaler     = StandardScaler()
    X_tr_d     = {"KNN": scaler.fit_transform(X_all[tr_idx]), "RandomForest": X_all[tr_idx], "XGBoost": X_all[tr_idx]}
    X_te_d     = {"KNN": scaler.transform(X_all[te_idx]),     "RandomForest": X_all[te_idx], "XGBoost": X_all[te_idx]}
    X_pred_raw = np.nan_to_num(df_predict[FEATURE_COLS].values.copy())

    models = build_models()
    cv_summary, test_summary, perclass_df, roc_df, perclass_cv_df, test_preds = evaluate_models(
        X_tr_d, X_te_d, y_tr, y_te, models, le, mode_name, SAVE_PATH)

    te_feature_ids = df_labeled.iloc[te_idx]["FeatureID"].values
    test_pred_df = pd.DataFrame({"FeatureID": te_feature_ids,
                                 "TrueClass": le.inverse_transform(y_te), "Mode": mode_name})
    for model_name, y_pred in test_preds.items():
        test_pred_df[f"Predicted_{model_name}"] = le.inverse_transform(y_pred)

    save_confusion_matrices(y_te, test_preds, list(le.classes_), mode_name)

    # Retrain on all labeled features for final predictions
    scaler_full = StandardScaler()
    X_full = {"KNN": scaler_full.fit_transform(X_all), "RandomForest": X_all, "XGBoost": X_all}
    X_pr   = {"KNN": scaler_full.transform(X_pred_raw) if len(df_predict) > 0 else np.empty((0, len(FEATURE_COLS))),
              "RandomForest": X_pred_raw, "XGBoost": X_pred_raw}

    df_predict_out = df_predict.copy().set_index("FeatureID")
    df_predict_out["Split"] = "Predict"
    for model_name, model in models.items():
        if model_name == "XGBoost":
            model = make_xgb()
        model.fit(X_full[model_name], y_all)
        if len(df_predict) > 0:
            confidence = np.max(model.predict_proba(X_pr[model_name]), axis=1)
            labels     = le.inverse_transform(model.predict(X_pr[model_name]))
            df_predict_out[f"Predicted_{model_name}"]  = np.where(confidence >= CONFIDENCE_THRESHOLD,
                                                                  labels, "Low Confidence")
            df_predict_out[f"Confidence_{model_name}"] = confidence.round(4)

    df_labeled_out = df_labeled.copy().set_index("FeatureID")
    split_label = pd.Series("Train", index=df_labeled.index)
    split_label.iloc[te_idx] = "Test"
    df_labeled_out["Split"] = split_label.values
    for model_name, y_pred in test_preds.items():
        te_map = dict(zip(te_feature_ids, le.inverse_transform(y_pred)))
        df_labeled_out[f"Predicted_{model_name}"]  = df_labeled["Class"].values
        df_labeled_out[f"Confidence_{model_name}"] = 1.0
        mask_te = df_labeled_out["Split"] == "Test"
        df_labeled_out.loc[mask_te, f"Predicted_{model_name}"] = df_labeled_out.loc[mask_te].index.map(te_map)

    df_final = pd.concat([df_labeled_out, df_predict_out]).sort_index().reset_index()
    return df_final, cv_summary, test_summary, perclass_df, roc_df, perclass_cv_df, test_pred_df


if __name__ == "__main__":
    print("Loading data...")
    raw = pd.read_excel(PROCESSED_PATH, sheet_name=None)
    curation_labels = load_curation(CURATION_PATH)

    out_xlsx = SAVE_PATH / "metabolite_class_predictions_accuracy.xlsx"
    collected = {k: [] for k in ["cv", "test", "perclass", "roc", "perclass_cv", "test_pred"]}

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for mode in ["HILIC", "RP"]:
            result = predict_metabolite_classes(raw, mode, curation_labels)
            if result is None:
                continue
            df_out, cv_sum, test_sum, perclass, roc_df, perclass_cv, test_pred_df = result
            cv_sum.insert(0, "Mode", mode)
            test_sum.insert(0, "Mode", mode)
            for k, v in zip(collected, [cv_sum, test_sum, perclass, roc_df, perclass_cv, test_pred_df]):
                collected[k].append(v)
            df_out.to_excel(writer, sheet_name=f"{mode}_Predictions", index=False)

        if collected["cv"]:
            cv_all   = pd.concat(collected["cv"],   ignore_index=True)
            test_all = pd.concat(collected["test"], ignore_index=True)
            overview = cv_all.merge(test_all, on=["Mode", "Model"])
            cv_all.to_excel(writer,   sheet_name="CVSummary",           index=False)
            test_all.to_excel(writer, sheet_name="TestSetSummary",      index=False)
            overview.to_excel(writer, sheet_name="PerformanceOverview", index=False)
            pd.concat(collected["perclass"],    ignore_index=True).to_excel(writer, sheet_name="PerClassTestSetScores", index=False)
            pd.concat(collected["perclass_cv"], ignore_index=True).to_excel(writer, sheet_name="PerClassCVAccuracy",    index=False)
            pd.concat(collected["roc"],         ignore_index=True).to_excel(writer, sheet_name="ROCRawData",            index=False)
            pd.concat(collected["test_pred"],   ignore_index=True).to_excel(writer, sheet_name="TestSet_Predictions",   index=False)

            print("\n" + "=" * 75 + "\nPERFORMANCE SUMMARY\n" + "=" * 75)
            for _, r in overview.iterrows():
                print(f"{r.Mode:<6} {r.Model:<14} "
                      f"CV MacroF1 {r.MeanF1MacroCV:.4f} ± {r.StdF1MacroCV:.4f}  "
                      f"CV Acc {r.MeanAccuracyCV:.4f} ± {r.StdAccuracyCV:.4f}  "
                      f"Test MacroF1 {r.MacroF1Test:.4f}  Test Acc {r.AccuracyTest:.4f}")

    print(f"\nAll results saved to: {out_xlsx}")
