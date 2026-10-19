"""
Log2 fold changes per nanotube: corona vs plasma and QWD vs pristine.

Input:  Master_Processed_Conservative_Data.xlsx, metabolite_class_predictions.xlsx
Output: {RP,HILIC}_{ALL,NEG,POS}_Log2FC.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp

from config import DEFECTS, OUT_LOG2FC, POLYMERS, PRED_PATH, PRISTINE, PROCESSED_PATH

OUT_DIR = OUT_LOG2FC
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODES       = ["RP", "HILIC"]
ION_FILTERS = {"ALL": None, "NEG": "-", "POS": "+"}

BASE_DEFECT  = PRISTINE
COMP_DEFECTS = [d for d in DEFECTS if d != BASE_DEFECT]
REF_DEFECT   = "Ref"
REF_POLYMER  = "Ref"
VALID_GAP    = ["No gap", "Missing ions", "No Gap", "Missing Ions"]
BEST_MODEL   = {"HILIC": "RandomForest", "RP": "RandomForest"}


# Class labels
def load_class_lookup(classification_path: Path, mode: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(classification_path, sheet_name=f"{mode}_Predictions")
    except Exception as e:
        print(f"  Could not load class predictions for {mode}: {e}. Proceeding without class annotation.")
        return pd.DataFrame(columns=["MetaboliteClass", "ClassSource", "ClassConfidence"])

    best_model = BEST_MODEL[mode]
    pred_col, conf_col = f"Predicted_{best_model}", f"Confidence_{best_model}"
    is_labeled = df["Split"].isin(["Train", "Test"])

    meta_class = df["Class"].where(is_labeled, df[pred_col])
    source = pd.Series("Low_Confidence", index=df.index)
    source[is_labeled] = "Labeled"
    source[~is_labeled & (meta_class != "Low Confidence")] = f"ML_{best_model}"

    result = pd.DataFrame({"MetaboliteClass": meta_class.values,
                           "ClassSource": source.values,
                           "ClassConfidence": df[conf_col].values},
                          index=df["FeatureID"].astype(str).values)
    n_ml  = (~is_labeled & (source != "Low_Confidence")).sum()
    n_low = (source == "Low_Confidence").sum()
    print(f"  Class lookup ({mode}): {is_labeled.sum()} labeled | {n_ml} ML-predicted | {n_low} Low Confidence")
    return result


# Helpers
def _pick_feature_cols_by_ion(df_full, feature_cols_all, ion_key):
    sign = ION_FILTERS[ion_key]
    if sign is None:
        return feature_cols_all
    # the "mode_m/z" header row holds labels such as "HILIC-_123.0456"
    mz_row = next(i for i in range(min(8, len(df_full)))
                  if "m/z" in str(df_full.iloc[i, 2]).lower())
    ion_labels = df_full.iloc[mz_row][feature_cols_all].astype(str)
    return [c for c in feature_cols_all if sign in ion_labels.get(c, "")]


def _nanmean_var_ddof1(X):
    X    = np.asarray(X, dtype=float)
    n    = np.sum(np.isfinite(X), axis=0).astype(float)
    mean = np.nanmean(X, axis=0)
    var  = np.full(mean.shape, np.nan)
    ok   = n >= 2
    if np.any(ok):
        var[ok] = np.nanvar(X[:, ok], axis=0, ddof=1)
    return mean, var, n


def _combine(d_mat, se_mat):
    """IVW combination across extraction methods. d_mat, se_mat: (n_methods, n_features)."""
    d_mat, se_mat = np.asarray(d_mat, float), np.asarray(se_mat, float)
    ok = np.isfinite(se_mat) & (se_mat > 0) & np.isfinite(d_mat)
    w  = np.zeros_like(se_mat)
    w[ok] = 1.0 / se_mat[ok] ** 2

    sum_w  = w.sum(axis=0)
    n_used = ok.sum(axis=0).astype(int)
    F = d_mat.shape[1]
    d_star, se_star, p_star, neglogp = (np.full(F, np.nan) for _ in range(4))

    ok2 = sum_w > 0
    if np.any(ok2):
        num = np.sum(np.where(ok[:, ok2], w[:, ok2] * d_mat[:, ok2], 0.0), axis=0)
        d_star[ok2]  = num / sum_w[ok2]
        se_star[ok2] = np.sqrt(1.0 / sum_w[ok2])
        logp = np.log(2.0) + norm.logsf(np.abs(d_star[ok2] / se_star[ok2]))
        neglogp[ok2] = -(logp / np.log(10.0))
        p_star[ok2]  = np.exp(logp)
    return d_star, se_star, p_star, neglogp, n_used


def _weight_summary_rows(comparison, polymer, defect, overlap_methods, d_mat, se_mat):
    d_mat, se_mat = np.asarray(d_mat, float), np.asarray(se_mat, float)
    ok = np.isfinite(se_mat) & (se_mat > 0) & np.isfinite(d_mat)
    w  = np.zeros_like(se_mat)
    w[ok] = 1.0 / se_mat[ok] ** 2
    sum_w   = w.sum(axis=0)
    feat_ok = sum_w > 0
    n_used_any = int(feat_ok.sum())

    w_norm = np.zeros_like(w)
    if n_used_any:
        w_norm[:, feat_ok] = w[:, feat_ok] / sum_w[feat_ok]

    rows = []
    for i, m in enumerate(overlap_methods):
        rows.append({
            "Comparison": comparison, "Polymer": polymer, "Defect": defect, "Method": m,
            "nOverlapMethods": len(overlap_methods), "nFeaturesTotal": int(d_mat.shape[1]),
            "nFeaturesUsedAny": n_used_any,
            "nFeaturesUsedThisMethod": int((w[i, :] > 0).sum()) if n_used_any else 0,
            "MeanWeightNorm":   float(np.mean(w_norm[i, feat_ok]))   if n_used_any else np.nan,
            "MedianWeightNorm": float(np.median(w_norm[i, feat_ok])) if n_used_any else np.nan,
        })
    return rows


def _prepend_class_columns(df_out, class_lookup, features):
    feat_str = [str(f) for f in features]
    get = lambda col: [class_lookup.at[f, col] if f in class_lookup.index else np.nan for f in feat_str]
    df_out.insert(1, "MetaboliteClass", get("MetaboliteClass"))
    df_out.insert(2, "ClassSource",     get("ClassSource"))
    df_out.insert(3, "ClassConfidence", [round(float(c), 4) if pd.notna(c) else np.nan
                                         for c in get("ClassConfidence")])
    return df_out


def _allnt_mean_ttest(fc_mat, se_mat):
    """Pooled ALL_NT Log2FC: mean + one-sample t-test (IVW z-test when n = 1)."""
    fc_mat, se_mat = np.asarray(fc_mat, float), np.asarray(se_mat, float)
    F = fc_mat.shape[0]
    mean_fc, p_val, neglogp = (np.full(F, np.nan) for _ in range(3))
    n_valid = np.zeros(F, dtype=int)

    for i in range(F):
        valid = np.isfinite(fc_mat[i])
        n = int(valid.sum())
        n_valid[i] = n
        if n == 0:
            continue
        vals = fc_mat[i][valid]
        mean_fc[i] = vals.mean()
        if n >= 2:
            _, p = ttest_1samp(vals, popmean=0.0)
            p_val[i] = p
            if p > 0:
                neglogp[i] = -np.log10(p)
        else:
            se_valid = se_mat[i][valid]
            ok_se = np.isfinite(se_valid) & (se_valid > 0)
            if ok_se.any():
                se_ivw = 1.0 / np.sqrt((1.0 / se_valid[ok_se] ** 2).sum())
                logp = np.log(2.0) + norm.logsf(abs(mean_fc[i] / se_ivw))
                p_val[i] = np.exp(logp)
                neglogp[i] = -(logp / np.log(10.0))
    return mean_fc, p_val, neglogp, n_valid


def _method_contrast(X_final, methods_vec, idx_a, idx_b, overlap_methods):
    """Per-method difference of means (a - b) and its SE, stacked over methods."""
    d_list, se_list = [], []
    for m in overlap_methods:
        A_mean, A_var, nA = _nanmean_var_ddof1(X_final.loc[idx_a & (methods_vec == m)].to_numpy(float))
        B_mean, B_var, nB = _nanmean_var_ddof1(X_final.loc[idx_b & (methods_vec == m)].to_numpy(float))
        with np.errstate(all="ignore"):
            se_m = np.sqrt(A_var / nA + B_var / nB)
        d_list.append(A_mean - B_mean)
        se_list.append(np.where(np.isfinite(se_m) & (se_m > 0), se_m, np.nan))
    return np.vstack(d_list), np.vstack(se_list)


def _fill_empty(out, polymer, defect):
    for prefix in ["Log2FC", "SE", "p", "negLogP"]:
        out[f"{prefix}_{polymer}_{defect}"] = np.nan
    out[f"nM_{polymer}_{defect}"] = 0


def _store(out, polymer, defect, d_star, se_star, p_star, nlp, n_used):
    out[f"Log2FC_{polymer}_{defect}"]  = d_star
    out[f"SE_{polymer}_{defect}"]      = se_star
    out[f"p_{polymer}_{defect}"]       = p_star
    out[f"negLogP_{polymer}_{defect}"] = nlp
    out[f"nM_{polymer}_{defect}"]      = n_used


# Main
def run_one(raw, mode_name, ion_key, class_lookup):
    log2_sheet, gap_sheet = f"{mode_name}_log2_1p", f"{mode_name}_gap"
    if log2_sheet not in raw or gap_sheet not in raw:
        print(f"[skip] missing sheets: {log2_sheet} / {gap_sheet}")
        return

    df_full  = pd.DataFrame(raw[log2_sheet])
    gap_full = pd.DataFrame(raw[gap_sheet])
    df, gap  = df_full.iloc[3:].copy(), gap_full.iloc[3:].copy()
    rename = {"Unnamed: 0": "Defect", "Unnamed: 1": "Polymer", "Peak idx/annotation": "Method"}
    df.rename(columns=rename, inplace=True)
    gap.rename(columns=rename, inplace=True)

    feature_cols_all = [c for c in df.columns if c not in ["Defect", "Polymer", "Method"]]
    feature_cols = _pick_feature_cols_by_ion(df_full, feature_cols_all, ion_key)
    if not feature_cols:
        print(f"[skip] no feature cols after ion filter: {mode_name} {ion_key}")
        return

    defect_vec, polymer_vec = df["Defect"].astype(str), df["Polymer"].astype(str)
    mask_all = (defect_vec.isin(DEFECTS) & polymer_vec.isin(POLYMERS)) | \
               ((defect_vec == REF_DEFECT) & (polymer_vec == REF_POLYMER))
    sub_df  = df.loc[mask_all].reset_index(drop=True)
    sub_gap = gap.loc[mask_all].reset_index(drop=True)
    if sub_df.empty:
        print(f"[skip] empty after NT+Ref filter: {mode_name} {ion_key}")
        return

    X_vals  = sub_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_final = X_vals.where(sub_gap[feature_cols].isin(VALID_GAP), np.nan)[feature_cols]
    if not np.isfinite(X_final.to_numpy(float)).any():
        print(f"[skip] all features full-gap after masking: {mode_name} {ion_key}")
        return

    features     = [str(c) for c in feature_cols]
    defects_vec  = sub_df["Defect"].astype(str)
    polymers_vec = sub_df["Polymer"].astype(str)
    methods_vec  = sub_df["Method"].astype(str).fillna("NA")
    weight_rows_nt, weight_rows_qwd = [], []

    # (A) NT vs plasma
    out_nt = pd.DataFrame({"Feature": features})
    ref_idx = (polymers_vec == REF_POLYMER) & (defects_vec == REF_DEFECT)
    methods_ref = set(methods_vec[ref_idx])

    for polymer in POLYMERS:
        for defect in DEFECTS:
            grp_idx = (polymers_vec == polymer) & (defects_vec == defect)
            if grp_idx.sum() == 0:
                continue
            overlap = sorted(set(methods_vec[grp_idx]) & methods_ref)
            if not overlap:
                _fill_empty(out_nt, polymer, defect)
                continue
            d_mat, se_mat = _method_contrast(X_final, methods_vec, grp_idx, ref_idx, overlap)
            _store(out_nt, polymer, defect, *_combine(d_mat, se_mat))
            weight_rows_nt.extend(_weight_summary_rows("NT_vs_Plasma", polymer, defect, overlap, d_mat, se_mat))

    fc_cols = [c for c in out_nt.columns if c.startswith("Log2FC_")]
    se_cols = [c for c in out_nt.columns if c.startswith("SE_")]
    if fc_cols and se_cols:
        mean_all, p_all, nlp_all, n_all = _allnt_mean_ttest(out_nt[fc_cols].to_numpy(float),
                                                            out_nt[se_cols].to_numpy(float))
        out_nt["Log2FC_ALL_NT"]  = mean_all
        out_nt["nM_ALL_NT"]      = n_all
        out_nt["p_ALL_NT"]       = p_all
        out_nt["negLogP_ALL_NT"] = nlp_all
        print(f"  ALL_NT ({mode_name} {ion_key}): {int((n_all >= 2).sum())} t-test | "
              f"{int((n_all == 1).sum())} IVW fallback | {int((n_all == 0).sum())} no data")

    out_nt = _prepend_class_columns(out_nt, class_lookup, features)

    # (B) QWD vs pristine
    out_qwd = pd.DataFrame({"Feature": features})
    for polymer in POLYMERS:
        idx_R = (polymers_vec == polymer) & (defects_vec == BASE_DEFECT)
        if idx_R.sum() == 0:
            continue
        methods_R = set(methods_vec[idx_R])
        for defect in COMP_DEFECTS:
            idx_D = (polymers_vec == polymer) & (defects_vec == defect)
            if idx_D.sum() == 0:
                continue
            overlap = sorted(set(methods_vec[idx_D]) & methods_R)
            if not overlap:
                _fill_empty(out_qwd, polymer, defect)
                continue
            d_mat, se_mat = _method_contrast(X_final, methods_vec, idx_D, idx_R, overlap)
            _store(out_qwd, polymer, defect, *_combine(d_mat, se_mat))
            weight_rows_qwd.extend(_weight_summary_rows("QWD_vs_Pristine", polymer, defect, overlap, d_mat, se_mat))

    out_qwd = _prepend_class_columns(out_qwd, class_lookup, features)

    # Class summary
    feat_class = pd.DataFrame({"Feature": features,
                               "MetaboliteClass": out_nt["MetaboliteClass"].values,
                               "ClassSource": out_nt["ClassSource"].values})
    class_rows = []
    for cls, grp in feat_class.groupby("MetaboliteClass", dropna=False):
        src = grp["ClassSource"].value_counts().to_dict()
        class_rows.append({"MetaboliteClass": cls if pd.notna(cls) else "Unclassified",
                           "TotalFeatures": len(grp),
                           "nLabeled": src.get("Labeled", 0),
                           "nML_RandomForest": src.get("ML_RandomForest", 0),
                           "nLowConfidence": src.get("Low_Confidence", 0)})
    df_class_summary = pd.DataFrame(class_rows).sort_values("TotalFeatures", ascending=False)

    out_xlsx = OUT_DIR / f"{mode_name}_{ion_key}_Log2FC.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        out_nt.to_excel(w,  sheet_name="NT_vs_Plasma",    index=False)
        out_qwd.to_excel(w, sheet_name="QWD_vs_Pristine", index=False)
        pd.DataFrame(weight_rows_nt).to_excel(w,  sheet_name="NT_vs_Plasma_MethodWeightSummary",    index=False)
        pd.DataFrame(weight_rows_qwd).to_excel(w, sheet_name="QWD_vs_Pristine_MethodWeightSummary", index=False)
        df_class_summary.to_excel(w, sheet_name="ClassSummary", index=False)
    print(f"  saved: {out_xlsx.name} (NT: {out_nt.shape}, QWD: {out_qwd.shape})")


if __name__ == "__main__":
    raw = pd.read_excel(PROCESSED_PATH, sheet_name=None)
    print(f"Loaded sheets: {list(raw.keys())}")

    class_lookups = {}
    for mode in MODES:
        if PRED_PATH.exists():
            class_lookups[mode] = load_class_lookup(PRED_PATH, mode)
        else:
            print(f"Classification file not found: {PRED_PATH}. Running without class annotation.")
            class_lookups[mode] = pd.DataFrame(columns=["MetaboliteClass", "ClassSource", "ClassConfidence"])

    for mode in MODES:
        for ion in ["ALL", "NEG", "POS"]:
            print(f"Processing {mode} {ion}...")
            run_one(raw, mode, ion, class_lookups[mode])
    print("\nDone.")
