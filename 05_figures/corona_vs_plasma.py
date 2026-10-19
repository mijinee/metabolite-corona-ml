"""
Corona vs plasma, all nanotubes pooled: volcano, RT distribution, class RT heatmap.

Input:  HILIC_ALL_Log2FC.xlsx, RP_ALL_Log2FC.xlsx, metabolite_class_predictions.xlsx
Output: corona_vs_plasma.xlsx, corona_vs_plasma_plot_data.xlsx, ALL_NT_summary_qvalue.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (DEFECTS, FC_THRESHOLD, HILIC_LOG2FC, HILIC_TRACK1, OUT_FIGURES,
                    POLYMERS, PRED_PATH, Q_THRESHOLD, RP_CLASSES, RP_LOG2FC)
from utils.dedup_features import add_dedup_group, deduplicate
from utils.stats import bh_fdr

OUT_DIR = OUT_FIGURES / "corona_vs_plasma"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG2FC_CLIP  = 4.5
NEGLOGQ_CLIP = 8.5
RT_BIN_WIDTH = 0.5
RT_BIN_MAX   = 11.0   # fixed bins 0-0.5 ... 10.5-11.0 min


def add_qvalue_columns(df):
    """BH-FDR per condition, across the features tested in that condition."""
    df = df.copy()
    if "p_ALL_NT" in df.columns:
        df["q_ALL_NT"] = bh_fdr(df["p_ALL_NT"].values)
    for poly in POLYMERS:
        for defect in DEFECTS:
            if f"p_{poly}_{defect}" in df.columns:
                df[f"q_{poly}_{defect}"] = bh_fdr(df[f"p_{poly}_{defect}"].values)
    return df


def load_and_join_rt(log2fc_path, pred_path, mode):
    nt = pd.read_excel(log2fc_path, sheet_name="NT_vs_Plasma")
    pred = pd.read_excel(pred_path, sheet_name=f"{mode}_Predictions")
    pred["FeatureID"] = pred["FeatureID"].astype(str)
    rt_lookup = pred.set_index("FeatureID")[["RT", "MZ"]].to_dict("index")
    nt["Feature"] = nt["Feature"].astype(str)
    nt["RT"] = nt["Feature"].map(lambda f: rt_lookup.get(f, {}).get("RT", np.nan))
    nt["MZ"] = nt["Feature"].map(lambda f: rt_lookup.get(f, {}).get("MZ", np.nan))
    print(f"  {mode}: {len(nt)} features, RT joined for {nt['RT'].notna().sum()}")
    return nt


def rejoin_rt_after_dedup(df_dedup, df_orig):
    df_dedup["RT"] = df_dedup["Feature"].map(df_orig.groupby("DedupGroup")["RT"].median())
    df_dedup["MZ"] = df_dedup["Feature"].map(df_orig.groupby("DedupGroup")["MZ"].median())
    return df_dedup


def classify_significance(log2fc, q_val):
    if pd.isna(log2fc) or pd.isna(q_val):
        return "No data"
    if q_val < Q_THRESHOLD and log2fc > FC_THRESHOLD:
        return "Corona-enriched"
    if q_val < Q_THRESHOLD and log2fc < -FC_THRESHOLD:
        return "Plasma-enriched"
    return "Non-significant"


def build_volcano_data(df, fc_col, p_col, q_col, condition_label):
    sub = df[["Feature", "MetaboliteClass", "ClassSource", "RT", "MZ", fc_col, p_col, q_col]].copy()
    sub = sub.rename(columns={fc_col: "Log2FC", p_col: "p_value", q_col: "q_value"})
    sub["Condition"] = condition_label
    sub["Significance"] = sub.apply(lambda r: classify_significance(r["Log2FC"], r["q_value"]), axis=1)
    sub["negLogQ"] = np.nan
    ok = sub["q_value"].notna() & (sub["q_value"] > 0)
    sub.loc[ok, "negLogQ"] = -np.log10(sub.loc[ok, "q_value"])
    return sub.dropna(subset=["Log2FC"])


def build_rt_class_heatmap(df, classes, conditions):
    rows = []
    for cls in classes:
        sub     = df[df["MetaboliteClass"] == cls]
        vals    = sub["Log2FC_ALL_NT"].dropna()
        rt_vals = sub["RT"].dropna()
        row = {"Class": cls,
               "median_RT":            rt_vals.median() if len(rt_vals) else np.nan,
               "n_features":           len(vals),
               "median_Log2FC_ALL_NT": vals.median() if len(vals) else np.nan,
               "pct_positive":         100 * (vals > 0).mean() if len(vals) else np.nan}
        for cond in conditions:
            if f"Log2FC_{cond}" in sub.columns:
                cv = sub[f"Log2FC_{cond}"].dropna()
                row[f"median_Log2FC_{cond}"] = cv.median() if len(cv) >= 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("median_RT")


def build_rt_distribution(df):
    sub = df.dropna(subset=["Log2FC_ALL_NT", "q_ALL_NT", "RT"]).copy()
    sub["Direction"] = "Non-significant"
    sig = sub["q_ALL_NT"] < Q_THRESHOLD
    sub.loc[sig & (sub["Log2FC_ALL_NT"] >  FC_THRESHOLD), "Direction"] = "Corona-enriched"
    sub.loc[sig & (sub["Log2FC_ALL_NT"] < -FC_THRESHOLD), "Direction"] = "Plasma-enriched"

    sig_df = sub[sub["Direction"].isin(["Corona-enriched", "Plasma-enriched"])].copy()
    rt_features = sig_df[["Feature", "MetaboliteClass", "RT", "MZ", "Log2FC_ALL_NT", "Direction"]].copy()

    summary_rows = []
    for direction in ["Corona-enriched", "Plasma-enriched"]:
        d = sig_df[sig_df["Direction"] == direction]
        if len(d):
            summary_rows.append({"Direction": direction, "n_features": len(d),
                                 "median_RT": d["RT"].median(),
                                 "q25_RT": d["RT"].quantile(0.25), "q75_RT": d["RT"].quantile(0.75),
                                 "min_RT": d["RT"].min(), "max_RT": d["RT"].max(),
                                 "median_Log2FC": d["Log2FC_ALL_NT"].median(),
                                 "median_MZ": d["MZ"].median()})

    n_bins = 20
    edges = np.linspace(sub["RT"].min(), sub["RT"].max(), n_bins + 1)
    bin_rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = sig_df[(sig_df["RT"] >= lo) & (sig_df["RT"] < hi if i < n_bins - 1 else sig_df["RT"] <= hi)]
        n_c = (in_bin["Direction"] == "Corona-enriched").sum()
        n_p = (in_bin["Direction"] == "Plasma-enriched").sum()
        n_t = n_c + n_p
        bin_rows.append({"RT_bin_lo": round(lo, 2), "RT_bin_hi": round(hi, 2),
                         "RT_bin_mid": round((lo + hi) / 2, 2),
                         "n_corona_enriched": n_c, "n_plasma_enriched": n_p, "n_total": n_t,
                         "pct_corona": 100 * n_c / n_t if n_t else np.nan,
                         "pct_plasma": 100 * n_p / n_t if n_t else np.nan})
    return rt_features, pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


# Plot-ready tables
def volcano_region_table(df_volcano, mode_label):
    """One X/Y column pair per region, clipped, ragged length."""
    sub = df_volcano[(df_volcano["Mode"] == mode_label) & (df_volcano["Condition"] == "ALL_NT")].copy()
    region_map = {"Corona-enriched": "CoronaEnriched", "Plasma-enriched": "PlasmaEnriched"}
    sub["Region"] = sub["Significance"].map(region_map).fillna("NonSignificant")
    sub["Log2FC_clipped"]  = sub["Log2FC"].clip(-LOG2FC_CLIP, LOG2FC_CLIP)
    sub["negLogQ_clipped"] = sub["negLogQ"].clip(upper=NEGLOGQ_CLIP)

    series = {}
    for region in ["CoronaEnriched", "PlasmaEnriched", "NonSignificant"]:
        vals = sub[sub["Region"] == region]
        series[f"X_{region}"] = vals["Log2FC_clipped"].reset_index(drop=True)
        series[f"Y_{region}"] = vals["negLogQ_clipped"].reset_index(drop=True)
    print(f"  {mode_label}: Corona={len(series['X_CoronaEnriched'])}, "
          f"Plasma={len(series['X_PlasmaEnriched'])}, NonSig={len(series['X_NonSignificant'])}")
    n = max(len(v) for v in series.values())
    return pd.DataFrame({k: v.reindex(range(n)) for k, v in series.items()})


def rt_raw_table(df_rt_features, mode_label):
    sub = df_rt_features[df_rt_features["Mode"] == mode_label]
    corona = sub[sub["Direction"] == "Corona-enriched"]["RT"].reset_index(drop=True)
    plasma = sub[sub["Direction"] == "Plasma-enriched"]["RT"].reset_index(drop=True)
    n = max(len(corona), len(plasma))
    return pd.DataFrame({"Corona_RT": corona.reindex(range(n)), "Plasma_RT": plasma.reindex(range(n))})


def rt_binned_fraction(df_rt_features, mode_label):
    sub = df_rt_features[df_rt_features["Mode"] == mode_label]
    edges = np.round(np.arange(0, RT_BIN_MAX + RT_BIN_WIDTH, RT_BIN_WIDTH), 4)
    n_corona_total = (sub["Direction"] == "Corona-enriched").sum()
    n_plasma_total = (sub["Direction"] == "Plasma-enriched").sum()
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = sub[(sub["RT"] >= lo) & (sub["RT"] < hi if i < len(edges) - 2 else sub["RT"] <= hi)]
        rows.append({"bin_lo": lo, "bin_hi": hi, "bin_mid": round((lo + hi) / 2, 4),
                     "Corona_fraction": (in_bin["Direction"] == "Corona-enriched").sum() / n_corona_total
                     if n_corona_total else np.nan,
                     "Plasma_fraction": (in_bin["Direction"] == "Plasma-enriched").sum() / n_plasma_total
                     if n_plasma_total else np.nan})
    return pd.DataFrame(rows)


# Main
print("Loading data...")
hilic_nt = add_dedup_group(load_and_join_rt(HILIC_LOG2FC, PRED_PATH, "HILIC"))
rp_nt    = add_dedup_group(load_and_join_rt(RP_LOG2FC,    PRED_PATH, "RP"))

print("Deduplicating and applying BH-FDR...")
hilic_dedup = rejoin_rt_after_dedup(add_qvalue_columns(deduplicate(hilic_nt, method="median")), hilic_nt)
rp_dedup    = rejoin_rt_after_dedup(add_qvalue_columns(deduplicate(rp_nt,    method="median")), rp_nt)

print("Building volcano data...")
volcano_frames = []
for mode_label, df in [("RP", rp_dedup), ("HILIC", hilic_dedup)]:
    v = build_volcano_data(df, "Log2FC_ALL_NT", "p_ALL_NT", "q_ALL_NT", "ALL_NT")
    v.insert(0, "Mode", mode_label)
    volcano_frames.append(v)
    for poly in POLYMERS:
        for defect in DEFECTS:
            fc_col, p_col, q_col = f"Log2FC_{poly}_{defect}", f"p_{poly}_{defect}", f"q_{poly}_{defect}"
            if all(c in df.columns for c in [fc_col, p_col, q_col]):
                vc = build_volcano_data(df, fc_col, p_col, q_col, f"{poly}_{defect}")
                vc.insert(0, "Mode", mode_label)
                volcano_frames.append(vc)
df_volcano = pd.concat(volcano_frames, ignore_index=True)
volcano_summary = (df_volcano[df_volcano["Significance"] != "No data"]
                   .groupby(["Mode", "Condition", "Significance"]).size()
                   .unstack(fill_value=0).reset_index())

print("Building RT distribution data...")
rt_feat, rt_summ, rt_bins = [], [], []
for mode_label, df in [("RP", rp_dedup), ("HILIC", hilic_dedup)]:
    f, s, b = build_rt_distribution(df)
    for frame in [f, s, b]:
        frame.insert(0, "Mode", mode_label)
    rt_feat.append(f); rt_summ.append(s); rt_bins.append(b)
df_rt_features = pd.concat(rt_feat, ignore_index=True)
df_rt_summary  = pd.concat(rt_summ, ignore_index=True)
df_rt_binned   = pd.concat(rt_bins, ignore_index=True)

print("Building class-level RT heatmap data...")
conditions = [f"{p}_{d}" for p in POLYMERS for d in DEFECTS]
rp_heatmap    = build_rt_class_heatmap(rp_dedup,    RP_CLASSES,   conditions)
hilic_heatmap = build_rt_class_heatmap(hilic_dedup, HILIC_TRACK1, conditions)
rp_heatmap.insert(0, "Mode", "RP")
hilic_heatmap.insert(0, "Mode", "HILIC")
df_heatmap = pd.concat([rp_heatmap, hilic_heatmap], ignore_index=True)

print("Building ALL_NT summary...")
summary_rows = []
for mode_label, n_full, df in [("RP", len(rp_nt), rp_dedup), ("HILIC", len(hilic_nt), hilic_dedup)]:
    allnt = df["Log2FC_ALL_NT"].dropna()
    n_sig = int(((df["q_ALL_NT"] < Q_THRESHOLD) & (df["Log2FC_ALL_NT"].abs() > FC_THRESHOLD)).sum())
    summary_rows.append({"Mode": mode_label, "Total_features_IVW": n_full,
                         "Total_features_dedup": len(df), "Features_with_ALL_NT": len(allnt),
                         "Median_Log2FC_ALL_NT": allnt.median(), "Mean_Log2FC_ALL_NT": allnt.mean(),
                         "Sig_features_q05_FC1": n_sig,
                         "Pct_sig": round(100 * n_sig / len(allnt), 1) if len(allnt) else 0})
df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))

print(f"Exporting to: {OUT_DIR}")
with pd.ExcelWriter(OUT_DIR / "corona_vs_plasma.xlsx", engine="openpyxl") as w:
    df_volcano[df_volcano["Condition"] == "ALL_NT"].to_excel(w, sheet_name="Volcano_ALL_NT",       index=False)
    df_volcano[df_volcano["Condition"] != "ALL_NT"].to_excel(w, sheet_name="Volcano_PerCondition", index=False)
    volcano_summary.to_excel(w, sheet_name="Volcano_Summary", index=False)
    df_rt_features.to_excel(w,  sheet_name="RT_Features",     index=False)
    df_rt_summary.to_excel(w,   sheet_name="RT_Summary",      index=False)
    df_rt_binned.to_excel(w,    sheet_name="RT_Binned",       index=False)
    df_heatmap.to_excel(w,      sheet_name="RT_ClassHeatmap", index=False)

with pd.ExcelWriter(OUT_DIR / "corona_vs_plasma_plot_data.xlsx", engine="openpyxl") as w:
    for mode_label in ["RP", "HILIC"]:
        volcano_region_table(df_volcano, mode_label).to_excel(w, sheet_name=f"Volcano_{mode_label}", index=False)
        rt_raw_table(df_rt_features, mode_label).to_excel(w, sheet_name=f"RT_raw_{mode_label}", index=False)
        rt_binned_fraction(df_rt_features, mode_label).to_excel(w, sheet_name=f"RT_binned_{mode_label}", index=False)

df_summary.to_excel(OUT_DIR / "ALL_NT_summary_qvalue.xlsx", sheet_name="ALL_NT_summary", index=False)
print("Done.")
