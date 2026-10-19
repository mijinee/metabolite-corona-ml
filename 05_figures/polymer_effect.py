"""
Polymer effect on the corona: PCA input, ssDNA vs PEG volcano, RT distribution, bubble plot, GSEA.

Input:  HILIC_ALL_Log2FC.xlsx, RP_ALL_Log2FC.xlsx, metabolite_class_predictions.xlsx
Output: polymer_effect.xlsx, polymer_effect_plot_data.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import norm

from config import (DEFECTS, DNA_POLYMERS, FC_THRESHOLD, HILIC_LOG2FC, HILIC_TRACK1,
                    HILIC_TRACK2, OUT_FIGURES, PEG_POLYMERS, POLYMERS, PRED_PATH,
                    Q_THRESHOLD, RP_CLASSES, RP_LOG2FC)
from utils.dedup_features import add_dedup_group, deduplicate
from utils.stats import bh_fdr, running_enrichment_score

OUT_DIR = OUT_FIGURES / "polymer_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG2FC_CLIP  = 4.0
NEGLOGQ_CLIP = 8.0
RT_BIN_WIDTH = 0.5
RT_BIN_MAX   = 11.0


def polymer_type(poly):
    return "DNA" if poly in DNA_POLYMERS else "PEG"


def load_and_join_rt(log2fc_path, pred_path, mode):
    nt = pd.read_excel(log2fc_path, sheet_name="NT_vs_Plasma")
    pred = pd.read_excel(pred_path, sheet_name=f"{mode}_Predictions")
    pred["FeatureID"] = pred["FeatureID"].astype(str)
    rt_lookup = pred.set_index("FeatureID")[["RT", "MZ"]].to_dict("index")
    nt["Feature"] = nt["Feature"].astype(str)
    nt["RT"] = nt["Feature"].map(lambda f: rt_lookup.get(f, {}).get("RT", np.nan))
    nt["MZ"] = nt["Feature"].map(lambda f: rt_lookup.get(f, {}).get("MZ", np.nan))
    return nt


def dedup_with_rt(df):
    if "DedupGroup" not in df.columns:
        df = add_dedup_group(df)
    df_d = deduplicate(df, method="median")
    df_d["RT"] = df_d["Feature"].map(df.groupby("DedupGroup")["RT"].median())
    df_d["MZ"] = df_d["Feature"].map(df.groupby("DedupGroup")["MZ"].median())
    return df, df_d


print("Loading data...")
hilic_nt, hilic_d = dedup_with_rt(load_and_join_rt(HILIC_LOG2FC, PRED_PATH, "HILIC"))
rp_nt,    rp_d    = dedup_with_rt(load_and_join_rt(RP_LOG2FC,    PRED_PATH, "RP"))


# PCA input matrices
print("\nPCA data")
pca_sheets = {}
for mode_label, df in [("HILIC", hilic_d), ("RP", rp_d)]:
    meta = df[[c for c in ["Feature", "MetaboliteClass", "ClassSource", "RT", "MZ"] if c in df.columns]]

    pristine_cols = [f"Log2FC_{p}_R" for p in POLYMERS if f"Log2FC_{p}_R" in df.columns]
    pca_sheets[f"{mode_label}_PCA_Pristine"] = (
        pd.concat([meta, df[pristine_cols]], axis=1).dropna(subset=pristine_cols, how="all"))

    all_cols = [f"Log2FC_{p}_{d}" for p in POLYMERS for d in DEFECTS if f"Log2FC_{p}_{d}" in df.columns]
    pca_sheets[f"{mode_label}_PCA_AllCond"] = (
        pd.concat([meta, df[all_cols]], axis=1).dropna(subset=all_cols, how="all"))
    print(f"  {mode_label}: pristine {len(pca_sheets[f'{mode_label}_PCA_Pristine'])} features, "
          f"all conditions {len(pca_sheets[f'{mode_label}_PCA_AllCond'])} features")


# ssDNA vs PEG volcano
print("\nssDNA vs PEG volcano")


def dna_vs_peg_test(feat, dna_fc, peg_fc, dna_se, peg_se):
    dna_vals = feat[dna_fc].dropna().values.astype(float)
    peg_vals = feat[peg_fc].dropna().values.astype(float)
    if len(dna_vals) == 0 or len(peg_vals) == 0:
        return None
    diff = dna_vals.mean() - peg_vals.mean()
    if len(dna_vals) >= 2 and len(peg_vals) >= 2:
        _, p_val = sp_stats.ttest_ind(dna_vals, peg_vals, equal_var=False)
    else:
        dna_se_vals = feat[dna_se].dropna().values.astype(float)
        peg_se_vals = feat[peg_se].dropna().values.astype(float)
        ok_dna = np.isfinite(dna_se_vals) & (dna_se_vals > 0)
        ok_peg = np.isfinite(peg_se_vals) & (peg_se_vals > 0)
        if ok_dna.any() and ok_peg.any():
            se_dna = 1.0 / np.sqrt((1.0 / dna_se_vals[ok_dna] ** 2).sum())
            se_peg = 1.0 / np.sqrt((1.0 / peg_se_vals[ok_peg] ** 2).sum())
            p_val = 2 * norm.sf(abs(diff / np.sqrt(se_dna ** 2 + se_peg ** 2)))
        else:
            p_val = np.nan
    return diff, dna_vals.mean(), peg_vals.mean(), len(dna_vals), len(peg_vals), p_val


def classify_dna_peg(log2fc, q_val):
    if pd.isna(log2fc) or pd.isna(q_val):
        return "Non-significant"
    if q_val < Q_THRESHOLD and log2fc > FC_THRESHOLD:
        return "DNA-enriched"
    if q_val < Q_THRESHOLD and log2fc < -FC_THRESHOLD:
        return "PEG-enriched"
    return "Non-significant"


volcano_frames = []
for mode_label, df in [("RP", rp_d), ("HILIC", hilic_d)]:
    dna_fc = [f"Log2FC_{p}_R" for p in DNA_POLYMERS if f"Log2FC_{p}_R" in df.columns]
    peg_fc = [f"Log2FC_{p}_R" for p in PEG_POLYMERS if f"Log2FC_{p}_R" in df.columns]
    dna_se = [f"SE_{p}_R" for p in DNA_POLYMERS if f"SE_{p}_R" in df.columns]
    peg_se = [f"SE_{p}_R" for p in PEG_POLYMERS if f"SE_{p}_R" in df.columns]

    rows = []
    for _, feat in df.iterrows():
        res = dna_vs_peg_test(feat, dna_fc, peg_fc, dna_se, peg_se)
        if res is None:
            continue
        diff, dna_mean, peg_mean, n_dna, n_peg, p_val = res
        rows.append({"Mode": mode_label, "Feature": feat["Feature"],
                     "MetaboliteClass": feat.get("MetaboliteClass", np.nan),
                     "ClassSource": feat.get("ClassSource", np.nan),
                     "RT": feat.get("RT", np.nan), "MZ": feat.get("MZ", np.nan),
                     "Log2FC_DNA_vs_PEG": diff, "DNA_mean_Log2FC": dna_mean, "PEG_mean_Log2FC": peg_mean,
                     "n_DNA": n_dna, "n_PEG": n_peg, "p_value": p_val,
                     **{f"Log2FC_{p}_R": feat.get(f"Log2FC_{p}_R", np.nan) for p in POLYMERS}})
    df_volc = pd.DataFrame(rows)
    df_volc["q_value"] = bh_fdr(df_volc["p_value"].values)       # BH-FDR within mode
    volcano_frames.append(df_volc)

df_volcano = pd.concat(volcano_frames, ignore_index=True)
with np.errstate(divide="ignore"):
    df_volcano["negLogQ"] = np.where(df_volcano["q_value"].notna() & (df_volcano["q_value"] > 0),
                                     -np.log10(df_volcano["q_value"]), np.nan)
df_volcano["Significance"] = df_volcano.apply(
    lambda r: classify_dna_peg(r["Log2FC_DNA_vs_PEG"], r["q_value"]), axis=1)

for mode_label in ["RP", "HILIC"]:
    sub = df_volcano[df_volcano["Mode"] == mode_label]
    print(f"  {mode_label}: {(sub['Significance'] == 'DNA-enriched').sum()} DNA-enriched, "
          f"{(sub['Significance'] == 'PEG-enriched').sum()} PEG-enriched, "
          f"{(sub['Significance'] == 'Non-significant').sum()} non-sig")

volcano_summary = (df_volcano.groupby(["Mode", "Significance"]).size()
                   .unstack(fill_value=0).reset_index())


# RT distribution
df_rt = df_volcano[df_volcano["Significance"].isin(["DNA-enriched", "PEG-enriched"]) &
                   df_volcano["RT"].notna()][
    ["Mode", "Feature", "MetaboliteClass", "RT", "Log2FC_DNA_vs_PEG", "Significance"]].copy()


def volcano_region_table(mode_label):
    sub = df_volcano[df_volcano["Mode"] == mode_label]
    series = {}
    for label, tag in [("DNA-enriched", "DNAenriched"), ("PEG-enriched", "PEGenriched"),
                       ("Non-significant", "NonSignificant")]:
        vals = sub[sub["Significance"] == label]
        series[f"X_{tag}"] = vals["Log2FC_DNA_vs_PEG"].clip(-LOG2FC_CLIP, LOG2FC_CLIP).reset_index(drop=True)
        series[f"Y_{tag}"] = vals["negLogQ"].clip(upper=NEGLOGQ_CLIP).reset_index(drop=True)
    n = max(len(v) for v in series.values())
    return pd.DataFrame({k: v.reindex(range(n)) for k, v in series.items()})


def rt_raw_table(mode_label):
    sub = df_rt[df_rt["Mode"] == mode_label]
    dna = sub[sub["Significance"] == "DNA-enriched"]["RT"].reset_index(drop=True)
    peg = sub[sub["Significance"] == "PEG-enriched"]["RT"].reset_index(drop=True)
    n = max(len(dna), len(peg))
    return pd.DataFrame({"DNA_RT": dna.reindex(range(n)), "PEG_RT": peg.reindex(range(n))})


def rt_binned_fraction(mode_label):
    sub = df_rt[df_rt["Mode"] == mode_label]
    edges = np.round(np.arange(0, RT_BIN_MAX + RT_BIN_WIDTH, RT_BIN_WIDTH), 4)
    n_dna = (sub["Significance"] == "DNA-enriched").sum()
    n_peg = (sub["Significance"] == "PEG-enriched").sum()
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = sub[(sub["RT"] >= lo) & (sub["RT"] < hi if i < len(edges) - 2 else sub["RT"] <= hi)]
        rows.append({"bin_lo": lo, "bin_hi": hi, "bin_mid": round((lo + hi) / 2, 4),
                     "DNA_fraction": (in_bin["Significance"] == "DNA-enriched").sum() / n_dna if n_dna else np.nan,
                     "PEG_fraction": (in_bin["Significance"] == "PEG-enriched").sum() / n_peg if n_peg else np.nan})
    return pd.DataFrame(rows)


# Bubble plot (pristine, per polymer x class)
print("\nBubble plot data")
bubble_rows = []
for mode_label, df, classes in [("RP", rp_d, RP_CLASSES), ("HILIC", hilic_d, HILIC_TRACK1)]:
    for cls_idx, cls in enumerate(classes):
        mask_cls = df["MetaboliteClass"] == cls
        all_vals = np.concatenate([df.loc[mask_cls, f"Log2FC_{p}_R"].dropna().values
                                   for p in POLYMERS if f"Log2FC_{p}_R" in df.columns] or [np.array([])])
        global_mean = all_vals.mean() if len(all_vals) > 0 else 0
        global_std  = all_vals.std() if len(all_vals) > 1 else 1
        class_rt = df.loc[mask_cls, "RT"].dropna()
        median_rt = class_rt.median() if len(class_rt) else np.nan

        for poly in POLYMERS:
            col = f"Log2FC_{poly}_R"
            if col not in df.columns:
                continue
            vals = df.loc[mask_cls, col].dropna()
            if len(vals) < 3:
                continue
            med = vals.median()
            bubble_rows.append({"MetaboliteClass": cls, "Number_of_Features": len(vals),
                                "Median_Log2FC": med, "Enriched_Features": int((vals > 0).sum()),
                                "Enriched_Fraction": (vals > 0).mean(), "Mode": mode_label,
                                "Polymer": poly, "Median_RT": median_rt, "Class_Index": cls_idx,
                                "z_score_Median_Log2FC": (med - global_mean) / global_std if global_std > 0 else 0})
df_bubble = pd.DataFrame(bubble_rows)


# GSEA, HILIC Track 2
print("\nGSEA (HILIC Track 2, pristine)")


def flag_col(cls):
    return f"is_{cls.replace(' ', '_').replace('&', 'and')}"


gsea_frames, es_curve_rows, es_summary_rows = [], [], []
for poly in POLYMERS:
    fc_col = f"Log2FC_{poly}_R"
    if fc_col not in hilic_d.columns:
        continue
    sub = (hilic_d[["Feature", "MetaboliteClass", fc_col]].dropna(subset=[fc_col])
           .sort_values(fc_col, ascending=False).reset_index(drop=True)
           .rename(columns={fc_col: "Log2FC"}))
    sub["Rank"] = range(1, len(sub) + 1)
    sub["Polymer"], sub["PolymerType"] = poly, polymer_type(poly)

    for cls in HILIC_TRACK2:
        mask = sub["MetaboliteClass"] == cls
        sub[flag_col(cls)] = mask.astype(int)
        if mask.sum() < 5:
            continue
        scores, es, es_idx = running_enrichment_score(sub["Log2FC"].values, mask.values)
        for i in range(len(scores)):
            es_curve_rows.append({"Polymer": poly, "PolymerType": polymer_type(poly), "Class": cls,
                                  "Rank": i + 1, "Rank_Fraction": (i + 1) / len(scores),
                                  "Running_ES": scores[i], "Log2FC_at_rank": sub["Log2FC"].iloc[i]})
        es_summary_rows.append({"Polymer": poly, "PolymerType": polymer_type(poly), "Class": cls,
                                "n_class": int(mask.sum()), "n_total": len(sub),
                                "ES": es, "ES_rank": es_idx + 1})
    gsea_frames.append(sub)

df_gsea       = pd.concat(gsea_frames, ignore_index=True)
df_es_curves  = pd.DataFrame(es_curve_rows)
df_es_summary = pd.DataFrame(es_summary_rows)


# Export
print(f"\nExporting to: {OUT_DIR}")
guide = pd.DataFrame([
    ("HILIC_PCA_Pristine / RP_PCA_Pristine", "Features x 5 pristine (R) polymer Log2FC columns."),
    ("HILIC_PCA_AllCond / RP_PCA_AllCond",   "Features x 25 nanotube Log2FC columns."),
    ("Volcano_DNA_vs_PEG", f"Log2FC_DNA_vs_PEG = mean(AT, CTT, GT) - mean(DMG, DSPE), pristine. "
                           f"Welch t-test or IVW-propagated SE; BH-FDR q per mode. "
                           f"Significance: |FC| > {FC_THRESHOLD}, q < {Q_THRESHOLD}."),
    ("Volcano_Summary", "Counts per mode x significance."),
    ("RT_Distribution", "RT of significant features from the ssDNA vs PEG volcano."),
    ("Bubble_PerPolymer", "Per polymer x class (pristine). RP classes + HILIC Track 1."),
    ("GSEA_PerPolymer_Ranked", "Ranked features per polymer with Track 2 class flags."),
    ("GSEA_RunningES_PerPoly / GSEA_ES_Summary_PerPoly", "Running ES curves and ES per polymer x class. "
                                             "Positive ES = class at the high-FC end."),
    ("polymer_effect_plot_data.xlsx", "Clipped volcano regions (|log2FC| <= 4, -log10 q <= 8) and RT lists / 0.5-min bin fractions."),
], columns=["Sheet", "Content"])

with pd.ExcelWriter(OUT_DIR / "polymer_effect.xlsx", engine="openpyxl") as w:
    for name, df_pca in pca_sheets.items():
        df_pca.to_excel(w, sheet_name=name, index=False)
    df_volcano.to_excel(w,        sheet_name="Volcano_DNA_vs_PEG",       index=False)
    volcano_summary.to_excel(w,   sheet_name="Volcano_Summary",          index=False)
    df_rt.to_excel(w,             sheet_name="RT_Distribution",          index=False)
    df_bubble.to_excel(w,         sheet_name="Bubble_PerPolymer",        index=False)
    df_gsea.to_excel(w,           sheet_name="GSEA_PerPolymer_Ranked",   index=False)
    df_es_curves.to_excel(w,      sheet_name="GSEA_RunningES_PerPoly",   index=False)
    df_es_summary.to_excel(w,     sheet_name="GSEA_ES_Summary_PerPoly",  index=False)
    guide.to_excel(w,             sheet_name="Guide",                    index=False)

with pd.ExcelWriter(OUT_DIR / "polymer_effect_plot_data.xlsx", engine="openpyxl") as w:
    for mode_label in ["RP", "HILIC"]:
        volcano_region_table(mode_label).to_excel(w, sheet_name=f"Volcano_{mode_label}", index=False)
        rt_raw_table(mode_label).to_excel(w, sheet_name=f"RT_raw_{mode_label}", index=False)
        rt_binned_fraction(mode_label).to_excel(w, sheet_name=f"RT_binned_{mode_label}", index=False)

print("Done.")
