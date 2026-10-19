"""
Class-level enrichment, corona vs plasma (labeled vs ML-predicted features).

Input:  HILIC_ALL_Log2FC.xlsx, RP_ALL_Log2FC.xlsx
Output: class_enrichment.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (DEFECTS, HILIC_LOG2FC, HILIC_TRACK1, HILIC_TRACK2,
                    OUT_FIGURES, POLYMERS, Q_THRESHOLD, RP_CLASSES, RP_LOG2FC)
from utils.dedup_features import add_dedup_group, deduplicate
from utils.stats import bh_fdr

OUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUT_FIGURES / "class_enrichment.xlsx"
CONDITIONS = [f"{p}_{d}" for p in POLYMERS for d in DEFECTS]


def prepare_subsets(df):
    if "DedupGroup" not in df.columns:
        df = add_dedup_group(df)
    df_all     = deduplicate(df, method="median")
    df_all["q_ALL_NT"] = bh_fdr(df_all["p_ALL_NT"].values)
    df_labeled = df[df["ClassSource"] == "Labeled"].copy()
    df_labeled = add_dedup_group(df_labeled) if "DedupGroup" not in df_labeled.columns else df_labeled
    df_ml      = df[df["ClassSource"].str.startswith("ML", na=False)].copy()
    df_ml      = add_dedup_group(df_ml) if "DedupGroup" not in df_ml.columns else df_ml
    return df, df_all, deduplicate(df_labeled, method="median"), deduplicate(df_ml, method="median")


def class_stats_allnt(df, cls):
    vals = df[df["MetaboliteClass"] == cls]["Log2FC_ALL_NT"].dropna()
    if len(vals) == 0:
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan,
                "q10": np.nan, "q90": np.nan, "pct_positive": np.nan}
    return {"n": len(vals), "median": vals.median(),
            "q25": vals.quantile(0.25), "q75": vals.quantile(0.75),
            "q10": vals.quantile(0.10), "q90": vals.quantile(0.90),
            "pct_positive": 100 * (vals > 0).mean()}


def class_stats_significance(df, cls):
    both = df[df["MetaboliteClass"] == cls].dropna(subset=["Log2FC_ALL_NT", "q_ALL_NT"])
    n = len(both)
    if n == 0:
        return {"n_tested": 0, "n_sig_enriched": 0, "n_sig_depleted": 0, "n_nonsig": 0}
    return {
        "n_tested":       n,
        "n_sig_enriched": int(((both["q_ALL_NT"] < Q_THRESHOLD) & (both["Log2FC_ALL_NT"] > 0)).sum()),
        "n_sig_depleted": int(((both["q_ALL_NT"] < Q_THRESHOLD) & (both["Log2FC_ALL_NT"] < 0)).sum()),
        "n_nonsig":       int((both["q_ALL_NT"] >= Q_THRESHOLD).sum()),
    }


def class_stats_conditions(df, cls, conditions):
    sub = df[df["MetaboliteClass"] == cls]
    n_pos = n_tested = 0
    details = []
    for cond in conditions:
        col = f"Log2FC_{cond}"
        if col not in sub.columns:
            continue
        vals = sub[col].dropna()
        if len(vals) >= 3:
            n_tested += 1
            med = vals.median()
            if med > 0:
                n_pos += 1
            details.append({"Condition": cond, "median_Log2FC": med, "n": len(vals)})
    return {"n_conditions_tested": n_tested, "n_conditions_positive": n_pos,
            "pct_conditions_positive": 100 * n_pos / n_tested if n_tested > 0 else np.nan}, details


def build_class_summary(df_dedup, classes, mode_label):
    rows = []
    for cls in classes:
        s, sig = class_stats_allnt(df_dedup, cls), class_stats_significance(df_dedup, cls)
        cond, _ = class_stats_conditions(df_dedup, cls, CONDITIONS)
        rows.append({"Mode": mode_label, "Class": cls, **s, **sig, **cond})
    return rows


def build_split_summary(df_lab, df_ml, df_all, classes, mode_label):
    rows = []
    for cls in classes:
        for src, df_src in [("Labeled", df_lab), ("ML_predicted", df_ml), ("All", df_all)]:
            cond, _ = class_stats_conditions(df_src, cls, CONDITIONS)
            rows.append({"Mode": mode_label, "Class": cls, "Source": src,
                         **class_stats_allnt(df_src, cls), **cond})
    return rows


def build_condition_detail(df_lab, df_ml, df_all, classes, mode_label):
    rows = []
    for cls in classes:
        for src, df_src in [("Labeled", df_lab), ("ML_predicted", df_ml), ("All", df_all)]:
            _, details = class_stats_conditions(df_src, cls, CONDITIONS)
            for cd in details:
                rows.append({"Mode": mode_label, "Class": cls, "Source": src, **cd})
    return rows


def build_raw_features(df_lab, df_ml, classes, mode_label):
    rows = []
    for cls in classes:
        for src, df_src in [("Labeled", df_lab), ("ML_predicted", df_ml)]:
            sub = df_src[df_src["MetaboliteClass"] == cls]
            for _, row in sub.iterrows():
                val = row.get("Log2FC_ALL_NT", np.nan)
                if pd.notna(val):
                    rows.append({"Mode": mode_label, "Class": cls, "Source": src,
                                 "Feature": row["Feature"], "Log2FC_ALL_NT": val})
    return rows


# Main

print("Loading data...")
hilic_nt = pd.read_excel(HILIC_LOG2FC, sheet_name="NT_vs_Plasma")
rp_nt    = pd.read_excel(RP_LOG2FC,    sheet_name="NT_vs_Plasma")

print("Deduplicating...")
hilic_nt, hilic_all, hilic_lab, hilic_ml = prepare_subsets(hilic_nt)
rp_nt,    rp_all,   rp_lab,   rp_ml     = prepare_subsets(rp_nt)

hilic_t1_rows = build_class_summary(hilic_all, HILIC_TRACK1, "HILIC")
hilic_t2_rows = build_split_summary(hilic_lab, hilic_ml, hilic_all, HILIC_TRACK2, "HILIC")
hilic_t2_cond = build_condition_detail(hilic_lab, hilic_ml, hilic_all, HILIC_TRACK2, "HILIC")
hilic_t2_raw  = build_raw_features(hilic_lab, hilic_ml, HILIC_TRACK2, "HILIC")

rp_class_rows = build_class_summary(rp_all, RP_CLASSES, "RP")
rp_split_rows = build_split_summary(rp_lab, rp_ml, rp_all, RP_CLASSES, "RP")
rp_cond_rows  = build_condition_detail(rp_lab, rp_ml, rp_all, RP_CLASSES, "RP")
rp_raw_rows   = build_raw_features(rp_lab, rp_ml, RP_CLASSES, "RP")

print(f"Exporting to: {OUT_PATH}")
with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    pd.DataFrame(hilic_t1_rows + rp_class_rows).to_excel(writer, sheet_name="ClassLevel_Summary",    index=False)
    pd.DataFrame(hilic_t2_rows + rp_split_rows).to_excel(writer, sheet_name="SplitBySource_Summary", index=False)
    pd.DataFrame(hilic_t2_cond + rp_cond_rows).to_excel( writer, sheet_name="PerCondition_Detail",   index=False)
    pd.DataFrame(hilic_t2_raw  + rp_raw_rows).to_excel(  writer, sheet_name="RawPerFeature",         index=False)
print("Done.")
