"""
Share of features per class affected by QWDs.

Input:  HILIC_ALL_Log2FC.xlsx, RP_ALL_Log2FC.xlsx
Output: qwd_affected_fraction.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (HILIC_LOG2FC, HILIC_TRACK1, OUT_FIGURES, POLYMERS, Q_THRESHOLD,
                    QWD_DEFECTS, QWD_FC_THRESHOLD, RP_CLASSES, RP_LOG2FC)
from utils.stats import bh_fdr

OUT_DIR = OUT_FIGURES / "qwd_affected_fraction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot order (top to bottom is reversed in barh)
RP_ORDER    = ["Acylcarnitines", "Fatty Acids", "Lysophospholipids", "Phospholipids", "Complex Lipids"]
HILIC_ORDER = ["Fatty Acids & Amides", "Phospholipids", "Lysophospholipids",
               "Secondary & Xenobiotic", "Organic Acids"]
assert set(RP_ORDER) == set(RP_CLASSES) and set(HILIC_ORDER) == set(HILIC_TRACK1)

PAIRS = [(p, d) for p in POLYMERS for d in QWD_DEFECTS]


def add_qvalue_columns(df):
    """BH-FDR per polymer x defect comparison, across features."""
    df = df.copy()
    for poly, defect in PAIRS:
        if f"p_{poly}_{defect}" in df.columns:
            df[f"q_{poly}_{defect}"] = bh_fdr(df[f"p_{poly}_{defect}"].values)
    return df


def classify_feature(row):
    sig_dirs = []
    for poly, defect in PAIRS:
        fc, q = row.get(f"Log2FC_{poly}_{defect}", np.nan), row.get(f"q_{poly}_{defect}", np.nan)
        if pd.isna(fc) or pd.isna(q):
            continue
        if q < Q_THRESHOLD and abs(fc) > QWD_FC_THRESHOLD:
            sig_dirs.append(1 if fc > 0 else -1)
    if not sig_dirs:
        return "not_affected"
    if all(d > 0 for d in sig_dirs):
        return "enriched"
    if all(d < 0 for d in sig_dirs):
        return "depleted"
    return "mixed"


def compute_fractions(df, classes):
    df = df[df["MetaboliteClass"].isin(classes)].copy()
    df = df[~df["ClassSource"].str.startswith("Low", na=False)].copy()
    df["_label"] = df.apply(classify_feature, axis=1)

    rows = []
    for cls in classes:
        sub = df[df["MetaboliteClass"] == cls]
        n = len(sub)
        if n == 0:
            rows.append({"MetaboliteClass": cls, "n_total": 0,
                         "pct_enriched": 0, "pct_depleted": 0, "pct_mixed": 0})
            continue
        counts = sub["_label"].value_counts()
        n_enr, n_dep, n_mix = counts.get("enriched", 0), counts.get("depleted", 0), counts.get("mixed", 0)
        n_aff = n - counts.get("not_affected", 0)
        rows.append({"MetaboliteClass": cls, "n_total": n, "n_affected": n_aff,
                     "n_enriched": n_enr, "n_depleted": n_dep, "n_mixed": n_mix,
                     "pct_enriched": 100 * n_enr / n, "pct_depleted": 100 * n_dep / n,
                     "pct_mixed": 100 * n_mix / n, "pct_affected": 100 * n_aff / n})
    return pd.DataFrame(rows)


print("Loading data and applying BH-FDR...")
rp_df    = add_qvalue_columns(pd.read_excel(RP_LOG2FC,    sheet_name="QWD_vs_Pristine"))
hilic_df = add_qvalue_columns(pd.read_excel(HILIC_LOG2FC, sheet_name="QWD_vs_Pristine"))

frac_rp    = compute_fractions(rp_df,    RP_ORDER)
frac_hilic = compute_fractions(hilic_df, HILIC_ORDER)
for label, frac in [("RP", frac_rp), ("HILIC Track 1", frac_hilic)]:
    print(f"\n{label}:")
    print(frac[["MetaboliteClass", "n_total", "pct_enriched", "pct_depleted", "pct_mixed", "pct_affected"]])

frac_rp["Mode"], frac_hilic["Mode"] = "RP", "HILIC"
pd.concat([frac_rp, frac_hilic], ignore_index=True).to_excel(OUT_DIR / "qwd_affected_fraction.xlsx", index=False)
print(f"\nSaved: {OUT_DIR}")
