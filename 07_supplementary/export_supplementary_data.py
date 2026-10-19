"""
Supplementary data tables.

Input:  HILIC_ALL_Log2FC.xlsx, RP_ALL_Log2FC.xlsx, metabolite_class_predictions.xlsx
Output: supplementary_data.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (DEFECTS, FC_THRESHOLD, HILIC_LOG2FC,
                    HILIC_TRACK1, HILIC_TRACK2, OUT_SUPPLEMENTARY, POLYMERS, PRED_PATH,
                    Q_THRESHOLD, QWD_DEFECTS, RP_CLASSES, RP_LOG2FC)
from utils.dedup_features import add_dedup_group, deduplicate
from utils.stats import bh_fdr

OUT_SUPPLEMENTARY.mkdir(parents=True, exist_ok=True)
OUT_PATH    = OUT_SUPPLEMENTARY / "supplementary_data.xlsx"
DEFECTS_QWD = QWD_DEFECTS


# Helpers
def load_and_join_rt(log2fc_path, pred_path, mode, sheet='NT_vs_Plasma'):
    df = pd.read_excel(log2fc_path, sheet_name=sheet)
    pred = pd.read_excel(pred_path, sheet_name=f'{mode}_Predictions')
    pred['FeatureID'] = pred['FeatureID'].astype(str)
    rt_lookup = pred.set_index('FeatureID')[['RT', 'MZ', 'Polarity']].to_dict('index')
    df['Feature'] = df['Feature'].astype(str)
    df['RT'] = df['Feature'].map(lambda f: rt_lookup.get(f, {}).get('RT', np.nan))
    df['MZ'] = df['Feature'].map(lambda f: rt_lookup.get(f, {}).get('MZ', np.nan))
    df['Polarity'] = df['Feature'].map(lambda f: rt_lookup.get(f, {}).get('Polarity', np.nan))
    return df


# Load data
print("Loading data...")
rp_nt = load_and_join_rt(RP_LOG2FC, PRED_PATH, 'RP', 'NT_vs_Plasma')
hilic_nt = load_and_join_rt(HILIC_LOG2FC, PRED_PATH, 'HILIC', 'NT_vs_Plasma')
rp_qwd = load_and_join_rt(RP_LOG2FC, PRED_PATH, 'RP', 'QWD_vs_Pristine')
hilic_qwd = load_and_join_rt(HILIC_LOG2FC, PRED_PATH, 'HILIC', 'QWD_vs_Pristine')

# Full feature sets
print("Adding DedupGroups...")
rp_nt = add_dedup_group(rp_nt)
hilic_nt = add_dedup_group(hilic_nt)
rp_qwd = add_dedup_group(rp_qwd)
hilic_qwd = add_dedup_group(hilic_qwd)

# Deduplicated
print("Deduplicating...")
rp_nt_d = deduplicate(rp_nt, method='median')
hilic_nt_d = deduplicate(hilic_nt, method='median')
rp_qwd_d = deduplicate(rp_qwd, method='median')
hilic_qwd_d = deduplicate(hilic_qwd, method='median')

# Re-join RT/MZ after dedup
for df_d, df_orig in [(rp_nt_d, rp_nt), (hilic_nt_d, hilic_nt),
                       (rp_qwd_d, rp_qwd), (hilic_qwd_d, hilic_qwd)]:
    if 'DedupGroup' in df_orig.columns:
        rt_g = df_orig.groupby('DedupGroup')['RT'].median()
        mz_g = df_orig.groupby('DedupGroup')['MZ'].median()
        df_d['RT'] = df_d['Feature'].map(rt_g)
        df_d['MZ'] = df_d['Feature'].map(mz_g)
    else:
        # Fallback: direct feature-level lookup
        rt_map = df_orig.set_index('Feature')['RT'].to_dict()
        mz_map = df_orig.set_index('Feature')['MZ'].to_dict()
        df_d['RT'] = df_d['Feature'].map(rt_map)
        df_d['MZ'] = df_d['Feature'].map(mz_map)

# BH-FDR for the ALL_NT test (per mode)
for df_d in [rp_nt_d, hilic_nt_d]:
    df_d['q_ALL_NT'] = bh_fdr(df_d['p_ALL_NT'].values)

# Predictions
pred_rp = pd.read_excel(PRED_PATH, sheet_name='RP_Predictions')
pred_hilic = pd.read_excel(PRED_PATH, sheet_name='HILIC_Predictions')

print("Data loaded.\n")


# Supplementary tables
sheets = {}

# IVW Log2FC summary
print("IVW summary...")
st2_rows = []
for mode, df_full, df_ded in [('RP', rp_nt, rp_nt_d), ('HILIC', hilic_nt, hilic_nt_d)]:
    allnt = df_ded['Log2FC_ALL_NT'].dropna()
    n_sig = ((df_ded['q_ALL_NT'] < Q_THRESHOLD) & (df_ded['Log2FC_ALL_NT'].abs() > FC_THRESHOLD)).sum()
    st2_rows.append({
        'Mode': mode,
        'Total_features_IVW': len(df_full),
        'Total_features_dedup': len(df_ded),
        'Features_with_ALL_NT': len(allnt),
        'Median_Log2FC_ALL_NT': allnt.median(),
        'Mean_Log2FC_ALL_NT': allnt.mean(),
        'Sig_features_q05_FC1': int(n_sig),
        'Pct_sig': 100 * n_sig / len(allnt) if len(allnt) > 0 else 0,
    })
    # Per condition
    for poly in POLYMERS:
        for defect in DEFECTS:
            fc_col = f'Log2FC_{poly}_{defect}'
            p_col = f'p_{poly}_{defect}'
            if fc_col in df_ded.columns:
                vals = df_ded[fc_col].dropna()
                st2_rows.append({
                    'Mode': mode,
                    'Condition': f'{poly}_{defect}',
                    'n_features': len(vals),
                    'Median_Log2FC': vals.median() if len(vals) > 0 else np.nan,
                    'Mean_Log2FC': vals.mean() if len(vals) > 0 else np.nan,
                })

sheets['IVW_Log2FC_Summary'] = pd.DataFrame(st2_rows)

# Deduplication counts
print("Dedup details...")
st3_rows = []
for mode, df_full, df_ded, classes in [
    ('RP', rp_nt, rp_nt_d, RP_CLASSES),
    ('HILIC', hilic_nt, hilic_nt_d, HILIC_TRACK1 + HILIC_TRACK2),
]:
    for cls in classes:
        n_full = (df_full['MetaboliteClass'] == cls).sum()
        n_ded = (df_ded['MetaboliteClass'] == cls).sum()
        n_labeled_full = ((df_full['MetaboliteClass'] == cls) &
                          (df_full['ClassSource'] == 'Labeled')).sum()
        n_labeled_ded = ((df_ded['MetaboliteClass'] == cls) &
                         (df_ded['ClassSource'] == 'Labeled')).sum()
        st3_rows.append({
            'Mode': mode,
            'Class': cls,
            'n_full': n_full,
            'n_dedup': n_ded,
            'n_collapsed': n_full - n_ded,
            'pct_reduction': 100 * (n_full - n_ded) / n_full if n_full > 0 else 0,
            'n_labeled_full': n_labeled_full,
            'n_labeled_dedup': n_labeled_ded,
        })
    # Total row
    st3_rows.append({
        'Mode': mode, 'Class': 'TOTAL',
        'n_full': len(df_full), 'n_dedup': len(df_ded),
        'n_collapsed': len(df_full) - len(df_ded),
        'pct_reduction': 100 * (len(df_full) - len(df_ded)) / len(df_full),
    })

sheets['Dedup_Counts_PerClass'] = pd.DataFrame(st3_rows)

# Dedup sensitivity
print("Dedup sensitivity...")
st5_rows = []
for mode, df_full, df_ded, classes in [
    ('RP', rp_nt, rp_nt_d, RP_CLASSES),
    ('HILIC', hilic_nt, hilic_nt_d, HILIC_TRACK1),
]:
    for cls in classes:
        for label, df_src in [('Full', df_full), ('Dedup', df_ded)]:
            vals = df_src.loc[df_src['MetaboliteClass'] == cls, 'Log2FC_ALL_NT'].dropna()
            if len(vals) < 3:
                continue
            st5_rows.append({
                'Mode': mode,
                'Class': cls,
                'Dataset': label,
                'n_features': len(vals),
                'median_Log2FC': vals.median(),
                'q25': vals.quantile(0.25),
                'q75': vals.quantile(0.75),
                'pct_positive': 100 * (vals > 0).mean(),
                'direction': 'Enriched' if vals.median() > 0 else 'Depleted',
            })

df_st5 = pd.DataFrame(st5_rows)
# Add same_direction flag
pivot = df_st5.pivot_table(index=['Mode', 'Class'], columns='Dataset',
                            values='direction', aggfunc='first')
if 'Full' in pivot.columns and 'Dedup' in pivot.columns:
    pivot['Same_Direction'] = pivot['Full'] == pivot['Dedup']
    sheets['Dedup_Sensitivity'] = pd.DataFrame(st5_rows)
    sheets['Dedup_Direction_Check'] = pivot.reset_index()

# Supplementary figure data

# Per-feature significance proportions by class
print("Significance proportions...")
sf7_rows = []
for mode, df, classes in [
    ('RP', rp_nt_d, RP_CLASSES),
    ('HILIC', hilic_nt_d, HILIC_TRACK1),
]:
    for cls in classes:
        sub = df[df['MetaboliteClass'] == cls].dropna(subset=['Log2FC_ALL_NT', 'q_ALL_NT'])
        n = len(sub)
        if n == 0:
            continue
        n_sig_enr = int(((sub['q_ALL_NT'] < Q_THRESHOLD) & (sub['Log2FC_ALL_NT'] > 0)).sum())
        n_sig_dep = int(((sub['q_ALL_NT'] < Q_THRESHOLD) & (sub['Log2FC_ALL_NT'] < 0)).sum())
        n_ns = n - n_sig_enr - n_sig_dep
        sf7_rows.append({
            'Mode': mode, 'Class': cls, 'n_total': n,
            'n_sig_enriched': n_sig_enr, 'pct_sig_enriched': 100 * n_sig_enr / n,
            'n_sig_depleted': n_sig_dep, 'pct_sig_depleted': 100 * n_sig_dep / n,
            'n_nonsig': n_ns, 'pct_nonsig': 100 * n_ns / n,
        })

sheets['Significance_Props_PerClass'] = pd.DataFrame(sf7_rows)

# RP vs HILIC phospholipid subpopulations
print("Phospholipid subpopulations...")
sf10_rows = []
for mode, df in [('RP', rp_nt_d), ('HILIC', hilic_nt_d)]:
    sub = df[df['MetaboliteClass'] == 'Phospholipids']
    vals_mz = sub['MZ'].dropna()
    vals_rt = sub['RT'].dropna()
    vals_fc = sub['Log2FC_ALL_NT'].dropna()
    n_labeled = (sub['ClassSource'] == 'Labeled').sum()

    sf10_rows.append({
        'Mode': mode,
        'n_features': len(sub),
        'n_labeled': n_labeled,
        'n_predicted': len(sub) - n_labeled,
        'median_MZ': vals_mz.median() if len(vals_mz) > 0 else np.nan,
        'q25_MZ': vals_mz.quantile(0.25) if len(vals_mz) > 0 else np.nan,
        'q75_MZ': vals_mz.quantile(0.75) if len(vals_mz) > 0 else np.nan,
        'median_RT': vals_rt.median() if len(vals_rt) > 0 else np.nan,
        'median_Log2FC_ALL_NT': vals_fc.median() if len(vals_fc) > 0 else np.nan,
    })

# Per-feature m/z and RT
sf10_features = []
for mode, df in [('RP', rp_nt_d), ('HILIC', hilic_nt_d)]:
    sub = df[df['MetaboliteClass'] == 'Phospholipids']
    for _, row in sub.iterrows():
        if pd.notna(row.get('MZ')):
            sf10_features.append({
                'Mode': mode,
                'Feature': row['Feature'],
                'MZ': row.get('MZ', np.nan),
                'RT': row.get('RT', np.nan),
                'ClassSource': row.get('ClassSource', np.nan),
                'Log2FC_ALL_NT': row.get('Log2FC_ALL_NT', np.nan),
            })

sheets['PL_RPvsHILIC_Summary'] = pd.DataFrame(sf10_rows)
sheets['PL_RPvsHILIC_Features'] = pd.DataFrame(sf10_features)

# QWD response comparison for phospholipids
sf10_qwd = []
for mode, df in [('RP', rp_qwd_d), ('HILIC', hilic_qwd_d)]:
    mask = df['MetaboliteClass'] == 'Phospholipids'
    for poly in POLYMERS:
        for d in DEFECTS_QWD:
            col = f'Log2FC_{poly}_{d}'
            if col in df.columns:
                vals = df.loc[mask, col].dropna()
                if len(vals) >= 3:
                    sf10_qwd.append({
                        'Mode': mode, 'Polymer': poly, 'Defect': d,
                        'n': len(vals), 'median_Log2FC': vals.median(),
                    })

sheets['PL_RPvsHILIC_QWD_Response'] = pd.DataFrame(sf10_qwd)

# Guide sheet
guide_rows = [
    ('IVW_Log2FC_Summary', 'IVW Log2FC summary per mode and per nanotube (significance: BH-FDR q < 0.05, |log2FC| > 1).'),
    ('Dedup_Counts_PerClass', 'Feature counts per class before and after name-based deduplication.'),
    ('Dedup_Sensitivity', 'Class-level enrichment with full vs deduplicated feature sets.'),
    ('Dedup_Direction_Check', 'Whether the direction of class-level enrichment is preserved after deduplication.'),
    ('Significance_Props_PerClass', 'Per-class proportions of enriched / depleted / non-significant features (BH-FDR q < 0.05).'),
    ('PL_RPvsHILIC_Summary', 'RP vs HILIC phospholipid subpopulations: summary.'),
    ('PL_RPvsHILIC_Features', 'RP vs HILIC phospholipid subpopulations: individual features (m/z, RT).'),
    ('PL_RPvsHILIC_QWD_Response', 'RP vs HILIC phospholipid subpopulations: QWD response per mode.'),
]
sheets['Guide'] = pd.DataFrame(guide_rows, columns=['Sheet', 'Description'])


# Export
print(f"\nExporting {len(sheets)} sheets to: {OUT_PATH}")
with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    for name, df_out in sheets.items():
        # Truncate sheet name to 31 chars (Excel limit)
        sheet_name = name[:31]
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  {sheet_name}: {len(df_out)} rows")

print("\nDone.")